#!/usr/bin/env python3
"""Rescue Type-II SGNs in M29L/R without modifying existing project data.

The current M29 segmentations were formed from CR and Ntng1 detections. This script adds
non-overlapping detections from the existing PV-derived SGN_v2 segmentation, measures the CR and
Ntng1 median intensities needed for subtype classification, reapplies the current spatially varying
thresholds, and compares subtype fractions before and after the rescue.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import threading
from concurrent import futures
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import zarr
from numcodecs import GZip
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

from flamingo_tools.intensity_annotation import eval_annotations as eval_utils
from flamingo_tools.postprocessing.sgn_subtype_utils import STAIN_TO_TYPE
from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, create_s3_target, get_s3_path


DATA_ROOT = Path("/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet")
MOBIE_ROOT = DATA_ROOT / "mobie_project" / "cochlea-lightsheet"
PREDICTION_ROOT = DATA_ROOT / "predictions"
THRESHOLD_ROOT = MOBIE_ROOT / "tables" / "annotator_variance"
SCRIPT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_ROOT / "output"

COCHLEAE = ("M_AMD_N190_L", "M_AMD_N190_R")
CURRENT_SEGMENTATION = "CR_Ntng1_SGN_v2"
PV_SEGMENTATION = "SGN_v2"
# M29R's SGN_v2 and CR_SGN_v2 prediction arrays were written under one another's names. The
# current CR+Ntng1 segmentation confirms this: its 13,634 max ID is 6,734 (the mislabeled
# SGN_v2/biological-CR array) + 6,900 Ntng1 IDs. The biological-PV array is therefore CR_SGN_v2.
PV_ARRAY_SEGMENTATION = {
    "M_AMD_N190_L": "SGN_v2",
    "M_AMD_N190_R": "CR_SGN_v2",
}
OUTPUT_SEGMENTATION = "CR_Ntng1_PV_SGN_v2"
STAINS = ("CR", "Ntng1")
VOXEL_SIZE = 0.38
OVERLAP_SCALE = "s2"
OVERLAP_THRESHOLD = 0.25
CHUNK_SHAPE = (64, 64, 64)
MAX_EDGE_DISTANCE = 30.0
MIN_SIZE = 1000
MIN_COMPONENT_LENGTH = 50
N_THREADS = int(os.environ.get("SLURM_CPUS_ON_NODE", "12"))

BASE_COLUMNS = [
    "label_id", "anchor_x", "anchor_y", "anchor_z",
    "bb_min_x", "bb_min_y", "bb_min_z", "bb_max_x", "bb_max_y", "bb_max_z", "n_pixels",
]
MAPPING_COLUMNS = ["offset", "length_fraction", "length[µm]", "frequency[kHz]"]
SUBTYPES = ("Type Ia", "Type Ib", "Type Ic", "Type II")


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value)}")


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_json_default)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def side(cochlea: str) -> str:
    return cochlea.rsplit("_", 1)[1]


def s3_key(cochlea: str, segmentation: str, scale: str) -> str:
    return f"{cochlea}/images/ome-zarr/{segmentation}.ome.zarr/{scale}"


def open_s3_array(cochlea: str, segmentation: str, scale: str):
    store, _ = get_s3_path(f"{cochlea}/images/ome-zarr/{segmentation}.ome.zarr")
    return zarr.open(store, mode="r")[scale]


def read_s3_table(fs, path: str) -> pd.DataFrame:
    with fs.open(f"{BUCKET_NAME}/{path}", "r") as f:
        return pd.read_csv(f, sep="\t")


def current_table(fs, cochlea: str) -> pd.DataFrame:
    return read_s3_table(fs, f"{cochlea}/tables/{CURRENT_SEGMENTATION}/default.tsv")


def current_measurements(fs, cochlea: str, stain: str) -> pd.DataFrame:
    name = f"{stain}_{CURRENT_SEGMENTATION.replace('_', '-')}_object-measures.tsv"
    return read_s3_table(fs, f"{cochlea}/tables/{CURRENT_SEGMENTATION}/{name}")


def threshold_path(cochlea: str, stain: str) -> Path:
    cochlea_name = cochlea.replace("_", "-")
    seg_name = CURRENT_SEGMENTATION.replace("_", "-")
    return THRESHOLD_ROOT / f"{cochlea_name}_{stain}_{seg_name}_annot.json"


def pv_array_path(cochlea: str) -> Path:
    return PREDICTION_ROOT / cochlea / PV_ARRAY_SEGMENTATION[cochlea] / "segmentation.zarr"


def pv_table_path(cochlea: str) -> Path:
    return MOBIE_ROOT / cochlea / "tables" / PV_SEGMENTATION / "default.tsv"


def validate_pv_table_array_pair(cochlea: str, table: pd.DataFrame, segmentation) -> None:
    """Verify deterministic table rows against the local array before any merge is attempted."""
    indices = np.linspace(0, len(table) - 1, min(24, len(table)), dtype=int)
    missing = []
    for _, row in table.iloc[indices].iterrows():
        bb = row_bbox(row, segmentation.shape)
        if not np.any(np.asarray(segmentation[bb]) == int(row.label_id)):
            missing.append(int(row.label_id))
    if missing:
        raise AssertionError(
            f"{cochlea}: PV table does not match {PV_ARRAY_SEGMENTATION[cochlea]} array; "
            f"sample IDs absent from their bounding boxes: {missing}"
        )


def chunk_bounds(coord: Tuple[int, int, int], shape: Sequence[int], chunks=CHUNK_SHAPE):
    begin = tuple(c * ch for c, ch in zip(coord, chunks))
    end = tuple(min(b + ch, sh) for b, ch, sh in zip(begin, chunks, shape))
    return tuple(slice(b, e) for b, e in zip(begin, end))


def remote_chunk_coords(fs, cochlea: str, segmentation: str, scale: str) -> set:
    prefix = f"{BUCKET_NAME}/{s3_key(cochlea, segmentation, scale)}"
    coords = set()
    for key in fs.find(prefix, detail=False):
        relative = key[len(prefix):].lstrip("/")
        parts = relative.replace(".", "/").split("/")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            coords.add(tuple(int(part) for part in parts))
    return coords


def local_chunk_coords(array_path: Path) -> set:
    data_path = array_path / "segmentation"
    coords = set()
    for path in data_path.iterdir():
        if path.name.startswith(".") or not path.is_file():
            continue
        parts = path.name.split(".")
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            coords.add(tuple(int(part) for part in parts))
    return coords


def table_chunk_coords(
    table: pd.DataFrame,
    shape: Sequence[int],
    chunks: Sequence[int],
    voxel_size: float = VOXEL_SIZE,
) -> set:
    """Resolve foreground candidate chunks from exact object bounding boxes.

    Some of the M29R OME-Zarr scales explicitly store compressed fill chunks, so store keys cannot
    be used to distinguish foreground from background. Bounding boxes are both faster and exact.
    """
    coords = set()
    for row in table.itertuples(index=False):
        mins = (
            max(0, int(math.floor(row.bb_min_z / voxel_size)) - 1),
            max(0, int(math.floor(row.bb_min_y / voxel_size)) - 1),
            max(0, int(math.floor(row.bb_min_x / voxel_size)) - 1),
        )
        maxs = (
            min(shape[0], int(math.ceil(row.bb_max_z / voxel_size)) + 1),
            min(shape[1], int(math.ceil(row.bb_max_y / voxel_size)) + 1),
            min(shape[2], int(math.ceil(row.bb_max_x / voxel_size)) + 1),
        )
        starts = tuple(value // chunk for value, chunk in zip(mins, chunks))
        stops = tuple((value - 1) // chunk for value, chunk in zip(maxs, chunks))
        for cz in range(starts[0], stops[0] + 1):
            for cy in range(starts[1], stops[1] + 1):
                for cx in range(starts[2], stops[2] + 1):
                    coords.add((cz, cy, cx))
    return coords


def encoded_background_labels(table: pd.DataFrame, shape: Sequence[int]) -> set:
    """Find labels that encode the image background rather than an SGN object.

    M29R's current CR+Ntng1 array contains one full-volume label (6734) in place of zero-valued
    background. Its table row covers more than half the entire image. This conservative size test
    detects that artifact without treating ordinary disconnected component-0 SGNs as background.
    """
    volume = int(np.prod(np.asarray(shape, dtype=np.int64)))
    return {
        int(label_id)
        for label_id in table.loc[table.n_pixels > (volume / 2), "label_id"].to_numpy()
    }


def expand_chunk_coords(
    source_coords: Iterable[Tuple[int, int, int]],
    source_chunks: Sequence[int],
    target_chunks: Sequence[int],
    shape: Sequence[int],
) -> set:
    assert all(src % dst == 0 for src, dst in zip(source_chunks, target_chunks))
    factors = tuple(src // dst for src, dst in zip(source_chunks, target_chunks))
    target_grid = tuple(math.ceil(sh / ch) for sh, ch in zip(shape, target_chunks))
    result = set()
    for coord in source_coords:
        start = tuple(c * factor for c, factor in zip(coord, factors))
        for dz in range(factors[0]):
            for dy in range(factors[1]):
                for dx in range(factors[2]):
                    target = (start[0] + dz, start[1] + dy, start[2] + dx)
                    if all(c < n for c, n in zip(target, target_grid)):
                        result.add(target)
    return result


def _overlap_chunk(
    a, b, coord, has_a: bool, has_b: bool, max_a: int, max_b: int,
    background_a: Sequence[int],
):
    bb = chunk_bounds(coord, a.shape, a.chunks)
    block_shape = tuple(sl.stop - sl.start for sl in bb)
    aa = np.asarray(a[bb]) if has_a else np.zeros(block_shape, dtype=np.uint64)
    bb_arr = np.asarray(b[bb]) if has_b else np.zeros(block_shape, dtype=np.uint64)
    for label_id in background_a:
        aa[aa == label_id] = 0

    a_values, a_counts = np.unique(aa[aa != 0], return_counts=True)
    b_values, b_counts = np.unique(bb_arr[bb_arr != 0], return_counts=True)

    mask = (aa != 0) & (bb_arr != 0)
    if mask.any():
        codes = aa[mask].astype(np.uint64) * np.uint64(max_b + 1) + bb_arr[mask].astype(np.uint64)
        pair_codes, pair_counts = np.unique(codes, return_counts=True)
    else:
        pair_codes = np.empty(0, dtype=np.uint64)
        pair_counts = np.empty(0, dtype=np.int64)
    return a_values, a_counts, b_values, b_counts, pair_codes, pair_counts


def select_pv_ids(fs, cochlea: str, out_dir: Path) -> dict:
    selection_path = out_dir / "selection.json"
    seg_a = open_s3_array(cochlea, CURRENT_SEGMENTATION, OVERLAP_SCALE)
    pv_array_name = PV_ARRAY_SEGMENTATION[cochlea]
    seg_b = open_s3_array(cochlea, pv_array_name, OVERLAP_SCALE)
    assert seg_a.shape == seg_b.shape
    assert tuple(seg_a.chunks) == tuple(seg_b.chunks) == CHUNK_SHAPE

    table_a_all = current_table(fs, cochlea)
    # Table sizes are measured at s0, so use the full-resolution shape for background detection.
    full_shape = open_s3_array(cochlea, CURRENT_SEGMENTATION, "s0").shape
    background_a = encoded_background_labels(table_a_all, full_shape)
    if selection_path.exists():
        with open(selection_path) as f:
            cached = json.load(f)
        if (
            cached.get("source_a_background_labels", []) == sorted(background_a)
            and cached.get("segmentation_b") == pv_array_name
        ):
            return cached
        print(f"[{cochlea}] Cached overlap selection predates background normalization; recomputing.", flush=True)

    print(f"[{cochlea}] Computing {OVERLAP_SCALE} overlaps.", flush=True)
    table_a = table_a_all[~table_a_all.label_id.isin(background_a)]
    table_b = pd.read_csv(pv_table_path(cochlea), sep="\t")
    max_a = int(table_a_all.label_id.max())
    max_b = int(table_b.label_id.max())

    # s2 is downsampled by four relative to the full-resolution tables.
    overlap_voxel_size = VOXEL_SIZE * 4
    chunks_a = table_chunk_coords(table_a, seg_a.shape, seg_a.chunks, overlap_voxel_size)
    chunks_b = table_chunk_coords(table_b, seg_b.shape, seg_b.chunks, overlap_voxel_size)
    coords = sorted(chunks_a | chunks_b)
    print(f"[{cochlea}] Reading {len(coords)} non-empty overlap-scale chunks.", flush=True)

    size_a = np.zeros(max_a + 1, dtype=np.uint64)
    size_b = np.zeros(max_b + 1, dtype=np.uint64)
    intersections: Dict[int, int] = {}

    def work(coord):
        return _overlap_chunk(
            seg_a, seg_b, coord, coord in chunks_a, coord in chunks_b, max_a=max_a, max_b=max_b,
            background_a=sorted(background_a),
        )

    with futures.ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        results = pool.map(work, coords)
        for result in tqdm(results, total=len(coords), desc=f"{cochlea} overlap chunks"):
            a_values, a_counts, b_values, b_counts, codes, counts = result
            size_a[a_values.astype(int)] += a_counts.astype(np.uint64)
            size_b[b_values.astype(int)] += b_counts.astype(np.uint64)
            for code, count in zip(codes, counts):
                code_int = int(code)
                intersections[code_int] = intersections.get(code_int, 0) + int(count)

    cumulative_iou = np.zeros(max_b + 1, dtype=np.float64)
    for code, intersection in intersections.items():
        a_id, b_id = divmod(code, max_b + 1)
        union = int(size_a[a_id]) + int(size_b[b_id]) - intersection
        cumulative_iou[b_id] += intersection / union

    all_ids_b = np.flatnonzero(size_b)
    selected = all_ids_b[cumulative_iou[all_ids_b] < OVERLAP_THRESHOLD]
    rejected = all_ids_b[cumulative_iou[all_ids_b] >= OVERLAP_THRESHOLD]
    observed_a = np.flatnonzero(size_a)
    assert len(observed_a) > 0 and int(observed_a.max()) == max_a

    result = {
        "cochlea": cochlea,
        "segmentation_a": CURRENT_SEGMENTATION,
        "segmentation_b": pv_array_name,
        "overlap_scale": OVERLAP_SCALE,
        "overlap_threshold": OVERLAP_THRESHOLD,
        "offset": max_a,
        "source_a_background_labels": sorted(background_a),
        "n_nonempty_chunks_a": len(chunks_a),
        "n_nonempty_chunks_b": len(chunks_b),
        "n_pv_ids_at_overlap_scale": len(all_ids_b),
        "n_selected_pv_ids": len(selected),
        "n_rejected_pv_ids": len(rejected),
        "selected_pv_ids": selected.tolist(),
        "selected_cumulative_iou_summary": {
            "min": float(cumulative_iou[selected].min()) if len(selected) else None,
            "median": float(np.median(cumulative_iou[selected])) if len(selected) else None,
            "max": float(cumulative_iou[selected].max()) if len(selected) else None,
        },
        "rejected_cumulative_iou_summary": {
            "min": float(cumulative_iou[rejected].min()) if len(rejected) else None,
            "median": float(np.median(cumulative_iou[rejected])) if len(rejected) else None,
            "max": float(cumulative_iou[rejected].max()) if len(rejected) else None,
        },
    }
    write_json(selection_path, result)
    return result


def merge_full_resolution(fs, cochlea: str, out_dir: Path, selection: dict) -> Path:
    seg_dir = out_dir / OUTPUT_SEGMENTATION
    output_path = seg_dir / "segmentation.zarr"
    success_path = output_path / "_SUCCESS"
    progress_path = output_path / "completed_chunks.txt"
    seg_dir.mkdir(parents=True, exist_ok=True)

    seg_a = open_s3_array(cochlea, CURRENT_SEGMENTATION, "s0")
    pv_container = zarr.open(str(pv_array_path(cochlea)), mode="r")
    seg_b = pv_container["segmentation"]
    assert seg_a.shape == seg_b.shape
    pv_table = pd.read_csv(pv_table_path(cochlea), sep="\t")
    validate_pv_table_array_pair(cochlea, pv_table, seg_b)

    table_a = current_table(fs, cochlea)
    background_a = set(selection.get("source_a_background_labels", []))
    foreground_a = table_a[~table_a.label_id.isin(background_a)]
    chunks_a = table_chunk_coords(foreground_a, seg_a.shape, CHUNK_SHAPE)
    native_b = local_chunk_coords(pv_array_path(cochlea))
    chunks_b = expand_chunk_coords(native_b, seg_b.chunks, CHUNK_SHAPE, seg_b.shape)
    coords = sorted(chunks_a | chunks_b)

    if success_path.exists():
        print(f"[{cochlea}] Full-resolution merge already complete.", flush=True)
        return output_path

    if output_path.exists():
        group = zarr.open_group(str(output_path), mode="a")
        if "segmentation" not in group:
            raise RuntimeError(f"Incomplete output without segmentation array: {output_path}")
        output = group["segmentation"]
    else:
        group = zarr.open_group(str(output_path), mode="w", zarr_format=2)
        output = group.create_array(
            "segmentation", shape=seg_a.shape, chunks=CHUNK_SHAPE, dtype=np.uint64,
            compressor=GZip(level=1), fill_value=0,
        )
        group.attrs.update({
            "cochlea": cochlea,
            "source_a": f"s3://{BUCKET_NAME}/{s3_key(cochlea, CURRENT_SEGMENTATION, 's0')}",
            "source_b": str(pv_array_path(cochlea)),
            "overlap_scale": OVERLAP_SCALE,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "pv_label_offset": int(selection["offset"]),
            "normalized_source_a_background_labels": sorted(background_a),
        })

    completed = set()
    if progress_path.exists():
        with open(progress_path) as f:
            completed = {tuple(map(int, line.strip().split("."))) for line in f if line.strip()}
    pending = [coord for coord in coords if coord not in completed]
    print(
        f"[{cochlea}] Merging {len(pending)}/{len(coords)} full-resolution non-empty candidate chunks.",
        flush=True,
    )

    selected = np.asarray(selection["selected_pv_ids"], dtype=np.int64)
    max_b = int(pv_table.label_id.max())
    selected_lut = np.zeros(max_b + 1, dtype=bool)
    selected_lut[selected] = True
    offset = int(selection["offset"])
    write_lock = threading.Lock()
    inserted_voxels = 0

    def merge_chunk(coord):
        bb = chunk_bounds(coord, seg_a.shape, CHUNK_SHAPE)
        block_shape = tuple(sl.stop - sl.start for sl in bb)
        block_a = np.asarray(seg_a[bb]) if coord in chunks_a else np.zeros(block_shape, dtype=np.uint64)
        for label_id in background_a:
            block_a[block_a == label_id] = 0
        if coord in chunks_b:
            block_b = np.asarray(seg_b[bb])
            insert = selected_lut[block_b] & (block_a == 0)
            n_inserted = int(insert.sum())
            if n_inserted:
                block_a[insert] = block_b[insert].astype(np.uint64) + np.uint64(offset)
        else:
            n_inserted = 0
        if np.any(block_a):
            output[bb] = block_a
        with write_lock:
            with open(progress_path, "a") as f:
                f.write(".".join(map(str, coord)) + "\n")
        return n_inserted

    with futures.ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        for count in tqdm(
            pool.map(merge_chunk, pending), total=len(pending), desc=f"{cochlea} full-resolution merge"
        ):
            inserted_voxels += count

    # On a resumed run, recompute the total from the exact added-object table later; this value is
    # only a progress diagnostic.
    with open(success_path, "w") as f:
        f.write(datetime.now(timezone.utc).isoformat() + "\n")
    print(f"[{cochlea}] Merge complete; inserted {inserted_voxels} voxels in this run.", flush=True)
    return output_path


def row_bbox(row: pd.Series, shape: Sequence[int]) -> Tuple[slice, slice, slice]:
    bb_min = [row.bb_min_z / VOXEL_SIZE, row.bb_min_y / VOXEL_SIZE, row.bb_min_x / VOXEL_SIZE]
    bb_max = [row.bb_max_z / VOXEL_SIZE, row.bb_max_y / VOXEL_SIZE, row.bb_max_x / VOXEL_SIZE]
    begin = [max(0, int(round(value)) - 1) for value in bb_min]
    end = [min(shape[i], int(round(value)) + 1) for i, value in enumerate(bb_max)]
    return tuple(slice(b, e) for b, e in zip(begin, end))


def exact_added_row(output, source_row: pd.Series, offset: int) -> dict | None:
    new_id = int(source_row.label_id) + offset
    bb = row_bbox(source_row, output.shape)
    block = np.asarray(output[bb])
    coords = np.nonzero(block == new_id)
    if len(coords[0]) == 0:
        return None
    starts = np.asarray([sl.start for sl in bb], dtype=np.int64)
    global_coords = [axis.astype(np.int64) + starts[num] for num, axis in enumerate(coords)]
    mins = np.asarray([axis.min() for axis in global_coords])
    maxs = np.asarray([axis.max() + 1 for axis in global_coords])
    centers = np.asarray([axis.mean() for axis in global_coords])
    return {
        "label_id": new_id,
        "anchor_x": centers[2] * VOXEL_SIZE,
        "anchor_y": centers[1] * VOXEL_SIZE,
        "anchor_z": centers[0] * VOXEL_SIZE,
        "bb_min_x": mins[2] * VOXEL_SIZE,
        "bb_min_y": mins[1] * VOXEL_SIZE,
        "bb_min_z": mins[0] * VOXEL_SIZE,
        "bb_max_x": maxs[2] * VOXEL_SIZE,
        "bb_max_y": maxs[1] * VOXEL_SIZE,
        "bb_max_z": maxs[0] * VOXEL_SIZE,
        "n_pixels": len(coords[0]),
        "source": "PV_rescue",
        "source_label_id": int(source_row.label_id),
    }


def label_components_fast(table: pd.DataFrame) -> np.ndarray:
    """Exact cKDTree equivalent of the repository's O(N^2) SGN component labeling."""
    valid_mask = table.n_pixels.to_numpy() >= MIN_SIZE
    valid_indices = np.flatnonzero(valid_mask)
    points = table.loc[valid_mask, ["anchor_x", "anchor_y", "anchor_z"]].to_numpy()
    pairs = cKDTree(points).query_pairs(MAX_EDGE_DISTANCE, output_type="ndarray")
    if len(pairs):
        rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
        cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
        graph = coo_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)), shape=(len(points), len(points))
        ).tocsr()
    else:
        graph = coo_matrix((len(points), len(points)), dtype=np.uint8).tocsr()
    _, raw_labels = connected_components(graph, directed=False)
    sizes = np.bincount(raw_labels)
    ordered = [label for label in np.argsort(-sizes) if sizes[label] >= MIN_COMPONENT_LENGTH]
    remap = {old: new + 1 for new, old in enumerate(ordered)}
    labels = np.zeros(len(table), dtype=np.int64)
    labels[valid_indices] = [remap.get(label, 0) for label in raw_labels]
    return labels


def build_merged_table(fs, cochlea: str, out_dir: Path, output_path: Path, selection: dict) -> pd.DataFrame:
    table_path = out_dir / OUTPUT_SEGMENTATION / "default_geometry.tsv"
    if table_path.exists():
        return pd.read_csv(table_path, sep="\t")

    print(f"[{cochlea}] Building the exact merged-object table.", flush=True)
    table_a = current_table(fs, cochlea)
    table_a = table_a[~table_a.label_id.isin(selection.get("source_a_background_labels", []))]
    table_a = table_a.copy()
    table_a["source"] = "CR_Ntng1"
    table_a["source_label_id"] = table_a.label_id.astype(int)
    table_a["component_labels_before"] = table_a.component_labels.astype(int)

    table_b = pd.read_csv(pv_table_path(cochlea), sep="\t").set_index("label_id", drop=False)
    selected_rows = [table_b.loc[label_id] for label_id in selection["selected_pv_ids"]]
    output = zarr.open(str(output_path), mode="r")["segmentation"]
    offset = int(selection["offset"])

    with futures.ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        added_rows = list(tqdm(
            pool.map(lambda row: exact_added_row(output, row, offset), selected_rows),
            total=len(selected_rows), desc=f"{cochlea} exact added geometry",
        ))
    added_rows = [row for row in added_rows if row is not None]
    added = pd.DataFrame(added_rows)
    added["component_labels_before"] = 0
    for column in MAPPING_COLUMNS:
        added[column] = 0.0

    combined = pd.concat([table_a, added], ignore_index=True, sort=False)
    combined = combined.sort_values("label_id").reset_index(drop=True)
    combined["component_labels"] = label_components_fast(combined)

    # Keep the old path fixed for the causal comparison. New cells (and any old cells newly joining
    # recomputed component 1) inherit the path position of their nearest old component-1 cell.
    reference = table_a[table_a.component_labels_before == 1].copy()
    reference_tree = cKDTree(reference[["anchor_x", "anchor_y", "anchor_z"]].to_numpy())
    distances, indices = reference_tree.query(combined[["anchor_x", "anchor_y", "anchor_z"]].to_numpy(), k=1)
    nearest = reference.iloc[indices].reset_index(drop=True)
    combined["distance_to_before_component1[µm]"] = distances
    for column in MAPPING_COLUMNS:
        preserve = (combined.source == "CR_Ntng1") & (combined.component_labels_before == 1)
        combined.loc[~preserve, column] = nearest.loc[~preserve, column].to_numpy()

    combined["fixed_cohort"] = (
        ((combined.source == "CR_Ntng1") & (combined.component_labels_before == 1)) |
        ((combined.source == "PV_rescue") &
         (combined["distance_to_before_component1[µm]"] <= MAX_EDGE_DISTANCE) &
         (combined.n_pixels >= MIN_SIZE))
    )
    combined.to_csv(table_path, sep="\t", index=False)
    return combined


def fixed_threshold_regions(thresholds: dict, before_table: pd.DataFrame) -> List[dict]:
    mapped = eval_utils.map_crops_to_length_fraction(copy.deepcopy(thresholds), before_table)
    fractions = list(mapped)
    limits = eval_utils.length_fraction_limits(fractions)
    return [
        {
            "center": mapped[fraction]["center"],
            "threshold": float(mapped[fraction]["threshold"]),
            "length_fraction": float(fraction),
            "lower": float(limits[num]),
            "upper": float(limits[num + 1]),
        }
        for num, fraction in enumerate(fractions)
    ]


def apply_regions(table: pd.DataFrame, measurements: pd.DataFrame, regions: List[dict], stain: str) -> pd.DataFrame:
    table = table.copy()
    column = f"marker_{stain}"
    table[column] = 0
    medians = measurements.set_index("label_id")["median"]
    for region in regions:
        in_region = (
            (table["length_fraction"] > region["lower"]) &
            (table["length_fraction"] < region["upper"])
        )
        ids = table.loc[in_region, "label_id"]
        values = medians.reindex(ids).dropna()
        positive = values.index[values >= region["threshold"]]
        negative = values.index[values < region["threshold"]]
        table.loc[table.label_id.isin(positive), column] = 1
        table.loc[table.label_id.isin(negative), column] = 2
    return table


def measure_one_median(image, segmentation, row: pd.Series) -> Tuple[int, float]:
    bb = row_bbox(row, segmentation.shape)
    last_error = None
    for _ in range(3):
        try:
            local_seg = np.asarray(segmentation[bb])
            mask = local_seg == int(row.label_id)
            if not mask.any():
                raise RuntimeError(f"Segmentation ID {int(row.label_id)} is empty in its bounding box")
            local_image = np.asarray(image[bb])
            return int(row.label_id), float(np.median(local_image[mask]))
        except Exception as error:  # retry transient S3 reads, then preserve the original traceback
            last_error = error
    raise last_error


def measure_needed_medians(
    fs, cochlea: str, stain: str, table: pd.DataFrame, output_path: Path, segmentation_path: Path,
) -> Tuple[pd.DataFrame, dict]:
    if output_path.exists():
        table_out = pd.read_csv(output_path, sep="\t")
        existing = current_measurements(fs, cochlea, stain)[["label_id", "median"]].copy()
        existing.label_id = existing.label_id.astype(int)
        existing_ids = set(existing.label_id)
        validation_path = output_path.with_name(output_path.stem + "_unchanged-validation.tsv")
        validation = pd.read_csv(validation_path, sep="\t")
        return table_out, {
            "loaded_existing_output": True,
            "n_needed": len(table_out),
            "n_reused_unchanged": int(table_out.label_id.isin(existing_ids).sum()),
            "n_recomputed": int((~table_out.label_id.isin(existing_ids)).sum()),
            "n_unchanged_validation": len(validation),
            "max_unchanged_median_difference": float(validation.absolute_difference.max()),
        }

    existing = current_measurements(fs, cochlea, stain)[["label_id", "median"]].copy()
    existing.label_id = existing.label_id.astype(int)
    needed_mask = table.fixed_cohort | (table.component_labels == 1)
    needed = table[needed_mask].copy()
    existing_ids = set(existing.label_id)
    missing = needed[~needed.label_id.isin(existing_ids)]

    segmentation = zarr.open(str(segmentation_path), mode="r")["segmentation"]
    image = open_s3_array(cochlea, stain, "s0")
    print(
        f"[{cochlea}] {stain}: measuring {len(missing)} new/newly-required objects; "
        f"reusing {needed.label_id.isin(existing_ids).sum()} unchanged measurements.", flush=True,
    )
    rows = [row for _, row in missing.iterrows()]
    with futures.ThreadPoolExecutor(max_workers=N_THREADS) as pool:
        measured = list(tqdm(
            pool.map(lambda row: measure_one_median(image, segmentation, row), rows),
            total=len(rows), desc=f"{cochlea} {stain} medians",
        ))
    measured = pd.DataFrame(measured, columns=["label_id", "median"])
    reused = existing[existing.label_id.isin(needed.label_id)]
    complete = pd.concat([reused, measured], ignore_index=True).sort_values("label_id").reset_index(drop=True)
    assert len(complete) == len(needed)
    complete.to_csv(output_path, sep="\t", index=False)

    # Remeasure a deterministic sample of unchanged objects to prove that keeping their original
    # measurements is exact: the merge never overwrites an A pixel.
    reusable = needed[needed.label_id.isin(existing_ids)].sort_values("label_id")
    sample_indices = np.linspace(0, len(reusable) - 1, min(32, len(reusable)), dtype=int)
    sample_rows = [row for _, row in reusable.iloc[sample_indices].iterrows()]
    with futures.ThreadPoolExecutor(max_workers=min(N_THREADS, 8)) as pool:
        sample = list(pool.map(lambda row: measure_one_median(image, segmentation, row), sample_rows))
    sample = pd.DataFrame(sample, columns=["label_id", "remeasured_median"])
    check = sample.merge(existing, on="label_id", how="left")
    check["absolute_difference"] = (check.remeasured_median - check["median"]).abs()
    check_path = output_path.with_name(output_path.stem + "_unchanged-validation.tsv")
    check.to_csv(check_path, sep="\t", index=False)
    max_difference = float(check.absolute_difference.max()) if len(check) else 0.0
    if max_difference != 0.0:
        raise AssertionError(f"Unchanged {stain} medians differ by up to {max_difference}")
    return complete, {
        "loaded_existing_output": False,
        "n_needed": len(needed),
        "n_reused_unchanged": len(reused),
        "n_recomputed": len(measured),
        "n_unchanged_validation": len(check),
        "max_unchanged_median_difference": max_difference,
    }


def assign_subtypes(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    labels = []
    for cr, ntng1 in zip(table.marker_CR, table.marker_Ntng1):
        if cr not in (1, 2) or ntng1 not in (1, 2):
            labels.append(None)
            continue
        cr_sign = "+" if cr == 1 else "-"
        ntng1_sign = "+" if ntng1 == 1 else "-"
        labels.append(STAIN_TO_TYPE[f"CR{cr_sign}/Ntng1{ntng1_sign}"])
    table["subtype_label"] = labels
    return table


def subtype_summary(table: pd.DataFrame, cohort: pd.Series, scenario: str) -> Tuple[dict, List[dict]]:
    cohort_table = table[cohort].copy()
    assigned = cohort_table[cohort_table.subtype_label.isin(SUBTYPES)]
    counts = assigned.subtype_label.value_counts().reindex(SUBTYPES, fill_value=0)
    total = int(counts.sum())
    summary = {
        "scenario": scenario,
        "n_cohort": len(cohort_table),
        "n_assigned": total,
        "n_unassigned": len(cohort_table) - total,
        "counts": {key: int(value) for key, value in counts.items()},
        "percent": {key: (100.0 * int(value) / total if total else 0.0) for key, value in counts.items()},
    }
    rows = []
    for subtype in SUBTYPES:
        rows.append({
            "scenario": scenario,
            "subtype": subtype,
            "count": int(counts[subtype]),
            "percent": summary["percent"][subtype],
            "n_cohort": len(cohort_table),
            "n_assigned": total,
            "n_unassigned": len(cohort_table) - total,
        })
    return summary, rows


def analyze_cochlea(fs, cochlea: str) -> Tuple[dict, List[dict]]:
    out_dir = OUTPUT_ROOT / cochlea
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = select_pv_ids(fs, cochlea, out_dir)
    segmentation_path = merge_full_resolution(fs, cochlea, out_dir, selection)
    merged = build_merged_table(fs, cochlea, out_dir, segmentation_path, selection)
    before = current_table(fs, cochlea)

    regions = {}
    before_rethresholded = before.copy()
    merged_thresholded = merged.copy()
    measurement_info = {}
    for stain in STAINS:
        path = threshold_path(cochlea, stain)
        with open(path) as f:
            threshold_dic = json.load(f)
        regions[stain] = fixed_threshold_regions(threshold_dic, before)

        before_measures = current_measurements(fs, cochlea, stain)[["label_id", "median"]]
        before_rethresholded = apply_regions(before_rethresholded, before_measures, regions[stain], stain)

        measure_path = out_dir / OUTPUT_SEGMENTATION / f"{stain}_object-measures.tsv"
        merged_measures, measurement_info[stain] = measure_needed_medians(
            fs, cochlea, stain, merged, measure_path, segmentation_path,
        )
        merged_thresholded = apply_regions(merged_thresholded, merged_measures, regions[stain], stain)

        # Reapplying the current thresholds must reproduce all existing component-1 marker labels.
        component1 = before.component_labels == 1
        mismatch = int((
            before_rethresholded.loc[component1, f"marker_{stain}"].to_numpy() !=
            before.loc[component1, f"marker_{stain}"].to_numpy()
        ).sum())
        if mismatch:
            raise AssertionError(f"{cochlea} {stain}: {mismatch} baseline marker labels changed")
        measurement_info[stain]["baseline_marker_mismatches"] = mismatch

    before_rethresholded = assign_subtypes(before_rethresholded)
    merged_thresholded = assign_subtypes(merged_thresholded)

    # Existing A objects in the fixed cohort must retain their labels exactly.
    retained = merged_thresholded[
        (merged_thresholded.source == "CR_Ntng1") & merged_thresholded.fixed_cohort
    ][["label_id", "marker_CR", "marker_Ntng1", "subtype_label"]]
    old = before_rethresholded[["label_id", "marker_CR", "marker_Ntng1", "subtype_label"]]
    check = retained.merge(old, on="label_id", suffixes=("_after", "_before"), validate="one_to_one")
    retained_mismatches = int((
        (check.marker_CR_after != check.marker_CR_before) |
        (check.marker_Ntng1_after != check.marker_Ntng1_before) |
        (check.subtype_label_after.fillna("") != check.subtype_label_before.fillna(""))
    ).sum())
    if retained_mismatches:
        raise AssertionError(f"{cochlea}: {retained_mismatches} retained A subtype labels changed")

    final_table_path = out_dir / OUTPUT_SEGMENTATION / "default.tsv"
    merged_thresholded.to_csv(final_table_path, sep="\t", index=False)
    before_rethresholded.to_csv(out_dir / "before_rethresholded.tsv", sep="\t", index=False)

    before_summary, before_rows = subtype_summary(
        before_rethresholded, before_rethresholded.component_labels == 1, "before",
    )
    fixed_summary, fixed_rows = subtype_summary(
        merged_thresholded, merged_thresholded.fixed_cohort, "after_fixed_cohort",
    )
    recomputed_summary, recomputed_rows = subtype_summary(
        merged_thresholded, merged_thresholded.component_labels == 1, "after_recomputed_component",
    )

    source_breakdown = {}
    for name, cohort in (
        ("after_fixed_cohort", merged_thresholded.fixed_cohort),
        ("after_recomputed_component", merged_thresholded.component_labels == 1),
    ):
        source_breakdown[name] = {}
        for source, subset in merged_thresholded[cohort].groupby("source"):
            counts = subset.subtype_label.value_counts().reindex(SUBTYPES, fill_value=0)
            source_breakdown[name][source] = {key: int(value) for key, value in counts.items()}

    exact_added = merged_thresholded.source == "PV_rescue"
    provenance = {
        "cochlea": cochlea,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "current_segmentation": f"s3://{BUCKET_NAME}/{s3_key(cochlea, CURRENT_SEGMENTATION, 's0')}",
            "pv_segmentation": str(pv_array_path(cochlea)),
            "pv_table": str(pv_table_path(cochlea)),
            "pv_table_sha256": sha256(pv_table_path(cochlea)),
            "thresholds": {
                stain: {"path": str(threshold_path(cochlea, stain)), "sha256": sha256(threshold_path(cochlea, stain))}
                for stain in STAINS
            },
        },
        "parameters": {
            "overlap_scale": OVERLAP_SCALE,
            "overlap_threshold": OVERLAP_THRESHOLD,
            "voxel_size_um": VOXEL_SIZE,
            "max_edge_distance_um": MAX_EDGE_DISTANCE,
            "min_size_voxels": MIN_SIZE,
            "min_component_length": MIN_COMPONENT_LENGTH,
            "threads": N_THREADS,
        },
        "selection": {key: value for key, value in selection.items() if key != "selected_pv_ids"},
        "merged_table": {
            "n_total": len(merged_thresholded),
            "n_current": int((merged_thresholded.source == "CR_Ntng1").sum()),
            "n_pv_rescue_with_pixels": int(exact_added.sum()),
            "n_pv_rescue_fixed_cohort": int((exact_added & merged_thresholded.fixed_cohort).sum()),
            "n_pv_rescue_recomputed_component1": int((exact_added & (merged_thresholded.component_labels == 1)).sum()),
            "inserted_pv_voxels": int(merged_thresholded.loc[exact_added, "n_pixels"].sum()),
            "component_counts": merged_thresholded.component_labels.value_counts().sort_index().to_dict(),
            "retained_a_subtype_mismatches": retained_mismatches,
        },
        "measurements": measurement_info,
        "threshold_regions": regions,
        "source_breakdown": source_breakdown,
    }
    write_json(out_dir / "provenance.json", provenance)

    result = {
        "cochlea": cochlea,
        "before": before_summary,
        "after_fixed_cohort": fixed_summary,
        "after_recomputed_component": recomputed_summary,
        "source_breakdown": source_breakdown,
    }
    write_json(out_dir / "result.json", result)
    rows = []
    for row in before_rows + fixed_rows + recomputed_rows:
        row["cochlea"] = cochlea
        rows.append(row)
    return result, rows


def write_report(results: List[dict], rows: List[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(rows)[[
        "cochlea", "scenario", "subtype", "count", "percent", "n_cohort", "n_assigned", "n_unassigned",
    ]]
    comparison.to_csv(OUTPUT_ROOT / "subtype_comparison.tsv", sep="\t", index=False)
    write_json(OUTPUT_ROOT / "subtype_comparison.json", {"results": results})

    lines = [
        "# M29 SGN Type-II rescue report", "",
        "The primary after-rescue result uses the fixed cohort (old CR+Ntng1 component 1 plus PV",
        "objects attached within 30 µm). The recomputed-component result is reported as a sensitivity",
        "analysis.", "",
        "| Cochlea | Scenario | Assigned SGNs | Type Ia | Type Ib | Type Ic | Type II |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        for key in ("before", "after_fixed_cohort", "after_recomputed_component"):
            summary = result[key]
            perc = summary["percent"]
            lines.append(
                f"| {result['cochlea']} | {summary['scenario']} | {summary['n_assigned']} | "
                f"{perc['Type Ia']:.3f}% | {perc['Type Ib']:.3f}% | {perc['Type Ic']:.3f}% | "
                f"{perc['Type II']:.3f}% |"
            )
        lines.append("")
    lines.extend([
        "All outputs are local to this directory. No existing segmentation, MoBIE metadata, table,",
        "prediction, or S3 object was modified.", "",
    ])
    with open(OUTPUT_ROOT / "report.md", "w") as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all", "cochlea", "report"))
    parser.add_argument("--cochlea", choices=COCHLEAE, help="Required for the 'cochlea' command.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "cochlea" and args.cochlea is None:
        raise ValueError("--cochlea is required for the 'cochlea' command")
    if args.command == "report":
        results = []
        rows = []
        for cochlea in COCHLEAE:
            with open(OUTPUT_ROOT / cochlea / "result.json") as f:
                result = json.load(f)
            results.append(result)
            for key in ("before", "after_fixed_cohort", "after_recomputed_component"):
                summary = result[key]
                for subtype in SUBTYPES:
                    rows.append({
                        "cochlea": cochlea,
                        "scenario": summary["scenario"],
                        "subtype": subtype,
                        "count": summary["counts"][subtype],
                        "percent": summary["percent"][subtype],
                        "n_cohort": summary["n_cohort"],
                        "n_assigned": summary["n_assigned"],
                        "n_unassigned": summary["n_unassigned"],
                    })
        write_report(results, rows)
        print(f"Saved the comparison to {OUTPUT_ROOT / 'report.md'}", flush=True)
        return
    cochleae = COCHLEAE if args.command == "all" else (args.cochlea,)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fs = create_s3_target(url=SERVICE_ENDPOINT, anon=False)
    results, rows = [], []
    for cochlea in cochleae:
        result, cochlea_rows = analyze_cochlea(fs, cochlea)
        results.append(result)
        rows.extend(cochlea_rows)
    write_report(results, rows)
    print(f"Saved the comparison to {OUTPUT_ROOT / 'report.md'}", flush=True)


if __name__ == "__main__":
    main()
