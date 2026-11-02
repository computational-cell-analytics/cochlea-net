"""Measure the quality of an IHC row segmentation from its tables, with the IHC_v11 training crops as reference.

The measure is `ihc_row_quality`. It flags row instances whose size, main-axis length, elongation or axis
direction differs from their neighbours, gaps in the row, and extra instances next to the row. The ellipsoids of
the row instances come from the feature set 'ellipsoid' of `compute_object_measures`.

Steps:
- ellipsoids: the ellipsoid tables of the row instances, from the S3 segmentations.
- reference: statistics of the annotated IHCs in the training crops. They give the defaults of `ihc_row_quality`.
- validate: the flags of `ihc_row_quality` against the annotations of the training crops. Then a comparison with
  the IHC_v9 row of the MWfLS cochleae, only as a secondary check, because IHC_v9 has its own errors.
- candidates: crop candidates for the MWfLS cochleae, away from the existing correction crops.
- tips: the IHC tips of the annotated IHCs (point 4) and the scores of `correct_ihc_row`. It writes to
  <output>/tips:
  - gt_cells.tsv: the tips and the shape of each complete annotated IHC in the path frame;
  - grid.tsv: tip cells per annotated IHC for a grid of prominence and spacing, for both tip detectors;
  - partition.tsv: the scores of `correct_ihc_row` per crop and scenario, see `evaluate_tip_crop`.
- standard: the 'standard' IHC instance (point 5), from gt_objects.tsv and tips/gt_cells.tsv. It writes
  standard_ihc.tsv: the percentiles of each feature per species and tonotopic third.

Validation protocol (leave one animal out): the percentile settings of `correct_ihc_row` and `ihc_row_quality`
for the crops of one animal come from the annotated IHCs of the other animals (`fold_thresholds`). The left
and right cochlea of one animal are one group. Decide between two variants with the paired per-crop deltas
(`--baseline`), with a 95% CI from resampling the animals. The IHC_v11 val crops are the only fair crops for
the output of IHC_v11 itself, because IHC_v11 trained on the other crops.

The annotations are split into connected components, like the training transform: some annotations reuse a
label id for two separate cells.

The test data of scripts/validation/IHCs/run_evaluation.py are not used: their cochleae are excluded.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
import pandas as pd
import tifffile
from elf.evaluation import matching
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as label_structures
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.measure import label as label_pieces
from skimage.segmentation import relabel_sequential

from flamingo_tools.analysis.training_data_utils import (
    AXIS_WINDOW, PATH_SMOOTHING, TIP_DISTANCE, _row_frame, _row_voxels, correct_ihc_row, find_crop_centers_quality,
    find_ihc_tips_height, find_ihc_tips_path, foot_points, ihc_row_quality, ihc_row_reference,
)
from flamingo_tools.export_data_utils import parse_crop_file
from flamingo_tools.measurements import compute_object_measures, compute_object_measures_impl
from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping
from flamingo_tools.postprocessing.label_components import compute_table_on_the_fly, ihc_row_path
from flamingo_tools.s3_utils import default_table_path, read_table

from export_training_data import DATASETS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "figures"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "validation", "IHCs"))
from run_evaluation import COMPONENT_DICT  # noqa: E402
from util import cochlea_components  # noqa: E402

CROP_ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea/IHC_v11_training_data_refined")
MWFLS_CROPS = os.path.expanduser("~/Documents/MWfLS/ihc_crops")
TRAINING_CROP_JSONS = os.path.join(os.path.dirname(__file__), "..", "..", "reproducibility", "training_crops")
MWFLS = ["M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R", "M_AMD_000137_L", "M_AMD_000137_R"]
VOXEL_SIZE = 0.38
# The size of the MWfLS correction crops, 512 x 512 x 256 pixels, in um in (x, y, z) order.
CROP_SIZE = [2 * halo * VOXEL_SIZE for halo in [128, 256, 256]]
COLUMNS = ["anchor_x", "anchor_y", "anchor_z"]
ERROR_CLASSES = ["off_row", "merged", "fragment", "shape", "extra"]
ERROR_COLUMNS = ["missed", "merged", "fragment", "shape", "off_row", "extra"]
# An S3 object is correct if the nearest annotated IHC is within MATCH_DISTANCE um and has 0.75-1.33 of its size.
MATCH_DISTANCE = 3.0
# An annotated IHC is missed if no S3 object is within MISS_DISTANCE um.
MISS_DISTANCE = 5.0
# Smaller annotations are partial cells at the crop border or annotation debris.
MIN_IHC_SIZE = 5000
# The cochleae of the IHC test data in scripts/validation/IHCs/run_evaluation.py.
TEST_COCHLEAE = {cochlea for components in COMPONENT_DICT.values() for cochlea in components}
TIP_METHODS = ["height", "path"]
# The grid of the tip detection on the train crops, in µm.
TIP_PROMINENCES = [0.5, 1.0, 1.5, 2.0, 3.0]
TIP_SPACINGS = [3.0, 4.5, 6.0]
# Simulated IHC_v11 errors: the probability that a run of 2 or 3 neighbours is merged, and that a single IHC is split.
MERGE_PROBABILITY = 0.3
SPLIT_PROBABILITY = 0.2
SEEDS = [0, 1, 2]
# An input instance is correct if it matches an annotated IHC with this IoU (class confusion of the correction).
CORRECT_IOU = 0.9
N_BOOTSTRAP = 2000


def row_definitions():
    """The segmentation and the row components of each cochlea."""
    rows = {cochlea: ("IHC_v11", [1]) for cochlea in MWFLS}
    for path in glob.glob(os.path.join(TRAINING_CROP_JSONS, "*_ihc.json")):
        with open(path) as f:
            params = json.load(f)[0]
        rows[params["dataset_name"]] = (params["segmentation_channel"], params["component_list"])
    rows["G_EK_000233_L"] = ("IHC_v11", cochlea_components("G_EK_000233_L", "IHC", "IHC_v11"))
    return rows


ROWS = row_definitions()


@lru_cache(maxsize=None)
def load_table(cochlea, segmentation):
    return read_table(default_table_path(cochlea, segmentation, s3=True), s3=True)


def ellipsoid_path(out_dir, cochlea):
    return os.path.join(out_dir, "ellipsoid", f"{cochlea}_{ROWS[cochlea][0]}.tsv")


@lru_cache(maxsize=None)
def row_quality(out_dir, cochlea, with_shape=True, window=100.0, settings=()):
    """`ihc_row_quality` of a cochlea. `settings` is a tuple of keyword items, for example of a fold."""
    segmentation, components = ROWS[cochlea]
    ellipsoids = pd.read_csv(ellipsoid_path(out_dir, cochlea), sep="\t") if with_shape else None
    return ihc_row_quality(load_table(cochlea, segmentation), components, ellipsoids, window=window, **dict(settings))


@lru_cache(maxsize=None)
def row_path(cochlea):
    """The path of the IHC row of a cochlea in µm, in (x, y, z) order."""
    segmentation, components = ROWS[cochlea]
    table = load_table(cochlea, segmentation)
    return ihc_row_path(table.loc[table["component_labels"].isin(components), COLUMNS].values)


@lru_cache(maxsize=None)
def row_path_tree(cochlea):
    return cKDTree(row_path(cochlea))


def compute_ellipsoids(out_dir):
    os.makedirs(os.path.join(out_dir, "ellipsoid"), exist_ok=True)
    for cochlea, (segmentation, components) in ROWS.items():
        start = time.time()
        seg_path = f"{cochlea}/images/ome-zarr/{segmentation}.ome.zarr"
        compute_object_measures(
            seg_path, seg_path, default_table_path(cochlea, segmentation, s3=True), ellipsoid_path(out_dir, cochlea),
            image_key="s0", segmentation_key="s0", feature_set="ellipsoid", component_list=components, s3=True,
        )
        print(f"{cochlea} {segmentation}: {time.time() - start:.0f} s")


def gt_crops():
    """The annotated training crops whose name gives the crop center in um, without the test cochleae.

    'val' marks the val crops of the IHC_v11 checkpoint, 'animal' is the cochlea without its side.
    """
    val_names = set(DATASETS["IHC_v11"]["val"])
    crops = []
    for path in sorted(glob.glob(os.path.join(CROP_ROOT, "*", "*_annotations.tif"))):
        name = os.path.basename(path)[:-len("_annotations.tif")]
        info = parse_crop_file(name)
        if info is None or name.endswith("_empty"):
            continue
        cochlea = info["source"].replace("-", "_")
        if cochlea in TEST_COCHLEAE:
            print(f"Skipping {name}: {cochlea} holds IHC test data.")
            continue
        crops.append({
            "name": name, "path": path, "cochlea": cochlea, "animal": re.sub(r"_[LR]$", "", cochlea),
            "center": info["crop_center"], "val": name in val_names,
        })
    return crops


def annotation_cells(crop):
    """The annotated IHCs of a crop with MIN_IHC_SIZE pixels or more, one label per connected component.

    Like the training transform, a label id that marks two separate cells gives two cells.
    """
    annotation = tifffile.imread(crop["path"])
    pieces = label_pieces(annotation, background=0)
    ids, counts = np.unique(pieces, return_counts=True)
    keep = ids[(ids > 0) & (counts >= MIN_IHC_SIZE)]
    owner = np.zeros(pieces.max() + 1, dtype=annotation.dtype)
    owner[pieces.ravel()] = annotation.ravel()
    owners, n_pieces = np.unique(owner[keep], return_counts=True)
    if (n_pieces > 1).any():
        print(f"{crop['name']}: the annotation ids {owners[n_pieces > 1].tolist()} hold several cells.")
    return relabel_sequential(np.where(np.isin(pieces, keep), pieces, 0))[0].astype("uint32")


def load_gt(crop):
    """Measure the annotated IHCs of a crop. Positions are in um in the volume, in (x, y, z) order."""
    annotation = annotation_cells(crop)
    shape = np.array(annotation.shape)
    start = crop_start(crop, shape)
    table = compute_table_on_the_fly(annotation, VOXEL_SIZE)
    border = (np.isclose(table[["bb_min_z", "bb_min_y", "bb_min_x"]].values / VOXEL_SIZE, 0) |
              np.isclose(table[["bb_max_z", "bb_max_y", "bb_max_x"]].values / VOXEL_SIZE, shape)).any(axis=1)
    ellipsoids = compute_object_measures_impl(
        annotation, annotation, n_threads=8, voxel_size=(VOXEL_SIZE,) * 3, table=table, feature_set="ellipsoid",
    )
    gt = table.merge(ellipsoids, on="label_id").assign(border=border)
    extent = gt[["bb_max_x", "bb_max_y", "bb_max_z"]].values - gt[["bb_min_x", "bb_min_y", "bb_min_z"]].values
    gt["eta"] = np.linalg.norm(extent, axis=1) / (6 * gt["n_pixels"] * VOXEL_SIZE ** 3 / np.pi) ** (1 / 3)
    gt[COLUMNS] += start[::-1] * VOXEL_SIZE
    box = (start[::-1] * VOXEL_SIZE, (start + shape)[::-1] * VOXEL_SIZE)
    return gt, box


def crop_start(crop, shape):
    """The start of a crop in the volume in pixels, in (z, y, x) order."""
    return np.round(np.array(crop["center"][::-1]) / VOXEL_SIZE).astype(int) - np.array(shape) // 2


def _angle(axis, reference):
    return np.degrees(np.arccos(min(1.0, abs(axis @ reference))))


def reference_statistics(out_dir):
    """Measure the annotated IHCs relative to the other complete IHCs of their crop."""
    rows = []
    for crop in gt_crops():
        gt, _ = load_gt(crop)
        inner = ~gt["border"].values
        if inner.sum() < 2:
            continue
        axes = gt[["axis_x", "axis_y", "axis_z"]].values
        coords = gt[COLUMNS].values
        tree = cKDTree(coords)
        spacing = tree.query(coords, k=2)[0][:, 1]
        on_row = crop["cochlea"] in ROWS
        path_distance = row_path_tree(crop["cochlea"]).query(coords)[0] if on_row else np.full(len(gt), np.nan)
        for i in np.flatnonzero(inner):
            neighbours = [j for j in tree.query(coords[i], k=min(7, len(gt)))[1] if j != i][:6]
            mean_axis = np.linalg.eigh(axes[neighbours].T @ axes[neighbours])[1][:, -1]
            rows.append({
                "crop": crop["name"], "cochlea": crop["cochlea"], "animal": crop["animal"], "val": crop["val"],
                "species": "gerbil" if crop["cochlea"].startswith("G") else "mouse",
                "label_id": gt.at[i, "label_id"], "n_pixels": gt.at[i, "n_pixels"],
                "n_rel": gt.at[i, "n_pixels"] / np.median(gt.loc[inner, "n_pixels"]), "eta": gt.at[i, "eta"],
                "a": gt.at[i, "a"], "b": gt.at[i, "b"], "c": gt.at[i, "c"],
                "a_rel": gt.at[i, "a"] / np.median(gt.loc[inner, "a"]), "elongation": gt.at[i, "a"] / gt.at[i, "b"],
                "axis_angle": _angle(axes[i], mean_axis), "spacing": spacing[i],
                "path_distance": path_distance[i],
            })
    objects = pd.DataFrame(rows)
    objects.to_csv(os.path.join(out_dir, "gt_objects.tsv"), sep="\t", index=False)

    percentiles = [1, 5, 50, 95, 99]
    summary = objects[["n_rel", "eta", "a", "b", "c", "a_rel", "elongation", "axis_angle", "spacing",
                       "path_distance"]].quantile(np.array(percentiles) / 100).T
    summary.columns = [f"p{p}" for p in percentiles]
    print(f"{len(objects)} complete annotated IHCs in {objects['crop'].nunique()} crops")
    print(summary.round(2).to_string())
    print(objects.groupby("species")[["n_pixels", "a", "b", "c", "elongation", "axis_angle", "spacing"]]
          .median().round(2).to_string())


def _percentile(table, column, p):
    return float(table[column].quantile(p / 100))


def quality_settings(objects, animal=None):
    """The ranges of `ihc_row_quality`: 1st-99th percentile of the annotated IHCs of the other animals."""
    other = objects[objects["animal"] != animal]
    return {
        "size_range": (_percentile(other, "n_rel", 1), _percentile(other, "n_rel", 99)),
        "axis_range": (_percentile(other, "a_rel", 1), _percentile(other, "a_rel", 99)),
        "min_elongation": _percentile(other, "elongation", 1),
    }


def correction_settings(cells, animal=None):
    """The shape settings of `correct_ihc_row` from the complete annotated IHCs of the other animals.

    The size, the length along the axis and the elongation are 99th (and 1st) percentiles, the path width the
    98.5th percentile. The trim radius keeps 98% of the voxels of 99% of the cells. `min_cell_width` (between a
    half cell and the 1st percentile), `max_axis_angle` (a gap in the angles) and the tip settings stay fixed.
    """
    other = cells[cells["animal"] != animal]
    return {
        "max_cell_size": _percentile(other, "n_rel", 99),
        "max_cell_width": _percentile(other, "path_width_rel", 98.5),
        "max_length_axis": _percentile(other, "length_axis_rel", 99),
        "elongation": (_percentile(other, "elongation", 1), _percentile(other, "elongation", 99)),
        "max_radius_factor": _percentile(other, "radius_p98", 99),
    }


def _match_s3_objects(crop, gt, box, table):
    """The S3 objects inside the crop with at least 1000 pixels, and if they match an annotated IHC."""
    lower, upper = box
    inside = ((table[["bb_min_x", "bb_min_y", "bb_min_z"]].values > lower + 1) &
              (table[["bb_max_x", "bb_max_y", "bb_max_z"]].values < upper - 1)).all(axis=1)
    objects = table.loc[inside & (table["n_pixels"] >= 1000).values, ["label_id", "n_pixels"] + COLUMNS].copy()
    distance, index = cKDTree(gt[COLUMNS].values).query(objects[COLUMNS].values)
    ratio = objects["n_pixels"].values / gt["n_pixels"].values[index]
    objects["correct"] = (distance < MATCH_DISTANCE) & (ratio > 0.75) & (ratio < 1.33)
    objects["crop"] = crop["name"]
    return objects


def validate_on_crops(out_dir):
    """Compare the flags of ihc_row_quality with the errors that the annotations show.

    The ranges of `ihc_row_quality` for a cochlea come from the annotated IHCs of the other animals.
    """
    gt_objects = pd.read_csv(os.path.join(out_dir, "gt_objects.tsv"), sep="\t")
    object_rows, crop_rows = [], []
    for crop in gt_crops():
        if crop["cochlea"] not in ROWS:
            print(f"Skipping {crop['name']}: no IHC row for {crop['cochlea']}.")
            continue
        gt, (lower, upper) = load_gt(crop)
        median_distance = np.median(row_path_tree(crop["cochlea"]).query(gt[COLUMNS].values)[0])
        if median_distance > 15:
            print(f"Skipping {crop['name']}: the annotated IHCs are {median_distance:.0f} um from the IHC row.")
            continue

        table = load_table(crop["cochlea"], ROWS[crop["cochlea"]][0])
        objects = _match_s3_objects(crop, gt, (lower, upper), table)
        inner = gt[~gt["border"]]
        gt_missed = int((cKDTree(table[COLUMNS].values).query(inner[COLUMNS].values)[0] >= MISS_DISTANCE).sum())
        crop_row = {
            "crop": crop["name"], "cochlea": crop["cochlea"], "val": crop["val"], "n_gt": len(inner),
            "n_objects": len(objects), "true_errors": int((~objects["correct"]).sum()) + gt_missed,
            "gt_missed": gt_missed,
        }
        settings = tuple(quality_settings(gt_objects, crop["animal"]).items())
        for mode, with_shape in [("size", False), ("shape", True)]:
            quality = row_quality(out_dir, crop["cochlea"], with_shape, settings=settings)[0].set_index("label_id")
            objects[f"class_{mode}"] = quality["class"].reindex(objects["label_id"]).fillna("none").values
            objects[f"flag_{mode}"] = objects[f"class_{mode}"].isin(ERROR_CLASSES)
            row = quality[quality["class"] != "extra"]
            anchors = table.set_index("label_id").loc[row.index, COLUMNS].values
            in_box = ((anchors >= lower) & (anchors < upper)).all(axis=1)
            crop_row[f"flagged_{mode}"] = int(objects[f"flag_{mode}"].sum())
            crop_row[f"gap_missed_{mode}"] = int(row.loc[in_box, "missed"].sum())
        object_rows.append(objects)
        crop_rows.append(crop_row)

    objects, crops = pd.concat(object_rows, ignore_index=True), pd.DataFrame(crop_rows)
    objects.to_csv(os.path.join(out_dir, "validation_objects.tsv"), sep="\t", index=False)
    crops.to_csv(os.path.join(out_dir, "validation_crops.tsv"), sep="\t", index=False)

    val_crops = set(crops.loc[crops["val"], "crop"])
    for subset, selection in [("all crops", objects["crop"]), ("IHC_v11 val crops", val_crops)]:
        part = objects[objects["crop"].isin(selection)]
        crop_part = crops[crops["crop"].isin(part["crop"])]
        wrong = ~part["correct"]
        print(f"{subset}: {crop_part.shape[0]} crops, {len(part)} S3 objects, {wrong.sum()} wrong, "
              f"{crop_part['gt_missed'].sum()} annotated IHCs missed")
        # The error counts grow with the number of objects of a crop, so the rates are compared too.
        n_crop = crop_part["n_objects"] + crop_part["gt_missed"]
        for mode in ["size", "shape"]:
            flag = part[f"flag_{mode}"]
            metric = crop_part[f"flagged_{mode}"] + crop_part[f"gap_missed_{mode}"]
            precision, recall = (flag & wrong).sum() / max(flag.sum(), 1), (flag & wrong).sum() / wrong.sum()
            print(f"  {'size + shape' if mode == 'shape' else 'size'}: flagged {flag.sum()}, "
                  f"precision {precision:.2f}, recall {recall:.2f}, "
                  f"per-crop Spearman {spearmanr(metric, crop_part['true_errors'])[0]:.2f} (counts), "
                  f"{spearmanr(metric / n_crop, crop_part['true_errors'] / n_crop)[0]:.2f} (rates)")
    print("Precision per class (size + shape, all crops):")
    for name, group in objects.groupby("class_shape"):
        print(f"  {name}: {len(group)} objects, {(~group['correct']).mean():.2f} wrong")


def compare_ihc_v9(out_dir):
    """Secondary check on MWfLS: the errors per window against the count deficit to the IHC_v9 row."""
    context = pd.read_csv(os.path.join(MWFLS_CROPS, "crop_context.tsv"), sep="\t")
    for cochlea in MWFLS[:4]:
        table = load_table(cochlea, "IHC_v11")
        reference = load_table(cochlea, "IHC_v9")
        reference = reference[reference["component_labels"] == 1]
        path = ihc_row_path(table.loc[table["component_labels"] == 1, COLUMNS].values)
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))])
        tree = cKDTree(path)
        for window in [100.0, 200.0]:
            _, windows = row_quality(out_dir, cochlea, True, window)
            edges = np.append(windows["start"].values, windows["stop"].values[-1])
            deficit = np.histogram(arc[tree.query(reference[COLUMNS].values)[1]], edges)[0] - windows["n_row"]
            print(f"{cochlea}, {window:.0f} um windows: Spearman with the IHC_v9 count deficit "
                  f"{spearmanr(windows['errors'], deficit)[0]:.2f}")
        _, windows = row_quality(out_dir, cochlea, True)
        edges = np.append(windows["start"].values, windows["stop"].values[-1])
        centers = context.loc[context["Cochlea"] == cochlea, ["Center_x", "Center_y", "Center_z"]].values
        crop_errors = windows["errors"].values[np.digitize(arc[tree.query(centers)[1]], edges) - 1]
        ranks = [(windows["errors"] > errors).mean() for errors in crop_errors]
        print(f"  correction crops: errors {crop_errors.tolist()}, "
              f"fraction of windows with more errors {np.round(ranks, 2).tolist()}")


def export_candidates(out_dir, n_crops):
    existing = {}
    for path in glob.glob(os.path.join(MWFLS_CROPS, "json", "*.json")):
        with open(path) as f:
            for params in json.load(f):
                existing.setdefault(params["dataset_name"], []).extend(params["crop_centers"])
    candidates = []
    for cochlea in MWFLS:
        objects, windows = row_quality(out_dir, cochlea)
        windows.to_csv(os.path.join(out_dir, f"{cochlea}_windows.tsv"), sep="\t", index=False)
        totals = windows[["n_row", "expected", "errors"] + ERROR_COLUMNS].sum()
        print(f"{cochlea}: " + ", ".join(f"{name} {value:.0f}" for name, value in totals.items()))
        centers, _ = find_crop_centers_quality(windows, n_crops, CROP_SIZE, exclude=existing.get(cochlea, []))
        for center in centers:
            window = windows.iloc[np.argmin(np.linalg.norm(windows[["x", "y", "z"]].values - center, axis=1))]
            candidates.append({"cochlea": cochlea, "center_x": center[0], "center_y": center[1], "center_z": center[2],
                               **window[["length_fraction", "errors"] + ERROR_COLUMNS].to_dict()})
    candidates = pd.DataFrame(candidates)
    candidates.to_csv(os.path.join(out_dir, "candidates.tsv"), sep="\t", index=False)
    print(candidates.to_string(index=False))


def crop_info(crop):
    species = "gerbil" if crop["cochlea"].startswith("G") else "mouse"
    return {"crop": crop["name"], "val": crop["val"], "animal": crop["animal"], "species": species}


def gt_row(crop):
    """The annotated IHCs of a crop (`annotation_cells`), the path through them and the reference IHC.

    The path is in pixels in (z, y, x) order. It goes 20 µm beyond the outer IHCs, so their tips are not at the
    path end.
    """
    gt = annotation_cells(crop)
    if gt.max() < 3:
        return None
    centers = np.array(center_of_mass(gt > 0, gt, np.arange(1, gt.max() + 1)))
    path = ihc_row_path(centers[:, ::-1] * VOXEL_SIZE)
    # ihc_row_path samples at 1 µm, so 20 steps are 20 µm.
    path = np.concatenate([path[:1] - 20 * (path[1] - path[0]), path, path[-1:] + 20 * (path[-1] - path[-2])])
    path = path[:, ::-1] / VOXEL_SIZE
    return gt, path, ihc_row_reference([gt], path, VOXEL_SIZE)


def s3_row_path(crop, shape, centers):
    """The path of the S3 IHC row in a crop, in pixels in (z, y, x) order, or None.

    The path is the longest run of path points within 20 µm of the crop. It is None if the cochlea has no IHC
    row, if the run is shorter than 50 µm, or if the cell centers (pixels, (z, y, x) order) are more than 15 µm
    from it (median).
    """
    if crop["cochlea"] not in ROWS:
        return None
    path = row_path(crop["cochlea"])[:, ::-1] / VOXEL_SIZE - crop_start(crop, shape)
    margin = 20.0 / VOXEL_SIZE
    inside = np.all((path >= -margin) & (path <= np.array(shape) + margin), axis=1)
    runs = np.split(np.arange(len(path)), np.flatnonzero(np.diff(inside.astype(int))) + 1)
    run = max((run for run in runs if inside[run[0]]), key=len, default=[])
    # The row path is sampled at 1 µm.
    if len(run) < 50 or np.median(cKDTree(path[run]).query(centers)[0]) * VOXEL_SIZE > 15:
        return None
    return path[run]


def find_tips(method, gt, frame, reference, **params):
    features, coords, points, axes = frame
    voxels, labels = _row_voxels(coords, list(features))
    if method == "height":
        return find_ihc_tips_height(voxels, labels, reference["axis"], gt.shape, VOXEL_SIZE, **params)
    tip_distance = tuple(np.array(TIP_DISTANCE) * reference["length_axis"])
    return find_ihc_tips_path(voxels, labels, points, axes, gt.shape, VOXEL_SIZE, tip_distance=tip_distance, **params)


def tip_counts(tips, features):
    """Count the complete annotated IHCs with exactly one tip cell inside them, without one, and with more."""
    complete = [i for i, feature in features.items() if not feature["border"]]
    cells = tips.groupby("label_id")["cell"].unique().reindex(complete)
    inside = tips.groupby("cell")["label_id"].nunique() == 1
    paired = tips.groupby("cell").size() == 2
    n_cells = cells.map(lambda c: 0 if not isinstance(c, np.ndarray) else len(c))
    one = cells[n_cells == 1].map(lambda c: inside[c[0]])
    return {
        "n_gt": len(complete), "one": int(one.sum()),
        "none": int((n_cells == 0).sum()), "multi": int((n_cells > 1).sum()),
        "one_paired": int(cells[n_cells == 1][one.values].map(lambda c: paired[c[0]]).sum()),
    }


def simulate_errors(gt, frame, seed=0):
    """Merge runs of 2 or 3 neighbours along the path and split single cells in half, like the IHC_v11 errors."""
    features, coords, _, _ = frame
    rng = np.random.default_rng(seed)
    order = sorted(features, key=lambda i: features[i]["index"])
    segmentation = gt.copy()
    next_id = int(gt.max()) + 1
    i = 0
    while i < len(order):
        run = (3 if rng.random() < 0.3 else 2) if rng.random() < MERGE_PROBABILITY else 1
        for member in order[i + 1:i + run]:
            segmentation[tuple(coords[member].T)] = order[i]
        if run == 1 and rng.random() < SPLIT_PROBABILITY:
            feature, voxels = features[order[i]], coords[order[i]]
            # Half of the splits cut across the long axis, the other half along the path.
            direction = feature["axis"] if rng.random() < 0.5 else feature["tangent"]
            projection = voxels @ direction
            segmentation[tuple(voxels[projection > np.median(projection)].T)] = next_id
            next_id += 1
        i += run
    return segmentation


def scores(labels, gt):
    """The matched annotated IHCs at IoU 0.5 and 0.75 (suffix '_75'), as counts and F1."""
    n_gt, n_out = int(gt.max()), len(np.unique(labels[labels > 0]))
    result = {"n_gt": n_gt, "n_out": n_out}
    for threshold, suffix in ((0.5, ""), (0.75, "_75")):
        tp = int(round(matching(labels, gt, threshold=threshold)["recall"] * n_gt))
        result.update({f"tp{suffix}": tp, f"fp{suffix}": n_out - tp, f"fn{suffix}": n_gt - tp,
                       f"f1{suffix}": 2 * tp / max(n_out + n_gt, 1)})
    return result


def _best_iou(segmentation, gt):
    """The largest IoU of each instance of the segmentation with an annotated IHC."""
    n_ids = int(gt.max()) + 1
    foreground = segmentation > 0
    pairs, overlap = np.unique(segmentation[foreground].astype(np.int64) * n_ids + gt[foreground], return_counts=True)
    seg_ids, gt_ids = pairs // n_ids, pairs % n_ids
    on_gt = gt_ids > 0
    union = np.bincount(segmentation.ravel())[seg_ids] + np.bincount(gt.ravel(), minlength=n_ids)[gt_ids] - overlap
    return pd.Series((overlap / union)[on_gt]).groupby(seg_ids[on_gt]).max()


def class_confusion(segmentation, labels, gt, report, reference):
    """Count the kept and the flagged instances of the input and of the output, and if they are correct.

    An instance is correct if it matches an annotated IHC with CORRECT_IOU or more.
    - Input ('in_'): a row instance is kept if its class is 'normal' or 'border' and it has no 'review' mark.
    - Output ('out_'): a cell is flagged if most of its voxels come from an 'uncertain' or a kept 'off_row'
      instance, or if its size is outside 0.6-1.4 of the reference IHC (the size rule of 'review'). The flagged
      cells are the candidates for a new segmentation. The 'review' column itself only covers the outputs with
      10% of an input instance, so it misses the cells of a large component.
    """
    row = report[report["class"] != "off_row"]
    correct = _best_iou(segmentation, gt).reindex(row["label_id"]).fillna(0.0).values >= CORRECT_IOU
    kept = row["class"].isin(["normal", "border"]).values & ~row["review"].values.astype(bool)

    n_ids = int(segmentation.max()) + 1
    foreground = labels > 0
    pairs, overlap = np.unique(labels[foreground].astype(np.int64) * n_ids + segmentation[foreground],
                               return_counts=True)
    source = pd.Series(pairs % n_ids).groupby(pairs // n_ids).agg(lambda ids: ids.iloc[np.argmax(overlap[ids.index])])
    output_ids = source.index.values
    classes = report.set_index("label_id")["class"].reindex(source.values).values
    size = np.bincount(labels.ravel())[output_ids] / reference["n_pixels"]
    out_kept = ~np.isin(classes, ["uncertain", "off_row"]) & (size >= 0.6) & (size <= 1.4)
    out_correct = _best_iou(labels, gt).reindex(output_ids).fillna(0.0).values >= CORRECT_IOU
    counts = {}
    for prefix, is_kept, is_correct in (("in", kept, correct), ("out", out_kept, out_correct)):
        counts.update({f"{prefix}_kept_correct": int((is_kept & is_correct).sum()),
                       f"{prefix}_kept_wrong": int((is_kept & ~is_correct).sum()),
                       f"{prefix}_flagged_correct": int((~is_kept & is_correct).sum()),
                       f"{prefix}_flagged_wrong": int((~is_kept & ~is_correct).sum())})
    return counts


def cell_geometry(gt, frame, reference, tips):
    """The tips and the shape of each complete annotated IHC in the path frame.

    The tips are the extreme voxels along the local axis. The offsets along the path ('ds') are between the
    foot points of the tip and of the cell center. 'radius_p98' is the 98th percentile of the voxel distances
    from the cell center in the plane perpendicular to the local axis, relative to the reference path length
    (the trim of `correct_ihc_row`).
    """
    features, coords, points, axes = frame
    detected = tips.groupby(["label_id", "side"]).size()
    rows = []
    for label_id, feature in features.items():
        if feature["border"]:
            continue
        index, offset, height = foot_points(coords[label_id].astype(float), points, axes)
        projection = coords[label_id] @ np.linalg.svd(axes[feature["index"]][None])[2][1:].T
        radius = np.linalg.norm(projection - np.median(projection, axis=0), axis=1) * VOXEL_SIZE
        row = {"label_id": label_id, "n_rel": feature["n_rel"], "volume": feature["n_pixels"] * VOXEL_SIZE ** 3,
               "length": (height.max() - height.min()) * VOXEL_SIZE,
               "length_axis_rel": feature["length_axis"] / reference["length_axis"],
               "path_width_rel": feature["length_path"] / reference["length_path"],
               "elongation": feature["elongation"], "axis_angle": _angle(feature["axis"], axes[feature["index"]]),
               "radius_p98": np.percentile(radius, 98) / reference["length_path"]}
        for side, tip in (("top", np.argmax(height)), ("bottom", np.argmin(height))):
            sign = 1 if side == "top" else -1
            row[f"{side}_distance"] = sign * height[tip] * VOXEL_SIZE
            row[f"{side}_distance_rel"] = sign * height[tip] * VOXEL_SIZE / reference["length_axis"]
            cosine = sign * height[tip] / np.linalg.norm(offset[tip])
            row[f"{side}_angle"] = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
            row[f"{side}_ds"] = (index[tip] - feature["index"]) * VOXEL_SIZE
            row[f"{side}_detected"] = int(detected.get((label_id, side), 0))
        row["pair_ds"] = abs(row["top_ds"] - row["bottom_ds"])
        rows.append(row)
    return rows


def measure_tip_crop(crop):
    """Pass 1 for one annotated crop: the grid of the tip detectors, the cell geometry and the reference IHC."""
    loaded = gt_row(crop)
    if loaded is None:
        return [], [], []
    gt, path, reference = loaded
    frame = _row_frame(gt, path, reference, VOXEL_SIZE, 15.0, PATH_SMOOTHING, AXIS_WINDOW)
    info = crop_info(crop)
    grid = []
    for method in TIP_METHODS:
        for prominence in TIP_PROMINENCES:
            for spacing in TIP_SPACINGS:
                tips = find_tips(method, gt, frame, reference, min_prominence=prominence, min_spacing=spacing)
                grid.append({**info, "method": method, "prominence": prominence, "spacing": spacing,
                             **tip_counts(tips, frame[0])})
    cells = cell_geometry(gt, frame, reference, find_tips("path", gt, frame, reference))
    sizes = {key: reference[key] for key in ("n_pixels", "length_axis", "length_path")}
    return grid, [{**info, **row} for row in cells], [{**info, "n_cells": int(gt.max()), **sizes}]


def evaluate_tip_crop(crop, settings):
    """Pass 2 for one annotated crop: the scores of `correct_ihc_row` with the settings of its fold.

    The scenarios:
    - union: the connected components of the annotated cells, with the reference IHC of the annotation (oracle);
    - union_selfref: the same input, with the reference IHC estimated from the simulated input of seed 0;
    - simulated: the annotation with simulated merges and splits (SEEDS), with its own reference IHC;
    - identity: the annotation itself, which the correction must keep.
    The path goes through the annotated cells ('gt'). For 'path' 's3', union_selfref and simulated (seed 0) use
    the path of the S3 IHC row, as in production. 'ref_ratio' is the size of the reference IHC of the scenario
    relative to the reference IHC of the annotation.
    """
    loaded = gt_row(crop)
    if loaded is None:
        return []
    gt, path, reference = loaded
    frame = _row_frame(gt, path, reference, VOXEL_SIZE, 15.0, PATH_SMOOTHING, AXIS_WINDOW)
    union = label_structures(gt > 0)[0].astype("uint32")
    simulated = {seed: simulate_errors(gt, frame, seed) for seed in SEEDS}
    own = {seed: ihc_row_reference([simulated[seed]], path, VOXEL_SIZE) for seed in SEEDS}
    inputs = [("union", "gt", 0, union, path, reference), ("union_selfref", "gt", 0, union, path, own[0])]
    inputs += [("simulated", "gt", seed, simulated[seed], path, own[seed]) for seed in SEEDS]
    inputs.append(("identity", "gt", 0, gt, path, reference))
    centers = np.array([voxels.mean(axis=0) for voxels in frame[1].values()])
    s3_path = s3_row_path(crop, gt.shape, centers)
    if s3_path is not None:
        s3_reference = ihc_row_reference([simulated[0]], s3_path, VOXEL_SIZE)
        inputs += [("union_selfref", "s3", 0, union, s3_path, s3_reference),
                   ("simulated", "s3", 0, simulated[0], s3_path, s3_reference)]

    rows = []
    for scenario, path_name, seed, segmentation, scenario_path, scenario_reference in inputs:
        info = {**crop_info(crop), "scenario": scenario, "path": path_name, "seed": seed,
                "ref_ratio": scenario_reference["n_pixels"] / reference["n_pixels"]}
        rows.append({**info, "method": "input", **scores(segmentation, gt)})
        labels, report, _ = correct_ihc_row(segmentation, scenario_path, scenario_reference, VOXEL_SIZE, **settings)
        rows.append({**info, "method": "path", **scores(labels, gt),
                     **class_confusion(segmentation, labels, gt, report, scenario_reference)})
    return rows


def tip_reference(out_dir, baseline=None):
    """Find the IHC tips in the annotated training crops and score the tip-based correction, one animal out.

    Pass 1 measures the annotated cells. Pass 2 corrects the scenario inputs of each crop with the settings from
    the cells of the other animals (`correction_settings`).
    """
    crops = gt_crops()
    folder = os.path.join(out_dir, "tips")
    os.makedirs(folder, exist_ok=True)
    # Each worker holds a few GB for a 512 x 512 x 256 crop.
    with ProcessPoolExecutor(3) as executor:
        results = list(executor.map(measure_tip_crop, crops))
    grid, cells, references = (pd.DataFrame(sum(parts, [])) for parts in zip(*results))
    grid.to_csv(os.path.join(folder, "grid.tsv"), sep="\t", index=False)
    cells.to_csv(os.path.join(folder, "gt_cells.tsv"), sep="\t", index=False)
    references.to_csv(os.path.join(folder, "gt_references.tsv"), sep="\t", index=False)

    settings = [correction_settings(cells, crop["animal"]) for crop in crops]
    with ProcessPoolExecutor(3) as executor:
        partition = pd.DataFrame(sum(executor.map(evaluate_tip_crop, crops, settings), []))
    partition.to_csv(os.path.join(folder, "partition.tsv"), sep="\t", index=False)
    summarize_tips(out_dir, baseline)


def _animal_ci(values, animals, rng):
    """The 95% CI of the mean of per-crop values, from resampling the animals."""
    groups = [values[animals == animal] for animal in np.unique(animals)]
    means = [np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))]).mean()
             for _ in range(N_BOOTSTRAP)]
    return np.percentile(means, [2.5, 97.5])


def crop_scores(partition):
    """The scores per crop, scenario, path and method, averaged over the seeds."""
    keys = ["crop", "val", "animal", "species", "scenario", "path", "method"]
    return partition.groupby(keys, as_index=False).mean(numeric_only=True)


def summarize_tips(out_dir, baseline=None):
    """Print the annotated cells, the tip grid and the out-of-fold scores.

    With `baseline` (a partition.tsv of another variant), also print the paired per-crop F1 deltas.
    """
    folder = os.path.join(out_dir, "tips")
    cells = pd.read_csv(os.path.join(folder, "gt_cells.tsv"), sep="\t")
    columns = [c for c in cells.columns if c not in ("crop", "val", "animal", "species", "label_id")]
    print(f"{len(cells)} complete annotated IHCs in {cells['crop'].nunique()} crops of {cells['animal'].nunique()} "
          "animals")
    print(cells[columns].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T.round(2).to_string())
    print("Settings of correct_ihc_row from all animals (the module defaults):")
    print({key: np.round(value, 3).tolist() for key, value in correction_settings(cells).items()})

    grid = pd.read_csv(os.path.join(folder, "grid.tsv"), sep="\t")
    counts = ["n_gt", "one", "none", "multi", "one_paired"]
    totals = grid.groupby(["method", "prominence", "spacing"])[counts].sum()
    for column in ["one", "none", "multi", "one_paired"]:
        totals[column] = (totals[column] / totals["n_gt"]).round(3)
    print("Tip cells per complete annotated IHC (fractions), all crops:")
    print(totals.sort_values("one", ascending=False).to_string())

    partition = pd.read_csv(os.path.join(folder, "partition.tsv"), sep="\t")
    per_crop = crop_scores(partition)
    rng = np.random.default_rng(0)
    keys = ["scenario", "path", "method", "species"]
    rows = []
    for key, group in per_crop.groupby(keys):
        low, high = _animal_ci(group["f1"].values, group["animal"].values, rng)
        rows.append({**dict(zip(keys, key)), "crops": len(group), "animals": group["animal"].nunique(),
                     "f1": group["f1"].mean(), "ci_low": low, "ci_high": high,
                     "f1_micro": 2 * group["tp"].sum() / (group["n_out"].sum() + group["n_gt"].sum()),
                     "f1_75": group["f1_75"].mean()})
    print("Partition scores out of fold (mean over crops of the mean over seeds; 95% CI from resampling animals):")
    print(pd.DataFrame(rows).round(3).to_string(index=False))

    corrected = partition[partition["method"] == "path"]
    estimated = corrected[corrected["scenario"].isin(["union_selfref", "simulated"])]
    off = (estimated["ref_ratio"] > 1.5) | (estimated["ref_ratio"] < 0.7)
    print(f"Estimated reference IHCs outside 0.7-1.5 of the annotation: {int(off.sum())} of {len(estimated)}")
    for prefix, name in (("in", "input instances"), ("out", "output cells (the ROI signal)")):
        columns = [f"{prefix}_{kind}" for kind in ("kept_correct", "kept_wrong", "flagged_correct", "flagged_wrong")]
        confusion = corrected.groupby(keys[:2] + ["species"])[columns].sum()
        confusion.columns = ["kept_correct", "kept_wrong", "flagged_correct", "flagged_wrong"]
        confusion["kept_precision"] = confusion["kept_correct"] / (confusion["kept_correct"] + confusion["kept_wrong"])
        confusion["error_recall"] = confusion["flagged_wrong"] / (confusion["flagged_wrong"] + confusion["kept_wrong"])
        confusion["flagged_precision"] = confusion["flagged_wrong"] / (
            confusion["flagged_wrong"] + confusion["flagged_correct"])
        print(f"Class confusion of the {name} (correct: IoU >= {CORRECT_IOU} with an annotated IHC):")
        print(confusion.round(3).to_string())

    if baseline is not None:
        merged = per_crop.merge(crop_scores(pd.read_csv(baseline, sep="\t"))[
            ["crop", "scenario", "path", "method", "f1"]], on=["crop", "scenario", "path", "method"],
            suffixes=("", "_baseline"))
        merged["delta"] = merged["f1"] - merged["f1_baseline"]
        rows = []
        for key, group in merged.groupby(keys):
            low, high = _animal_ci(group["delta"].values, group["animal"].values, rng)
            rows.append({**dict(zip(keys, key)), "crops": len(group), "delta": group["delta"].mean(),
                         "ci_low": low, "ci_high": high, "better": int((group["delta"] > 0.005).sum()),
                         "worse": int((group["delta"] < -0.005).sum())})
        print(f"Paired F1 deltas against {baseline}:")
        print(pd.DataFrame(rows).round(3).to_string(index=False))


def crop_length_fractions():
    """The length fraction (0 apex, 1 base) of the row instance nearest to each annotated crop center."""
    crops = gt_crops()
    fractions = {}
    for cochlea in sorted({crop["cochlea"] for crop in crops} & set(ROWS)):
        segmentation, components = ROWS[cochlea]
        mapped, _ = tonotopic_mapping(load_table(cochlea, segmentation), component_label=components, cell_type="ihc")
        mapped = mapped[mapped["component_labels"].isin(components) & mapped["length_fraction"].notna()]
        tree = cKDTree(mapped[COLUMNS].values)
        for crop in crops:
            if crop["cochlea"] == cochlea:
                fractions[crop["name"]] = mapped["length_fraction"].values[tree.query(crop["center"])[1]]
    return fractions


def standard_ihc(out_dir):
    """Percentiles of the annotated IHCs per species and tonotopic third: the 'standard' IHC instance.

    The features come from gt_objects.tsv (step reference) and tips/gt_cells.tsv (step tips). Relative features
    are relative to the typical IHC of the crop. The Spearman correlation with the length fraction shows if a
    feature depends on the tonotopy.
    """
    fractions = crop_length_fractions()
    objects = pd.read_csv(os.path.join(out_dir, "gt_objects.tsv"), sep="\t")
    objects["volume"] = objects["n_pixels"] * VOXEL_SIZE ** 3
    cells = pd.read_csv(os.path.join(out_dir, "tips", "gt_cells.tsv"), sep="\t")
    tables = [
        (objects, ["volume", "n_rel", "a", "b", "c", "a_rel", "elongation", "axis_angle", "spacing"]),
        (cells, ["length", "path_width_rel", "top_distance", "bottom_distance", "top_angle", "bottom_angle",
                 "pair_ds"]),
    ]
    rows, correlations = [], []
    for table, features in tables:
        table = table.assign(length_fraction=table["crop"].map(fractions))
        table["tonotopy"] = pd.cut(table["length_fraction"], [0, 1 / 3, 2 / 3, 1], include_lowest=True,
                                   labels=["apex", "mid", "base"]).astype(object).fillna("unknown")
        for (species, tonotopy), group in pd.concat([table, table.assign(tonotopy="all")]).groupby(
                ["species", "tonotopy"]):
            for feature in features:
                values = group[feature].dropna()
                rows.append({"species": species, "tonotopy": tonotopy, "feature": feature, "n": len(values),
                             **{f"p{p}": values.quantile(p / 100) for p in (1, 5, 50, 95, 99)}})
        for species, group in table.dropna(subset=["length_fraction"]).groupby("species"):
            for feature in features:
                rho = spearmanr(group["length_fraction"], group[feature], nan_policy="omit")[0]
                correlations.append({"species": species, "feature": feature, "spearman": rho,
                                     "crops": group["crop"].nunique()})
    standard = pd.DataFrame(rows)
    standard.to_csv(os.path.join(out_dir, "standard_ihc.tsv"), sep="\t", index=False)
    print(standard.round(2).to_string(index=False))
    print("Spearman correlation with the length fraction:")
    print(pd.DataFrame(correlations).round(2).to_string(index=False))


STEPS = ["ellipsoids", "reference", "validate", "candidates", "tips", "standard"]
# 'summary' prints the summary of the tips step again, for example with another --baseline.


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/Documents/lightsheet-cochlea/ihc_quality"))
    parser.add_argument("-s", "--steps", nargs="+", choices=STEPS + ["summary"], default=STEPS)
    parser.add_argument("-n", "--n_crops", type=int, default=3, help="Crop candidates per cochlea.")
    parser.add_argument("-b", "--baseline", default=None,
                        help="partition.tsv of another variant, for the paired per-crop F1 deltas.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if "ellipsoids" in args.steps:
        compute_ellipsoids(args.output)
    if "reference" in args.steps:
        reference_statistics(args.output)
    if "validate" in args.steps:
        validate_on_crops(args.output)
        compare_ihc_v9(args.output)
    if "candidates" in args.steps:
        export_candidates(args.output, args.n_crops)
    if "tips" in args.steps:
        tip_reference(args.output, args.baseline)
    if "summary" in args.steps:
        summarize_tips(args.output, args.baseline)
    if "standard" in args.steps:
        standard_ihc(args.output)


if __name__ == "__main__":
    main()
