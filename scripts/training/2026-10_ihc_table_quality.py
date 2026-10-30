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
- tips: the IHC tips of the annotated IHCs (point 4). It writes to <output>/tips:
  - gt_cells.tsv: the tips of each complete annotated IHC in the path frame;
  - grid.tsv: tip cells per annotated IHC for a grid of prominence and spacing, for both tip detectors;
  - partition.tsv: the scores of `correct_ihc_row` on the connected components of the annotations (union) and
    on the annotations with simulated merges and splits (simulated). A file partition_<suffix>.tsv from another
    method, for example the k-means version of 2026-10-29, joins the printed summary.

The test data of scripts/validation/IHCs/run_evaluation.py are not used.
"""
import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache, partial

import numpy as np
import pandas as pd
import tifffile
from elf.evaluation import matching
from scipy.ndimage import center_of_mass
from scipy.ndimage import label as label_structures
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from skimage.segmentation import relabel_sequential

from flamingo_tools.analysis.training_data_utils import (
    TIP_DISTANCE, _row_frame, correct_ihc_row, find_crop_centers_quality, find_ihc_tips_height, find_ihc_tips_path,
    foot_points, ihc_row_quality, ihc_row_reference,
)
from flamingo_tools.export_data_utils import parse_crop_file
from flamingo_tools.measurements import compute_object_measures, compute_object_measures_impl
from flamingo_tools.postprocessing.label_components import compute_table_on_the_fly, ihc_row_path
from flamingo_tools.s3_utils import default_table_path, read_table

from export_training_data import DATASETS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "figures"))
from util import cochlea_components  # noqa: E402

CROP_ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea/IHC_v11_training_data")
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
TIP_METHODS = ["height", "path"]
# The grid of the tip detection on the train crops, in µm.
TIP_PROMINENCES = [0.5, 1.0, 1.5, 2.0, 3.0]
TIP_SPACINGS = [3.0, 4.5, 6.0]
# Simulated IHC_v11 errors: the probability that a run of 2 or 3 neighbours is merged, and that a single IHC is split.
MERGE_PROBABILITY = 0.3
SPLIT_PROBABILITY = 0.2


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
def row_quality(out_dir, cochlea, with_shape=True, window=100.0):
    segmentation, components = ROWS[cochlea]
    ellipsoids = pd.read_csv(ellipsoid_path(out_dir, cochlea), sep="\t") if with_shape else None
    return ihc_row_quality(load_table(cochlea, segmentation), components, ellipsoids, window=window)


@lru_cache(maxsize=None)
def row_path_tree(cochlea):
    segmentation, components = ROWS[cochlea]
    table = load_table(cochlea, segmentation)
    return cKDTree(ihc_row_path(table.loc[table["component_labels"].isin(components), COLUMNS].values))


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
    """The annotated training crops whose name gives the crop center in um."""
    val_names = set(DATASETS["IHC_v11"]["val"])
    crops = []
    for path in sorted(glob.glob(os.path.join(CROP_ROOT, "*", "*_annotations.tif"))):
        name = os.path.basename(path)[:-len("_annotations.tif")]
        info = parse_crop_file(name)
        if info is None or name.endswith("_empty"):
            continue
        crops.append({
            "name": name, "path": path, "cochlea": info["source"].replace("-", "_"),
            "center": info["crop_center"], "val": name in val_names,
        })
    return crops


def load_gt(crop):
    """Measure the annotated IHCs of a crop. Positions are in um in the volume, in (x, y, z) order."""
    # Some annotations have very large label ids, and find_objects allocates one entry per id.
    annotation = relabel_sequential(tifffile.imread(crop["path"]))[0].astype("uint32")
    shape = np.array(annotation.shape)
    start = np.round(np.array(crop["center"][::-1]) / VOXEL_SIZE).astype(int) - shape // 2
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
    return gt[gt["n_pixels"] >= MIN_IHC_SIZE].reset_index(drop=True), box


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
                "crop": crop["name"], "cochlea": crop["cochlea"], "val": crop["val"],
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
    """Compare the flags of ihc_row_quality with the errors that the annotations show."""
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
            "true_errors": int((~objects["correct"]).sum()) + gt_missed, "gt_missed": gt_missed,
        }
        for mode, with_shape in [("size", False), ("shape", True)]:
            quality = row_quality(out_dir, crop["cochlea"], with_shape)[0].set_index("label_id")
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
        for mode in ["size", "shape"]:
            flag = part[f"flag_{mode}"]
            metric = crop_part[f"flagged_{mode}"] + crop_part[f"gap_missed_{mode}"]
            precision, recall = (flag & wrong).sum() / max(flag.sum(), 1), (flag & wrong).sum() / wrong.sum()
            print(f"  {'size + shape' if mode == 'shape' else 'size'}: flagged {flag.sum()}, "
                  f"precision {precision:.2f}, recall {recall:.2f}, "
                  f"per-crop Spearman {spearmanr(metric, crop_part['true_errors'])[0]:.2f}")
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


def gt_row(crop):
    """The annotated IHCs of a crop with MIN_IHC_SIZE pixels or more, the path through them and the reference IHC.

    The path is in pixels in (z, y, x) order. It goes 20 µm beyond the outer IHCs, so their tips are not at the
    path end.
    """
    gt = relabel_sequential(tifffile.imread(crop["path"]))[0].astype("uint32")
    ids, counts = np.unique(gt, return_counts=True)
    gt[~np.isin(gt, ids[(ids > 0) & (counts >= MIN_IHC_SIZE)])] = 0
    gt = relabel_sequential(gt)[0]
    if gt.max() < 3:
        return None
    centers = np.array(center_of_mass(gt > 0, gt, np.arange(1, gt.max() + 1)))
    path = ihc_row_path(centers[:, ::-1] * VOXEL_SIZE)
    # ihc_row_path samples at 1 µm, so 20 steps are 20 µm.
    path = np.concatenate([path[:1] - 20 * (path[1] - path[0]), path, path[-1:] + 20 * (path[-1] - path[-2])])
    path = path[:, ::-1] / VOXEL_SIZE
    return gt, path, ihc_row_reference([gt], path, VOXEL_SIZE)


def find_tips(method, gt, frame, reference, **params):
    features, coords, points, axes = frame
    ids = list(features)
    voxels = np.concatenate([coords[i] for i in ids])
    labels = np.concatenate([np.full(len(coords[i]), i) for i in ids])
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


def keep_input(segmentation, path, reference):
    return segmentation


def correct_with_tips(segmentation, path, reference, method):
    return correct_ihc_row(segmentation, path, reference, VOXEL_SIZE, tip_method=method)[0]


def evaluate_tip_crop(crop, methods, with_grid=True):
    """The tip grid, the partition scores of all methods and the cell geometry for one annotated crop."""
    loaded = gt_row(crop)
    if loaded is None:
        return [], [], []
    gt, path, reference = loaded
    frame = _row_frame(gt, path, reference, VOXEL_SIZE, 15.0, 10.0, 15.0)
    species = "gerbil" if crop["cochlea"].startswith("G") else "mouse"
    info = {"crop": crop["name"], "val": crop["val"], "species": species}

    grid = []
    for method in TIP_METHODS if with_grid else []:
        for prominence in TIP_PROMINENCES:
            for spacing in TIP_SPACINGS:
                tips = find_tips(method, gt, frame, reference, min_prominence=prominence, min_spacing=spacing)
                grid.append({**info, "method": method, "prominence": prominence, "spacing": spacing,
                             **tip_counts(tips, frame[0])})

    simulated = simulate_errors(gt, frame)
    inputs = {
        "union": (label_structures(gt > 0)[0].astype("uint32"), reference),
        "simulated": (simulated, ihc_row_reference([simulated], path, VOXEL_SIZE)),
    }
    partition = []
    for scenario, (segmentation, scenario_reference) in inputs.items():
        for name, correct in methods.items():
            labels = correct(segmentation, path, scenario_reference)
            scores = matching(labels, gt)
            partition.append({**info, "scenario": scenario, "method": name, "n_gt": int(gt.max()),
                              "n_out": len(np.unique(labels)) - 1, **{k: float(v) for k, v in scores.items()}})

    if not with_grid:
        return grid, partition, []
    tips = find_tips("path", gt, frame, reference)
    cells = cell_geometry(gt, frame, reference, tips)
    return grid, partition, [{**info, **row} for row in cells]


def cell_geometry(gt, frame, reference, tips):
    """The tips of each complete annotated IHC in the path frame: its extreme voxels along the local axis.

    The offsets along the path ('ds') are between the foot points of the tip and of the cell center.
    """
    features, coords, points, axes = frame
    detected = tips.groupby(["label_id", "side"]).size()
    rows = []
    for label_id, feature in features.items():
        if feature["border"]:
            continue
        index, offset, height = foot_points(coords[label_id].astype(float), points, axes)
        length = (height.max() - height.min()) * VOXEL_SIZE
        row = {"label_id": label_id, "n_rel": feature["n_rel"], "length": length,
               "axis_angle": _angle(feature["axis"], axes[feature["index"]])}
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


def tip_reference(out_dir, methods=None, suffix=""):
    """Find the IHC tips in the annotated training crops, tune the detectors and score the tip-based correction.

    The grid and the cell geometry use the annotations directly. The partition test gives the correction two
    inputs: the connected components of the annotated cells, and the annotations with simulated merges and splits.
    """
    if methods is None:
        methods = {"input": keep_input, **{m: partial(correct_with_tips, method=m) for m in TIP_METHODS}}
    crops = gt_crops()
    # Each worker holds a few GB for a 512 x 512 x 256 crop.
    with ProcessPoolExecutor(3) as executor:
        results = list(executor.map(partial(evaluate_tip_crop, methods=methods, with_grid=not suffix), crops))
    grid, partition, cells = (pd.DataFrame(sum(parts, [])) for parts in zip(*results))
    os.makedirs(os.path.join(out_dir, "tips"), exist_ok=True)
    partition.to_csv(os.path.join(out_dir, "tips", f"partition{suffix}.tsv"), sep="\t", index=False)
    if suffix:
        return
    grid.to_csv(os.path.join(out_dir, "tips", "grid.tsv"), sep="\t", index=False)
    cells.to_csv(os.path.join(out_dir, "tips", "gt_cells.tsv"), sep="\t", index=False)
    summarize_tips(out_dir)


def summarize_tips(out_dir):
    folder = os.path.join(out_dir, "tips")
    cells = pd.read_csv(os.path.join(folder, "gt_cells.tsv"), sep="\t")
    columns = [c for c in cells.columns if c not in ("crop", "val", "species", "label_id")]
    print(f"{len(cells)} complete annotated IHCs in {cells['crop'].nunique()} crops")
    print(cells[columns].quantile([0.01, 0.05, 0.5, 0.95, 0.99]).T.round(2).to_string())

    grid = pd.read_csv(os.path.join(folder, "grid.tsv"), sep="\t")
    counts = ["n_gt", "one", "none", "multi", "one_paired"]
    totals = grid.groupby(["val", "method", "prominence", "spacing"])[counts].sum()
    for column in ["one", "none", "multi", "one_paired"]:
        totals[column] = (totals[column] / totals["n_gt"]).round(3)
    print("Tip cells per complete annotated IHC (fractions), train crops:")
    print(totals.loc[False].sort_values("one", ascending=False).to_string())

    paths = sorted(glob.glob(os.path.join(folder, "partition*.tsv")))
    partition = pd.concat([pd.read_csv(path, sep="\t") for path in paths])
    summary = partition.groupby(["scenario", "val", "method"])[["precision", "recall", "f1"]].mean().round(3)
    summary["crops"] = partition.groupby(["scenario", "val", "method"]).size()
    print("Partition scores (mean over crops, IoU 0.5):")
    print(summary.to_string())


STEPS = ["ellipsoids", "reference", "validate", "candidates", "tips"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", default=os.path.expanduser("~/Documents/lightsheet-cochlea/ihc_quality"))
    parser.add_argument("-s", "--steps", nargs="+", choices=STEPS, default=STEPS)
    parser.add_argument("-n", "--n_crops", type=int, default=3, help="Crop candidates per cochlea.")
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
        tip_reference(args.output)


if __name__ == "__main__":
    main()
