"""Evaluate the current IHC post-processing on the IHC evaluation planes of run_evaluation.py.

The 12 consensus annotations are points in one z-plane of the test cochleae M_LR_000226/227. The script scores
the S3 segmentation in each plane like run_evaluation.py (all consensus rows, matching tolerance 5 pixels), for
these variants of the IHC row:
- 'components': the objects of the COMPONENT_DICT components, the current state. They must give the scores of
  reproducibility/model_accuracy/IHC_3D.json, else the script stops.
- 'refined_<min_size>': `refine_ihc_row`, seeded with the same components, with max_path_deviation 15 um and
  the row merged into one component.
- '<variant>+correct': `correct_ihc_row` with its defaults, in 3D windows around the row crossings of the plane.
  A crossing is a run of plane objects along the row path. Its window holds the plane objects and the row
  objects within CONTEXT um along the path, for the reference IHC and the local axis. Only the plane voxels of
  the crossing's plane objects take the corrected labels.

The settings are the current ones, without tuning (point 8 of ihc_segmentation_to-do.md).
"""
import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.spatial import cKDTree
from skimage.segmentation import relabel_sequential

from flamingo_tools.analysis.training_data_utils import correct_ihc_row, ihc_row_reference
from flamingo_tools.postprocessing.label_components import ihc_row_path, refine_ihc_row
from flamingo_tools.s3_utils import default_table_path, get_s3_path, read_table
from flamingo_tools.validation import (
    _parse_annotation_path, compute_scores_for_annotated_slice, compute_scores_from_counts,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from export_evaluation_data import ALIASES  # noqa: E402
from run_evaluation import COMPONENT_DICT  # noqa: E402

ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea")
ANNOTATIONS = os.path.join(ROOT, "final_submission", "evaluation_data_standardized", "IHC_evaluation", "consensus")
ACCURACY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "reproducibility", "model_accuracy", "IHC_3D.json"
)
VOXEL_SIZE = 0.38
MATCHING_TOLERANCE = 5
MIN_SIZES = [1000, 5000, 10000]
MAX_PATH_DEVIATION = 15.0
CORRECTED = ["components", "refined_5000"]
COLUMNS = ["anchor_x", "anchor_y", "anchor_z"]
# Lengths along the row path in um: a larger gap between plane objects starts a new crossing; a crossing is
# split after MAX_CROSSING, so that the windows stay small; the context and the path reach beyond the crossing.
CROSSING_GAP = 50.0
# ponytail: a merge of two plane objects across a split of a long crossing gives two labels. Upgrade: split at
# the largest gap of a long crossing.
MAX_CROSSING = 150.0
CONTEXT = 40.0
PATH_MARGIN = 60.0
HALO = 5.0


def evaluation_planes(folder):
    """The annotated planes, with the original name of run_evaluation.py, the cochlea and the plane index."""
    original = {alias: cochlea_id for cochlea_id, alias in ALIASES.items()}
    planes = []
    for path in sorted(glob(os.path.join(folder, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        alias, rest = stem[:5], stem[6:]
        name = f"{original[alias]}_{rest}"
        cochlea, z = _parse_annotation_path(f"{name}.csv")
        planes.append({"name": name, "path": path, "cochlea": cochlea, "z": z})
    return planes


def read_volume(cochlea, segmentation_name, roi, cache_path):
    if os.path.exists(cache_path):
        return tifffile.imread(cache_path)
    store, _ = get_s3_path(f"{cochlea}/images/ome-zarr/{segmentation_name}.ome.zarr")
    data = zarr.open(store, mode="r")["s0"][roi]
    assert data.max() < 2 ** 32
    data = data.astype("uint32")
    tifffile.imwrite(cache_path, data, compression="zlib")
    return data


def volume_shape(cochlea, segmentation_name):
    store, _ = get_s3_path(f"{cochlea}/images/ome-zarr/{segmentation_name}.ome.zarr")
    return zarr.open(store, mode="r")["s0"].shape


def row_variants(table, components):
    """The label ids and the central path (um, (x, y, z) order) of the IHC row for each variant."""
    label_ids = table["label_id"].values.astype("int64")
    row = table["component_labels"].isin(components).values
    variants = {"components": (label_ids[row], ihc_row_path(table[COLUMNS].values[row]))}
    for min_size in MIN_SIZES:
        refined, path, _ = refine_ihc_row(table.copy(), components, min_size, MAX_PATH_DEVIATION,
                                          merge_components=True)
        variants[f"refined_{min_size}"] = (label_ids[refined["component_labels"].values == 1], path)
    return variants


def crossings(plane_ids, position):
    """Group the plane objects into runs along the path. `position` maps the row ids to the path index (um)."""
    order = position.loc[plane_ids].sort_values()
    groups, start, last = [], None, None
    for label_id, index in order.items():
        if start is None or index - last > CROSSING_GAP or index - start > MAX_CROSSING:
            groups.append([])
            start = index
        groups[-1].append(label_id)
        last = index
    return groups


def window_box(table, ids, shape):
    """The bounding box of the objects plus HALO, in pixels, (z, y, x) order, clipped to the volume."""
    rows = table[table["label_id"].isin(ids)]
    low = np.floor(rows[["bb_min_z", "bb_min_y", "bb_min_x"]].values.min(axis=0) / VOXEL_SIZE - HALO / VOXEL_SIZE)
    high = np.ceil(rows[["bb_max_z", "bb_max_y", "bb_max_x"]].values.max(axis=0) / VOXEL_SIZE + HALO / VOXEL_SIZE)
    low = np.maximum(low.astype(int), 0)
    high = np.minimum(high.astype(int) + 1, shape)
    return low, high


def correct_plane(plane, table, row_ids, path, cochlea, segmentation_name, z, cache_folder, shape):
    """`correct_ihc_row` in a window around each crossing; returns the corrected plane and the action counts."""
    plane_ids = np.intersect1d(np.unique(plane), row_ids)
    rows = table[table["label_id"].isin(row_ids)]
    position = pd.Series(cKDTree(path).query(rows[COLUMNS].values)[1], index=rows["label_id"].values.astype("int64"))
    corrected = np.zeros(plane.shape, dtype="uint32")
    actions, offset = {"failed": 0}, 0
    for group in crossings(plane_ids, position):
        low_index, high_index = position.loc[group].min(), position.loc[group].max()
        context = position.index[(position >= low_index - CONTEXT) & (position <= high_index + CONTEXT)]
        low, high = window_box(table, context, shape)
        key = "_".join(f"{a}-{b}" for a, b in zip(low, high))
        window = read_volume(cochlea, segmentation_name, tuple(slice(a, b) for a, b in zip(low, high)),
                             os.path.join(cache_folder, f"{cochlea}_{segmentation_name}_{key}.tif"))
        window, forward, _ = relabel_sequential(np.where(np.isin(window, row_ids), window, 0))
        window = window.astype("uint32")
        local_path = path[max(low_index - int(PATH_MARGIN), 0):high_index + int(PATH_MARGIN) + 1]
        local_path = local_path[:, ::-1] / VOXEL_SIZE - low
        group_ids = forward[np.array(group)]
        mask = np.isin(window[z - low[0]], group_ids)
        target = corrected[low[1]:high[1], low[2]:high[2]]
        try:
            reference = ihc_row_reference([window], local_path, VOXEL_SIZE)
            labels, report, _ = correct_ihc_row(window, local_path, reference, VOXEL_SIZE)
        except Exception as error:
            # One failed window keeps its input, so the other crossings of the plane are still scored.
            print(f"  correct_ihc_row failed for {len(group)} plane objects at z {z}, window {key}: {error!r}")
            actions["failed"] += 1
            labels, report = window, None
        output = labels[z - low[0]]
        target[mask] = np.where(output[mask] > 0, output[mask] + offset, 0)
        offset += int(labels.max()) + 1
        if report is not None:
            for action, count in report[report["label_id"].isin(group_ids)]["action"].value_counts().items():
                actions[action] = actions.get(action, 0) + int(count)
        print(f"  crossing of {len(group)} plane objects, window {high - low} px")
    return corrected, actions


def score(labels, annotations):
    labels = relabel_sequential(labels)[0]
    result = compute_scores_for_annotated_slice(labels, annotations, matching_tolerance=MATCHING_TOLERANCE)
    return {**result, "n_objects": len(np.unique(labels)) - 1, "n_points": len(annotations)}


def check_components(results, segmentation_name):
    """Stop if the 'components' variant does not give the scores of IHC_3D.json."""
    with open(ACCURACY) as f:
        published = json.load(f).get(segmentation_name.split("_", 1)[1])
    if published is None:
        print(f"IHC_3D.json has no entry for {segmentation_name}, no check.")
        return
    expected = pd.DataFrame({key: published[key] for key in ("tp", "fp", "fn")}, index=published["crops"])
    current = results[results["variant"] == "components"].set_index("name")[["tp", "fp", "fn"]]
    difference = current.reindex(expected.index) - expected
    if difference.abs().values.sum() > 0:
        print(difference)
        raise RuntimeError("The 'components' variant does not give the scores of IHC_3D.json.")
    print(f"The 'components' variant gives the scores of IHC_3D.json for {segmentation_name}.")


def summarize(results):
    rows = []
    for variant, group in results.groupby("variant", sort=False):
        error = group["n_objects"] - group["n_points"]
        rows.append({
            "variant": variant, "tp": group["tp"].sum(), "fp": group["fp"].sum(), "fn": group["fn"].sum(),
            **compute_scores_from_counts(int(group["tp"].sum()), int(group["fp"].sum()), int(group["fn"].sum())),
            "n_objects": group["n_objects"].sum(), "n_points": group["n_points"].sum(),
            "count_error": error.sum(), "abs_count_error": error.abs().sum(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    columns = ["tp", "fp", "fn", "n_objects", "n_points"]
    print(results.pivot(index="name", columns="variant", values="fn").to_string())
    print(results.groupby("variant", sort=False)[columns].sum().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default=ANNOTATIONS, help="Folder of the standardized consensus CSVs.")
    parser.add_argument("-o", "--output", default=os.path.join(ROOT, "ihc_postprocessing"))
    parser.add_argument("--segmentation_name", default="IHC_v11")
    args = parser.parse_args()

    cache_folder = os.path.join(args.output, "cache")
    plane_folder = os.path.join(args.output, f"planes_{args.segmentation_name}")
    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(plane_folder, exist_ok=True)
    tables, shapes, results = {}, {}, []
    for plane_info in evaluation_planes(args.input):
        name, cochlea, z = plane_info["name"], plane_info["cochlea"], plane_info["z"]
        print(f"{name}: {cochlea}, z {z}")
        if cochlea not in tables:
            table = read_table(default_table_path(cochlea, args.segmentation_name, s3=True), s3=True)
            tables[cochlea] = (table, row_variants(table, COMPONENT_DICT[args.segmentation_name][cochlea]))
            shapes[cochlea] = volume_shape(cochlea, args.segmentation_name)
        table, variants = tables[cochlea]
        plane = read_volume(cochlea, args.segmentation_name, z,
                            os.path.join(cache_folder, f"{cochlea}_{args.segmentation_name}_z{z}.tif"))
        annotations = pd.read_csv(plane_info["path"]).drop(columns=["index", "axis-0"], errors="ignore")

        for variant, (row_ids, path) in variants.items():
            labels = np.where(np.isin(plane, row_ids), plane, 0)
            results.append({"name": name, "variant": variant, **score(labels, annotations)})
            if variant in CORRECTED:
                corrected, actions = correct_plane(plane, table, row_ids, path, cochlea, args.segmentation_name,
                                                   z, cache_folder, shapes[cochlea])
                tifffile.imwrite(os.path.join(plane_folder, f"{name}_{variant}+correct.tif"), corrected,
                                 compression="zlib")
                results.append({"name": name, "variant": f"{variant}+correct", **score(corrected, annotations),
                                **actions})
        print(pd.DataFrame(results[-len(variants) - len(CORRECTED):]).to_string(index=False))

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.output, f"evaluation_{args.segmentation_name}.tsv"), sep="\t", index=False)
    check_components(results, args.segmentation_name)
    summarize(results)


if __name__ == "__main__":
    main()
