"""Verify the IHC row correction on the IHC evaluation planes of run_evaluation.py (the test data).

The 12 consensus annotations are points in one z-plane of the test cochleae M_LR_000226/227. The script scores
the S3 segmentation in each plane like run_evaluation.py (all consensus rows, matching tolerance 5 pixels), for
these variants of the IHC row:
- 'components': the published IHC_v11 rows (PUBLISHED_ROWS). They must give the scores of
  reproducibility/model_accuracy/IHC_3D.json, else the script stops.
- 'refined_<min_size>': `refine_ihc_row`, seeded with the same components, with max_path_deviation 15 um and
  the row merged into one component.
- '<variant>+correct': `correct_ihc_row` with its defaults, in 3D windows around the row crossings of the plane
  (`common.correct_region`), along the row path of `common.row_path`.

'count_error' is the number of row objects in the plane minus the number of annotated points. A missed point
(FN) is a model miss ('fn_miss') if no row object is within the matching tolerance. Else the nearest object
matched another point, and the point is part of a merge ('fn_merge').

The S3 tables of the test cochleae got new IHC rows on 2026-11-11 (refine_row with min_size 5000). The script
replays the labeling of the published rows from the S3 instances, so it does not depend on the S3 component labels.
"""
import argparse
import json
import os
from glob import glob

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import relabel_sequential

from flamingo_tools.postprocessing.label_components import label_components_ihc
from flamingo_tools.s3_utils import default_table_path, read_table
from flamingo_tools.validation import (
    _parse_annotation_path, compute_matches_for_annotated_slice, compute_scores_from_counts,
)

from common import REPOSITORY, correct_region, read_volume, row_variants, volume_shape
from export_evaluation_data import ALIASES

ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea")
ANNOTATIONS = os.path.join(ROOT, "final_submission", "evaluation_data_standardized", "IHC_evaluation", "consensus")
ACCURACY = os.path.join(REPOSITORY, "reproducibility", "model_accuracy", "IHC_3D.json")
SEGMENTATION = "IHC_v11"
# The labeling parameters and the row components of the published IHC_v11 rows (the state of IHC_3D.json).
PUBLISHED_ROWS = {
    "M_LR_000226_L": ({"min_size": 1000, "min_component_length": 20, "max_edge_distance": 30}, [1, 3]),
    "M_LR_000226_R": ({"min_size": 1000, "min_component_length": 50, "max_edge_distance": 30}, [1]),
    "M_LR_000227_L": ({"min_size": 1000, "min_component_length": 50, "max_edge_distance": 30}, [1]),
    "M_LR_000227_R": ({"min_size": 1000, "min_component_length": 50, "max_edge_distance": 30}, [1]),
}
MATCHING_TOLERANCE = 5
MIN_SIZES = [1000, 5000, 10000]
CORRECTED = ["components", "refined_5000"]


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


def published_table(cochlea):
    """The S3 table of the cochlea with the component labels of the published IHC row, and the row components."""
    table = read_table(default_table_path(cochlea, SEGMENTATION, s3=True), s3=True)
    params, components = PUBLISHED_ROWS[cochlea]
    return label_components_ihc(table, **params), components


def score(labels, annotations):
    labels = relabel_sequential(labels)[0]
    matches = compute_matches_for_annotated_slice(labels, annotations, matching_tolerance=MATCHING_TOLERANCE)
    points = np.round(annotations[["axis-1", "axis-2"]].values[matches["fn"]]).astype(int)
    near = distance_transform_edt(labels == 0)[tuple(points.T)] <= MATCHING_TOLERANCE
    return {"tp": len(matches["tp_objects"]), "fp": len(matches["fp"]), "fn": len(matches["fn"]),
            "fn_miss": int((~near).sum()), "fn_merge": int(near.sum()),
            "n_objects": len(np.unique(labels)) - 1, "n_points": len(annotations)}


def check_components(results):
    """Stop if the 'components' variant does not give the scores of IHC_3D.json."""
    with open(ACCURACY) as f:
        published = json.load(f)[SEGMENTATION.split("_", 1)[1]]
    expected = pd.DataFrame({key: published[key] for key in ("tp", "fp", "fn")}, index=published["crops"])
    current = results[results["variant"] == "components"].set_index("name")[["tp", "fp", "fn"]]
    difference = current.reindex(expected.index) - expected
    if difference.abs().values.sum() > 0:
        print(difference)
        raise RuntimeError("The 'components' variant does not give the scores of IHC_3D.json.")
    print(f"The 'components' variant gives the scores of IHC_3D.json for {SEGMENTATION}.")


def summarize(results):
    rows = []
    for variant, group in results.groupby("variant", sort=False):
        error = group["n_objects"] - group["n_points"]
        rows.append({
            "variant": variant, "tp": group["tp"].sum(), "fp": group["fp"].sum(), "fn": group["fn"].sum(),
            **compute_scores_from_counts(int(group["tp"].sum()), int(group["fp"].sum()), int(group["fn"].sum())),
            "n_objects": group["n_objects"].sum(), "n_points": group["n_points"].sum(),
            "count_error": error.sum(), "abs_count_error": error.abs().sum(),
            "fn_miss": group["fn_miss"].sum(), "fn_merge": group["fn_merge"].sum(),
        })
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    columns = ["tp", "fp", "fn", "n_objects", "n_points"]
    print(results.pivot(index="name", columns="variant", values="fn").to_string())
    print(results.groupby("variant", sort=False)[columns].sum().to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default=ANNOTATIONS, help="Folder of the standardized consensus CSVs.")
    parser.add_argument("-o", "--output", default=os.path.join(ROOT, "ihc_postprocessing"))
    parser.add_argument("--cache", default=None, help="Folder of the cached S3 planes and windows. "
                        "Default: <output>/cache.")
    args = parser.parse_args()

    cache_folder = args.cache or os.path.join(args.output, "cache")
    plane_folder = os.path.join(args.output, f"planes_{SEGMENTATION}")
    os.makedirs(cache_folder, exist_ok=True)
    os.makedirs(plane_folder, exist_ok=True)
    tables, shapes, results = {}, {}, []
    for plane_info in evaluation_planes(args.input):
        name, cochlea, z = plane_info["name"], plane_info["cochlea"], plane_info["z"]
        print(f"{name}: {cochlea}, z {z}")
        if cochlea not in tables:
            table, components = published_table(cochlea)
            tables[cochlea] = (table, row_variants(table, components, MIN_SIZES))
            shapes[cochlea] = volume_shape(cochlea, SEGMENTATION)
        table, variants = tables[cochlea]
        plane = read_volume(cochlea, SEGMENTATION, z, os.path.join(cache_folder, f"{cochlea}_{SEGMENTATION}_z{z}.tif"))
        annotations = pd.read_csv(plane_info["path"]).drop(columns=["index", "axis-0"], errors="ignore")

        for variant, (row_ids, path) in variants.items():
            labels = np.where(np.isin(plane, row_ids), plane, 0)
            results.append({"name": name, "variant": variant, **score(labels, annotations)})
            if variant in CORRECTED:
                corrected, actions = correct_region(plane[None], (z, 0, 0), table, row_ids, path, cochlea,
                                                    SEGMENTATION, cache_folder, shapes[cochlea])
                corrected = corrected[0]
                tifffile.imwrite(os.path.join(plane_folder, f"{name}_{variant}+correct.tif"), corrected,
                                 compression="zlib")
                results.append({"name": name, "variant": f"{variant}+correct", **score(corrected, annotations),
                                **actions})
        print(pd.DataFrame(results[-len(variants) - len(CORRECTED):]).to_string(index=False))

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.output, "evaluation_planes.tsv"), sep="\t", index=False)
    check_components(results)
    summarize(results)


if __name__ == "__main__":
    main()
