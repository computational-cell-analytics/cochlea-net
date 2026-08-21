import argparse
import os
from glob import glob
from typing import List, Optional

import pandas as pd
from flamingo_tools.json_util import update_json
from flamingo_tools.validation import (
    _parse_annotation_path, compute_scores_for_annotated_slice, compute_scores_from_counts,
    fetch_data_for_evaluation,
)

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/final_annotations"  # noqa
ANNOTATION_FOLDERS = ["final_consensus_annotations"]
MATCHING_TOLERANCE = 5

# The MoBIE dataset name differs from the cochlea name parsed from the annotation file.
MOBIE_NAMES = {"M_LR_000169_R": "M_LR_000169_R_fused"}

COMPONENT_DICT = {
    "SGN_v2": {
        "M_AMD_000058_L": [1],
        "M_LR_000169_R": [1],
        "M_LR_000226_L": [1],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
    "SGN_v2-1": {
        "M_AMD_000058_L": [1],
        "M_LR_000169_R": [1],
        "M_LR_000226_L": [1],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
    "SGN_v2-2": {
        "M_AMD_000058_L": [1],
        "M_LR_000169_R": [1],
        "M_LR_000226_L": [1],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
    "SGN_v2-3": {
        "M_AMD_000058_L": [1],
        "M_LR_000169_R": [1],
        "M_LR_000226_L": [1],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
    "SGN_v2-4": {
        "M_AMD_000058_L": [1],
        "M_LR_000169_R": [1],
        "M_LR_000226_L": [1],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
}


def run_evaluation(
    root: str,
    annotation_folders: List[str],
    output_file: Optional[str],
    cache_folder: Optional[str],
    segmentation_name: str,
) -> None:
    """Evaluate an SGN segmentation against the manual annotations of thin validation slices.

    The annotated slices are thin slabs of a full cochlea volume. A network applied to such a
    slab cannot reach its full accuracy, so the segmentation is taken from the full 3D volume
    on S3 instead.

    Args:
        root: Root directory with the annotation folders.
        annotation_folders: Folders with the annotations in CSV format, relative to root.
        output_file: Optional path of the accuracy JSON file.
        cache_folder: Optional folder for caching the downloaded segmentation slices.
        segmentation_name: Name of the segmentation in the S3 bucket.
    """
    results = {
        "annotator": [],
        "cochlea": [],
        "slice": [],
        "tp": [],
        "fp": [],
        "fn": [],
        "name": [],
    }

    if cache_folder is not None:
        os.makedirs(cache_folder, exist_ok=True)

    for folder in annotation_folders:
        annotator = "consensus" if "consensus" in folder else folder[len("Annotations"):]
        annotation_paths = sorted(glob(os.path.join(root, folder, "*.csv")))
        for annotation_path in annotation_paths:
            print(annotation_path)
            cochlea, slice_id = _parse_annotation_path(annotation_path)
            component = COMPONENT_DICT.get(segmentation_name, {}).get(cochlea, [1])

            print(f"Run evaluation for {annotator}, {cochlea}, z={slice_id}")
            segmentation, annotations = fetch_data_for_evaluation(
                annotation_path, components_for_postprocessing=component,
                seg_name=segmentation_name, cochlea=MOBIE_NAMES.get(cochlea),
                cache_path=None if cache_folder is None else os.path.join(cache_folder, f"{cochlea}_{slice_id}.tif"),
            )
            print(f"Evaluating segmentation with shape {segmentation.shape}")
            scores = compute_scores_for_annotated_slice(
                segmentation, annotations, matching_tolerance=MATCHING_TOLERANCE
            )
            results["annotator"].append(annotator)
            results["name"].append(os.path.splitext(os.path.basename(annotation_path))[0])
            results["cochlea"].append(cochlea)
            results["slice"].append(slice_id)
            results["tp"].append(scores["tp"])
            results["fp"].append(scores["fp"])
            results["fn"].append(scores["fn"])

    table = pd.DataFrame(results)
    scores = compute_scores_from_counts(int(table.tp.sum()), int(table.fp.sum()), int(table.fn.sum()))

    print("All results:")
    print(table)
    print("Evaluation:")
    print("Precision:", scores["precision"])
    print("Recall:", scores["recall"])
    print("F1-Score:", scores["f1-score"])

    if output_file is not None:
        version_key = "_".join(segmentation_name.split("_")[1:])
        update_json({version_key: {
            "crops": table["name"].tolist(),
            "tp": [int(v) for v in table["tp"]],
            "fp": [int(v) for v in table["fp"]],
            "fn": [int(v) for v in table["fn"]],
            **scores,
        }}, output_file)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an SGN segmentation against the manual annotations of validation slices.")
    parser.add_argument("-i", "--input", default=ROOT,
                        help="Root directory with the annotation folders.")
    parser.add_argument("--folders", default=ANNOTATION_FOLDERS, nargs="+",
                        help="Folders with the annotations in CSV format, relative to the input directory.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Optional directory to save the accuracy JSON file (SGN_3D.json).")
    parser.add_argument("--segmentation_name", default="SGN_v2",
                        help="Name of the segmentation in the S3 bucket.")
    parser.add_argument("--cache_folder", default=None,
                        help="Optional folder for caching the downloaded segmentation slices.")
    args = parser.parse_args()

    output_file = None
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, "SGN_3D.json")

    run_evaluation(args.input, args.folders, output_file, args.cache_folder, args.segmentation_name)


if __name__ == "__main__":
    main()
