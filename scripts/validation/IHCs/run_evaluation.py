import argparse
import os
from glob import glob

import pandas as pd
from flamingo_tools.json_util import update_json
from flamingo_tools.validation import (
    _parse_annotation_path, compute_scores_for_annotated_slice, compute_scores_from_counts,
    fetch_data_for_evaluation,
)

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationIHCs"
# ANNOTATION_FOLDERS = ["AnnotationsEK", "AnnotationsAMD", "AnnotationsLR"]
# ANNOTATION_FOLDERS = ["Annotations_AMD", "Annotations_LR"]
ANNOTATION_FOLDERS = ["consensus_annotation"]

COMPONENT_DICT = {
    "IHC_v9": {
        "M_LR_000226_L": [1, 3],
        "M_LR_000226_R": [2],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
    "IHC_v10": {
        "M_LR_000226_L": [1, 3],
        "M_LR_000226_R": [2],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
    "IHC_v11": {
        "M_LR_000226_L": [1, 3],
        "M_LR_000226_R": [1],
        "M_LR_000227_L": [1],
        "M_LR_000227_R": [1],
    },
}


def run_evaluation(root, annotation_folders, output_file, cache_folder, segmentation_name, exclude):
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
        annotator = "consensus" if folder == "consensus_annotation" else folder[len("Annotations"):]
        annotations = sorted(glob(os.path.join(root, folder, "*.csv")))
        for annotation_path in annotations:
            print(annotation_path)
            cochlea, slice_id = _parse_annotation_path(annotation_path)

            if segmentation_name in list(COMPONENT_DICT.keys()):
                component = COMPONENT_DICT[segmentation_name][cochlea]
            else:
                # For the cochlea M_LR_000226_R the actual component is 2, not 1. (Only for IHC_v2).
                component = [2] if ("226_R" in cochlea and segmentation_name == "IHC_v2") else [1]

            print(f"Run evaluation for {annotator}, {cochlea}, z={slice_id}")
            segmentation, annotations = fetch_data_for_evaluation(
                annotation_path, components_for_postprocessing=component,
                seg_name=segmentation_name,
                cache_path=None if cache_folder is None else os.path.join(cache_folder, f"{cochlea}_{slice_id}.tif"),
                exclude_zero_synapse_count=exclude,
            )
            print(f"Evaluating segmentation with shape {segmentation.shape}")
            scores = compute_scores_for_annotated_slice(segmentation, annotations, matching_tolerance=5)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=ROOT)
    parser.add_argument("--folders", default=ANNOTATION_FOLDERS, nargs="+")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Optional directory to save accuracy JSON file (IHC_3D.json).")
    parser.add_argument("--segmentation_name", default="IHC_v4c")
    parser.add_argument("--cache_folder", default=None)
    parser.add_argument("--exclude", action="store_true")
    args = parser.parse_args()

    output_file = None
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, "IHC_3D.json")

    run_evaluation(args.input, args.folders, output_file, args.cache_folder, args.segmentation_name, args.exclude)


if __name__ == "__main__":
    main()
