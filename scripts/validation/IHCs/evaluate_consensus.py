import argparse
import os
from glob import glob
from typing import Optional

import pandas as pd
from flamingo_tools.json_util import update_json
from flamingo_tools.validation import compute_consensus_scores, evaluate_pairwise_agreement, match_detections

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationIHCs"
ANNOTATION_FOLDERS = ["Annotations_AMD", "Annotations_EK", "Annotations_LR"]
CONSENSUS_FOLDER = "consensus_annotation"
MATCHING_DISTANCE = 12.0


def match_annotations(consensus_path, root=ROOT, annotation_folders=ANNOTATION_FOLDERS):
    annotations = {}
    prefix = os.path.basename(consensus_path).split("_")[:3]
    prefix = "_".join(prefix)

    annotations = {}
    for annotation_folder in annotation_folders:
        all_annotations = glob(os.path.join(root, annotation_folder, "*.csv"))
        matches = [ann for ann in all_annotations if os.path.basename(ann).startswith(prefix)]
        assert len(matches) == 1
        annotation_path = matches[0]
        annotator = annotation_folder.split("_")[-1]
        annotations[annotator] = annotation_path

    return annotations


def annotations_per_crop(root: str = ROOT) -> dict:
    """Map each crop to the individual annotations of all annotators.

    Returns:
        Dictionary that maps a crop name to a dictionary of annotator name and annotation path.
    """
    return {
        os.path.splitext(os.path.basename(consensus_file))[0]: match_annotations(consensus_file, root=root)
        for consensus_file in sorted(glob(os.path.join(root, CONSENSUS_FOLDER, "*.csv")))
    }


def evaluate_consensus(
    root: str = ROOT,
    output_dir: Optional[str] = None,
    max_dist: float = MATCHING_DISTANCE,
) -> None:
    """Evaluate consensus annotation by comparing it to the individual annotations.

    Args:
        root: Root directory with the annotation folders and the consensus annotations.
        output_dir: Optional output directory for consensus_IHC.json.
        max_dist: Maximal matching distance in voxels for annotations.
    """
    consensus_dir = os.path.join(root, CONSENSUS_FOLDER)
    consensus_files = sorted(glob(os.path.join(consensus_dir, "*.csv")))
    if len(consensus_files) == 0:
        raise ValueError(f"Could not find any consensus annotation in {consensus_dir}.")

    results = {
        "annotator": [],
        "file_name": [],
        "tps": [],
        "fps": [],
        "fns": [],
    }
    for consensus_file in consensus_files:
        consensus = pd.read_csv(consensus_file)
        consensus = consensus[consensus.annotator == "consensus"][["axis-0", "axis-1", "axis-2"]]

        annotations = match_annotations(consensus_file, root=root)
        for name, annotation_path in annotations.items():
            annotation = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]]
            tp, _, fp, fn = match_detections(annotation, consensus, max_dist=max_dist)
            results["annotator"].append(name)
            results["file_name"].append(os.path.splitext(os.path.basename(consensus_file))[0])
            results["tps"].append(len(tp))
            results["fps"].append(len(fp))
            results["fns"].append(len(fn))

    results = pd.DataFrame(results)
    results.to_csv(os.path.join(root, "consensus_evaluation.csv"), index=False)

    scores = compute_consensus_scores(results)

    print("All results:")
    print(results)
    print("Evaluation:")
    print(pd.DataFrame(scores).T[["precision", "recall", "f1-score"]])

    if output_dir is not None:
        update_json(scores, os.path.join(output_dir, "consensus_IHC.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate IHC consensus annotations against the individual annotations.")
    parser.add_argument("-i", "--input", type=str, default=ROOT,
                        help="Root directory with the annotation folders and the consensus annotations.")
    parser.add_argument("-d", "--matching_distance", type=float, default=MATCHING_DISTANCE,
                        help="Matching distance in voxels for annotations.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Optional directory to save the accuracy JSON file (consensus_IHC.json).")
    parser.add_argument("--pairwise", action="store_true",
                        help="Also evaluate the direct agreement between all annotator pairs.")
    args = parser.parse_args()

    evaluate_consensus(root=args.input, output_dir=args.output_dir, max_dist=args.matching_distance)

    # The consensus is derived from the same annotations it is compared against, so the scores
    # above are correlated with the individual annotations. The pairwise agreement is not.
    if args.pairwise:
        scores = evaluate_pairwise_agreement(
            annotations_per_crop(args.input), table_dir=args.input,
            matching_distance=args.matching_distance,
        )
        if args.output_dir is not None:
            update_json({"pairwise": scores}, os.path.join(args.output_dir, "consensus_IHC.json"))


if __name__ == "__main__":
    main()
