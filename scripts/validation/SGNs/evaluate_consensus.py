import argparse
import os
from glob import glob
from pathlib import Path
from typing import Optional

import pandas as pd
from flamingo_tools.json_util import update_json
from flamingo_tools.validation import compute_consensus_scores, evaluate_pairwise_agreement, match_detections

# The regular root folder.
ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
# The root folder for the new annotations for data with scaling issues.
ROOT2 = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"  # noqa

ANNOTATION_FOLDERS = ["AnnotationsAMD", "AnnotationsEK", "AnnotationsLR"]
CONSENSUS_FOLDER = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs/final_annotations/final_consensus_annotations"  # noqa
MATCHING_DISTANCE = 8.0


def match_annotations(consensus_path, sample_name, root=ROOT, annotation_folders=ANNOTATION_FOLDERS):
    annotations = {}
    prefix = os.path.basename(consensus_path).split("_")[:3]
    prefix = "_".join(prefix)

    if sample_name in ("MLR169R_PV_z1913_base_full_rescaled", "MLR169R_PV_z2594_mid_full_rescaled"):
        root = ROOT2

    annotations = {}
    for annotation_folder in annotation_folders:
        all_annotations = glob(os.path.join(root, annotation_folder, "*.csv"))
        matches = [
            ann for ann in all_annotations if (os.path.basename(ann).startswith(prefix) and "negative" not in ann)
        ]
        assert len(matches) == 1, f"Expected exactly one annotation for {prefix} in {annotation_folder}: {matches}"
        annotation_path = matches[0]
        annotator = annotation_folder[len("Annotations"):]
        annotations[annotator] = annotation_path

    return annotations


def annotations_per_crop(consensus_dir: str = CONSENSUS_FOLDER, root: str = ROOT) -> dict:
    """Map each sample to the individual annotations of all annotators.

    Returns:
        Dictionary that maps a sample name to a dictionary of annotator name and annotation path.
    """
    annotations = {}
    for consensus_file in sorted(glob(os.path.join(consensus_dir, "*.csv"))):
        sample_name = Path(consensus_file).stem
        annotations[sample_name] = match_annotations(consensus_file, sample_name, root=root)
    return annotations


def evaluate_consensus(
    root: str = ROOT,
    consensus_dir: str = CONSENSUS_FOLDER,
    output_dir: Optional[str] = None,
    max_dist: float = MATCHING_DISTANCE,
) -> None:
    """Evaluate consensus annotation by comparing it to the individual annotations.

    Args:
        root: Root directory with the annotation folders. Also holds consensus_evaluation.csv.
        consensus_dir: Directory with the consensus annotations in CSV format.
        output_dir: Optional output directory for consensus_SGN.json.
        max_dist: Maximal matching distance in voxels for annotations.
    """
    consensus_files = sorted(glob(os.path.join(consensus_dir, "*.csv")))
    assert len(consensus_files) > 0

    results = {
        "annotator": [],
        "file_name": [],
        "tps": [],
        "fps": [],
        "fns": [],
    }
    for consensus_file in consensus_files:
        consensus = pd.read_csv(consensus_file)
        consensus = consensus[["axis-0", "axis-1", "axis-2"]]
        sample_name = Path(consensus_file).stem

        annotations = match_annotations(consensus_file, sample_name, root=root)
        for name, annotation_path in annotations.items():
            annotation = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]]
            tp, _, fp, fn = match_detections(annotation, consensus, max_dist=max_dist)
            results["annotator"].append(name)
            results["file_name"].append(sample_name)
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
        update_json(scores, os.path.join(output_dir, "consensus_SGN.json"))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SGN consensus annotations against the individual annotations.")
    parser.add_argument("-i", "--input", type=str, default=ROOT,
                        help="Root directory with the annotation folders. Also holds consensus_evaluation.csv.")
    parser.add_argument("--consensus_dir", type=str, default=CONSENSUS_FOLDER,
                        help="Directory with the consensus annotations in CSV format.")
    parser.add_argument("-d", "--matching_distance", type=float, default=MATCHING_DISTANCE,
                        help="Matching distance in voxels for annotations.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Optional directory to save the accuracy JSON file (consensus_SGN.json).")
    parser.add_argument("--pairwise", action="store_true",
                        help="Also evaluate the direct agreement between all annotator pairs.")
    args = parser.parse_args()

    evaluate_consensus(
        root=args.input, consensus_dir=args.consensus_dir,
        output_dir=args.output_dir, max_dist=args.matching_distance,
    )

    # The consensus is derived from the same annotations it is compared against, so the scores
    # above are correlated with the individual annotations. The pairwise agreement is not.
    if args.pairwise:
        scores = evaluate_pairwise_agreement(
            annotations_per_crop(args.consensus_dir, root=args.input), table_dir=args.input,
            matching_distance=args.matching_distance,
        )
        if args.output_dir is not None:
            update_json({"pairwise": scores}, os.path.join(args.output_dir, "consensus_SGN.json"))


if __name__ == "__main__":
    main()
