import argparse
import os
from glob import glob
from typing import Optional

import pandas as pd
from flamingo_tools.json_util import update_json
from flamingo_tools.validation import compute_consensus_scores, match_detections

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
    consensus_files = sorted(glob(os.path.join(root, CONSENSUS_FOLDER, "*.csv")))

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
    args = parser.parse_args()

    evaluate_consensus(root=args.input, output_dir=args.output_dir, max_dist=args.matching_distance)


if __name__ == "__main__":
    main()
