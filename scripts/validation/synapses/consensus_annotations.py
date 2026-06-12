import argparse
import os
from glob import glob
from typing import Optional

import pandas as pd
from flamingo_tools.validation import create_consensus_annotations, match_detections
from flamingo_tools.synapse_detection.read_imaris_data import extract_training_data

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/Synapses_2026-04"
ANNOTATION_FOLDERS = [
    "for_consensus_annotations_synapses_AMD",
    "for_consensus_annotations_synapses_EK",
    "for_consensus_annotations_synapses_LR",
]

OUTPUT_FOLDER = os.path.join(ROOT, "consensus_annotation")


def create_images_and_labels(
    input_dir: str,
    output_dir: Optional[str] = None,
) -> None:
    """Create images and labels for all IMARIS files in a given directory.

    Args:
        input_dir: Input directory wih IMARIS files.
        output_dir: Output directory in which the subfolders "images" and "labels" are created.
    """
    if output_dir is None:
        output_dir = input_dir
    file_paths = [entry.path for entry in os.scandir(input_dir) if ".ims" in entry.name]
    for imaris_file in file_paths:
        extract_training_data(imaris_file, output_dir, crop=False)


def match_annotations(
    image_path: str,
    annotation_folders: list[str] = ANNOTATION_FOLDERS,
) -> dict:
    """Match annotations of multiple annotators.

    Args:
        image_path: Path to image from which its prefix is extracted.
        annotation_folder: Folders which are searched to find matching files with the same prefix.

    Returns:
        dictionary containing annotators and file path to annotation.
    """
    annotations = {}
    prefix = os.path.basename(image_path).split("_")[:3]
    prefix = "_".join(prefix)

    annotations = {}
    for annotation_folder in annotation_folders:
        all_annotations = glob(os.path.join(ROOT, annotation_folder, "labels", "*.csv"))
        matches = [ann for ann in all_annotations if os.path.basename(ann).startswith(prefix)]
        if len(matches) == 0:
            continue
        assert len(matches) == 1
        annotation_path = matches[0]
        annotator = annotation_folder.split("_")[-1]
        annotations[annotator] = annotation_path

    return annotations


def consensus_annotations(
    image_path: str,
    matching_distance: float = 2.0,
    annotation_folders: list[str] = ANNOTATION_FOLDERS,
) -> None:
    """Create consensus annotation for a single image.

    Args:
        image_path: Image in ZARR format.
        matching_distance: Maximal distance to match annotations.
    """
    annotation_paths = match_annotations(image_path)
    # assert len(annotation_paths) == len(ANNOTATION_FOLDERS)
    if len(annotation_paths) != len(annotation_folders):
        print(f"Incomplete annotations for {os.path.basename(image_path)}")

    matching_distance = matching_distance
    consensus_annotations, unmatched_annotations = create_consensus_annotations(
        annotation_paths, matching_distance=matching_distance, min_matches_for_consensus=2,
    )
    fname = os.path.basename(image_path)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    consensus_annotations = consensus_annotations[["axis-0", "axis-1", "axis-2"]]
    consensus_annotations.insert(0, "annotator", ["consensus"] * len(consensus_annotations))
    unmatched_annotations = unmatched_annotations[["axis-0", "axis-1", "axis-2", "annotator"]]
    annotations = pd.concat([consensus_annotations, unmatched_annotations])
    output_path = os.path.join(OUTPUT_FOLDER, fname.replace(".zarr", ".csv"))
    annotations.to_csv(output_path, index=False)
    print("Saved to", output_path)


def evaluate_consensus(
    consensus_dir: str,
    output_dir: str,
    max_dist: float = 2.0,
) -> None:
    """Evaluate consensus annotation by comparing it to the individual annotations.
    Creates an output file in which the analysis is saved.

    Args:
        consensus_dir: Directory containing annotations in CSV format.
        output_dir: Output directory for consensus_evaluation.csv
        max_dist: Maximal matching distance for spots.
    """
    consensus_files = sorted(glob(os.path.join(consensus_dir, "*.csv")))
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

        annotations = match_annotations(consensus_file)
        for name, annotation_path in annotations.items():
            annotation = pd.read_csv(annotation_path)[["axis-0", "axis-1", "axis-2"]]
            tp, _, fp, fn = match_detections(annotation, consensus, max_dist=max_dist)
            results["annotator"].append(name)
            file_name = os.path.splitext(os.path.basename(consensus_file))[0]
            results["file_name"].append(file_name)
            results["tps"].append(len(tp))
            results["fps"].append(len(fp))
            results["fns"].append(len(fn))

    results = pd.DataFrame(results)
    output_path = os.path.join(output_dir, "consensus_evaluation.csv")
    results.to_csv(output_path, index=False)

    tp = results.tps.sum()
    fp = results.fps.sum()
    fn = results.fns.sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("All results:")
    print(results)
    print("Evaluation:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", default=None)
    parser.add_argument("--annotation_dirs", nargs="+", default=None,
                        help="Annotation directories")

    parser.add_argument("-d", "--matching_distance", type=float, default=2.,
                        help="Matching distance in µm for annotations.")
    args = parser.parse_args()

    if args.annotation_dirs is None:
        annotation_folders = [os.path.join(ROOT, f) for f in ANNOTATION_FOLDERS]
    else:
        annotation_folders = args.annotation_dirs

    # extract image and annotation data from imaris files
    for annotation_folder in annotation_folders:
        number_imaris_files = len([entry for entry in os.scandir(annotation_folder) if ".ims" in entry.name])
        basename = os.path.basename(annotation_folder)
        image_dir = os.path.join(annotation_folder, "images")
        if not os.path.isdir(image_dir):
            print(f"Extracting image and label data for {basename}.")
            create_images_and_labels(annotation_folder)
        elif len(os.listdir(image_dir)) != number_imaris_files:
            print(f"Not all annotations have been extracted yet. Extracting image and label data for {basename}.")
            create_images_and_labels(annotation_folder)
        else:
            print(f"Image and label data are already extracted for {basename}.")

    # create consensus annotations based on individual annotations
    if args.images is None:
        image_paths = sorted(glob(os.path.join(ROOT, ANNOTATION_FOLDERS[0], "images", "*.zarr")))
    else:
        image_paths = args.images
    print("Creating consensus annotations.")
    for image_path in image_paths:
        consensus_annotations(image_path, args.matching_distance)

    # evaluate consensus annotation
    evaluate_consensus(consensus_dir=OUTPUT_FOLDER, output_dir=ROOT, max_dist=args.matching_distance)


if __name__ == "__main__":
    main()
