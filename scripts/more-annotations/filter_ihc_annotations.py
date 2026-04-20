import argparse
import os

import imageio.v3 as imageio

from flamingo_tools.analysis.training_data_utils import filter_segmentation_3d

INPUT_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/IHC_training_crops_2026-04/annotations" # noqa
OUTPUT_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/IHC_training_crops_2026-04/edited_annotations" # noqa


def filter_ihc_annotations(
    input_dir: str,
    output_dir: str,
    force_overwrite: bool = False,
):
    """Filter IHC annotations by removing small artifacts and isolated components.

    Args:
        input_dir: Directory containing annotations in TIF format.
        output_dir: Output directory for filtered annotations.
    """
    os.makedirs(output_dir, exist_ok=True)
    annotation_paths = [entry.path for entry in os.scandir(input_dir)]
    annotation_names = [entry.name for entry in os.scandir(input_dir)]
    annotation_paths.sort()
    annotation_names.sort()
    for annotation_path, annotation_name in zip(annotation_paths, annotation_names):
        out_path = os.path.join(output_dir, annotation_name)
        if os.path.isfile(out_path) and not force_overwrite:
            print(f"Skipping {annotation_name}, output already exists.")
            continue
        print(annotation_name)
        arr = imageio.imread(annotation_path)
        filtered_array = filter_segmentation_3d(arr)

        imageio.imwrite(out_path, filtered_array)


def main():
    parser = argparse.ArgumentParser(
        description="Script to filter IHC annotations by removing small artifacts and isolated components.")

    parser.add_argument("-i", "--input", type=str, default=INPUT_DIR,
                        help="Input directory containing annotations.")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                        help="Output directory for JSON dictionaries which feature parameters for crop extraction.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    args = parser.parse_args()

    filter_ihc_annotations(
        input_dir=args.input,
        output_dir=args.output,
        force_overwrite=args.force,
    )


if __name__ == "__main__":
    main()
