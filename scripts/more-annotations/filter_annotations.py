import argparse

from flamingo_tools.analysis.training_data_utils import filter_annotations

INPUT_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/IHC_training_crops_2026-04/annotations" # noqa
OUTPUT_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/IHC_training_crops_2026-04/edited_annotations" # noqa


def main():
    parser = argparse.ArgumentParser(
        description="Filter segmentation annotations by removing small instances and, "
                    "optionally, small disconnected sub-components within an instance.")

    parser.add_argument("-i", "--input", type=str, default=INPUT_DIR,
                        help="Input directory containing annotations.")
    parser.add_argument("-o", "--output", type=str, default=OUTPUT_DIR,
                        help="Output directory for refined annotations.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--min_pixels_per_instance", type=int, default=100,
                        help="Minimum number of pixels an instance must have to be kept.")
    parser.add_argument("--min_pixels_per_component", type=int, default=100,
                        help="Minimum number of pixels a disconnected sub-component of an "
                             "instance must have to be kept. Only used with split filtering.")
    parser.add_argument("--no_split_filter", action="store_true",
                        help="Only filter by instance size, skip filtering disconnected "
                             "sub-components within an instance (e.g. for SGNs).")

    args = parser.parse_args()

    filter_annotations(
        input_dir=args.input,
        output_dir=args.output,
        force_overwrite=args.force,
        min_pixels_per_instance=args.min_pixels_per_instance,
        min_pixels_per_component=args.min_pixels_per_component,
        filter_split_components=not args.no_split_filter,
    )


if __name__ == "__main__":
    main()
