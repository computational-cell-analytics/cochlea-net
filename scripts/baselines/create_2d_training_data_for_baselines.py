import argparse
import os

from flamingo_tools.analysis.training_data_utils import create_2d_training_data

INPUT_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/SGN/2025-05_supervised" # noqa
OUTPUT_MICROSAM = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/SGN/2026-04_SGN-v2-data_micro-sam" # noqa
OUTPUT_CELLPOSE = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/SGN/2026-04_SGN-v2-data_cellpose3" # noqa


def create_2d_training_data_for_baselines(
    input_dir: str,
    output_dir: str,
    skip_empty: bool = False,
    empty_blocks: int = 2,
):
    """Create 2D training data for the baseline methods Cellpose3 and µSAM.

    Args:
        input_dir: Input directory featuring train and validation subdirectories.
        output_dir: Output directory for new training data
        """
    os.makedirs(output_dir, exist_ok=True)
    subdirs = ["train", "val"]
    for subdir in subdirs:
        in_dir = os.path.join(input_dir, subdir)
        out_dir = os.path.join(output_dir, subdir)
        print(f"Copying files from {in_dir} to {out_dir}.")
        create_2d_training_data(in_dir, out_dir, skip_empty=skip_empty, empty_blocks=empty_blocks)


def main():
    parser = argparse.ArgumentParser(
        description="Script to filter IHC annotations by removing small artifacts and isolated components.")

    parser.add_argument("-i", "--input", type=str, default=INPUT_DIR,
                        help="Input directory containing annotations.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for JSON dictionaries which feature parameters for crop extraction.")
    parser.add_argument("--skip_empty", action="store_true", help="Skip empty label data.")
    parser.add_argument("--empty_blocks", type=int, default=0, help="Create label data for first n empty 3D blocks.")

    args = parser.parse_args()

    if args.output is None:
        create_2d_training_data_for_baselines(
            input_dir=args.input,
            output_dir=OUTPUT_MICROSAM,
            skip_empty=True,
            empty_blocks=5,
        )

        create_2d_training_data_for_baselines(
            input_dir=args.input,
            output_dir=OUTPUT_CELLPOSE,
            skip_empty=True,
            empty_blocks=0,
        )

    else:
        create_2d_training_data_for_baselines(
            input_dir=args.input,
            output_dir=args.output,
            skip_empty=args.skip_empty,
            empty_blocks=args.empty_blocks,
        )


if __name__ == "__main__":
    main()
