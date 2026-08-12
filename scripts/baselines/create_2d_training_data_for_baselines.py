import argparse
import os

from flamingo_tools.analysis.training_data_utils import create_2d_training_data


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
        description="Script to prepare 3D training and validation data for finetuning of Cellpose3 and micro-sam. "
        "Both networks expect 2D training data so every 3D block ")

    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Input directory containing annotations.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for JSON dictionaries which feature parameters for crop extraction.")
    parser.add_argument("--skip_empty", action="store_true", help="Skip empty label data.")
    parser.add_argument("--empty_blocks", type=int, default=0, help="Create label data for first n empty 3D blocks.")
    parser.add_argument("--cell_type", type=str, default="sgn", help="Select cell type for default data preparation.")

    args = parser.parse_args()

    if args.output is None and args.input is None:
        training_data_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data"
        if args.cell_type == "sgn":
            input_dir = os.path.join(training_data_dir, "SGN", "2025-05_supervised")
            output_microsam = os.path.join(training_data_dir, "SGN", "2026-04_SGN-v2-data_micro-sam")
            output_cellpose = os.path.join(training_data_dir, "SGN", "2026-04_SGN-v2-data_cellpose3")
        elif args.cell_type == "ihc":
            input_dir = os.path.join(training_data_dir, "IHC", "IHC_v11_2026-07")
            output_microsam = os.path.join(training_data_dir, "IHC", "2026-07_IHC-v11-data_micro-sam")
            output_cellpose = os.path.join(training_data_dir, "IHC", "2026-07_IHC-v11-data_cellpose3")

        create_2d_training_data_for_baselines(
            input_dir=input_dir,
            output_dir=output_microsam,
            skip_empty=True,
            empty_blocks=5,
        )

        create_2d_training_data_for_baselines(
            input_dir=input_dir,
            output_dir=output_cellpose,
            skip_empty=True,
            empty_blocks=0,
        )

    elif args.output is not None and args.input is not None:
        create_2d_training_data_for_baselines(
            input_dir=args.input,
            output_dir=args.output,
            skip_empty=args.skip_empty,
            empty_blocks=args.empty_blocks,
        )

    else:
        raise ValueError("Either provide --input and --output or neither for default processing using --cell_type.")


if __name__ == "__main__":
    main()
