import argparse
import os
import pybdv
import sys
import imageio.v3 as imageio


def main(input_path: str, output_path: str = None) -> None:
    """Convert tif file to n5 format.
    If no output_path is supplied, the output file is created in the same directory as the input.

    Args:
        input_path: Input path to tif.
        output_path: Output path for n5 format.
    """
    if not os.path.isfile(input_path):
        sys.exit("Input file does not exist.")

    if input_path.split(".")[-1] not in ["TIFF", "TIF", "tiff", "tif"]:
        sys.exit("Input file must be in tif format.")

    basename = ".".join(input_path.split("/")[-1].split(".")[:-1])
    input_dir = os.path.abspath(input_path).split(basename)[0]

    if output_path is None:
        output_path = os.path.join(input_dir, f"{basename}.n5")

    img = imageio.imread(input_path)
    pybdv.make_bdv(img, output_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script to transform file from tif into n5 format.")

    parser.add_argument('-i', '--input', required=True, type=str, help="Input file")
    parser.add_argument('-o', "--output", type=str, default=None, help="Output file. Default: <basename>.n5")

    args = parser.parse_args()

    main(args.input, args.output)
