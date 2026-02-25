import argparse
import os

import zarr

from elf.parallel import unique
from elf.io import open_file

import flamingo_tools.s3_utils as s3_utils


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_folder", type=str, default=None,
                        help="Output directory containing segmentation.zarr.")

    parser.add_argument("-k", "--input_key", type=str, default="segmentation",
                        help="The key / internal path of the segmentation.")
    parser.add_argument("-m", "--min_size", type=int, default=1000,
                        help="Minimal number of voxel size for counting object.")

    parser.add_argument("--s3_input", type=str, default=None, help="Input file path on S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    if args.output_folder is not None:
        seg_path = os.path.join(args.output_folder, "segmentation.zarr")
    elif args.s3_input is None:
        raise ValueError("Either provide an output_folder containing 'segmentation.zarr' or an S3 input.")

    if args.s3_input is not None:
        s3_path, fs = s3_utils.get_s3_path(args.s3_input, bucket_name=args.s3_bucket_name,
                                           service_endpoint=args.s3_service_endpoint,
                                           credential_file=args.s3_credentials)

        with zarr.open(s3_path, mode="r") as f:
            dataset = f[args.input_key]

    else:
        segmentation = open_file(seg_path, mode='r')
        dataset = segmentation[args.input_key]

    ids, counts = unique(dataset, return_counts=True)

    # You can change the minimal size for objects to be counted here:
    min_size = args.min_size

    counts = counts[counts > min_size]
    print("Number of objects:", len(counts))


if __name__ == "__main__":
    main()
