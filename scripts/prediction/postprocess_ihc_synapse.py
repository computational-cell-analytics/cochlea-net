"""This script post-processes IHC segmentation with too many synapses based on a base segmentation and a reference.
"""
import argparse

import imageio.v3 as imageio
import pandas as pd
from elf.io import open_file

import flamingo_tools.segmentation.ihc_synapse_postprocessing as ihc_synapse_postprocessing
from flamingo_tools.file_utils import read_image_data


def main():
    parser = argparse.ArgumentParser(
        description="Script to postprocess IHC segmentation based on the number of synapses per IHC.")

    parser.add_argument("--base_path", type=str, required=True, help="Base segmentation. WARNING: Will be edited.")
    parser.add_argument("--ref_path", type=str, required=True, help="Reference segmentation.")
    parser.add_argument("--out_path_tif", type=str, default=None, help="Output segmentation for tif output.")

    parser.add_argument("--base_table", type=str, required=True, help="Synapse per IHC table of base segmentation.")

    parser.add_argument("--base_key", type=str, default=None,
                        help="Input key for data in base segmentation.")
    parser.add_argument("--ref_key", type=str, default=None,
                        help="Input key for data in reference segmentation.")

    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer.")
    parser.add_argument("--tif", action="store_true", help="Store output as tif file.")
    parser.add_argument("--crop", action="store_true", help="Process crop of original array.")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    if args.tif:
        if args.out_path_tif is None:
            raise ValueError("Specify out_path_tif for saving TIF file.")

    if args.base_key is None:
        data_base = read_image_data(args.base_path, args.base_key)
    else:
        data_base = open_file(args.base_path, "a")[args.base_key]
    data_ref = read_image_data(args.ref_path, args.ref_key)

    with open(args.base_table, "r") as f:
        table_base = pd.read_csv(f, sep="\t")

    if args.crop:
        output_ = ihc_synapse_postprocessing.postprocess_ihc_synapse_crop(
            data_base, data_ref, table_base=table_base, synapse_limit=25, min_overlap=0.5,
        )
    else:
        output_ = ihc_synapse_postprocessing.postprocess_ihc_synapse(
            data_base, data_ref, table_base=table_base, synapse_limit=25, min_overlap=0.5,
            voxel_size=args.voxel_size, roi_pad=40,
        )

    if args.tif:
        imageio.imwrite(args.out_path, output_, compression="zlib")


if __name__ == "__main__":

    main()
