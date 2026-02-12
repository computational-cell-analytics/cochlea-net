"""This script extracts data around an input center coordinate in a given ROI halo.
"""
import argparse
import json

from flamingo_tools.extract_block_util import extract_block_single


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('-i', '--input', type=str, required=True, help="Input file in n5 / ome-zarr format.")
    parser.add_argument('-o', "--output", type=str, default="", help="Output directory.")
    parser.add_argument('-c', "--coord", type=str, required=True,
                        help="3D coordinate as center of extracted block, json-encoded.")

    parser.add_argument('-k', "--input_key", type=str, default=None,
                        help="Input key for data in input file.")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Output key for data in output file.")
    parser.add_argument('-r', "--resolution", type=float, default=0.38, help="Resolution of input in micrometer.")
    parser.add_argument("--roi_halo", type=str, default="[128,128,64]",
                        help="ROI halo around center coordinate, json-encoded.")
    parser.add_argument("--tif", action="store_true", help="Store output as tif file.")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    coords = json.loads(args.coord)
    roi_halo = json.loads(args.roi_halo)

    extract_block_single(
        args.input, coords, args.output, args.input_key, args.output_key, args.resolution, roi_halo,
        args.tif, args.s3, args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
