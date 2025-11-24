import argparse

import pandas as pd

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping


def main():

    parser = argparse.ArgumentParser(
        description="Script for the tonotopic mapping of IHCs and SGNs. "
        "Either locally or on an S3 bucket.")

    parser.add_argument("-i", "--input", required=True, help="Input table with IHC segmentation.")
    parser.add_argument("-o", "--output", required=True, help="Output path for json file with cropping parameters.")

    parser.add_argument("-t", "--type", type=str, default="sgn", help="Cell type of segmentation.")

    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    if args.s3:
        tsv_path, fs = s3_utils.get_s3_path(args.input, bucket_name=args.s3_bucket_name,
                                            service_endpoint=args.s3_service_endpoint,
                                            credential_file=args.s3_credentials)
        with fs.open(tsv_path, 'r') as f:
            tsv_table = pd.read_csv(f, sep="\t")
    else:
        with open(args.input, 'r') as f:
            tsv_table = pd.read_csv(f, sep="\t")

    table = tonotopic_mapping(
        tsv_table, cell_type=args.type,
    )

    table.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
