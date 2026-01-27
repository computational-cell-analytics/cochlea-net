import argparse

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import get_s3_path


def eval_table(table_path, column, s3, s3_credentials=None, s3_bucket_name=None, s3_service_endpoint=None,):

    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table = pd.read_csv(table_path, sep="\t")

    if column not in table.columns:
        raise ValueError(f"Unknown column {column}. Please check columns in the segmentation table: {table.columns}.")

    values = table[column].values
    print(f"Mean value of column '{column}' in segmentation table {table_path}: {np.mean(values)}")


def main():
    parser = argparse.ArgumentParser(
        description="Start a GUI for determining an intensity threshold for positive "
        "/ negative transduction in segmented cells.")
    parser.add_argument("--table_path", help="Path to segmentation table.")
    parser.add_argument("--column", type=str, default="median",
                        help="Supply column name for calculation.")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    eval_table(
        args.table_path, args.column,
        args.s3, args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
