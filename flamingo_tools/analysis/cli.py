"""private
"""
import argparse

from .seg_table_utils import print_table_info


def table_info():
    parser = argparse.ArgumentParser(
        description="Find the segmentation table row closest to a given column value "
        "and print its label ID and central coordinate in micrometer.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Input path to segmentation table.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Optional output path for a JSON file with the results.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for table lookup
    parser.add_argument("--column", type=str, default="length_fraction",
                        help="Column name to match against, e.g. 'length_fraction' or 'frequency[kHz]'. "
                        "Default: length_fraction")
    parser.add_argument("--value", type=float, nargs="+", required=True,
                        help="Target value(s) to find the closest row for.")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=None,
                        help="Optional list of connected components to filter the table by before matching.")

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

    print_table_info(
        table_path=args.input,
        column=args.column,
        values=args.value,
        component_list=args.components,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
        output_path=args.output,
        force_overwrite=args.force,
    )
