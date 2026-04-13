import argparse

from flamingo_tools.analysis.central_path_utils import insert_coordinates_to_central_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input path to table with central path spots.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path to main table.")
    parser.add_argument("-c", "--coordinates", type=str, nargs="+", default=None,
                        help="List of coordinates, coordinates are seperated by comma, e.g. 100,340,87.")

    # options for data access on S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    if args.coordinates is not None:
        coordinates = [[float(c) for c in coord_str.split(",")] for coord_str in args.coordinates]

        insert_coordinates_to_central_path(
            table_path=args.input,
            output_path=args.output,
            coordinates=coordinates,
            s3=args.s3,
            s3_credentials=args.s3_credentials,
            s3_bucket_name=args.s3_bucket_name,
            s3_service_endpoint=args.s3_service_endpoint,
        )


if __name__ == "__main__":
    main()
