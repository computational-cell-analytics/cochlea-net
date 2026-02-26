import argparse

from flamingo_tools.analysis.seg_table_utils import add_column_from_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--ref", type=str, required=True,
                        help="Input path to reference table.")
    parser.add_argument("-t", "--target", type=str, required=True,
                        help="Input path to target table.")
    parser.add_argument("--ref_column", type=str, required=True,
                        help="Column name to copy.")
    parser.add_argument("--target_column", type=str, default=None,
                        help="Column name for output table. Default: Keep column name.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path to final table. Default: Extend target table.")

    # options for data access on S3 bucket
    parser.add_argument("--s3_ref", action="store_true", help="Reference table path on S3 bucket.")
    parser.add_argument("--s3_target", action="store_true", help="Target table path on S3 bucket.")

    args = parser.parse_args()

    add_column_from_ref(
        ref_path=args.ref,
        target_path=args.target,
        ref_column=args.ref_column,
        target_column=args.target_column,
        output_path=args.output,
        s3_ref=args.s3_ref,
        s3_target=args.s3_target,
    )


if __name__ == "__main__":
    main()
