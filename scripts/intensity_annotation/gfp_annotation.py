import argparse
import os
from typing import Optional

import flamingo_tools.intensity_annotation.annotation_utils as annotation_utils


def gfp_annotation(
    prefix: str,
    measurement_table_path: str,
    statistics_keyword: str = "median",
    is_otof: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    direc = os.path.dirname(os.path.abspath(prefix))
    basename = os.path.basename(prefix)
    file_names = [entry.name for entry in os.scandir(direc)]
    if is_otof:  # OTOF cochlea with VGlut3, Alphatag and IHC segmentation.
        stain1_file = [name for name in file_names if basename in name and "pha" in name][0]
        stain2_file = [name for name in file_names if basename in name and "lut3" in name][0]
        seg_file = [name for name in file_names if basename in name and "IHC" in name][0]

        stain1_name = "Alphatag"
        stain2_name = "Vglut3"
        seg_name = "IHC"

    else:  # ChReef cochlea with PV, GFP and SGN segmentation
        stain1_file = [name for name in file_names if basename in name and "GFP" in name][0]
        stain2_file = [name for name in file_names if basename in name and "PV" in name][0]
        seg_file = [name for name in file_names if basename in name and "SGN" in name][0]

        stain1_name = "GFP"
        stain2_name = "PV"
        seg_name = "SGN"

    stain_dict = {
        stain1_name: os.path.join(direc, stain1_file),
        stain2_name: os.path.join(direc, stain2_file),
    }
    seg_file = os.path.join(direc, seg_file)

    annotation_utils.annotation_napari(
        stain_dict=stain_dict,
        measurement_table_path=measurement_table_path,
        seg_name=seg_name,
        seg_file=seg_file,
        statistics_keyword=statistics_keyword,
        is_otof=is_otof,
        s3=s3,
        s3_credentials=s3_credentials,
        s3_bucket_name=s3_bucket_name,
        s3_service_endpoint=s3_service_endpoint,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Start a GUI for determining an intensity threshold for positive "
        "/ negative transduction in segmented cells.")
    parser.add_argument("-p", "--prefix", type=str, required=True,
                        help="The prefix of the files to open with the annotation tool.")
    parser.add_argument("-m", "--meas_table", type=str, default=None,
                        help="Measurement table containing intensity information about segmentation.")
    parser.add_argument("--otof", action="store_true",
                        help="Whether to run the annotation tool for otof samples with VGlut3, "
                        "Alphatag and IHC segmentation.")

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

    gfp_annotation(
        args.prefix,
        measurement_table_path=args.meas_table,
        is_otof=args.otof,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
