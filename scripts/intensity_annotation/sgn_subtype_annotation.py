import argparse
import os
from typing import Optional

import flamingo_tools.intensity_annotation.annotation_utils as annotation_utils


def sgn_subtype_annotation(
    prefix: str,
    measurement_table_path: str,
    subtype: str,
    statistics_keyword: str = "median",
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Function for SGN subtype annotation.
    The function requires crops of the same dimension, which are labeled using a specific naming scheme.
    The files should have the common prefix: <cochlea>_crop_xxx-yyy-zzz
    Files required for the annotation: <subtype>, SGN, optional: PV

    Args:
        prefix: Common file prefix of a specific crop.
        measurement_table_path: Measurement table of object measures for stain-segmentation combination.
            Use subtype_ratio table, if PV stain is available.
        subtype: Subtype stain, e.g. CR, Lypd1, etc.
        statistics_keyword: Column keyword for pandas dataframe of object measures.
        s3: Use S3 file path for measurement table.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """

    direc = os.path.dirname(os.path.abspath(prefix))
    basename = os.path.basename(prefix)
    file_names = [entry.name for entry in os.scandir(direc)]

    stain1_file = [name for name in file_names if basename in name and subtype in name][0]
    pv_files = [name for name in file_names if basename in name and "PV.tif" in name]
    if len(pv_files) == 0:
        ref_stain_file = None
    else:
        ref_stain = "PV"
        ref_stain_file = [name for name in file_names if basename in name and "PV.tif" in name][0]
        statistics_keyword = f"{subtype}_ratio_PV"

    seg_name = "SGN"
    seg_file = [name for name in file_names if basename in name and seg_name in name][0]

    stain_dict = {
        subtype: os.path.join(direc, stain1_file),
    }
    if ref_stain_file is not None:
        stain_dict[ref_stain] = os.path.join(direc, ref_stain)

    seg_file = os.path.join(direc, seg_file)

    annotation_utils.annotation_napari(
        stain_dict=stain_dict,
        measurement_table_path=measurement_table_path,
        seg_name=seg_name,
        seg_file=seg_file,
        statistics_keyword=statistics_keyword,
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
    parser.add_argument("--subtype", type=str, default=None,
                        help="Supply SGN subtype, e.g. Calb1, Prph, Lypd1, ...")

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

    sgn_subtype_annotation(
        args.prefix,
        measurement_table_path=args.meas_table,
        subtype=args.subtype,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
