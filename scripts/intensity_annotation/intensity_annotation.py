import argparse
import os
from typing import Optional

import flamingo_tools.intensity_annotation.annotation_utils as annotation_utils


def intensity_annotation(
    prefix: str,
    measurement_table_dir: str,
    statistics_keyword: str = "median",
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Function for GFP/Alphatag annotation.
    The function requires crops of the same dimension, which are labeled using a specific naming scheme.
    The files should have the common prefix: <cochlea>_crop_xxx-yyy-zzz
    The scenario is auto-detected from the stain files found for this prefix:
    Files required for the GFP annotation: GFP stain, PV stain, SGN segmentation
    Files required for the Alphatag annotation: Alphatag, Vglut3, IHC segmentation, optional: Otof

    Args:
        prefix: Common file prefix of a specific crop.
        measurement_table_dir: Directory containing per-channel object-measures tables for the
            stain-segmentation combinations, e.g. Alphatag_IHC-v11_object-measures-bg-mask.tsv.
        statistics_keyword: Column keyword for pandas dataframe of object measures.
        s3: Use S3 file path for measurement tables.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """
    direc = os.path.dirname(os.path.abspath(prefix))
    basename = os.path.basename(prefix)
    file_names = [entry.name for entry in os.scandir(direc)]

    has_alphatag = any(basename in name and "Alphatag" in name for name in file_names)
    has_gfp = any(basename in name and "GFP" in name for name in file_names)
    if has_alphatag and has_gfp:
        raise ValueError(f"Found both GFP and Alphatag files for prefix {basename}; cannot auto-detect scenario.")
    elif has_alphatag:
        is_otof = True
    elif has_gfp:
        is_otof = False
    else:
        raise ValueError(f"Found neither GFP nor Alphatag files for prefix {basename}; cannot auto-detect scenario.")

    stain3_file = None
    if is_otof:  # OTOF cochlea with VGlut3, Alphatag and IHC segmentation.
        stain1_file = [name for name in file_names if basename in name and "Alphatag" in name][0]
        stain2_file = [name for name in file_names if basename in name and "Vglut3" in name][0]
        otof_matches = [name for name in file_names if basename in name and "Otof" in name]
        if len(otof_matches) > 0:
            stain3_file = otof_matches[0]
        seg_file = [name for name in file_names if basename in name and "IHC" in name][0]

        stain1_name = "Alphatag"
        stain2_name = "Vglut3"
        stain3_name = "Otof"
        seg_name = "IHC"
        default_channel = "Alphatag"

    else:  # ChReef cochlea with PV, GFP and SGN segmentation
        stain1_file = [name for name in file_names if basename in name and "GFP" in name][0]
        stain2_file = [name for name in file_names if basename in name and "PV" in name][0]
        seg_file = [name for name in file_names if basename in name and "SGN" in name][0]

        stain1_name = "GFP"
        stain2_name = "PV"
        seg_name = "SGN"
        default_channel = "GFP"

    stain_dict = {
        stain1_name: os.path.join(direc, stain1_file),
        stain2_name: os.path.join(direc, stain2_file),
    }
    if stain3_file is not None:
        stain_dict[stain3_name] = os.path.join(direc, stain3_file)

    seg_file = os.path.join(direc, seg_file)

    # stain2_name (PV / Vglut3) is a reference channel only; it is never thresholded.
    channels = [name for name in stain_dict if name != stain2_name]
    measurement_tables = annotation_utils.find_channel_measurement_tables(
        measurement_table_dir,
        channels=channels,
        s3=s3,
        s3_credentials=s3_credentials,
        s3_bucket_name=s3_bucket_name,
        s3_service_endpoint=s3_service_endpoint,
    )
    if not measurement_tables:
        raise ValueError(f"No measurement tables found for channels {channels} in {measurement_table_dir}")

    annotation_utils.annotation_napari(
        stain_dict=stain_dict,
        measurement_tables=measurement_tables,
        seg_name=seg_name,
        seg_file=seg_file,
        default_channel=default_channel,
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
                        help="Directory containing per-channel object-measures tables, e.g. "
                        "Alphatag_IHC-v11_object-measures-bg-mask.tsv, Otof_IHC-v11_object-measures-bg-mask.tsv, "
                        "Vglut3_IHC-v11_object-measures-bg-mask.tsv.")
    parser.add_argument("--intensity_keyword", type=str, default="median",
                        help="Keyword for intensity information of object measures. Default: median")

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

    intensity_annotation(
        args.prefix,
        measurement_table_dir=args.meas_table,
        statistics_keyword=args.intensity_keyword,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
