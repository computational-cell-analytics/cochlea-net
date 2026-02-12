import argparse
import os
import warnings
from typing import List, Optional

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER
from flamingo_tools.postprocessing.sgn_subtype_utils import COCHLEAE


def get_subtype_stain_ratio(
    cochleae: List[str],
    output_dir: Optional[str] = None,
    seg_name: Optional[str] = None,
    img_channels: List[str] = [],
    mobie_dir: str = MOBIE_FOLDER,
    component_list: List[int] = [1],
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Calculate the intensity ratio of subtype stains to a reference stain for SGN segmentation.

    Args:
        cochleae: List of cochleae.
        output_dir: Output directory.
        seg_name: Identifier vor segmentation, e.g. SGN_v2.
        img_channels: List of image stains, e.g. [PV, CR, Ntng1].
        mobie_dir: Local directory of MoBIE project.
        component_list: List of connected components.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    for cochlea in cochleae:
        print(cochlea)
        if len(img_channels) == 0:
            img_channels = COCHLEAE[cochlea]["stains"]

        if cochlea not in COCHLEAE:
            warnings.warn(f"Cochlea {cochlea} not in COCHLEAE dictionary. "
                          "Please add entry in 'flamingo_tools/postprocessing/sgn_subtype_utils.py")

        # PV is the default reference channel. If PV is not available, CR has to be there
        if "PV" in img_channels:
            reference_channel = "PV"
        else:
            assert "CR" in img_channels
            reference_channel = "CR"

        if s3:
            table_seg_path = os.path.join(cochlea, "tables", seg_name, "default.tsv")
            table_path_s3, fs = get_s3_path(table_seg_path, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            with fs.open(table_path_s3, "r") as f:
                table_seg = pd.read_csv(f, sep="\t")
        else:
            table_seg_path = os.path.join(mobie_dir, cochlea, "tables", seg_name, "default.tsv")
            table_seg = pd.read_csv(table_seg_path, sep="\t")

        if "component_list" in list(COCHLEAE[cochlea].keys()):
            component_list = COCHLEAE[cochlea]["component_list"]
        table_seg = table_seg[table_seg["component_labels"].isin(component_list)]
        print("Number of SGNs", len(table_seg))
        valid_sgns = table_seg.label_id
        output_table = {"label_id": table_seg.label_id.values, "frequency[kHz]": table_seg["frequency[kHz]"]}

        # Analyze the different channels (= different subtypes).
        reference_intensity = None
        for channel in img_channels:

            table_name = f"{channel}_{seg_name.replace('_', '-')}_object-measures.tsv"

            if s3:
                table_meas_path = os.path.join(cochlea, "tables", seg_name, table_name)
                table_path_s3, fs = get_s3_path(table_meas_path, bucket_name=s3_bucket_name,
                                                service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
                with fs.open(table_path_s3, "r") as f:
                    table_meas = pd.read_csv(f, sep="\t")
            else:
                table_meas_path = os.path.join(mobie_dir, cochlea, "tables", seg_name, table_name)
                table_meas = pd.read_csv(table_meas_path, sep="\t")

            table_meas = table_meas[table_meas.label_id.isin(valid_sgns)]
            assert len(table_seg) == len(table_meas)
            assert (table_meas.label_id.values == table_seg.label_id.values).all()

            medians = table_meas["median"].values
            output_table[f"{channel}_median"] = medians
            if channel == reference_channel:
                reference_intensity = medians
            else:
                assert reference_intensity is not None
                output_table[f"{channel}_ratio_{reference_channel}"] = medians / reference_intensity

        cochlea_str = cochlea.replace('_', '-')
        if output_dir is None:
            if s3:
                raise ValueError("Specify an output directory, when data is accessed from the S3 bucket.")
            else:
                out_path = os.path.join(mobie_dir, cochlea, "tables", seg_name, "subtype_ratio.tsv")
        else:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{cochlea_str}_subtype_ratio.tsv")
        output_table = pd.DataFrame(output_table)
        output_table.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Calculate the intensity ratio of subtype stains to a reference stain."
    )
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output directory. Default: MoBIE directory")

    # options for creating data paths automatically
    parser.add_argument("--seg_name", type=str, default="SGN_v2")
    parser.add_argument("--img_channels", type=str, nargs="+", default=[],
                        help="Stains to analyze, e.g. PV, CR.")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project.")

    parser.add_argument("--component_list", type=int, nargs="+", default=[1],
                        help="List of connected components.")

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

    get_subtype_stain_ratio(
        cochleae=args.cochlea,
        output_dir=args.output,
        seg_name=args.seg_name,
        img_channels=args.img_channels,
        mobie_dir=args.mobie_dir,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
