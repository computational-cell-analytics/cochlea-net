import argparse
import copy
import json
import os
from typing import List, Optional

import pandas as pd

import flamingo_tools.intensity_annotation.eval_annotations as eval_utils
from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER
from flamingo_tools.file_utils import read_image_data

MARKER_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ChReef_PV-GFP/2025-07_PV_GFP_SGN"
# The cochlea for the CHReef analysis.
COCHLEAE = [
    "M_LR_000143_L",
    "M_LR_000144_L",
    "M_LR_000145_L",
    "M_LR_000153_L",
    "M_LR_000155_L",
    "M_LR_000189_L",
    "M_LR_000143_R",
    "M_LR_000144_R",
    "M_LR_000145_R",
    "M_LR_000153_R",
    "M_LR_000155_R",
    "M_LR_000189_R",
    "G_EK_000049_L",
    "G_EK_000049_R",
]


def eval_marker_annotation(
    cochleae: List[str],
    output_dir: Optional[str] = None,
    annotation_dirs: Optional[List[str]] = None,
    threshold_save_dir: Optional[str] = None,
    input_key: str = "s0",
    data_seg_path: Optional[str] = None,
    table_seg_path: Optional[str] = None,
    table_meas_path: Optional[str] = None,
    mobie_dir: str = MOBIE_FOLDER,
    seg_name: str = "SGN_v2",
    marker_name: str = "GFP",
    force_overwrite: bool = False,
    compute_variance: bool = False,
    s3: Optional[bool] = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> None:
    """Evaluate marker annotations of a single or multiple annotators.
    Segmentation instances are assigned a positive (1) or negative label (2)
    in form of the "marker_label" component of the output segmentation table.
    The assignment is based on the median intensity supplied by a measurement table.
    Instances not considered for the assignment are labeled as 0.

    Args:
        cochleae: List of cochlea
        output_dir: Output directory for segmentation table with "marker_label" in format <cochlea>_<marker>_<seg>.tsv
            If no output directory is passed, the table will be saved in the appropriate location in the MoBIE project.
        annotation_dirs: List of directories containing marker annotations by annotator(s).
        mobie_dir: Local MoBIE directory used for creating data paths.
        seg_name: Identifier for segmentation.
        marker_name: Identifier for marker stain.
        threshold_save_dir: Optional directory for saving the thresholds.
        force_overwrite: Whether to overwrite already existing results.
        compute_variance: Whether to compare the marker percentages of the individual annotators
            with each other and with the median thresholds. The result is saved as
            <cochlea>_<marker>_<seg>_variance.json next to the thresholds.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    marker_pattern = None
    if marker_name in ["OTOF", "Alphatag"]:
        marker_pattern = f"{marker_name}_"

    if marker_name == "rbOtof":
        halo_size = 150
        voxel_size = [1.887779, 1.887779, 3.0]
    else:
        halo_size = 20
        voxel_size = (0.38, 0.38, 0.38)

    if annotation_dirs is None:
        if "MARKER_DIR" in globals():
            marker_dir = MARKER_DIR
            annotation_dirs = [entry.path for entry in os.scandir(marker_dir)
                               if os.path.isdir(entry) and "Results" in entry.name]

    seg_string = seg_name.replace('_', '-')
    for cochlea in cochleae:
        cochlea_str = cochlea.replace('_', '-')

        if output_dir is None:
            if s3:
                raise ValueError("Specify an output directory, when data is accessed from the S3 bucket.")
            else:
                print(f"Using MoBIE directory {mobie_dir} for output paths.")
                out_dir = os.path.join(mobie_dir, cochlea, "tables", seg_name)
                os.makedirs(out_dir, exist_ok=True)
                # TODO: Overwrite default table after checking that other entries are identical.
                out_path = os.path.join(out_dir, f"{marker_name}_{seg_string}.tsv")
        else:
            out_dir = output_dir
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{cochlea_str}_{marker_name}_{seg_string}.tsv")

        if os.path.exists(out_path) and not force_overwrite:
            print(f"Skipping {out_path}. Table already exists.")
            continue

        # check for legacy formatting, e.g. M_LR_000143_L instead of M-LR-000143-L
        search_str = cochlea_str
        annotations = [a for a in annotation_dirs if
                       len(eval_utils.find_annotations(a, search_str, marker_pattern)["center_strings"]) != 0]
        if len(annotations) == 0:
            search_str = cochlea
            annotations = [a for a in annotation_dirs if
                           len(eval_utils.find_annotations(a, search_str, marker_pattern)["center_strings"]) != 0]

        print(f"Evaluating data for cochlea {cochlea} in {annotations}.")

        # get the segmentation data, the segmentation table, and the object measures for the marker
        # the paths are resolved per cochlea, so that the given paths stay valid for the next cochlea
        if data_seg_path is None:
            if s3:
                seg_path = f"{cochlea}/images/ome-zarr/{seg_name}.ome.zarr"
            else:
                seg_path = os.path.join(mobie_dir, cochlea, "images", "ome-zarr", f"{seg_name}.ome.zarr")
        else:
            seg_path = data_seg_path
        if s3:
            seg_path, fs = get_s3_path(seg_path, bucket_name=s3_bucket_name,
                                       service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        data_seg = read_image_data(seg_path, input_key)

        if table_seg_path is None:
            if s3:
                seg_table = f"{cochlea}/tables/{seg_name}/default.tsv"
            else:
                seg_table = os.path.join(mobie_dir, cochlea, "tables", seg_name, "default.tsv")
        else:
            seg_table = table_seg_path
        if s3:
            table_path_s3, fs = get_s3_path(seg_table, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            with fs.open(table_path_s3, "r") as f:
                table_seg = pd.read_csv(f, sep="\t")
        else:
            table_seg = pd.read_csv(seg_table, sep="\t")

        if table_meas_path is None:
            table_meas_name = f"{marker_name}_{seg_string}_object-measures.tsv"
            if s3:
                meas_table = f"{cochlea}/tables/{seg_name}/{table_meas_name}"
            else:
                meas_table = os.path.join(mobie_dir, cochlea, "tables", seg_name, table_meas_name)
        else:
            meas_table = table_meas_path
        if s3:
            table_path_s3, fs = get_s3_path(meas_table, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            with fs.open(table_path_s3, "r") as f:
                table_meas = pd.read_csv(f, sep="\t")
        else:
            table_meas = pd.read_csv(meas_table, sep="\t")

        # Find the thresholds from the annotated blocks and save it if specified.
        intensity_dic, _ = eval_utils.find_thresholds(annotations, search_str, data_seg, table_meas,
                                                      voxel_size=voxel_size, pattern=marker_pattern)
        if threshold_save_dir is not None:
            os.makedirs(threshold_save_dir, exist_ok=True)
            threshold_out_path = os.path.join(threshold_save_dir, f"{cochlea_str}_{marker_name}_{seg_string}.json")
            with open(threshold_out_path, "w") as f:
                json.dump(intensity_dic, f, sort_keys=True, indent=4)

        # Compare the thresholds of the individual annotators with the median thresholds.
        if compute_variance:
            variance_dir = out_dir if threshold_save_dir is None else threshold_save_dir
            os.makedirs(variance_dir, exist_ok=True)
            variance_dic = eval_utils.evaluate_annotator_variance(
                copy.deepcopy(intensity_dic), table_seg.copy(), table_meas,
                halo_size=halo_size, cochlea=cochlea, marker_name=marker_name, seg_name=seg_name,
            )
            variance_out_path = os.path.join(
                variance_dir, f"{cochlea_str}_{marker_name}_{seg_string}_variance.json"
            )
            # The keys are not sorted, so that the per-crop breakdown stays at the end of the file.
            with open(variance_out_path, "w") as f:
                json.dump(variance_dic, f, indent=4)

        # Apply the threshold to all SGNs.
        table_seg = eval_utils.apply_nearest_threshold(intensity_dic, table_seg, table_meas, halo_size=halo_size)

        # Save the table with positives / negatives for all SGNs.
        table_seg.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker based on annotation thresholds."
    )

    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output", type=str, help="Output directory.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    parser.add_argument("-a", "--annotation_dirs", type=str, nargs="+", default=None,
                        help="Directories containing marker annotations.")
    parser.add_argument("-t", "--threshold_save_dir")
    parser.add_argument("--variance", action="store_true",
                        help="Compare the marker percentages of the individual annotators "
                        "with each other and with the median thresholds.")

    # options for specific data paths
    parser.add_argument("--seg_data", type=str, default=None,
                        help="Path to segmentation data.")
    parser.add_argument("--seg_table", type=str, default=None,
                        help="Path to segmentation table.")
    parser.add_argument("--meas_table", type=str, default=None,
                        help="Path to table with object measures.")

    # options for creating data paths automatically
    parser.add_argument("--seg_name", type=str, default="SGN_v2")
    parser.add_argument("--marker_name", type=str, default="GFP")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project.")

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

    eval_marker_annotation(
        cochleae=args.cochlea,
        output_dir=args.output,
        annotation_dirs=args.annotation_dirs,
        threshold_save_dir=args.threshold_save_dir,
        data_seg_path=args.seg_data,
        table_seg_path=args.seg_table,
        table_meas_path=args.meas_table,
        mobie_dir=args.mobie_dir,
        seg_name=args.seg_name,
        marker_name=args.marker_name,
        force_overwrite=args.force,
        compute_variance=args.variance,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
