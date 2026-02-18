import argparse
import json
import os
from typing import List, Optional

import pandas as pd

import flamingo_tools.intensity_annotation.eval_annotations as eval_utils
from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.postprocessing.sgn_subtype_utils import CUSTOM_THRESHOLDS, COCHLEAE

MARKER_DIR_SUBTYPE = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes"


def eval_subtype_annotation(
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
    force_overwrite: bool = False,
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
        threshold_save_dir: Optional directory for saving the thresholds.
        force_overwrite: Whether to overwrite already existing results.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    if annotation_dirs is None:
        marker_dir = MARKER_DIR_SUBTYPE
        annotation_dirs = [entry.path for entry in os.scandir(marker_dir)
                           if os.path.isdir(entry) and "Result" in entry.name]

    for cochlea in cochleae:

        if cochlea not in list(COCHLEAE.keys()):
            subtype_utils = "flamingo_tools/postprocessing/sgn_subtype_utils.py"
            raise ValueError(f"Please add cochlea {cochlea} to the COCHLEAE dictionary in {subtype_utils}.")

        seg_name = COCHLEAE[cochlea]["seg_data"]
        if "output_seg" in list(COCHLEAE[cochlea].keys()):
            output_seg = COCHLEAE[cochlea]["output_seg"]
        else:
            output_seg = seg_name

        stains = COCHLEAE[cochlea]["subtype_stains"]
        seg_string = output_seg.replace('_', '-')
        cochlea_str = cochlea.replace('_', '-')
        subtype_str = "-".join(stains)
        print(f"Cochlea {cochlea} with subtype stains {stains}.")

        if output_dir is None:
            if s3:
                raise ValueError("Specify an output directory, when data is accessed from the S3 bucket.")
            else:
                print(f"Using MoBIE directory {mobie_dir} for output paths.")
                output_dir = os.path.join(mobie_dir, cochlea, "tables", seg_name)
                os.makedirs(output_dir, exist_ok=True)
                # TODO: Overwrite default table after checking that other entries are identical.
                out_path = os.path.join(output_dir, f"{subtype_str}_{seg_string}.tsv")
                annot_out = os.path.join(output_dir, f"{subtype_str}_{seg_string}_annotations.tsv")

        else:
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}.tsv")
            annot_out = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}_annot-overview.tsv")

        if os.path.exists(out_path) and os.path.exists(annot_out) and not force_overwrite:
            print(f"Skipping {out_path}. Output already exists.")
            continue

        # get the segmentation data and the segmentation table
        if data_seg_path is None:
            if s3:
                data_seg_path = os.path.join(cochlea, "images", "ome-zarr", f"{seg_name}.ome.zarr")
            else:
                data_seg_path = os.path.join(mobie_dir, cochlea, "images", "ome-zarr", f"{seg_name}.ome.zarr")
        if s3:
            data_seg_path, fs = get_s3_path(data_seg_path, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        data_seg = read_image_data(data_seg_path, input_key)

        if table_seg_path is None:
            if s3:
                table_seg_path = os.path.join(cochlea, "tables", seg_name, "default.tsv")
            else:
                table_seg_path = os.path.join(mobie_dir, cochlea, "tables", seg_name, "default.tsv")
        if s3:
            table_path_s3, fs = get_s3_path(table_seg_path, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            with fs.open(table_path_s3, "r") as f:
                table_seg = pd.read_csv(f, sep="\t")
        else:
            table_seg = pd.read_csv(table_seg_path, sep="\t")

        # Check whether to use intensity ratio of subtype / PV or object measures for thresholding
        intensity_mode = COCHLEAE[cochlea]["intensity"]

        # iterate through subtypes
        annot_table = None
        for stain in stains:

            if table_meas_path is None:
                if intensity_mode == "ratio":
                    table_meas_name = "subtype_ratio.tsv"
                    column = f"{stain}_ratio_PV"
                elif intensity_mode == "absolute":
                    table_meas_name = f"{stain}_{seg_string}_object-measures.tsv"
                    column = "median"
                else:
                    raise ValueError("Choose either 'ratio' or 'median' as intensity mode.")

                if s3:
                    table_meas_path = os.path.join(cochlea, "tables", seg_name, table_meas_name)
                else:
                    table_meas_path = os.path.join(mobie_dir, cochlea, "tables", seg_name, table_meas_name)

            if s3:
                table_path_s3, fs = get_s3_path(table_meas_path)
                with fs.open(table_path_s3, "r") as f:
                    table_meas = pd.read_csv(f, sep="\t")
            else:
                table_meas = pd.read_csv(table_meas_path, sep="\t")

            # check for legacy formatting, e.g. M_LR_000143_L instead of M-LR-000143-L
            search_str = cochlea_str
            annotations = [a for a in annotation_dirs
                           if len(eval_utils.find_annotations(a, search_str, stain)["center_strings"]) != 0]
            if len(annotations) == 0:
                search_str = cochlea
                annotations = [a for a in annotation_dirs
                               if len(eval_utils.find_annotations(a, search_str, stain)["center_strings"]) != 0]

            print(f"Evaluating data for cochlea {cochlea} in {annotations}.")

            # Find the thresholds from the annotated blocks and save them if specified.
            intensity_dic, annot_dic = eval_utils.find_thresholds(annotations, search_str, data_seg,
                                                                  table_meas, column=column, pattern=stain)

            if annot_table is None:
                annot_table = eval_utils.get_annotation_table(annot_dic, stain)
            else:
                annot_table = pd.concat(
                    [annot_table, eval_utils.get_annotation_table(annot_dic, stain)],
                    ignore_index=True,
                )

            # create dictionary containing median intensity and segmentation ids for every crop
            om_dic = eval_utils.get_object_measures(annot_dic, intensity_dic, intensity_mode, stain)
            om_out_path = os.path.join(output_dir, f"{cochlea_str}_{stain}_{seg_string}_crop-intensity.json")
            with open(om_out_path, "w") as f:
                json.dump(om_dic, f, sort_keys=True, indent=4)

            if threshold_save_dir is not None:
                os.makedirs(threshold_save_dir, exist_ok=True)
                threshold_out_path = os.path.join(threshold_save_dir, f"{cochlea_str}_{stain}_{seg_string}_annot.json")
                with open(threshold_out_path, "w") as f:
                    json.dump(intensity_dic, f, sort_keys=True, indent=4)

            # load measurement table of output segmentation
            # this step can (hopefully) be ignored for future analysis
            if "output_seg" in list(COCHLEAE[cochlea].keys()):
                output_seg = COCHLEAE[cochlea]["output_seg"]
                table_meas_path = os.path.join(cochlea, "tables", output_seg, "subtype_ratio.tsv")
                table_path_s3, fs = get_s3_path(table_meas_path)
                with fs.open(table_path_s3, "r") as f:
                    table_meas = pd.read_csv(f, sep="\t")

            # Apply the threshold to all SGNs.
            if CUSTOM_THRESHOLDS.get(cochlea, {}).get(stain) is not None:
                custom_threshold_dic = CUSTOM_THRESHOLDS[cochlea][stain]
            else:
                custom_threshold_dic = None

            table_seg = eval_utils.apply_nearest_threshold(
                intensity_dic, table_seg, table_meas, column=column, suffix=stain,
                threshold_dic=custom_threshold_dic,
            )

        # Save the table with positives / negatives for all SGNs.
        if not os.path.exists(out_path) or force_overwrite:
            table_seg.to_csv(out_path, sep="\t", index=False)

        if not os.path.exists(annot_out) or force_overwrite:
            annot_table.to_csv(annot_out, sep="\t", index=False)


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

    # options for specific data paths
    parser.add_argument("--seg_data", type=str, default=None,
                        help="Path to segmentation data.")
    parser.add_argument("--seg_table", type=str, default=None,
                        help="Path to segmentation table.")
    parser.add_argument("--meas_table", type=str, default=None,
                        help="Path to table with object measures.")

    # options for creating data paths automatically
    parser.add_argument("--seg_name", type=str, default="SGN_v2")
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

    eval_subtype_annotation(
        cochleae=args.cochlea,
        output_dir=args.output,
        annotation_dirs=args.annotation_dirs,
        threshold_save_dir=args.threshold_save_dir,
        data_seg_path=args.seg_data,
        table_seg_path=args.seg_table,
        table_meas_path=args.meas_table,
        mobie_dir=args.mobie_dir,
        seg_name=args.seg_name,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
