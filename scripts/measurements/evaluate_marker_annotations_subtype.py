import argparse
import json
import os
from typing import List, Optional

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.postprocessing.chreef_utils import localize_median_intensities, find_annotations
from flamingo_tools.postprocessing.sgn_subtype_utils import CUSTOM_THRESHOLDS, COCHLEAE

MARKER_DIR_SUBTYPE = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes"


def get_length_fraction_from_center(table, center_str):
    """Get 'length_fraction' parameter for center coordinate by averaging nearby segmentation instances.
    """
    center_coord = tuple([int(c) for c in center_str.split("-")])
    (cx, cy, cz) = center_coord
    offset = 20
    subset = table[
        (cx - offset < table["anchor_x"]) &
        (table["anchor_x"] < cx + offset) &
        (cy - offset < table["anchor_y"]) &
        (table["anchor_y"] < cy + offset) &
        (cz - offset < table["anchor_z"]) &
        (table["anchor_z"] < cz + offset)
    ]
    length_fraction = list(subset["length_fraction"])
    length_fraction = float(sum(length_fraction) / len(length_fraction))
    return length_fraction


def apply_nearest_threshold(intensity_dic, table_seg, table_meas,
                            column="median", suffix="labels", threshold_dic=None):
    """Apply threshold to nearest segmentation instances.
    Crop centers are transformed into the "length fraction" parameter of the segmentation table.
    This avoids issues with the spiral shape of the cochlea and maps the assignment onto the Rosenthal"s canal.
    """
    # assign crop centers to length fraction of Rosenthal"s canal
    lf_intensity = {}
    for key in intensity_dic.keys():
        length_fraction = get_length_fraction_from_center(table_seg, key)
        intensity_dic[key]["length_fraction"] = length_fraction
        if threshold_dic is None:
            lf_intensity[length_fraction] = {"threshold": intensity_dic[key]["median_intensity"]}
        else:
            if isinstance(threshold_dic, (int, float)):
                custom_threshold = threshold_dic
            else:
                custom_threshold = threshold_dic[key]["manual"]
            print(f"Using custom threshold {custom_threshold} for crop {key}.")
            lf_intensity[length_fraction] = {"threshold": custom_threshold}

    # get limits for checking marker thresholds
    lf_intensity = dict(sorted(lf_intensity.items()))
    lf_fractions = list(lf_intensity.keys())
    # start of cochlea
    lf_limits = [0]
    # half distance between block centers
    for i in range(len(lf_fractions) - 1):
        lf_limits.append((lf_fractions[i] + lf_fractions[i+1]) / 2)
    # end of cochlea
    lf_limits.append(1)

    marker_labels = [0 for _ in range(len(table_seg))]
    table_seg.loc[:, f"marker_{suffix}"] = marker_labels
    for num, fraction in enumerate(lf_fractions):
        subset_seg = table_seg[
            (table_seg["length_fraction"] > lf_limits[num]) &
            (table_seg["length_fraction"] < lf_limits[num + 1])
        ]
        # assign values based on limits
        threshold = lf_intensity[fraction]["threshold"]
        label_ids_seg = subset_seg["label_id"]

        subset_measurement = table_meas[table_meas["label_id"].isin(label_ids_seg)]
        subset_positive = subset_measurement[subset_measurement[column] >= threshold]
        subset_negative = subset_measurement[subset_measurement[column] < threshold]
        label_ids_pos = list(subset_positive["label_id"])
        label_ids_neg = list(subset_negative["label_id"])

        table_seg.loc[table_seg["label_id"].isin(label_ids_pos), f"marker_{suffix}"] = 1
        table_seg.loc[table_seg["label_id"].isin(label_ids_neg), f"marker_{suffix}"] = 2

    return table_seg


def find_thresholds(cochlea_annotations, cochlea, data_seg, table_meas, column="median", pattern=None):
    # Find the median intensities by averaging the individual annotations for specific crops
    annotation_dics = {}
    annotated_centers = []
    for annotation_dir in cochlea_annotations:
        print(f"Localizing threshold with median intensities for {os.path.basename(annotation_dir)}.")
        annotation_dic = localize_median_intensities(annotation_dir, cochlea, data_seg,
                                                     table_meas, column=column, pattern=pattern)
        annotated_centers.extend(annotation_dic["center_strings"])
        annotation_dics[annotation_dir] = annotation_dic

    annotated_centers = list(set(annotated_centers))
    intensity_dic = {}
    # loop over all annotated blocks
    for annotated_center in annotated_centers:
        intensities = []
        annotator_success = []
        annotator_failure = []
        annotator_missing = []
        # loop over annotated block from single user
        for annotator_key in annotation_dics.keys():
            if annotated_center not in annotation_dics[annotator_key]["center_strings"]:
                annotator_missing.append(os.path.basename(annotator_key))
                continue
            else:
                median_intensity = annotation_dics[annotator_key][annotated_center]["median_intensity"]
                if median_intensity is None:
                    print(f"No threshold for {os.path.basename(annotator_key)} and crop {annotated_center}.")
                    annotator_failure.append(os.path.basename(annotator_key))
                else:
                    intensities.append(median_intensity)
                    annotator_success.append(os.path.basename(annotator_key))

        if len(intensities) == 0:
            print(f"No viable annotation for cochlea {cochlea} and crop {annotated_center}.")
            median_int_avg = None
        else:
            median_int_avg = float(sum(intensities) / len(intensities)),

        intensity_dic[annotated_center] = {
            "median_intensity": median_int_avg,
            "annotation_success": annotator_success,
            "annotation_failure": annotator_failure,
            "annotation_missing": annotator_missing,
        }

    return intensity_dic, annotation_dics


def get_annotation_table(annotation_dics, subtype):
    """Create table containing information about SGNs within crops.
    """
    rows = []
    for annotation_dir, annotation_dic in annotation_dics.items():

        annotator_dir = os.path.basename(annotation_dir)
        annotator = annotator_dir.split("_")[1]
        for center_str in annotation_dic["center_strings"]:
            row = {"annotator": annotator}
            row["subtype_stains"] = subtype
            row["center_str"] = center_str
            row["median_intensity"] = annotation_dic[center_str]["median_intensity"]
            row["inbetween_ids"] = len(annotation_dic[center_str]["inbetween_ids"])
            row["allweak_pos"] = len(annotation_dic[center_str]["allweak_pos"])
            row["allweak_neg"] = len(annotation_dic[center_str]["allweak_neg"])
            row["negexc_pos"] = len(annotation_dic[center_str]["negexc_pos"])
            row["negexc_neg"] = len(annotation_dic[center_str]["negexc_neg"])

            row["allweak_pos_mean"] = annotation_dic[center_str]["allweak_pos_mean"]
            row["allweak_neg_mean"] = annotation_dic[center_str]["allweak_neg_mean"]
            row["negexc_pos_mean"] = annotation_dic[center_str]["negexc_pos_mean"]
            row["negexc_neg_mean"] = annotation_dic[center_str]["negexc_neg_mean"]
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def get_object_measures(annotation_dics, intensity_dic, intensity_mode, subtype_stain):
    """Get information to create table containing object measure information.
    """
    om_dic = {}
    center_strings = list(intensity_dic.keys())
    om_dic["center_strings"] = center_strings
    om_dic["subtype_stains"] = subtype_stain
    om_dic["intensity_mode"] = intensity_mode
    for center_str in center_strings:
        crop_dic = {}
        crop_dic["median_intensity"] = intensity_dic[center_str]["median_intensity"]
        for _, annotation_dic in annotation_dics.items():
            if center_str in list(annotation_dic.keys()):
                crop_dic["seg_ids"] = [int(i) for i in annotation_dic[center_str]["seg_ids"]]

        om_dic[center_str] = crop_dic
    return om_dic


def evaluate_subtype_annotation(
    cochleae: List[str],
    output_dir: str,
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
        annotation_dirs: List of directories containing marker annotations by annotator(s).
        mobie_dir: Local MoBIE directory used for creating data paths.
        seg_name: Identifier for segmentation.
        threshold_save_dir: Optional directory for saving the thresholds.
        force_overwrite: Whether to overwrite already existing results.
        legacy_formatting: Use legacy formatting of thresholds with underscores, e.g. M_LR_N127_L.
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

        seg_string = output_seg.replace('_', '-')
        cochlea_str = cochlea.replace('_', '-')
        stains = COCHLEAE[cochlea]["subtype_stains"]
        print(f"Cochlea {cochlea} with subtype stains {stains}.")
        subtype_str = "_".join(stains)
        out_path = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}.tsv")
        annot_out = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}_annotations.tsv")
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
                           if len(find_annotations(a, search_str, stain)["center_strings"]) != 0]
            if len(annotations) == 0:
                search_str = cochlea
                annotations = [a for a in annotation_dirs
                               if len(find_annotations(a, search_str, stain)["center_strings"]) != 0]

            print(f"Evaluating data for cochlea {cochlea} in {annotations}.")

            # Find the thresholds from the annotated blocks and save them if specified.
            intensity_dic, annot_dic = find_thresholds(annotations, search_str, data_seg,
                                                       table_meas, column=column, pattern=stain)

            if annot_table is None:
                annot_table = get_annotation_table(annot_dic, stain)
            else:
                annot_table = pd.concat([annot_table, get_annotation_table(annot_dic, stain)], ignore_index=True)

            # create dictionary containing median intensity and segmentation ids for every crop
            om_dic = get_object_measures(annot_dic, intensity_dic, intensity_mode, stain)
            om_out_path = os.path.join(output_dir, f"{cochlea_str}_{stain}_om.json")
            with open(om_out_path, "w") as f:
                json.dump(om_dic, f, sort_keys=True, indent=4)

            if threshold_save_dir is not None:
                os.makedirs(threshold_save_dir, exist_ok=True)
                threshold_out_path = os.path.join(threshold_save_dir, f"{cochlea_str}_{stain}_{seg_string}.json")
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

            table_seg = apply_nearest_threshold(
                intensity_dic, table_seg, table_meas, column=column, suffix=stain,
                threshold_dic=custom_threshold_dic,
            )

        # Save the table with positives / negatives for all SGNs.
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(out_path) or force_overwrite:
            table_seg.to_csv(out_path, sep="\t", index=False)
        if not os.path.exists(annot_out) or force_overwrite:
            annot_table.to_csv(annot_out, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker based on annotation thresholds."
    )
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
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

    evaluate_subtype_annotation(
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
