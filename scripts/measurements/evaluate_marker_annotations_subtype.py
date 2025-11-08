import argparse
import json
import os
from typing import List, Optional

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.segmentation.chreef_utils import localize_median_intensities, find_annotations
from flamingo_tools.segmentation.sgn_subtype_utils import CUSTOM_THRESHOLDS, COCHLEAE

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


def apply_nearest_threshold(intensity_dic, table_seg, table_measurement, column="median", suffix="labels", threshold_dic=None):
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

        subset_measurement = table_measurement[table_measurement["label_id"].isin(label_ids_seg)]
        subset_positive = subset_measurement[subset_measurement[column] >= threshold]
        subset_negative = subset_measurement[subset_measurement[column] < threshold]
        label_ids_pos = list(subset_positive["label_id"])
        label_ids_neg = list(subset_negative["label_id"])

        table_seg.loc[table_seg["label_id"].isin(label_ids_pos), f"marker_{suffix}"] = 1
        table_seg.loc[table_seg["label_id"].isin(label_ids_neg), f"marker_{suffix}"] = 2

    return table_seg


def find_thresholds(cochlea_annotations, cochlea, data_seg, table_measurement, column="median", pattern=None):
    # Find the median intensities by averaging the individual annotations for specific crops
    annotation_dics = {}
    annotated_centers = []
    for annotation_dir in cochlea_annotations:
        print(f"Localizing threshold with median intensities for {os.path.basename(annotation_dir)}.")
        annotation_dic = localize_median_intensities(annotation_dir, cochlea, data_seg,
                                                     table_measurement, column=column, pattern=pattern)
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
            row = {"annotator" : annotator}
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


def evaluate_marker_annotation(
    cochleae: List[str],
    output_dir: str,
    annotation_dirs: Optional[List[str]] = None,
    threshold_save_dir: Optional[str] = None,
    force: bool = False,
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
        threshold_save_dir: Optional directory for saving the thresholds.
        force: Whether to overwrite already existing results.
    """
    input_key = "s0"

    if annotation_dirs is None:
        marker_dir = MARKER_DIR_SUBTYPE
        annotation_dirs = [entry.path for entry in os.scandir(marker_dir)
                           if os.path.isdir(entry) and "Result" in entry.name]

    for cochlea in cochleae:
        data_name = COCHLEAE[cochlea]["seg_data"]
        if "output_seg" in list(COCHLEAE[cochlea].keys()):
            output_seg = COCHLEAE[cochlea]["output_seg"]
        else:
            output_seg = data_name

        seg_string = "-".join(output_seg.split("_"))
        cochlea_str = "-".join(cochlea.split("_"))
        stains = COCHLEAE[cochlea]["subtype_stain"]
        subtype_str = "_".join(stains)
        out_path = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}.tsv")
        annot_out = os.path.join(output_dir, f"{cochlea_str}_{subtype_str}_{seg_string}_annotations.tsv")
        if os.path.exists(out_path) and os.path.exists(annot_out) and not force:
            continue

        # Get the segmentation data and table.
        input_path = f"{cochlea}/images/ome-zarr/{data_name}.ome.zarr"
        input_path, fs = get_s3_path(input_path)
        data_seg = read_image_data(input_path, input_key)

        table_seg_path = f"{cochlea}/tables/{output_seg}/default.tsv"
        table_path_s3, fs = get_s3_path(table_seg_path)
        with fs.open(table_path_s3, "r") as f:
            table_seg = pd.read_csv(f, sep="\t")

        # Check whether to use intensity ratio of subtype / PV or object measures for thresholding
        intensity_mode = COCHLEAE[cochlea]["intensity"]

        # iterate through subtypes
        annot_table = None
        for stain in stains:
            if intensity_mode == "ratio":
                table_measurement_path = f"{cochlea}/tables/{data_name}/subtype_ratio.tsv"
                column = f"{stain}_ratio_PV"
            elif intensity_mode == "absolute":
                table_measurement_path = f"{cochlea}/tables/{data_name}/{stain}_{seg_string}_object-measures.tsv"
                column = "median"
            else:
                raise ValueError("Choose either 'ratio' or 'median' as intensity mode.")

            table_path_s3, fs = get_s3_path(table_measurement_path)
            with fs.open(table_path_s3, "r") as f:
                table_measurement = pd.read_csv(f, sep="\t")

            cochlea_annotations = [a for a in annotation_dirs
                                   if len(find_annotations(a, cochlea, stain)["center_strings"]) != 0]
            print(f"Evaluating data for cochlea {cochlea} in {cochlea_annotations}.")

            # Find the thresholds from the annotated blocks and save them if specified.
            intensity_dic, annot_dic = find_thresholds(cochlea_annotations, cochlea, data_seg,
                                            table_measurement, column=column, pattern=stain)

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
            if "output_seg" in list(COCHLEAE[cochlea].keys()):
                output_seg = COCHLEAE[cochlea]["output_seg"]
                table_measurement_path = f"{cochlea}/tables/{output_seg}/subtype_ratio.tsv"
                table_path_s3, fs = get_s3_path(table_measurement_path)
                with fs.open(table_path_s3, "r") as f:
                    table_measurement = pd.read_csv(f, sep="\t")

            # Apply the threshold to all SGNs.
            if CUSTOM_THRESHOLDS.get(cochlea, {}).get(stain) is not None:
                custom_threshold_dic = CUSTOM_THRESHOLDS[cochlea][stain]
            else:
                custom_threshold_dic = None

            table_seg = apply_nearest_threshold(
                intensity_dic, table_seg, table_measurement, column=column, suffix=stain,
                threshold_dic=custom_threshold_dic,
            )

        # Save the table with positives / negatives for all SGNs.
        os.makedirs(output_dir, exist_ok=True)

        if not os.path.exists(out_path) or force:
            table_seg.to_csv(out_path, sep="\t", index=False)
        if not os.path.exists(annot_out) or force:
            annot_table.to_csv(annot_out, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker based on annotation thresholds."
    )

    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-a", "--annotation_dirs", type=str, nargs="+", default=None,
                        help="Directories containing marker annotations.")
    parser.add_argument("--threshold_save_dir", "-t")
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    evaluate_marker_annotation(
        args.cochlea, args.output, args.annotation_dirs, threshold_save_dir=args.threshold_save_dir, force=args.force,
    )


if __name__ == "__main__":
    main()
