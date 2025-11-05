import argparse
import json
import os
from typing import List, Optional

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.segmentation.chreef_utils import localize_median_intensities, find_annotations

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


def get_length_fraction_from_center(table, center_str, halo_size=20):
    """Get 'length_fraction' parameter for center coordinate by averaging nearby segmentation instances.
    """
    center_coord = tuple([int(c) for c in center_str.split("-")])
    (cx, cy, cz) = center_coord
    subset = table[
        (cx - halo_size < table["anchor_x"]) &
        (table["anchor_x"] < cx + halo_size) &
        (cy - halo_size < table["anchor_y"]) &
        (table["anchor_y"] < cy + halo_size) &
        (cz - halo_size < table["anchor_z"]) &
        (table["anchor_z"] < cz + halo_size)
    ]
    length_fraction = list(subset["length_fraction"])
    length_fraction = float(sum(length_fraction) / len(length_fraction))
    return length_fraction


def apply_nearest_threshold(intensity_dic, table_seg, table_measurement, halo_size=20):
    """Apply threshold to nearest segmentation instances.
    Crop centers are transformed into the "length fraction" parameter of the segmentation table.
    This avoids issues with the spiral shape of the cochlea and maps the assignment onto the Rosenthal"s canal.
    """
    # assign crop centers to length fraction of Rosenthal"s canal
    lf_intensity = {}
    for key in intensity_dic.keys():
        length_fraction = get_length_fraction_from_center(table_seg, key, halo_size=halo_size)
        intensity_dic[key]["length_fraction"] = length_fraction
        lf_intensity[length_fraction] = {"threshold": intensity_dic[key]["median_intensity"]}

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
    table_seg.loc[:, "marker_labels"] = marker_labels
    for num, fraction in enumerate(lf_fractions):
        subset_seg = table_seg[
            (table_seg["length_fraction"] > lf_limits[num]) &
            (table_seg["length_fraction"] < lf_limits[num + 1])
        ]
        # assign values based on limits
        threshold = lf_intensity[fraction]["threshold"]
        label_ids_seg = subset_seg["label_id"]

        subset_measurement = table_measurement[table_measurement["label_id"].isin(label_ids_seg)]
        subset_positive = subset_measurement[subset_measurement["median"] >= threshold]
        subset_negative = subset_measurement[subset_measurement["median"] < threshold]
        label_ids_pos = list(subset_positive["label_id"])
        label_ids_neg = list(subset_negative["label_id"])

        table_seg.loc[table_seg["label_id"].isin(label_ids_pos), "marker_labels"] = 1
        table_seg.loc[table_seg["label_id"].isin(label_ids_neg), "marker_labels"] = 2

    return table_seg


def find_thresholds(cochlea_annotations, cochlea, data_seg, table_measurement, resolution=0.38):
    # Find the median intensities by averaging the individual annotations for specific crops
    annotation_dics = {}
    annotated_centers = []
    for annotation_dir in cochlea_annotations:
        print(f"Localizing threshold with median intensities for {os.path.basename(annotation_dir)}.")
        annotation_dic = localize_median_intensities(annotation_dir, cochlea, data_seg, table_measurement,
                                                     resolution=resolution)
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

    return intensity_dic


def evaluate_marker_annotation(
    cochleae: List[str],
    output_dir: str,
    annotation_dirs: Optional[List[str]] = None,
    seg_name: str = "SGN_v2",
    marker_name: str = "GFP",
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
        seg_name: Identifier for segmentation.
        marker_name: Identifier for marker stain.
        threshold_save_dir: Optional directory for saving the thresholds.
        force: Whether to overwrite already existing results.
    """
    input_key = "s0"

    if marker_name == "rbOtof":
        halo_size = 150
        resolution = [1.887779, 1.887779, 3.0]
    else:
        halo_size = 20
        resolution = 0.38

    if annotation_dirs is None:
        if "MARKER_DIR" in globals():
            marker_dir = MARKER_DIR
            annotation_dirs = [entry.path for entry in os.scandir(marker_dir)
                               if os.path.isdir(entry) and "Results" in entry.name]

    seg_string = "-".join(seg_name.split("_"))
    for cochlea in cochleae:
        cochlea_str = "-".join(cochlea.split("_"))
        out_path = os.path.join(output_dir, f"{cochlea_str}_{marker_name}_{seg_string}.tsv")
        if os.path.exists(out_path) and not force:
            continue

        cochlea_annotations = [a for a in annotation_dirs if len(find_annotations(a, cochlea)["center_strings"]) != 0]
        print(f"Evaluating data for cochlea {cochlea} in {cochlea_annotations}.")

        # Get the segmentation data and table.
        input_path = f"{cochlea}/images/ome-zarr/{seg_name}.ome.zarr"
        input_path, fs = get_s3_path(input_path)
        data_seg = read_image_data(input_path, input_key)

        table_seg_path = f"{cochlea}/tables/{seg_name}/default.tsv"
        table_path_s3, fs = get_s3_path(table_seg_path)
        with fs.open(table_path_s3, "r") as f:
            table_seg = pd.read_csv(f, sep="\t")

        table_measurement_path = f"{cochlea}/tables/{seg_name}/{marker_name}_{seg_string}_object-measures.tsv"
        table_path_s3, fs = get_s3_path(table_measurement_path)
        with fs.open(table_path_s3, "r") as f:
            table_measurement = pd.read_csv(f, sep="\t")

        # Find the thresholds from the annotated blocks and save it if specified.
        intensity_dic = find_thresholds(cochlea_annotations, cochlea, data_seg, table_measurement,
                                        resolution=resolution)
        if threshold_save_dir is not None:
            os.makedirs(threshold_save_dir, exist_ok=True)
            threshold_out_path = os.path.join(threshold_save_dir, f"{cochlea_str}_{marker_name}_{seg_string}.json")
            with open(threshold_out_path, "w") as f:
                json.dump(intensity_dic, f, sort_keys=True, indent=4)

        # Apply the threshold to all SGNs.
        table_seg = apply_nearest_threshold(intensity_dic, table_seg, table_measurement, halo_size=halo_size)

        # Save the table with positives / negatives for all SGNs.
        os.makedirs(output_dir, exist_ok=True)
        table_seg.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker based on annotation thresholds."
    )

    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")
    parser.add_argument("-a", "--annotation_dirs", type=str, nargs="+", default=None,
                        help="Directories containing marker annotations.")
    parser.add_argument("-t", "--threshold_save_dir")
    parser.add_argument("-s", "--seg_name", type=str, default="SGN_v2")
    parser.add_argument("-m", "--marker_name", type=str, default="GFP")
    parser.add_argument("-f", "--force", action="store_true")

    args = parser.parse_args()
    evaluate_marker_annotation(
        args.cochlea, args.output, args.annotation_dirs, threshold_save_dir=args.threshold_save_dir,
        seg_name=args.seg_name, marker_name=args.marker_name, force=args.force
    )


if __name__ == "__main__":
    main()
