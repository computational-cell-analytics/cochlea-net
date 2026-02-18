import os
import multiprocessing as mp
from concurrent import futures
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm


def coord_from_string(center_str: str) -> Tuple[int]:
    return tuple([int(c) for c in center_str.split("-")])


def find_annotations(annotation_dir: str, cochlea: str, pattern: str = None) -> dict:
    """Create a dictionary for the analysis of ChReef annotations.

    Annotations should have format positive-negative_<cochlea>_crop_<coord>_allNegativeExcluded_thr<thr>.tif

    Args:
        annotation_dir: Directory containing annotations.
        cochlea: The name of the cochlea to analyze.

    Returns:
        Dictionary with information about the intensity annotations.
    """

    def extract_center_string(cochlea, name):
        # Extract center crop coordinate from file name
        crop_suffix = name.split(f"{cochlea}_crop_")[1]
        center_str = crop_suffix.split("_")[0]
        return center_str

    if pattern is not None:
        cochlea_files = [entry.name for entry in os.scandir(annotation_dir) if cochlea in entry.name
                         and pattern in entry.name]
    else:
        cochlea_files = [entry.name for entry in os.scandir(annotation_dir) if cochlea in entry.name]
    dic = {"cochlea": cochlea}
    dic["cochlea_files"] = cochlea_files
    center_strings = list(set([extract_center_string(cochlea, name=f) for f in cochlea_files]))
    center_strings.sort()
    dic["center_strings"] = center_strings
    remove_strings = []
    for center_str in center_strings:
        files_neg = [c for c in cochlea_files if all(x in c for x in [cochlea, center_str, "NegativeExcluded"])]
        files_pos = [c for c in cochlea_files if all(x in c for x in [cochlea, center_str, "WeakPositive"])]
        if len(files_neg) != 1 or len(files_pos) != 1:
            print(f"Skipping crop {center_str} for cochlea {cochlea}. "
                  f"Missing or multiple annotation files in {annotation_dir}.")
            remove_strings.append(center_str)
        else:
            dic[center_str] = {"file_neg": os.path.join(annotation_dir, files_neg[0]),
                               "file_pos": os.path.join(annotation_dir, files_pos[0])}
    for rm_str in remove_strings:
        dic["center_strings"].remove(rm_str)

    return dic


def get_roi(
    coord: tuple,
    roi_halo: tuple,
    resolution: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> Tuple[int]:
    """Get parameters for loading ROI of segmentation.

    Args:
        coord: Center coordinate.
        roi_halo: Halo for roi.
        resolution: Resolution of array in µm.

    Returns:
        The region of interest.
    """
    coords = list(coord)
    # reverse dimensions for correct extraction
    coords.reverse()
    coords = np.array(coords)
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))
    return roi


def find_overlapping_masks(
    arr_base: np.ndarray,
    arr_ref: np.ndarray,
    label_id_base: int,
    min_overlap: float = 0.5,
) -> List[int]:
    """Find overlapping masks between base array and reference array.

    Args:
        arr_base: Base array.
        arr_ref: Reference array.
        label_id_base: ID of segmentation to check for overlap.
        min_overlap: Minimal overlap to consider segmentation ID as matching.

    Returns:
        Matching IDs of reference array.
    """
    arr_base_labeled = arr_base == label_id_base

    # iterate through segmentation ids in reference mask
    ref_ids = list(np.unique(arr_ref)[1:])

    def check_overlap(ref_id):
        # check overlap of reference ID and base
        arr_ref_instance = arr_ref == ref_id

        intersection = np.logical_and(arr_ref_instance, arr_base_labeled)
        overlap_ratio = np.sum(intersection) / np.sum(arr_ref_instance)
        if overlap_ratio >= min_overlap:
            return ref_id
        else:
            return None

    n_threads = min(16, mp.cpu_count())
    print(f"Finding overlapping masks with {n_threads} Threads.")
    with futures.ThreadPoolExecutor(n_threads) as pool:
        results = list(tqdm(pool.map(check_overlap, ref_ids), total=len(ref_ids)))

    matching_ids = [r for r in results if r is not None]
    return matching_ids


def find_inbetween_ids(
    arr_negexc: np.typing.ArrayLike,
    arr_allweak: np.typing.ArrayLike,
    roi_seg: np.typing.ArrayLike,
) -> List[int]:
    """Identify list of segmentation IDs inbetween thresholds.

    Args:
        arr_negexc: Array with all negatives excluded.
        arr_allweak: Array with all weak positives.
        roi_seg: Region of interest of segmentation.

    Returns:
        A list of the ids that are in between the respective thresholds.
    """
    # negative annotation == 1, positive annotation == 2
    negexc_neg = find_overlapping_masks(arr_negexc, roi_seg, label_id_base=1)
    allweak_pos = find_overlapping_masks(arr_allweak, roi_seg, label_id_base=2)

    negexc_pos = find_overlapping_masks(arr_negexc, roi_seg, label_id_base=2)
    allweak_neg = find_overlapping_masks(arr_allweak, roi_seg, label_id_base=1)
    inbetween_ids = [int(i) for i in set(negexc_neg).intersection(set(allweak_pos))]
    return inbetween_ids, allweak_pos, negexc_neg, allweak_neg, negexc_pos


def get_crop_parameters(
    file_negexc: str,
    file_allweak: str,
    center: Tuple[int],
    data_seg: np.typing.ArrayLike,
    table_measure: pd.DataFrame,
    column: str = "median",
    resolution: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> dict:
    """Obtain parameters for a specific crop by analyzing the intensity annotations.

    Args:
        file_negexc: File path to annotation created by excluding all negative instances.
        file_allweak: File path to annotation created by thresholding all weakly positive instances.
        center: Center coordinate.
        data_seg: Segmentation data.
        table_measure: Table containing object measures.
        column: Name of column in measurement table.
        resolution: Voxel size in micrometer.

    Returns:
        parameter dictionary featuring analysis of crop
    """
    arr_negexc = tifffile.imread(file_negexc)
    arr_allweak = tifffile.imread(file_allweak)

    roi_halo = tuple([r // 2 for r in arr_negexc.shape])
    roi = get_roi(center, roi_halo, resolution=resolution)

    roi_seg = data_seg[roi]
    inbetween_ids, allweak_pos, negexc_neg, allweak_neg, negexc_pos = find_inbetween_ids(arr_negexc,
                                                                                         arr_allweak, roi_seg)

    param_dic = {}
    param_dic["seg_ids"] = list(np.unique(roi_seg)[1:])
    param_dic["inbetween_ids"] = inbetween_ids
    param_dic["allweak_pos"] = allweak_pos
    param_dic["allweak_neg"] = allweak_neg
    param_dic["negexc_neg"] = negexc_neg
    param_dic["negexc_pos"] = negexc_pos

    subset_allweak_pos = table_measure[table_measure["label_id"].isin(allweak_pos)]
    subset_allweak_neg = table_measure[table_measure["label_id"].isin(allweak_neg)]
    subset_negexc_neg = table_measure[table_measure["label_id"].isin(negexc_neg)]
    subset_negexc_pos = table_measure[table_measure["label_id"].isin(negexc_pos)]
    param_dic["allweak_pos_mean"] = float(subset_allweak_pos[column].mean())
    param_dic["allweak_neg_mean"] = float(subset_allweak_neg[column].mean())
    param_dic["negexc_neg_mean"] = float(subset_negexc_neg[column].mean())
    param_dic["negexc_pos_mean"] = float(subset_negexc_pos[column].mean())

    if len(inbetween_ids) == 0:
        if len(allweak_pos) == 0 and len(negexc_neg) == 0:
            param_dic["median_intensity"] = None
            return param_dic

        subset_positive = table_measure[table_measure["label_id"].isin(allweak_pos)]
        subset_negative = table_measure[table_measure["label_id"].isin(negexc_neg)]
        lowest_positive = float(subset_positive[column].min())
        highest_negative = float(subset_negative[column].max())
        if np.isnan(lowest_positive) or np.isnan(highest_negative):
            param_dic["median_intensity"] = None
            return param_dic

        param_dic["median_intensity"] = np.average([lowest_positive, highest_negative])
        return param_dic

    subset = table_measure[table_measure["label_id"].isin(inbetween_ids)]
    intensities = list(subset[column])
    param_dic["median_intensity"] = np.median(list(intensities))

    return param_dic


def localize_median_intensities(
    annotation_dir: str,
    cochlea: str,
    data_seg: np.typing.ArrayLike,
    table_measure: pd.DataFrame,
    column: str = "median",
    pattern: Optional[str] = None,
    resolution: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> str:
    """Find median intensities in blocks and assign them to center positions of cropped block.
    """
    annotation_dic = find_annotations(annotation_dir, cochlea, pattern=pattern)
    # center_keys = [key for key in annotation_dic["center_strings"] if key in annotation_dic.keys()]

    for center_str in annotation_dic["center_strings"]:
        center_coord = coord_from_string(center_str)
        print(f"Getting median intensities for {center_coord}.")
        file_pos = annotation_dic[center_str]["file_pos"]
        file_neg = annotation_dic[center_str]["file_neg"]
        param_dic = get_crop_parameters(file_neg, file_pos, center_coord, data_seg,
                                        table_measure, column=column, resolution=resolution)

        median_intensity = param_dic["median_intensity"]
        if median_intensity is None:
            print(f"No threshold identified for {center_str}.")

        for key in param_dic.keys():
            annotation_dic[center_str][key] = param_dic[key]

    return annotation_dic


def get_length_fraction_from_center(
    table: pd.DataFrame,
    center_str: str,
    halo_size: int = 20,
) -> float:
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


def apply_nearest_threshold(
    intensity_dic: dict,
    table_seg: pd.DataFrame,
    table_meas: pd.DataFrame,
    column: str = "median",
    suffix: str = "labels",
    threshold_dic: Optional[dict] = None,
    halo_size: int = 20,
) -> pd.DataFrame:
    """Apply threshold to nearest segmentation instances.
    Crop centers are transformed into the "length fraction" parameter of the segmentation table.
    This avoids issues with the spiral shape of the cochlea and maps the assignment onto the Rosenthal"s canal.
    """
    # assign crop centers to length fraction of Rosenthal"s canal
    lf_intensity = {}
    for key in intensity_dic.keys():
        length_fraction = get_length_fraction_from_center(table_seg, key, halo_size=halo_size)
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


def find_thresholds(
    cochlea_annotations: List[str],
    cochlea: str,
    data_seg: np.typing.ArrayLike,
    table_meas: pd.DataFrame,
    column: str = "median",
    pattern: Optional[str] = None,
    resolution: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> Tuple[dict, dict]:
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
