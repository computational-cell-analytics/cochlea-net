import os
import multiprocessing as mp
import re
from concurrent import futures
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from tqdm import tqdm

from ..export_data_utils import normalize_voxel_size

# Label values used by the napari annotation layer, see `annotation_utils.annotation_napari`.
NEGATIVE_LABEL = 1
POSITIVE_LABEL = 2

# The threshold an annotator dialed in is appended to the file name, either as a bare number
# ("_46.tif") or with a "thr" prefix ("_thr39.tif").
_THRESHOLD_PATTERN = re.compile(r"^(?:thr)?(\d+(?:\.\d+)?)")


def coord_from_string(center_str: str) -> Tuple[int]:
    return tuple([int(c) for c in center_str.split("-")])


def threshold_from_filename(file_path: str) -> Optional[float]:
    """Read the threshold that an annotator recorded in the annotation file name.

    Args:
        file_path: File path of the annotation.

    Returns:
        The threshold, or None if the file name does not end in a numeric token.
    """
    stem = os.path.splitext(os.path.basename(file_path))[0]
    match = _THRESHOLD_PATTERN.match(stem.rsplit("_", 1)[-1])
    return None if match is None else float(match.group(1))


def threshold_from_filenames(file_paths: List[Optional[str]]) -> Optional[float]:
    """Average the thresholds recorded in the names of the annotation files of one crop.

    The "allNegativeExcluded" file holds the high threshold and the "allPositiveIncluded" file the
    low one, so the true threshold is inside that band, like the value derived from the annotations.

    Args:
        file_paths: File paths of the annotations of a single crop. Missing files may be None.

    Returns:
        The averaged threshold, or None if no file name contains one.
    """
    thresholds = [threshold_from_filename(path) for path in file_paths if path is not None]
    thresholds = [thr for thr in thresholds if thr is not None]
    return None if len(thresholds) == 0 else float(np.average(thresholds))


def find_annotations(annotation_dir: str, cochlea: str, pattern: str = None) -> dict:
    """Create a dictionary for the analysis of ChReef annotations.

    Annotations should have format positive-negative_<cochlea>_crop_<coord>_allNegativeExcluded_thr<thr>.tif

    A crop with only one of the two files is kept, with None for the missing side. Such a crop can
    still yield a threshold if the annotation is uniformly positive or negative,
    see `get_single_annotation_parameters`.

    Args:
        annotation_dir: Directory containing annotations.
        cochlea: The name of the cochlea to analyze.
        pattern: Optional substring that a file name must contain, used to select a single stain.

    Returns:
        Dictionary with information about the intensity annotations.
    """
    crop_token = f"{cochlea}_crop_"

    def extract_center_string(name):
        # Extract center crop coordinate from file name
        crop_suffix = name.split(crop_token)[1]
        center_str = crop_suffix.split("_")[0]
        return center_str

    cochlea_files = [entry.name for entry in os.scandir(annotation_dir) if crop_token in entry.name and
                     (pattern is None or pattern in entry.name)]
    dic = {"cochlea": cochlea}
    dic["cochlea_files"] = cochlea_files
    center_strings = list(set([extract_center_string(f) for f in cochlea_files]))
    center_strings.sort()
    dic["center_strings"] = center_strings
    remove_strings = []
    for center_str in center_strings:
        files_neg = [c for c in cochlea_files if center_str in c and "negativeexcluded" in c.lower()]
        files_pos = [c for c in cochlea_files if center_str in c and "positiveincluded" in c.lower()]
        if len(files_neg) > 1 or len(files_pos) > 1 or len(files_neg) + len(files_pos) == 0:
            print(f"Skipping crop {center_str} for cochlea {cochlea}. "
                  f"Missing or multiple annotation files in {annotation_dir}.")
            remove_strings.append(center_str)
            continue
        if len(files_neg) == 0 or len(files_pos) == 0:
            missing = "allPositiveIncluded" if len(files_pos) == 0 else "allNegativeExcluded"
            print(f"Crop {center_str} for cochlea {cochlea} has no {missing} annotation. "
                  "Trying to derive a threshold from the single annotation.")
        dic[center_str] = {
            "file_neg": os.path.join(annotation_dir, files_neg[0]) if files_neg else None,
            "file_pos": os.path.join(annotation_dir, files_pos[0]) if files_pos else None,
        }
    for rm_str in remove_strings:
        dic["center_strings"].remove(rm_str)

    return dic


def get_roi(
    coord: tuple,
    crop_shape: Tuple[int, ...],
    data_shape: Tuple[int, ...],
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> Tuple[slice, ...]:
    """Get the region of interest that an exported crop was taken from.

    The crop was written by `export_data_utils.compute_crop_bb`, which clamps the bounding box to
    the volume with `start = max(0, center - halo)` and `stop = min(shape, center + halo)`. A crop
    at a volume border is therefore smaller than 2 * halo and not centered on `coord`, so its halo
    cannot be recovered as `shape // 2`. Clamping the start into `[0, data_shape - crop_shape]`
    inverts that formula for an unclipped crop and for a crop clipped at either border, so the ROI
    always has exactly `crop_shape` and stays aligned with the crop.

    Args:
        coord: Center coordinate of the crop in micrometer, in (x, y, z) order.
        crop_shape: Shape of the exported crop in pixel, in (z, y, x) order.
        data_shape: Shape of the full volume in pixel, in (z, y, x) order.
        voxel_size: Voxel size of array in micrometer, in (x, y, z) order.

    Returns:
        The region of interest, in (z, y, x) order.
    """
    voxel_size_zyx = np.array(normalize_voxel_size(voxel_size)[::-1])
    center_zyx = np.round(np.array(coord[::-1]) / voxel_size_zyx).astype(np.int32)

    roi = []
    for center, extent, dim in zip(center_zyx, crop_shape, data_shape):
        if extent > dim:
            raise ValueError(f"The crop shape {tuple(crop_shape)} does not fit the volume {tuple(data_shape)}.")
        start = int(min(max(int(center) - extent // 2, 0), dim - extent))
        roi.append(slice(start, start + extent))
    return tuple(roi)


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
    if arr_base.shape != arr_ref.shape:
        raise ValueError(f"Shape mismatch between annotation {arr_base.shape} and segmentation {arr_ref.shape}.")

    arr_base_labeled = arr_base == label_id_base

    # iterate through segmentation ids in reference mask
    ref_ids = np.unique(arr_ref)
    ref_ids = list(ref_ids[ref_ids != 0])

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
    negexc_neg = find_overlapping_masks(arr_negexc, roi_seg, label_id_base=NEGATIVE_LABEL)
    allweak_pos = find_overlapping_masks(arr_allweak, roi_seg, label_id_base=POSITIVE_LABEL)

    negexc_pos = find_overlapping_masks(arr_negexc, roi_seg, label_id_base=POSITIVE_LABEL)
    allweak_neg = find_overlapping_masks(arr_allweak, roi_seg, label_id_base=NEGATIVE_LABEL)
    inbetween_ids = [int(i) for i in set(negexc_neg).intersection(set(allweak_pos))]
    return inbetween_ids, allweak_pos, negexc_neg, allweak_neg, negexc_pos


def read_annotation(file_path: str) -> np.ndarray:
    """Read an annotation volume, rejecting a file that is not a 3D volume.

    A truncated TIF makes `tifffile.imread` fall back to a single 2D plane instead of raising, which
    would otherwise be reconstructed into a wrong region of interest.

    Args:
        file_path: File path of the annotation.

    Returns:
        The annotation volume.
    """
    arr = tifffile.imread(file_path)
    if arr.ndim != 3:
        raise ValueError(f"Expected a 3D annotation in {file_path}, got shape {arr.shape}. "
                         "The file is probably truncated.")
    return arr


def _empty_crop_parameters() -> dict:
    """Build the crop parameters of a crop that could not be analyzed.

    All keys of `get_crop_parameters` are present, so that the consumers of the annotation
    dictionary do not have to special-case a failed crop.
    """
    param_dic = {key: [] for key in
                 ["seg_ids", "inbetween_ids", "allweak_pos", "allweak_neg", "negexc_neg", "negexc_pos"]}
    param_dic.update({key: float("nan") for key in
                      ["allweak_pos_mean", "allweak_neg_mean", "negexc_neg_mean", "negexc_pos_mean"]})
    param_dic["median_intensity"] = None
    param_dic["threshold_source"] = None
    return param_dic


def get_single_annotation_parameters(
    file_path: str,
    table_measure: pd.DataFrame,
    column: str = "median",
    all_negative_threshold_factor: float = 1.5,
) -> dict:
    """Derive the threshold of a crop that has only one of the two annotation files.

    One annotation is enough when it labels every instance the same way. An annotation without any
    positive instance means that the whole crop is negative, so the threshold is set above the
    highest measured value. An annotation without any negative instance means the opposite. The
    maximum is taken over the full measurement table, because the threshold is applied to a whole
    section of the cochlea and not only to the crop.

    A one-sided annotation that contains both classes does not define a threshold, because the
    second file is the one that bounds it from the other side.

    Args:
        file_path: File path of the single annotation.
        table_measure: Table containing object measures.
        column: Name of column in measurement table.
        all_negative_threshold_factor: Factor applied to the highest '{column}' value to estimate
            the threshold for an all-negative annotation.

    Returns:
        Parameter dictionary featuring analysis of crop.
    """
    param_dic = _empty_crop_parameters()
    arr = read_annotation(file_path)
    labels = set(np.unique(arr).tolist()) - {0}
    name = os.path.basename(file_path)

    if labels == {POSITIVE_LABEL}:
        param_dic["median_intensity"] = 0.0
        param_dic["threshold_source"] = "single-annotation-positive"
        print(f"All instances are annotated as positive in {name}. Using a threshold of 0.")
    elif labels == {NEGATIVE_LABEL}:
        param_dic["median_intensity"] = all_negative_threshold_factor * float(table_measure[column].max())
        param_dic["threshold_source"] = "single-annotation-negative"
        print(f"All instances are annotated as negative in {name}. "
              f"Using a threshold of {param_dic['median_intensity']}, "
              f"{all_negative_threshold_factor} times the highest '{column}' value.")
    elif len(labels) == 0:
        print(f"No annotated instance in {name}. No threshold can be derived from it.")
    else:
        print(f"Both positive and negative instances are annotated in {name}, "
              "but the second annotation file of this crop is missing. No threshold can be derived from it.")

    return param_dic


def get_crop_parameters(
    file_negexc: str,
    file_allweak: str,
    center: Tuple[int],
    data_seg: np.typing.ArrayLike,
    table_measure: pd.DataFrame,
    column: str = "median",
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> dict:
    """Obtain parameters for a specific crop by analyzing the intensity annotations.

    Args:
        file_negexc: File path to annotation created by excluding all negative instances.
        file_allweak: File path to annotation created by thresholding all weakly positive instances.
        center: Center coordinate.
        data_seg: Segmentation data.
        table_measure: Table containing object measures.
        column: Name of column in measurement table.
        voxel_size: Voxel size in micrometer.

    Returns:
        parameter dictionary featuring analysis of crop
    """
    arr_negexc = read_annotation(file_negexc)
    arr_allweak = read_annotation(file_allweak)
    if arr_negexc.shape != arr_allweak.shape:
        raise ValueError(f"The two annotations of this crop have different shapes, "
                         f"{arr_negexc.shape} and {arr_allweak.shape}.")

    roi = get_roi(center, arr_negexc.shape, data_seg.shape, voxel_size=voxel_size)

    roi_seg = data_seg[roi]
    inbetween_ids, allweak_pos, negexc_neg, allweak_neg, negexc_pos = find_inbetween_ids(arr_negexc,
                                                                                         arr_allweak, roi_seg)

    param_dic = {"threshold_source": "annotation"}
    seg_ids = np.unique(roi_seg)
    param_dic["seg_ids"] = list(seg_ids[seg_ids != 0])
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
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    use_filename_threshold: bool = False,
) -> str:
    """Find median intensities in blocks and assign them to center positions of cropped block.

    A crop that cannot be analyzed does not abort the other crops. Its threshold is None, unless
    `use_filename_threshold` allows to fall back to the threshold recorded in the file name.

    Args:
        annotation_dir: Directory containing annotations.
        cochlea: The name of the cochlea to analyze.
        data_seg: Segmentation data.
        table_measure: Table containing object measures.
        column: Name of column in measurement table.
        pattern: Optional substring that an annotation file name must contain.
        voxel_size: Voxel size in micrometer.
        use_filename_threshold: Whether to use the threshold recorded in the annotation file name
            when the annotations do not yield one.

    Returns:
        Dictionary with the analysis of the annotations per crop.
    """
    annotation_dic = find_annotations(annotation_dir, cochlea, pattern=pattern)

    for center_str in annotation_dic["center_strings"]:
        center_coord = coord_from_string(center_str)
        print(f"Getting median intensities for {center_coord}.")
        file_pos = annotation_dic[center_str]["file_pos"]
        file_neg = annotation_dic[center_str]["file_neg"]
        try:
            if file_pos is None or file_neg is None:
                param_dic = get_single_annotation_parameters(
                    file_neg if file_pos is None else file_pos, table_measure, column=column,
                )
            else:
                param_dic = get_crop_parameters(file_neg, file_pos, center_coord, data_seg,
                                                table_measure, column=column, voxel_size=voxel_size)
        except Exception as error:
            print(f"Could not analyze the annotations of crop {center_str}: {error}")
            param_dic = _empty_crop_parameters()

        if param_dic["median_intensity"] is None:
            param_dic["threshold_source"] = None
            print(f"No threshold identified for {center_str}.")
            if use_filename_threshold:
                threshold = threshold_from_filenames([file_neg, file_pos])
                if threshold is None:
                    print(f"No threshold in the annotation file names of crop {center_str} either.")
                else:
                    param_dic["median_intensity"] = threshold
                    param_dic["threshold_source"] = "filename"
                    print(f"Using the threshold {threshold} from the annotation file names of crop {center_str}.")

        for key in param_dic.keys():
            annotation_dic[center_str][key] = param_dic[key]

    return annotation_dic


def get_length_fraction_from_center(
    table: pd.DataFrame,
    center_str: str,
    halo_size: int = 20,
) -> Optional[float]:
    """Get 'length_fraction' parameter for center coordinate by averaging nearby segmentation instances.

    Args:
        table: Segmentation table.
        center_str: Center coordinate of crop as reference point.
        halo_size: Halo in micrometer to find nearby SGN instances.

    Returns:
        Average length fraction or None if no instance is in range.
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
    if len(length_fraction) == 0:
        return None
    length_fraction = float(sum(length_fraction) / len(length_fraction))
    return length_fraction


def map_crops_to_length_fraction(
    intensity_dic: dict,
    table_seg: pd.DataFrame,
    threshold_dic: Optional[dict] = None,
    halo_size: int = 20,
) -> dict:
    """Assign crop centers to the "length fraction" parameter of Rosenthal's canal.

    The crop center is only a point in space, so the length fraction is averaged over the
    segmentation instances around it. The halo is doubled until at least one instance is in range.

    A crop without a threshold is left out, so that the neighboring crops cover a larger part of the
    cochlea. Keeping it would let it claim a section in which no instance can be classified.

    Args:
        intensity_dic: Dictionary with the crop center strings as keys.
        table_seg: Segmentation table.
        threshold_dic: Optional custom thresholds, either a single value or a dictionary per crop.
        halo_size: Halo in micrometer to find nearby segmentation instances.

    Returns:
        Dictionary of the length fraction to the threshold and center string of the crop, sorted by length fraction.
    """
    lf_intensity = {}
    for key in intensity_dic.keys():
        if threshold_dic is None and _threshold_value(intensity_dic[key]["median_intensity"]) is None:
            print(f"Skipping crop {key}. No threshold available, "
                  "so the neighboring crops cover this part of the cochlea.")
            continue
        length_fraction = get_length_fraction_from_center(table_seg, key, halo_size=halo_size)
        while length_fraction is None:
            halo_size = 2 * halo_size
            print(f"Found no instance in ROI halo around center coordinate. Trying halo {halo_size}.")
            if halo_size > 150:
                raise ValueError("No SGN instance nearby.")
            length_fraction = get_length_fraction_from_center(table_seg, key, halo_size=halo_size)

        intensity_dic[key]["length_fraction"] = length_fraction
        if threshold_dic is None:
            threshold = _threshold_value(intensity_dic[key]["median_intensity"])
        else:
            if isinstance(threshold_dic, (int, float)):
                threshold = threshold_dic
            else:
                threshold = threshold_dic[key]["manual"]
            print(f"Using custom threshold {threshold} for crop {key}.")
        lf_intensity[length_fraction] = {"threshold": threshold, "center": key}

    return dict(sorted(lf_intensity.items()))


def length_fraction_limits(lf_fractions: List[float]) -> List[float]:
    """Get the limits of the cochlea sections that each crop threshold is applied to.

    Args:
        lf_fractions: Length fractions of the crop centers, sorted in ascending order.

    Returns:
        The section limits, from the start to the end of the cochlea.
    """
    # start of cochlea
    lf_limits = [0]
    # half distance between block centers
    for i in range(len(lf_fractions) - 1):
        lf_limits.append((lf_fractions[i] + lf_fractions[i + 1]) / 2)
    # end of cochlea
    lf_limits.append(1)
    return lf_limits


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
    lf_intensity = map_crops_to_length_fraction(
        intensity_dic, table_seg, threshold_dic=threshold_dic, halo_size=halo_size,
    )
    if len(lf_intensity) == 0:
        raise ValueError("No crop has a threshold, so no marker label can be assigned. "
                         "Check the annotation files of this cochlea.")
    lf_fractions = list(lf_intensity.keys())
    lf_limits = length_fraction_limits(lf_fractions)

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


def _threshold_value(value) -> Optional[float]:
    # find_thresholds stores the averaged threshold as a one-element sequence.
    if isinstance(value, (list, tuple)):
        value = value[0] if len(value) > 0 else None
    return None if value is None else float(value)


def percentage(count: int, total: int) -> float:
    return float(round(100 * count / total, 4)) if total > 0 else 0.0


def _count_marker_labels(
    thresholds: Dict[str, float],
    table_seg: pd.DataFrame,
    table_meas: pd.DataFrame,
    column: str = "median",
    halo_size: int = 20,
) -> Tuple[dict, dict]:
    """Count positive and negative instances for one set of crop thresholds.

    The instances are assigned to crops the same way as in `apply_nearest_threshold`.

    Args:
        thresholds: Threshold per crop center string.
        table_seg: Segmentation table.
        table_meas: Table containing object measures.
        column: Name of column in the measurement table.
        halo_size: Halo in micrometer to find nearby segmentation instances.

    Returns:
        The counts for the whole cochlea and the counts per crop.
    """
    intensity_dic = {center: {"median_intensity": threshold} for center, threshold in thresholds.items()}
    lf_intensity = map_crops_to_length_fraction(intensity_dic, table_seg, halo_size=halo_size)
    lf_fractions = list(lf_intensity.keys())
    lf_limits = length_fraction_limits(lf_fractions)

    crop_counts = {}
    n_positive = 0
    n_negative = 0
    for num, fraction in enumerate(lf_fractions):
        subset_seg = table_seg[
            (table_seg["length_fraction"] > lf_limits[num]) &
            (table_seg["length_fraction"] < lf_limits[num + 1])
        ]
        threshold = lf_intensity[fraction]["threshold"]
        subset_measurement = table_meas[table_meas["label_id"].isin(subset_seg["label_id"])]
        crop_positive = int((subset_measurement[column] >= threshold).sum())
        crop_negative = int((subset_measurement[column] < threshold).sum())
        n_positive += crop_positive
        n_negative += crop_negative

        crop_counts[lf_intensity[fraction]["center"]] = {
            "threshold": threshold,
            "length_fraction": fraction,
            "n_positive": crop_positive,
            "n_negative": crop_negative,
            "percent_positive": percentage(crop_positive, crop_positive + crop_negative),
            "percent_negative": percentage(crop_negative, crop_positive + crop_negative),
        }

    counts = {
        "n_crops": len(lf_fractions),
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_unassigned": int(len(table_seg) - n_positive - n_negative),
        "percent_positive": percentage(n_positive, n_positive + n_negative),
        "percent_negative": percentage(n_negative, n_positive + n_negative),
    }
    return counts, crop_counts


def evaluate_annotator_variance(
    intensity_dic: dict,
    table_seg: pd.DataFrame,
    table_meas: pd.DataFrame,
    column: str = "median",
    halo_size: int = 20,
    cochlea: Optional[str] = None,
    marker_name: Optional[str] = None,
    seg_name: Optional[str] = None,
    custom_thresholds: bool = False,
) -> dict:
    """Quantify how much the manual threshold annotation depends on the annotator.

    The thresholds of a single annotator are applied to the whole cochlea, which gives one
    scenario per annotator. These scenarios are compared to each other and to the "median"
    scenario, which uses the thresholds averaged over all annotators.

    An annotator does not have to annotate every crop. Crops without a threshold for this annotator
    are left out of the scenario, so that the neighboring crops cover a larger part of the cochlea.

    Args:
        intensity_dic: Threshold dictionary created by `find_thresholds`.
        table_seg: Segmentation table.
        table_meas: Table containing object measures.
        column: Name of column in the measurement table.
        halo_size: Halo in micrometer to find nearby segmentation instances.
        cochlea: Identifier for the cochlea, saved as metadata.
        marker_name: Identifier for the marker stain, saved as metadata.
        seg_name: Identifier for the segmentation, saved as metadata.
        custom_thresholds: Whether a custom threshold overrides the annotations for this stain.
            The "median" scenario always uses the annotation-derived threshold, so that the
            comparison stays between annotators. The flag records that the segmentation table
            was created with different values.

    Returns:
        Dictionary with the marker percentages per scenario, their variance, and a per-crop breakdown.
    """
    consensus_key = "median"

    consensus_thresholds = {}
    for center, crop_dic in intensity_dic.items():
        threshold = _threshold_value(crop_dic.get("median_intensity"))
        if threshold is None:
            print(f"Skipping crop {center} for the variance evaluation. No threshold available.")
            continue
        consensus_thresholds[center] = threshold

    annotators = sorted({
        annotator for crop_dic in intensity_dic.values() for annotator in crop_dic.get("annotator_intensities", {})
    })

    n_crops = len(consensus_thresholds)
    scenarios = {consensus_key: consensus_thresholds}
    for annotator in annotators:
        thresholds = {
            center: float(crop_dic["annotator_intensities"][annotator])
            for center, crop_dic in intensity_dic.items()
            if annotator in crop_dic.get("annotator_intensities", {})
        }
        if len(thresholds) < n_crops:
            print(f"Annotator {annotator} has thresholds for {len(thresholds)}/{n_crops} crops. "
                  "Not all crops of this annotator are available.")
        scenarios[annotator] = thresholds

    counts = {}
    crops = {}
    for name, thresholds in scenarios.items():
        if len(thresholds) == 0:
            print(f"Skipping scenario {name} for the variance evaluation. No threshold available.")
            continue
        scenario_counts, crop_counts = _count_marker_labels(
            thresholds, table_seg, table_meas, column=column, halo_size=halo_size,
        )
        counts[name] = scenario_counts
        for center, crop_dic in crop_counts.items():
            crops.setdefault(center, {})[name] = crop_dic

    evaluated = [annotator for annotator in annotators if annotator in counts]
    percentages = [counts[annotator]["percent_positive"] for annotator in evaluated]

    deviation = {}
    if consensus_key in counts:
        consensus_percentage = counts[consensus_key]["percent_positive"]
        deviation = {
            annotator: float(round(counts[annotator]["percent_positive"] - consensus_percentage, 4))
            for annotator in evaluated
        }

    pairwise_difference = {}
    for num, first in enumerate(evaluated):
        for second in evaluated[num + 1:]:
            difference = counts[first]["percent_positive"] - counts[second]["percent_positive"]
            pairwise_difference[f"{first}-{second}"] = float(round(difference, 4))

    # The percentages of positives and negatives add up to 100, so they have the same variance.
    if len(percentages) > 0:
        variance = float(round(np.var(percentages), 4))
        deviation_std = float(round(np.std(percentages), 4))
    else:
        variance, deviation_std = None, None

    return {
        "cochlea": cochlea,
        "marker": marker_name,
        "segmentation": seg_name,
        "custom_thresholds": custom_thresholds,
        "annotators": annotators,
        "n_crops": n_crops,
        "scenarios": counts,
        "deviation_from_median": deviation,
        "pairwise_difference": pairwise_difference,
        "variance": {
            "percent_positive": variance,
            "percent_negative": variance,
            "std_percent_positive": deviation_std,
            "std_percent_negative": deviation_std,
        },
        "crops": {center: crops[center] for center in sorted(crops)},
    }


def find_thresholds(
    cochlea_annotations: List[str],
    cochlea: str,
    data_seg: np.typing.ArrayLike,
    table_meas: pd.DataFrame,
    column: str = "median",
    pattern: Optional[str] = None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    use_filename_threshold: bool = False,
) -> Tuple[dict, dict]:
    """ Find the median intensities by averaging the individual annotations for specific crops.

    Args:
        cochlea_annotations: Directories containing the annotations of the individual annotators.
        cochlea: The name of the cochlea to analyze.
        data_seg: Segmentation data.
        table_meas: Table containing object measures.
        column: Name of column in measurement table.
        pattern: Optional substring that an annotation file name must contain.
        voxel_size: Voxel size in micrometer.
        use_filename_threshold: Whether to use the threshold recorded in the annotation file name
            when the annotations of a crop do not yield one.

    Returns:
        The threshold per crop, averaged over the annotators, and the analysis per annotator.
    """
    annotation_dics = {}
    annotated_centers = []
    for annotation_dir in cochlea_annotations:
        print(f"Localizing threshold with median intensities for {os.path.abspath(annotation_dir)}.")
        annotation_dic = localize_median_intensities(annotation_dir, cochlea, data_seg,
                                                     table_meas, column=column, pattern=pattern,
                                                     voxel_size=voxel_size,
                                                     use_filename_threshold=use_filename_threshold)
        annotated_centers.extend(annotation_dic["center_strings"])
        annotation_dics[annotation_dir] = annotation_dic

    annotated_centers = list(set(annotated_centers))
    intensity_dic = {}
    print("Annotator keys", list(annotation_dics.keys()))
    # loop over all annotated blocks
    for annotated_center in annotated_centers:
        intensities = []
        annotator_success = []
        annotator_failure = []
        annotator_missing = []
        annotator_intensities = {}
        threshold_sources = {}
        # loop over annotated block from single user
        for annotator_key in annotation_dics.keys():
            subdirname = os.path.basename(os.path.abspath(annotator_key))
            if annotated_center not in annotation_dics[annotator_key]["center_strings"]:
                annotator_missing.append(subdirname)
                continue
            else:
                crop_dic = annotation_dics[annotator_key][annotated_center]
                median_intensity = crop_dic["median_intensity"]
                if median_intensity is None:
                    print(f"No threshold for {subdirname} and crop {annotated_center}.")
                    annotator_failure.append(subdirname)
                else:
                    intensities.append(median_intensity)
                    annotator_success.append(subdirname)
                    annotator_intensities[subdirname] = float(median_intensity)
                    threshold_sources[subdirname] = crop_dic.get("threshold_source")

        if len(intensities) == 0:
            print(f"No viable annotation for cochlea {cochlea} and crop {annotated_center}.")
            median_int_avg = None
        else:
            median_int_avg = float(sum(intensities) / len(intensities))

        intensity_dic[annotated_center] = {
            "median_intensity": median_int_avg,
            "annotator_intensities": annotator_intensities,
            "threshold_sources": threshold_sources,
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
