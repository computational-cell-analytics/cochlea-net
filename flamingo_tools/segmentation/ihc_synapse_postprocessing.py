from typing import List, Tuple, Union

import numpy as np
import pandas as pd


def find_overlapping_masks(
    arr_base: np.ndarray,
    arr_ref: np.ndarray,
    label_id_base: int,
    running_label_id: int,
    min_overlap: float = 0.5,
) -> Tuple[List[dict], int]:
    """Find overlapping masks between a base array and a reference array.
    A label id of the base array is supplied and all unique IDs of the
    reference array are checked for a minimal overlap.
    Returns a list of all label IDs of the reference fulfilling this criteria.

    Args:
        arr_base: 3D array acting as base.
        arr_ref: 3D array acting as reference.
        label_id_base: Value of instance segmentation in base array.
        running_label_id: Unique label id for array, which replaces instance in base array.
        min_overlap: Minimal fraction of overlap between ref and base isntances to consider replacement.

    Returns:
        List of dictionaries containing reference label ID and new label ID in base array.
        The updated label ID for new arrays in base array.
    """
    edit_labels = []
    # base array containing only segmentation with too many synapses
    arr_base[arr_base != label_id_base] = 0
    if np.count_nonzero(arr_base) == 0:
        raise ValueError(f"Label id {label_id_base} not found in array. Wrong input?")
    arr_base = arr_base.astype(bool)

    edit_labels = []
    # iterate through segmentation ids in reference mask
    ref_ids = np.unique(arr_ref)[1:]
    for ref_id in ref_ids:
        arr_ref_instance = arr_ref.copy()
        arr_ref_instance[arr_ref_instance != ref_id] = 0
        arr_ref_instance = arr_ref_instance.astype(bool)

        intersection = np.logical_and(arr_ref_instance, arr_base)
        overlap_ratio = np.sum(intersection) / np.sum(arr_ref_instance)
        if overlap_ratio >= min_overlap:
            edit_labels.append({"ref_id": ref_id,
                                "new_label": running_label_id})
            running_label_id += 1

    return edit_labels, running_label_id


def replace_masks(
    arr_base: np.ndarray,
    arr_ref: np.ndarray,
    label_id_base: int,
    edit_labels: List[dict],
) -> np.ndarray:
    """Replace mask in base array with multiple masks from reference array.

    Args:
        data_base: Base array.
        data_ref: Reference array.
        label_id_base: Value of instance segmentation in base array to be replaced.
        edit_labels: List of dictionaries containing reference labels and new label ID.

    Returns:
        Base array with updated content.
    """
    print(f"Replacing {len(edit_labels)} instances")
    arr_base[arr_base == label_id_base] = 0
    for edit_dic in edit_labels:
        # bool array for new mask
        data_ref_id = arr_ref.copy()
        data_ref_id[data_ref_id != edit_dic["ref_id"]] = 0
        bool_ref = data_ref_id.astype(bool)

        arr_base[bool_ref] = edit_dic["new_label"]
    return arr_base


def postprocess_ihc_synapse_crop(
    data_base: np.typing.ArrayLike,
    data_ref: np.typing.ArrayLike,
    table_base: pd.DataFrame,
    synapse_limit: int = 25,
    min_overlap: float = 0.5,
) -> np.typing.ArrayLike:
    """Postprocess IHC segmentation based on number of synapse per IHC count.
    Segmentations from a base segmentation are analysed and replaced with
    instances from a reference segmentation, if suitable instances overlap with
    the base segmentation.

    Args:
        data_base_: Base array.
        data_ref_: Reference array.
        table_base: Segmentation table of base segmentation with synapse per IHC counts.
        synapse_limit: Limit of synapses per IHC to consider replacement of base segmentation.
        min_overlap: Minimal fraction of overlap between ref and base isntances to consider replacement.

    Returns:
        Base array with updated content.
    """
    # filter out problematic IHC segmentation
    table_edit = table_base[table_base["syn_per_IHC"] >= synapse_limit]

    running_label_id = int(table_base["label_id"].max() + 1)
    min_overlap = 0.5
    edit_labels = []

    seg_ids_base = np.unique(data_base)[1:]
    for seg_id_base in seg_ids_base:
        if seg_id_base in list(table_edit["label_id"]):

            edit_labels, running_label_id = find_overlapping_masks(
                data_base.copy(), data_ref.copy(), seg_id_base,
                running_label_id, min_overlap=min_overlap,
            )

            if len(edit_labels) > 1:
                data_base = replace_masks(data_base, data_ref, seg_id_base, edit_labels)
    return data_base


def postprocess_ihc_synapse(
    data_base: np.typing.ArrayLike,
    data_ref: np.typing.ArrayLike,
    table_base: pd.DataFrame,
    synapse_limit: int = 25,
    min_overlap: float = 0.5,
    roi_pad: int = 40,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
) -> np.typing.ArrayLike:
    """Postprocess IHC segmentation based on number of synapse per IHC count.
    Segmentations from a base segmentation are analysed and replaced with
    instances from a reference segmentation, if suitable instances overlap with
    the base segmentation.

    Args:
        data_base: Base array.
        data_ref: Reference array.
        table_base: Segmentation table of base segmentation with synapse per IHC counts.
        synapse_limit: Limit of synapses per IHC to consider replacement of base segmentation.
        min_overlap: Minimal fraction of overlap between ref and base isntances to consider replacement.
        roi_pad: Padding added to bounding box to analyze overlapping segmentation masks in a ROI.
        voxel_size: The voxel size of the data in micrometer.

    Returns:
        Base array with updated content.
    """
    # filter out problematic IHC segmentation
    table_edit = table_base[table_base["syn_per_IHC"] >= synapse_limit]

    running_label_id = int(table_base["label_id"].max() + 1)

    if not isinstance(voxel_size, float):
        if len(voxel_size) == 1:
            voxel_size = voxel_size * 3
        assert len(voxel_size) == 3
    else:
        voxel_size = (voxel_size,) * 3

    for _, row in table_edit.iterrows():
        # access array in image space (pixels)
        coords_max = [row["bb_max_x"], row["bb_max_y"], row["bb_max_z"]]
        coords_max = [int(round(c / voxel_size[num])) for num, c in enumerate(coords_max)]
        coords_min = [row["bb_min_x"], row["bb_min_y"], row["bb_min_z"]]
        coords_min = [int(round(c / voxel_size[num])) for num, c in enumerate(coords_min)]

        coords_max.reverse()
        coords_min.reverse()
        roi = tuple(slice(cmin - roi_pad, cmax + roi_pad) for cmax, cmin in zip(coords_max, coords_min))

        roi_base = data_base[roi]
        roi_ref = data_ref[roi]
        label_id_base = row["label_id"]

        edit_labels, running_label_id = find_overlapping_masks(
            roi_base.copy(), roi_ref.copy(), label_id_base,
            running_label_id, min_overlap=min_overlap,
        )

        if len(edit_labels) > 1:
            roi_base = replace_masks(roi_base, roi_ref, label_id_base, edit_labels)
            data_base[roi] = roi_base

    return data_base
