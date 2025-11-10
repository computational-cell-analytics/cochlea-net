import argparse
import json
import multiprocessing as mp
import os
from concurrent import futures
from typing import List

import numpy as np
import tifffile
from tqdm import tqdm

GT_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/2025-07_NIS3D/test"
PRED_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/val_nucleus/distance_unet_NIS3D"


def find_overlapping_masks(
    arr_base: np.ndarray,
    arr_ref: np.ndarray,
    label_id_base: int,
    min_overlap: float = 0.5,
) -> List[int]:
    """Find masks of segmentation, which have an overlap with undefined mask greater than 0.5.
    """
    labels_undefined_mask = []
    arr_base_undefined = arr_base == label_id_base

    # iterate through segmentation ids in reference mask
    ref_ids = list(np.unique(arr_ref)[1:])
    for ref_id in ref_ids:
        arr_ref_instance = arr_ref == ref_id

        intersection = np.logical_and(arr_ref_instance, arr_base_undefined)
        overlap_ratio = np.sum(intersection) / np.sum(arr_ref_instance)
        if overlap_ratio >= min_overlap:
            labels_undefined_mask.append(ref_id)

    return labels_undefined_mask


def find_matching_masks(arr_gt, arr_ref, out_path, labels_undefined_mask=[]):
    """For each instance in the reference array, the corresponding mask of the ground truth array,
    which has the biggest overlap, is identified.

    Args:
        arr_gt:
        arr_ref:
        out_path: Output path for saving dictionary.
        labels_undefined_mask: Labels of the reference array to exclude.
    """
    seg_ids_ref = [int(i) for i in np.unique(arr_ref)[1:]]
    print(f"total number of segmentation masks: {len(seg_ids_ref)}")
    seg_ids_ref = [s for s in seg_ids_ref if s not in labels_undefined_mask]
    print(f"number of segmentation masks after filtering undefined masks: {len(seg_ids_ref)}")

    def compute_overlap(ref_id):
        """Identify ID of segmentation mask with biggest overlap.
        Return matched IDs and overlap.
        """
        arr_ref_instance = arr_ref == ref_id

        seg_ids_gt = np.unique(arr_gt[arr_ref_instance])[1:]

        max_overlap = 0
        gt_id_match = None

        for gt_id in seg_ids_gt:
            arr_gt_instance = arr_gt == gt_id

            intersection = np.logical_and(arr_ref_instance, arr_gt_instance)
            overlap_ratio = np.sum(intersection) / np.sum(arr_ref_instance)
            if overlap_ratio > max_overlap:
                gt_id_match = int(gt_id.tolist())
                max_overlap = np.max([max_overlap, overlap_ratio])

        if gt_id_match is not None:
            return {
                "ref_id": ref_id,
                "gt_id": gt_id_match,
                "overlap": float(max_overlap.tolist())
            }
        else:
            return None

    n_threads = min(16, mp.cpu_count())
    print(f"Parallelizing with {n_threads} Threads.")
    with futures.ThreadPoolExecutor(n_threads) as pool:
        results = list(tqdm(pool.map(compute_overlap, seg_ids_ref), total=len(seg_ids_ref)))

    matching_masks = {r['ref_id']: r for r in results if r is not None}

    with open(out_path, "w") as f:
        json.dump(matching_masks, f, indent='\t', separators=(',', ': '))


def filter_true_positives(output_folder, prefixes, force_overwrite):
    """ Filter true positives from segmentation.
    Segmentation instances and ground truth labels are filtered symmetrically.
    The maximal overlap of each is computed and taken as a true positive if symmetric.
    The instance ID, the reference ID, and the overlap are saved in dictionaries.

    Args:
        output_folder: Output folder for dictionaries.
        prefixes: List of prefixes for evaluation. One or multiple of ["Drosophila", "MusMusculus", "Zebrafish"].
        force_overwrite: Flag for forced overwrite of existing output files.
    """
    if "PRED_DIR" in globals():
        pred_dir = PRED_DIR
    if "GT_DIR" in globals():
        gt_dir = GT_DIR

    if prefixes is None:
        prefixes = ["Drosophila", "MusMusculus", "Zebrafish"]

    for prefix in prefixes:
        conf_file = os.path.join(gt_dir, f"{prefix}_1_iitest_confidence.tif")
        annot_file = os.path.join(gt_dir, f"{prefix}_1_iitest_annotations.tif")
        conf_arr = tifffile.imread(conf_file)
        gt_arr = tifffile.imread(annot_file)

        seg_file = os.path.join(pred_dir, f"{prefix}_1_iitest_seg.tif")
        seg_arr = tifffile.imread(seg_file)

        # find largest overlap of ground truth mask with each segmentation instance
        out_path = os.path.join(output_folder, f"{prefix}_matching_ref_gt.json")
        if os.path.isfile(out_path) and not force_overwrite:
            print(f"Skipping the creation of {out_path}. File already exists.")
        else:
            # exclude detections with more than 50% of pixels in undefined category
            if 1 in np.unique(conf_arr)[1:]:
                labels_undefined_mask = find_overlapping_masks(conf_arr, seg_arr, label_id_base=1)
            else:
                labels_undefined_mask = []
                print("Array does not contain undefined mask")

            find_matching_masks(gt_arr, seg_arr, out_path, labels_undefined_mask=labels_undefined_mask)

        # find largest overlap of segmentation instance with each ground truth mask
        out_path = os.path.join(output_folder, f"{prefix}_matching_gt_ref.json")
        if os.path.isfile(out_path) and not force_overwrite:
            print(f"Skipping the creation of {out_path}. File already exists.")
        else:
            find_matching_masks(seg_arr, gt_arr, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--prefix", "-p", nargs="+", type=str, default=None)
    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")
    args = parser.parse_args()

    filter_true_positives(
        args.output_folder,
        args.prefix,
        args.force,
    )


if __name__ == "__main__":
    main()
