import json
import os
from typing import List, Optional

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import get_s3_path


def add_metadata_to_crop_table(
    table_in: str,
    data_dir: str,
    table_out: Optional[str] = None,
    min_size: int = 1000,
):
    """Add meta information like volume and crop dimension to an existing table,
    which compiles the crops used for training and validation of a segmentation network.

    Args:
        table_in: File path to TSV table.
        data_dir: Directory featuring sub-directories 'train' and 'val'.
        table_out: Output path for extended table.
        min_size: Minimal number of pixels per instance.
    """
    if table_out is None:
        table_out = table_in

    df = pd.read_csv(table_in, sep="\t")
    n_samples = []
    n_samples_undercut = []
    mean_vol = []
    min_vol = []
    max_vol = []
    dimensions = []
    for _, row in df.iterrows():
        file_name = row["Original"]
        dataset = row["Dataset"]
        seg_file = os.path.join(data_dir, dataset, f"{file_name}_annotations.tif")
        arr = imageio.imread(seg_file)
        unique_labels, counts = np.unique(arr, return_counts=True)
        samples = len(unique_labels) - 1
        counts_filtered = []
        if len(unique_labels) > 1 and unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
            counts = counts[1:]

        samples_undercut = 0
        samples = 0
        for (unique_label, count) in zip(unique_labels, counts):
            if count < min_size:
                samples_undercut += 1
                print(f"{file_name}: Pixel count {count} lower than minimal number {min_size} for ID {unique_label}.")
            else:
                samples += 1
                counts_filtered.append(count)

        n_samples.append(samples)
        n_samples_undercut.append(samples_undercut)
        mean_vol.append(round(np.mean(counts_filtered), 1))
        min_vol.append(min(counts_filtered))
        max_vol.append(max(counts_filtered))
        dimensions.append(str(arr.shape))

    df.loc[:, "dim"] = dimensions
    df.loc[:, "n_samples[>=1000px]"] = n_samples
    df.loc[:, "n_samples[<1000px]"] = n_samples_undercut
    df.loc[:, "mean_vol[px]"] = mean_vol
    df.loc[:, "min_vol[px]"] = min_vol
    df.loc[:, "max_vol[px]"] = max_vol

    df.to_csv(table_out, sep="\t", index=False)


def check_overlapping_crops(center1, center2, size):
    """
    Check if two 3D crops overlap in space.

    Parameters:
    -----------
    center1 : tuple or array-like
        Center coordinates of first crop (x, y, z)
    size1 : tuple or array-like
        Size of first crop (dx, dy, dz)
    center2 : tuple or array-like
        Center coordinates of second crop (x, y, z)
    size2 : tuple or array-like
        Size of second crop (dx, dy, dz)

    Returns:
    --------
    bool : True if crops overlap, False otherwise
    """
    center1 = np.array(center1)
    center2 = np.array(center2)
    size = np.array(size)

    # Calculate bounding box corners for both crops
    # For each dimension, calculate min and max coordinates
    half_size = size / 2
    min1 = center1 - half_size
    max1 = center1 + half_size
    min2 = center2 - half_size
    max2 = center2 + half_size

    # Check for overlap in each dimension
    # Crops overlap if they overlap in ALL dimensions
    overlap_x = (min1[0] < max2[0]) and (min2[0] < max1[0])
    overlap_y = (min1[1] < max2[1]) and (min2[1] < max1[1])
    overlap_z = (min1[2] < max2[2]) and (min2[2] < max1[2])

    # Crops overlap only if they overlap in all three dimensions
    return overlap_x and overlap_y and overlap_z


def check_all_crops_overlap(crop_centers, size=(128, 256, 256)):
    """
    Check if any pair of crops in a list overlaps.

    Parameters:
    -----------
    crops : list of tuples
        Each tuple contains (center, size) for a crop

    Returns:
    --------
    bool : True if any pair overlaps, False otherwise
    """
    n = len(crop_centers)
    if n == 1:
        return False
    overlap_list = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_list.append(check_overlapping_crops(crop_centers[i], crop_centers[j], size))
    return np.any(overlap_list)


def find_crop_centers_ihc(df, component_labels, crop_size=(128, 256, 256), max_crops_per_comp=10):
    n_blocks_try = [i + 1 for i in range(max_crops_per_comp)]
    n_blocks_try = sorted(n_blocks_try, reverse=True)
    total_centers = []
    for label in component_labels:
        subset = df[df["component_labels"] == label]
        length_sect = list(subset["length_fraction"])
        length_sect.sort()
        for n_blocks in n_blocks_try:
            target_s = np.linspace(length_sect[0], length_sect[-1], n_blocks * 2 + 1)
            target_s = [s for num, s in enumerate(target_s) if num % 2 == 1]
            centers = []
            for target in target_s:
                idx = (subset["length_fraction"] - target).abs().idxmin()
                closest_row = subset.loc[idx]
                center_physical = [closest_row["anchor_x"], closest_row["anchor_y"], closest_row["anchor_z"]]
                centers.append(center_physical)
            centers = [[round(c) for c in center] for center in centers]
            overlap = check_all_crops_overlap(centers, size=crop_size)
            if not overlap:
                print(f"Using {n_blocks} block(s) for label {label}.")
                best_centers = centers
                break
        total_centers.extend(best_centers)
    return total_centers


def export_crop_centers(
    cochlea: str,
    component_labels: List[int],
    out_dir: str,
    segmentation_channel: str = "IHC_v4b",
    halo_size: List[int] = [128, 256, 256],
    suffix: str = "crop",
):
    cell_type = segmentation_channel.split("_")[0]
    if cell_type in ["ihc", "IHC"]:
        # check training on PV
        image_channel = ["PV", "Vglut3"]
    elif cell_type in ["sgn", "SGN"]:
        image_channel = ["PV"]
    else:
        raise ValueError(f"Automatically determined cell type {cell_type} does not fit preset functions.")

    image_channel.append(segmentation_channel)
    seg_table_s3 = f"{cochlea}/tables/{segmentation_channel}/default.tsv"
    tsv_path, fs = get_s3_path(seg_table_s3)
    with fs.open(tsv_path, "r") as f:
        df = pd.read_csv(f, sep="\t")

    crop_size = [i * 2 for i in halo_size]
    total_centers = find_crop_centers_ihc(df, component_labels, crop_size=crop_size)
    n_blocks = len(total_centers)

    crop_dict = {}
    crop_dict["dataset_name"] = cochlea
    crop_dict["image_channel"] = image_channel
    crop_dict["segmentation_channel"] = segmentation_channel
    crop_dict["cell_type"] = cell_type.lower()
    crop_dict["n_blocks"] = n_blocks
    crop_dict["roi_halo"] = halo_size
    crop_dict["component_list"] = component_labels
    crop_dict["crop_centers"] = total_centers

    output_path = os.path.join(out_dir, f"{cochlea}_{suffix}_{cell_type.lower()}.json")
    with open(output_path, "w") as f:
        json.dump([crop_dict], f, indent='\t', separators=(',', ': '))
