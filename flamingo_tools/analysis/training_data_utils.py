import json
import os
from typing import List, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from scipy.ndimage import label as label_structures

from flamingo_tools.s3_utils import get_s3_path


def add_metadata_to_crop_table(
    table_in: str,
    data_dir: str,
    table_out: Optional[str] = None,
    min_size: int = 1000,
    label_dir: str = None,
):
    """Add meta information like volume and crop dimension to an existing table,
    which compiles the crops used for training and validation of a segmentation network.

    Args:
        table_in: File path to TSV table.
        data_dir: Directory featuring sub-directories with datasets, e.g. 'train' and 'val'.
        table_out: Output path for extended table.
        min_size: Minimal number of pixels per instance.
        label_dir: Directory containing annotations.
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
        if label_dir is None:
            seg_file = os.path.join(data_dir, dataset, f"{file_name}_annotations.tif")
        else:
            seg_file = os.path.join(label_dir, f"{file_name}_annotations.tif")
        arr = imageio.imread(seg_file)
        unique_labels, counts = np.unique(arr, return_counts=True)
        samples = len(unique_labels) - 1
        counts_filtered = []
        if len(unique_labels) > 1 and unique_labels[0] == 0:
            unique_labels = unique_labels[1:]
            counts = counts[1:]
        # empty crop
        else:
            unique_labels = []

        samples_undercut = 0
        samples = 0
        for (unique_label, count) in zip(unique_labels, counts):
            if count < min_size:
                samples_undercut += 1
                print(f"{file_name}: Pixel count {count} lower than minimal number {min_size} for ID {unique_label}.")
            else:
                samples += 1
                counts_filtered.append(count)

        n_samples.append(int(samples))
        n_samples_undercut.append(int(samples_undercut))
        mean_vol.append(round(np.mean(counts_filtered), 1))
        min_vol.append(int(min(counts_filtered)))
        max_vol.append(int(max(counts_filtered)))
        dimensions.append(str(arr.shape))

    df.loc[:, "dim"] = dimensions
    df.loc[:, "n_samples[>=1000px]"] = n_samples
    df.loc[:, "n_samples[<1000px]"] = n_samples_undercut
    df.loc[:, "mean_vol[px]"] = mean_vol
    df.loc[:, "min_vol[px]"] = min_vol
    df.loc[:, "max_vol[px]"] = max_vol

    df.to_csv(table_out, sep="\t", index=False)


def check_overlapping_crops(
    center1: Tuple[float],
    center2: Tuple[float],
    size: Tuple[int],
) -> bool:
    """Check if two 3D crops overlap in space.

    Args:
        center1: Center coordinates of first crop (x, y, z)
        center2: Center coordinates of second crop (x, y, z)
        size: size of crops (dx, dy, dz)

    Returns:
        True if crops overlap, False otherwise
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


def check_all_crops_overlap(
    crop_centers: List[Tuple[float]],
    size: Tuple[int] = (128, 256, 256),
) -> bool:
    """Check if any pair of crops in a list overlaps.

    Args:
        crops : List of tuples, each the center coordinates of a crop.
        size: Size of the crop.

    Returns:
        True if any pair overlaps, False otherwise
    """
    n = len(crop_centers)
    if n == 1:
        return False
    overlap_list = []
    for i in range(n):
        for j in range(i + 1, n):
            overlap_list.append(check_overlapping_crops(crop_centers[i], crop_centers[j], size))
    return np.any(overlap_list)


def find_crop_centers_ihc(
    df: pd.DataFrame,
    component_labels: List[int],
    crop_size: Tuple[int] = (128, 256, 256),
    max_crops_per_comp: int = 10,
) -> Tuple[List[Tuple[int]], List[float]]:
    """Find crop centers for IHC segmentation.
    The function will go through each component individually.
    It will find the maximal number of equidistant crops with the given size which do not overlap.

    Args:
        df: Dataframe of segmentation table.
        component_labels: List of components.
        crop_size: Size of the ROI.
        max_crops_per_comp: Maximum number of crops per component.

    Returns:
        List of crop centers for all components.
        List of length fractions for crop centers.
    """
    n_blocks_try = [i + 1 for i in range(max_crops_per_comp)]
    n_blocks_try = sorted(n_blocks_try, reverse=True)
    total_centers = []
    total_length_fractions = []
    # iterate through components
    for label in component_labels:
        subset = df[df["component_labels"] == label]
        length_sect = list(subset["length_fraction"])
        length_sect.sort()
        # try decreasing number of blocks
        for n_blocks in n_blocks_try:
            target_s = np.linspace(length_sect[0], length_sect[-1], n_blocks * 2 + 1)
            target_s = [s for num, s in enumerate(target_s) if num % 2 == 1]
            centers = []
            fractions = []
            for target in target_s:
                idx = (subset["length_fraction"] - target).abs().idxmin()
                closest_row = subset.loc[idx]
                center_physical = [closest_row["anchor_x"], closest_row["anchor_y"], closest_row["anchor_z"]]
                centers.append(center_physical)
                fractions.append(closest_row["length_fraction"])
            centers = [[round(c) for c in center] for center in centers]
            fractions = [round(fr, 3) for fr in fractions]
            overlap = check_all_crops_overlap(centers, size=crop_size)
            # found maximal number of blocks
            if not overlap:
                print(f"Using {n_blocks} block(s) for label {label}.")
                best_centers = centers
                centers_length_fraction = fractions
                break
        total_centers.extend(best_centers)
        total_length_fractions.extend(centers_length_fraction)
    return total_centers, total_length_fractions


def export_crop_centers(
    cochlea: str,
    component_labels: List[int],
    out_dir: str,
    segmentation_channel: str = "IHC_v4b",
    halo_size: List[int] = [128, 256, 256],
    suffix: str = "crop",
    force_overwrite: str = False,
) -> str:
    """Export JSON dictionary for the creation of crops for annotation.

    Args:
        cochlea: Name of the cochlea dataset.
        component_labels: List of component labels.
        out_dir: Output directory for JSONs.
        segmentation_channel: Name of the segmentation channel.
        halo_size: Size of the halo of the ROI. ROI will be twice the size.
        suffix: Suffix for JSON dictionary.
        force_overwrite: Forcefully overwrite JSON dictionary.

    Returns:
        Output path of the JSON dictionary.
    """
    cell_type = segmentation_channel.split("_")[0]
    output_path = os.path.join(out_dir, f"{cochlea}_{suffix}_{cell_type.lower()}.json")
    if os.path.isfile(output_path) and not force_overwrite:
        print(f"JSON dictionary {output_path} already exists. Skipping creation.")
        return output_path
    else:
        print(f"Exporting crop centers for cochlea {cochlea}.")

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
    total_centers, total_length_fractions = find_crop_centers_ihc(df, component_labels, crop_size=crop_size)
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
    crop_dict["length_fraction_centers"] = total_length_fractions

    with open(output_path, "w") as f:
        json.dump([crop_dict], f, indent='\t', separators=(',', ': '))

    return output_path


def export_position_for_crop_centers(
    json_files: List[str],
    save_path: str,
):
    """Create an Excel spreadsheet or a CSV table which summarizes the location of the crops.
    """
    dict_list = []
    for json_file in json_files:
        with open(json_file, "r") as f:
            param_dicts = json.loads(f.read())
        if not isinstance(param_dicts, list):
            param_dicts = [param_dicts]
        for params in param_dicts:
            cochlea = params["dataset_name"]
            total_centers = params["crop_centers"]
            total_length_fractions = params["length_fraction_centers"]
            for center, fraction in zip(total_centers, total_length_fractions):
                center_dic = {"Cochlea": cochlea}
                center_dic["X"] = center[0]
                center_dic["Y"] = center[1]
                center_dic["Z"] = center[2]
                center_dic["length_fraction"] = fraction
                dict_list.append(center_dic)

    data = pd.DataFrame(dict_list)

    ext = os.path.splitext(save_path)[1]
    if ext == "":  # No file extension given, By default we save to CSV.
        file_path = f"{save_path}.csv"
        data.to_csv(file_path, index=False)
    elif ext == ".csv":  # Extension was specified as csv
        file_path = save_path
        data.to_csv(file_path, index=False)
    elif ext == ".xlsx":  # We also support excel.
        file_path = save_path
        data.to_excel(file_path, index=False)
    else:
        raise ValueError("Invalid extension for table: {ext}. We support .csv or .xlsx.")


def filter_segmentation_3d(
    segmentation_array: np.ndarray,
    min_pixels_per_instance: int = 100,
    min_pixels_per_component: int = 100,
) -> np.ndarray:
    """
    Filter a 3D segmentation array by removing small instances and small components within instances.

    Params:
        segmentation_array: 3D numpy array (H, W, D) with integer label IDs.
        min_pixels_per_instance: Minimum number of pixels an entire instance must have to be kept.
        min_pixels_per_component: Minimum number of pixels a component within an instance must have to be kept.

    Returns:
        filtered_array: 3D numpy array with filtered components and original label IDs.
    """
    # Step 1: Get unique label IDs (excluding background, assuming 0 is background)
    unique_labels = np.unique(segmentation_array)
    # Remove background (0) if present
    labels_to_process = unique_labels[unique_labels != 0]

    # Create output array initialized with zeros (background)
    filtered_array = np.zeros_like(segmentation_array, dtype=segmentation_array.dtype)

    # Process each label ID
    filtered_ids = 0
    filtered_components = 0
    for label_id in labels_to_process:
        # Create binary mask for current label
        mask = (segmentation_array == label_id).astype(np.uint8)

        # Count total pixels for this instance
        total_pixels = np.sum(mask)

        # Skip if below threshold
        if total_pixels < min_pixels_per_instance:
            filtered_ids += 1
            continue

        # Step 2: Find connected components within this label
        labeled_components, num_components = label_structures(mask)

        # Step 3: Check each component
        for comp_id in range(1, num_components + 1):
            comp_mask = (labeled_components == comp_id).astype(np.uint8)
            comp_pixels = np.sum(comp_mask)

            # Keep component if it meets the threshold
            if comp_pixels >= min_pixels_per_component:
                # Add this component back with original label ID
                filtered_array += comp_mask * label_id
            else:
                filtered_components += 1
    print(f"Filtered {filtered_ids} IDs and {filtered_components} components.")

    return filtered_array
