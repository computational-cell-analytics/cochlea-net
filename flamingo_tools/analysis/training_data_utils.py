import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import find_objects
from scipy.ndimage import label as label_structures
from tqdm import tqdm

from flamingo_tools.s3_utils import get_s3_path

# Columns that add_metadata_to_crop_table derives from the annotation crops.
CROP_TABLE_COLUMNS = [
    "n_samples",
    "mean_vol[px]",
    "min_vol[px]",
    "max_vol[px]",
    "dim",
    "n_samples[>=1000px]",
    "n_samples[<1000px]",
]

# Columns that add_metadata_to_crop_table_synapses derives from the crops.
SYNAPSE_CROP_TABLE_COLUMNS = ["n_samples", "dim"]


def _measure_annotation_crop(seg_file: str, min_size: int) -> Tuple[Dict[str, Any], List[Tuple[int, int]]]:
    """Count the instances of an annotation crop and measure their volumes.

    The crop is read plane by plane and counted with np.bincount, so the peak memory is one plane
    instead of the full volume and the counting is a single pass instead of a sort.

    Args:
        seg_file: File path to the annotation TIF.
        min_size: Minimal number of pixels per instance.

    Returns:
        The measures for the columns in CROP_TABLE_COLUMNS.
        The label IDs and pixel counts of all instances below min_size.
    """
    counts = np.zeros(0, dtype=np.int64)
    with tifffile.TiffFile(seg_file) as f:
        series = f.series[0]
        shape = tuple(series.shape)
        # A multi-page series holds one plane per page. Everything else is read in one go.
        planes = (page.asarray() for page in series.pages) if len(series.pages) > 1 else [series.asarray()]
        for plane in planes:
            plane_counts = np.bincount(plane.ravel())
            if plane_counts.size > counts.size:
                counts = np.pad(counts, (0, plane_counts.size - counts.size))
            counts[:plane_counts.size] += plane_counts

    # Drop the background bin and all label IDs that are not present in the crop.
    label_ids = np.arange(1, counts.size, dtype=np.int64)
    counts = counts[1:]
    present = counts > 0
    label_ids, counts = label_ids[present], counts[present]

    kept = counts[counts >= min_size]
    undersized = [(int(i), int(c)) for i, c in zip(label_ids, counts) if c < min_size]
    measures = {
        "n_samples": int(counts.size),
        "mean_vol[px]": round(float(np.mean(kept)), 1) if kept.size else 0.0,
        "min_vol[px]": int(kept.min()) if kept.size else 0,
        "max_vol[px]": int(kept.max()) if kept.size else 0,
        "dim": str(shape),
        "n_samples[>=1000px]": int(kept.size),
        "n_samples[<1000px]": len(undersized),
    }
    return measures, undersized


def _crop_row_is_complete(row: pd.Series) -> bool:
    """Check whether the measures of a table row are present and consistent.

    A row with a total instance count that matches the sum of the two size-filtered counts was
    measured with the same criteria that are applied now, so it does not have to be measured again.
    """
    if any(column not in row.index or pd.isna(row[column]) for column in CROP_TABLE_COLUMNS):
        return False
    return float(row["n_samples"]) == float(row["n_samples[>=1000px]"]) + float(row["n_samples[<1000px]"])


def _apply_updates(df: pd.DataFrame, columns: List[str], updates: Dict[Any, Dict[str, Any]]) -> pd.DataFrame:
    """Write the new measures into the table and keep the values of the rows that were skipped."""
    for column in columns:
        df[column] = [
            updates[index][column] if index in updates else df.at[index, column] for index in df.index
        ]
    return df


def add_metadata_to_crop_table(
    table_in: str,
    data_dir: str,
    table_out: Optional[str] = None,
    min_size: int = 1000,
    label_dir: str = None,
    recompute: bool = False,
    n_workers: int = 4,
):
    """Add meta information like volume and crop dimension to an existing table,
    which compiles the crops used for training and validation of a segmentation network.

    Rows with complete and consistent measures are skipped, unless recompute is set.
    Their annotation crops are not read.

    Args:
        table_in: File path to TSV table.
        data_dir: Directory featuring sub-directories with datasets, e.g. 'train' and 'val'.
        table_out: Output path for extended table.
        min_size: Minimal number of pixels per instance.
        label_dir: Directory containing annotations.
        recompute: Measure all crops again, including the rows that are already complete.
        n_workers: Number of threads for reading the annotation crops.
    """
    if table_out is None:
        table_out = table_in

    df = pd.read_csv(table_in, sep="\t")
    for column in CROP_TABLE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    row_ids = [index for index in df.index if recompute or not _crop_row_is_complete(df.loc[index])]

    def measure_row(index):
        file_name = df.at[index, "Original"]
        if label_dir is None:
            seg_file = os.path.join(data_dir, df.at[index, "Dataset"], f"{file_name}_annotations.tif")
        else:
            seg_file = os.path.join(label_dir, f"{file_name}_annotations.tif")
        measures, undersized = _measure_annotation_crop(seg_file, min_size)
        for label_id, count in undersized:
            tqdm.write(f"{file_name}: Pixel count {count} lower than minimal number {min_size} for ID {label_id}.")
        return index, measures

    table_name = os.path.basename(table_in)
    print(f"{table_name}: measuring {len(row_ids)} of {len(df)} crops.")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(pool.map(measure_row, row_ids), total=len(row_ids), desc=f"Measure crops of {table_name}"))

    df = _apply_updates(df, CROP_TABLE_COLUMNS, dict(results))
    df_reordered = df.loc[:, ["Original", "Standardized", "Dataset", "Crop_center"] + CROP_TABLE_COLUMNS]
    df_reordered.to_csv(table_out, sep="\t", index=False)


def add_metadata_to_crop_table_synapses(
    table_in: str,
    train_dir: str,
    test_dir: Optional[str] = None,
    input_key: str = "raw",
    table_out: Optional[str] = None,
    recompute: bool = False,
):
    """Add meta information like the number of instances and crop dimensions to an existing table,
    which compiles the crops used for training, validation and testing of the network for synapse detection.

    Rows with complete measures are skipped, unless recompute is set.

    Args:
        table_in: File path to TSV table.
        train_dir: Directory used for training. Features sub-directories 'images' and 'labels'.
        test_dir: Directory used for testing. Features sub-directories 'images' and 'labels'.
        input_key: Input key for ZARR image file. Used to determine crop dimensions.
        table_out: Output path for extended table.
        recompute: Measure all crops again, including the rows that are already complete.
    """
    if table_out is None:
        table_out = table_in

    df = pd.read_csv(table_in, sep="\t")
    for column in SYNAPSE_CROP_TABLE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan

    row_ids = [
        index for index in df.index
        if recompute or any(pd.isna(df.at[index, column]) for column in SYNAPSE_CROP_TABLE_COLUMNS)
    ]

    table_name = os.path.basename(table_in)
    print(f"{table_name}: measuring {len(row_ids)} of {len(df)} crops.")
    updates = {}
    for index in tqdm(row_ids, desc=f"Measure crops of {table_name}"):
        file_name = df.at[index, "Original"]
        dataset = df.at[index, "Dataset"]
        if dataset in ["train", "val"]:
            data_dir = train_dir
        else:
            if test_dir is None:
                raise ValueError(f"Supply a test directory for {file_name}.")
            data_dir = test_dir

        image_path = os.path.join(data_dir, "images", f"{file_name}.zarr")
        image = zarr.open(image_path)[input_key]

        label_path = os.path.join(data_dir, "labels", f"{file_name}.csv")
        label_df = pd.read_csv(label_path, sep="\t")
        updates[index] = {"n_samples": len(label_df), "dim": str(tuple(image.shape))}

    df = _apply_updates(df, SYNAPSE_CROP_TABLE_COLUMNS, updates)
    df_reordered = df.loc[:, ["Original", "Standardized", "Dataset", "Crop_center"] + SYNAPSE_CROP_TABLE_COLUMNS]
    df_reordered.to_csv(table_out, sep="\t", index=False)


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


def create_2d_training_data(
    input_dir,
    output_dir,
    skip_empty=False,
    empty_blocks=0,
):
    """Create 2D training data based on 3D image crops.

    Args:
        input_dir: Directory containing data in TIF format.
        output_dir: Output directory for 2D slice data.
        skip_empty: Skip empty 3D blocks and 2D slices.
        empty_blocks: Create 2D data for the first n empty 3D blocks.
    """
    os.makedirs(output_dir, exist_ok=True)
    image_paths = [entry.path for entry in os.scandir(input_dir) if
                   ".tif" in entry.name and
                   "annotations" not in entry.name]
    label_paths = [entry.path for entry in os.scandir(input_dir) if
                   ".tif" in entry.name and
                   "annotations" in entry.name]
    image_paths.sort()
    label_paths.sort()

    def create_2d_data_from_3d(img, seg, output_dir, file_name, skip_empty_slices=False):
        z_dim = img.shape[0]
        removed_slices = 0
        for i in range(z_dim):
            zstr = str(i).zfill(3)
            seg_slice = seg[i, :, :]
            suffix = ""
            if len(np.unique(seg_slice)) == 1:
                if skip_empty_slices:
                    removed_slices += 1
                    continue
                elif "empty" not in file_name:
                    print(file_name)
                    suffix = "_empty"
            seg_out_path = os.path.join(output_dir, f"{file_name}_z{zstr}{suffix}_annotations.tif")
            imageio.imwrite(seg_out_path, seg_slice)

            img_out_path = os.path.join(output_dir, f"{file_name}_z{zstr}{suffix}.tif")
            img_slice = img[i, :, :]
            imageio.imwrite(img_out_path, img_slice)
        return removed_slices

    removed_data = 0
    removed_slices = 0
    shapes = []
    # iterate through all 3D datasets
    for image_path, label_path in tqdm(
        zip(image_paths, label_paths), total=len(image_paths), desc="Creating 2D data from training files",
    ):
        file_name = os.path.basename(image_path).split(".")[0]
        seg = imageio.imread(label_path)
        if seg.shape not in shapes:
            shapes.append(seg.shape)
        if len(np.unique(seg)) == 1:
            if skip_empty:
                if empty_blocks == 0:
                    removed_data += 1
                    continue
                else:
                    img = imageio.imread(image_path)
                    removed_slices += create_2d_data_from_3d(
                        img, seg, output_dir=output_dir, file_name=file_name, skip_empty_slices=False,
                    )
                    empty_blocks -= 1
                    continue

        img = imageio.imread(image_path)
        removed_slices += create_2d_data_from_3d(
            img, seg, output_dir=output_dir, file_name=file_name, skip_empty_slices=skip_empty,
        )

    print(f"Data shapes of crops: {shapes}")
    print(f"Removed crops: {removed_data}")
    print(f"Removed slices: {removed_slices}")


def filter_segmentation_3d(
    segmentation_array: np.ndarray,
    min_pixels_per_instance: int = 100,
    min_pixels_per_component: int = 100,
    filter_split_components: bool = True,
) -> np.ndarray:
    """
    Filter a 3D segmentation array by removing small instances and, optionally, small
    disconnected sub-components within an instance (e.g. annotation artifacts that ended up
    sharing a label with a larger, unrelated instance).

    Per-label work (pixel counting, connected-component analysis) is restricted to each
    label's bounding box via `scipy.ndimage.find_objects`, instead of scanning the full array
    once per label. Cost scales with the total instance volume rather than
    n_labels * array size, which matters once there are many instances.

    Note: `find_objects` allocates memory proportional to the *maximum label value*, not
    the number of labels present. Crops can carry large global instance IDs (from the
    whole-cochlea segmentation table) even though only a few instances appear locally, so
    labels are remapped to a small contiguous range before the bounding-box lookup to avoid
    a MemoryError.

    Params:
        segmentation_array: 3D numpy array (H, W, D) with integer label IDs.
        min_pixels_per_instance: Minimum number of pixels an entire instance must have to be kept.
        min_pixels_per_component: Minimum number of pixels a component within an instance must have
            to be kept. Only used if filter_split_components is True.
        filter_split_components: Whether to also remove small disconnected sub-components within
            an instance's label. Set to False to only filter by instance size.

    Returns:
        filtered_array: 3D numpy array with filtered components and original label IDs.
    """
    labels = np.unique(segmentation_array)
    labels = labels[labels != 0]

    filtered_array = np.zeros_like(segmentation_array)
    if len(labels) == 0:
        return filtered_array

    # Remap to a compact 1..len(labels) range so find_objects' internal allocation
    # scales with the number of instances, not the magnitude of their original IDs.
    compact_array = (np.searchsorted(labels, segmentation_array) + 1).astype(np.int32)
    compact_array[segmentation_array == 0] = 0

    objects = find_objects(compact_array)
    n_filtered_instances = 0
    n_filtered_components = 0

    for compact_id, label_id in enumerate(labels, start=1):
        bbox = objects[compact_id - 1]
        mask = segmentation_array[bbox] == label_id
        total_pixels = int(mask.sum())

        if total_pixels < min_pixels_per_instance:
            n_filtered_instances += 1
            continue

        if not filter_split_components:
            filtered_array[bbox][mask] = label_id
            continue

        labeled_components, num_components = label_structures(mask)
        component_counts = np.bincount(labeled_components.ravel())
        for comp_id in range(1, num_components + 1):
            if component_counts[comp_id] >= min_pixels_per_component:
                filtered_array[bbox][labeled_components == comp_id] = label_id
            else:
                n_filtered_components += 1

    print(f"Filtered {n_filtered_instances} instance(s) and {n_filtered_components} split-off component(s).")
    return filtered_array


def filter_annotations(
    input_dir: str,
    output_dir: str,
    force_overwrite: bool = False,
    min_pixels_per_instance: int = 100,
    min_pixels_per_component: int = 100,
    filter_split_components: bool = True,
):
    """Filter manual annotations by removing small instances and, optionally, small
    disconnected sub-components within an instance.

    Args:
        input_dir: Directory containing annotations in TIF format.
        output_dir: Output directory for filtered annotations.
        force_overwrite: Forcefully overwrite existing output files.
        min_pixels_per_instance: Minimum number of pixels an instance must have to be kept.
        min_pixels_per_component: Minimum number of pixels a sub-component within an instance
            must have to be kept. Only used if filter_split_components is True.
        filter_split_components: Whether to also remove small disconnected sub-components
            within an instance's label (relevant for IHCs). Set to False to only filter by
            instance size (sufficient for SGNs).
    """
    os.makedirs(output_dir, exist_ok=True)
    annotation_paths = [entry.path for entry in os.scandir(input_dir) if "annotation" in entry.name]
    annotation_names = [os.path.basename(p) for p in annotation_paths]
    annotation_paths.sort()
    annotation_names.sort()
    for annotation_path, annotation_name in zip(annotation_paths, annotation_names):
        out_path = os.path.join(output_dir, annotation_name)
        if os.path.isfile(out_path) and not force_overwrite:
            print(f"Skipping {annotation_name}, output already exists.")
            continue
        print(annotation_name)
        arr = imageio.imread(annotation_path)
        filtered_array = filter_segmentation_3d(
            arr,
            min_pixels_per_instance=min_pixels_per_instance,
            min_pixels_per_component=min_pixels_per_component,
            filter_split_components=filter_split_components,
        )

        imageio.imwrite(out_path, filtered_array, compression="zlib")
