import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.ndimage import find_objects, gaussian_filter, gaussian_filter1d, maximum_position
from scipy.ndimage import label as label_structures
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from skimage.morphology import h_maxima
from tqdm import tqdm

from flamingo_tools.postprocessing.cochlea_mapping import arc_length, moving_average_3d, path_tangents, resample_path
from flamingo_tools.s3_utils import default_table_path, read_table

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

# Upper bound for a np.bincount allocation, which holds one bin per label value.
MAX_BINS = 2 ** 20

# The columns of the IHC tip tables of find_ihc_tips_height and find_ihc_tips_path.
TIP_COLUMNS = ["side", "z", "y", "x", "label_id", "cell"]
# Range of the distance between an IHC tip and the central line, relative to the reference IHC length.
# The annotated IHCs of the IHC_v11 training crops have 0.26-0.82 (1st-99th percentile). The upper limit has a
# margin, because the path of a cochlea is not always at the cell centers: an upper limit of 0.85 misses the
# tips in the MWfLS crop M-AMD-000126-R_crop_0448-1443-0611.
TIP_DISTANCE = (0.25, 1.0)


def _count_labels_in_plane(plane: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Count the pixels per label ID in one plane.

    np.bincount allocates one bin per label value instead of one per label, so a single stray label ID
    would allocate an array of that size. It is only used where the label values are bounded.

    Args:
        plane: Array with label IDs.

    Returns:
        The label IDs present in the plane.
        The pixel count per label ID.
    """
    # For a small integer type the dtype alone bounds the allocation to 65536 bins.
    # For a wider type the maximum decides, which is much cheaper to determine than sorting.
    use_bincount = plane.dtype.kind in "ui" and (
        plane.dtype.itemsize <= 2 or (plane.size != 0 and int(plane.max()) < MAX_BINS)
    )
    if not use_bincount:
        return np.unique(plane, return_counts=True)

    counts = np.bincount(plane.ravel())
    label_ids = np.flatnonzero(counts)
    # Return the label IDs in the dtype of the plane, so that the planes of one crop stay comparable
    # even if they take different branches.
    return label_ids.astype(plane.dtype, copy=False), counts[label_ids]


def _measure_annotation_crop(seg_file: str, min_size: int) -> Tuple[Dict[str, Any], List[Tuple[int, int]]]:
    """Count the instances of an annotation crop and measure their volumes.

    The crop is read and counted plane by plane, so the peak memory is bound by one plane and by the
    number of labels, not by the size of the volume or by the largest label ID.

    Args:
        seg_file: File path to the annotation TIF.
        min_size: Minimal number of pixels per instance.

    Returns:
        The measures for the columns in CROP_TABLE_COLUMNS.
        The label IDs and pixel counts of all instances below min_size.
    """
    id_chunks, count_chunks = [], []
    with tifffile.TiffFile(seg_file) as f:
        series = f.series[0]
        shape = tuple(series.shape)
        # A multi-page series holds one plane per page. Everything else is read in one go.
        planes = (page.asarray() for page in series.pages) if len(series.pages) > 1 else [series.asarray()]
        for plane in planes:
            plane_ids, plane_counts = _count_labels_in_plane(plane)
            id_chunks.append(plane_ids)
            count_chunks.append(plane_counts)

    # Sum the pixel counts of each label over all planes. The bins of this np.bincount are the
    # positions in the compact list of label IDs, so its size is the number of labels.
    label_ids, inverse = np.unique(np.concatenate(id_chunks), return_inverse=True)
    counts = np.bincount(inverse, weights=np.concatenate(count_chunks)).astype(np.int64)

    foreground = label_ids != 0
    label_ids, counts = label_ids[foreground], counts[foreground]

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

    Returns:
        The names of the crops that could not be measured.
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
        # A crop that cannot be measured must not discard the measures of the other crops.
        try:
            measures, undersized = _measure_annotation_crop(seg_file, min_size)
        except Exception as error:
            tqdm.write(f"{file_name}: Measuring {seg_file} failed with {type(error).__name__}: {error}")
            return index, None
        for label_id, count in undersized:
            tqdm.write(f"{file_name}: Pixel count {count} lower than minimal number {min_size} for ID {label_id}.")
        return index, measures

    table_name = os.path.basename(table_in)
    print(f"{table_name}: measuring {len(row_ids)} of {len(df)} crops.")
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(tqdm(pool.map(measure_row, row_ids), total=len(row_ids), desc=f"Measure crops of {table_name}"))

    failed = [str(df.at[index, "Original"]) for index, measures in results if measures is None]
    df = _apply_updates(df, CROP_TABLE_COLUMNS, {i: measures for i, measures in results if measures is not None})
    df_reordered = df.loc[:, ["Original", "Standardized", "Dataset", "Crop_center"] + CROP_TABLE_COLUMNS]
    df_reordered.to_csv(table_out, sep="\t", index=False)
    return failed


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

    Returns:
        The names of the crops that could not be measured.
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
    updates, failed = {}, []
    for index in tqdm(row_ids, desc=f"Measure crops of {table_name}"):
        file_name = df.at[index, "Original"]
        dataset = df.at[index, "Dataset"]
        data_dir = train_dir if dataset in ["train", "val"] else test_dir
        # A crop that cannot be measured must not discard the measures of the other crops.
        if data_dir is None:
            tqdm.write(f"{file_name}: Supply a test directory for the dataset '{dataset}'.")
            failed.append(str(file_name))
            continue
        try:
            image = zarr.open(os.path.join(data_dir, "images", f"{file_name}.zarr"))[input_key]
            label_df = pd.read_csv(os.path.join(data_dir, "labels", f"{file_name}.csv"), sep="\t")
        except Exception as error:
            tqdm.write(f"{file_name}: Measuring the crop failed with {type(error).__name__}: {error}")
            failed.append(str(file_name))
            continue
        updates[index] = {"n_samples": len(label_df), "dim": str(tuple(image.shape))}

    df = _apply_updates(df, SYNAPSE_CROP_TABLE_COLUMNS, updates)
    df_reordered = df.loc[:, ["Original", "Standardized", "Dataset", "Crop_center"] + SYNAPSE_CROP_TABLE_COLUMNS]
    df_reordered.to_csv(table_out, sep="\t", index=False)
    return failed


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


def find_crop_centers_missed(
    reference: pd.DataFrame,
    table: pd.DataFrame,
    n_crops: int,
    crop_size: Tuple[float, float, float],
    match_distance: float = 5.0,
    run_distance: float = 20.0,
) -> Tuple[List[List[int]], List[int]]:
    """Find crop centers at the largest runs of reference objects that a segmentation misses.

    A reference object is missed if no object of the table has an anchor within `match_distance`.
    Missed objects within `run_distance` of each other form one run. A run of neighbouring misses
    breaks the connected components of the segmentation, so the largest runs are the most useful crops.

    Args:
        reference: Table of the reference segmentation, for example the curated component of an older model.
        table: Table of the segmentation to check.
        n_crops: Maximal number of crops.
        crop_size: Size of the crops in micrometer, in (x, y, z) order.
        match_distance: Maximal anchor distance in micrometer for a matched object.
        run_distance: Maximal distance in micrometer between the missed objects of one run.

    Returns:
        The crop centers in micrometer, in (x, y, z) order.
        The number of missed objects in the run of each crop.
    """
    columns = ["anchor_x", "anchor_y", "anchor_z"]
    reference_coords = reference[columns].values
    distances, _ = cKDTree(table[columns].values).query(reference_coords)
    missed = reference_coords[distances >= match_distance]
    if len(missed) == 0:
        return [], []

    tree = cKDTree(missed)
    _, run_ids = connected_components(tree.sparse_distance_matrix(tree, run_distance), directed=False)
    run_sizes = np.bincount(run_ids)

    centers, n_missed = [], []
    for run_id in np.argsort(-run_sizes, kind="stable"):
        if len(centers) == n_crops:
            break
        run = missed[run_ids == run_id]
        # The missed object closest to the mean of the run keeps the center on the cell row.
        center = run[np.argmin(np.linalg.norm(run - run.mean(axis=0), axis=1))]
        center = [int(round(c)) for c in center]
        if any(check_overlapping_crops(center, other, crop_size) for other in centers):
            continue
        centers.append(center)
        n_missed.append(int(run_sizes[run_id]))
    return centers, n_missed


def _orientation(axes: np.ndarray) -> np.ndarray:
    """Return the mean direction of unit axes with an arbitrary sign: the top eigenvector of their tensor."""
    return np.linalg.eigh(axes.T @ axes)[1][:, -1]


def _axis_angles(axes: np.ndarray, n_side: int = 3) -> np.ndarray:
    """Angle in degree between each axis and the mean axis of its n_side neighbours on each side."""
    angles = []
    for i, axis in enumerate(axes):
        neighbours = np.concatenate([axes[max(i - n_side, 0):i], axes[i + 1:i + 1 + n_side]])
        angles.append(np.degrees(np.arccos(min(1.0, abs(axis @ _orientation(neighbours))))))
    return np.array(angles)


def _rolling_median(values: pd.Series) -> pd.Series:
    """Median of the 21 nearest values in the order of the series, the value itself included."""
    return values.rolling(21, center=True, min_periods=1).median()


def ihc_row_quality(
    table: pd.DataFrame,
    component_list: List[int],
    ellipsoids: Optional[pd.DataFrame] = None,
    window: float = 100.0,
    size_range: Tuple[float, float] = (0.58, 1.52),
    axis_range: Tuple[float, float] = (0.70, 1.32),
    min_elongation: float = 1.2,
    max_axis_angle: float = 30.0,
    max_path_distance: float = 15.0,
    min_size: int = 1000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate the segmentation errors of an IHC row from the segmentation table.

    Each row instance is compared with its neighbours along the central path (`ihc_row_path`), because
    the size of the IHCs follows the tonotopy. The default ranges are the 1-99 percentiles of the annotated
    IHCs in the IHC_v11 training crops (`scripts/training/2026-10_ihc_table_quality.py`). The classes are,
    in this order:
    - 'off_row': the center is more than `max_path_distance` from the path.
    - 'merged': the size is above `size_range`. It counts round(size) - 1 errors.
    - 'fragment': the size is below `size_range`.
    - 'shape': the main axis is shorter or longer than `axis_range`, the cell is rounder than `min_elongation`,
      or its main axis is not parallel to the main axis of its neighbours. This needs the ellipsoids.
    - 'extra': an instance outside the row, with at least `min_size` pixels and within `max_path_distance`
      of the path. It is an artifact, a fragment or a cell that is missing from the row.
    A gap between two row instances counts the cells that fit in it as missed. The cell spacing is the
    median gap between normal neighbours. Cells that are missing at the ends of the row are not found.

    Args:
        table: Segmentation table with a 'component_labels' column.
        component_list: The components that form the IHC row.
        ellipsoids: Optional ellipsoid measures of all row instances, from the feature set 'ellipsoid' of
            `compute_object_measures`. Without them, no instance gets the class 'shape'.
        window: Length of the windows along the path in micrometers.
        size_range: Range of a normal size, relative to the median of the 21 nearest row instances along the path.
        axis_range: Range of a normal main-axis length, relative to the median of the 21 nearest row instances.
        min_elongation: Minimal ratio between the two largest semi-axes.
        max_axis_angle: Maximal angle in degree between the main axis and the mean main axis of the
            3 row neighbours on each side.
        max_path_distance: Maximal distance in micrometers between the center of a row instance and the path.
        min_size: Minimal number of pixels of an extra instance.

    Returns:
        One row per row instance and per extra instance, with 'label_id', 'n_pixels', 's' (position along
        the path in micrometers), 'distance' (to the path in micrometers), 'n_rel' (relative size), 'n_cells',
        'class' and 'missed' (the missed cells in the gap to the next row instance along the path).
        With ellipsoids also 'a_rel', 'elongation' and 'axis_angle'.
        One row per window along the path, with 'start' and 'stop' (in micrometers along the path), 'x', 'y' and
        'z' (the path point at the center), 'length_fraction' (median of the row instances, if the table has
        the column), 'n_row', 'expected' (cells for the median spacing), the error counts 'missed', 'merged',
        'fragment', 'shape', 'off_row' and 'extra', and their sum 'errors'.

    Raises:
        ValueError: If an instance of the row has no ellipsoid, or if no two neighbouring row instances are
            normal.
    """
    from flamingo_tools.postprocessing.label_components import ihc_row_path

    columns = ["anchor_x", "anchor_y", "anchor_z"]
    in_row = table["component_labels"].isin(component_list).values
    path = ihc_row_path(table.loc[in_row, columns].values)
    arc = arc_length(path)
    distances, index = cKDTree(path).query(table[columns].values)
    keep = ["label_id", "n_pixels"] + (["length_fraction"] if "length_fraction" in table else [])
    objects = table[keep].assign(s=arc[index], distance=distances)

    row = objects[in_row].sort_values("s", kind="stable").reset_index(drop=True)
    row["n_rel"] = row["n_pixels"] / _rolling_median(row["n_pixels"])
    row["n_cells"] = np.maximum(1, np.floor(row["n_rel"] + 0.5)).astype(int)
    shape = np.zeros(len(row), dtype=bool)
    if ellipsoids is not None:
        row = row.merge(ellipsoids[["label_id", "a", "b", "axis_x", "axis_y", "axis_z"]], on="label_id", how="left")
        if row["a"].isna().any():
            raise ValueError(f"{row['a'].isna().sum()} row instances have no ellipsoid.")
        row["a_rel"] = row["a"] / _rolling_median(row["a"])
        row["elongation"] = row["a"] / row["b"]
        row["axis_angle"] = _axis_angles(row[["axis_x", "axis_y", "axis_z"]].values)
        row = row.drop(columns=["a", "b", "axis_x", "axis_y", "axis_z"])
        shape = ((row["a_rel"] < axis_range[0]) | (row["a_rel"] > axis_range[1]) |
                 (row["elongation"] < min_elongation) | (row["axis_angle"] > max_axis_angle)).values
    row["class"] = np.select(
        [row["distance"] > max_path_distance, row["n_rel"] > size_range[1], row["n_rel"] < size_range[0], shape],
        ["off_row", "merged", "fragment", "shape"], "normal",
    )

    on_path = row[row["class"] != "off_row"]
    s, n_cells, normal = on_path["s"].values, on_path["n_cells"].values, (on_path["class"] == "normal").values
    gaps = np.diff(s)
    normal_pairs = normal[1:] & normal[:-1]
    if not normal_pairs.any():
        raise ValueError("The row has no two neighbouring normal instances, so the cell spacing is not defined.")
    spacing = np.median(gaps[normal_pairs])
    missed = np.maximum(0, np.floor(gaps / spacing - (n_cells[1:] + n_cells[:-1]) / 2 + 0.5))
    row["missed"] = 0
    row.loc[on_path.index[:-1], "missed"] = missed.astype(int)

    extra = objects[~in_row & (objects["distance"] <= max_path_distance) & (objects["n_pixels"] >= min_size)]
    objects = pd.concat([row, extra.assign(**{"class": "extra"})], ignore_index=True)

    edges = np.arange(0.0, arc[-1] + window, window)
    centers = path[np.minimum(np.searchsorted(arc, (edges[:-1] + edges[1:]) / 2), len(path) - 1)]
    windows = pd.DataFrame({
        "start": edges[:-1], "stop": edges[1:], "x": centers[:, 0], "y": centers[:, 1], "z": centers[:, 2],
    })
    if "length_fraction" in row:
        windows["length_fraction"] = row.groupby(np.digitize(row["s"], edges) - 1)["length_fraction"].median()
    windows["n_row"] = np.histogram(row["s"], edges)[0]
    windows["expected"] = np.diff(np.minimum(edges, arc[-1])) / spacing
    windows["missed"] = np.histogram((s[1:] + s[:-1]) / 2, edges, weights=missed)[0]
    extra_cells = np.where(row["class"] == "merged", row["n_cells"] - 1, 0)
    windows["merged"] = np.histogram(row["s"], edges, weights=extra_cells)[0]
    for name in ["fragment", "shape", "off_row"]:
        windows[name] = np.histogram(row["s"], edges, weights=(row["class"] == name).astype(float))[0]
    windows["extra"] = np.histogram(extra["s"], edges)[0]
    error_columns = ["missed", "merged", "fragment", "shape", "off_row", "extra"]
    windows[error_columns] = windows[error_columns].astype(int)
    windows["errors"] = windows[error_columns].sum(axis=1)
    return objects, windows


def find_crop_centers_quality(
    windows: pd.DataFrame,
    n_crops: int,
    crop_size: Tuple[float, float, float],
    exclude: List[List[float]] = (),
) -> Tuple[List[List[int]], List[int]]:
    """Find crop centers at the windows of `ihc_row_quality` with the most errors.

    Args:
        windows: The windows of `ihc_row_quality`.
        n_crops: Maximal number of crops.
        crop_size: Size of the crops in micrometer, in (x, y, z) order.
        exclude: Centers of existing crops in micrometer, in (x, y, z) order. A new crop does not overlap them.

    Returns:
        The crop centers in micrometer, in (x, y, z) order.
        The number of errors in the window of each crop.
    """
    centers, n_errors, taken = [], [], list(exclude)
    for _, window in windows.sort_values("errors", ascending=False, kind="stable").iterrows():
        if len(centers) == n_crops or window["errors"] == 0:
            break
        center = [int(round(window[axis])) for axis in "xyz"]
        if any(check_overlapping_crops(center, other, crop_size) for other in taken):
            continue
        centers.append(center)
        taken.append(center)
        n_errors.append(int(window["errors"]))
    return centers, n_errors


def export_crop_centers(
    cochlea: str,
    component_labels: List[int],
    out_dir: str,
    segmentation_channel: str = "IHC_v4b",
    halo_size: List[int] = [128, 256, 256],
    suffix: str = "crop",
    force_overwrite: str = False,
    voxel_size: float = 0.38,
) -> str:
    """Export JSON dictionary for the creation of crops for annotation.

    Args:
        cochlea: Name of the cochlea dataset.
        component_labels: List of component labels.
        out_dir: Output directory for JSONs.
        segmentation_channel: Name of the segmentation channel.
        halo_size: Size of the halo of the ROI in pixels, in (x, y, z) order. ROI will be twice the size.
        suffix: Suffix for JSON dictionary.
        force_overwrite: Forcefully overwrite JSON dictionary.
        voxel_size: The voxel size of the data in micrometer.

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
    seg_table_s3 = default_table_path(cochlea, segmentation_channel, s3=True)
    df = read_table(seg_table_s3, s3=True)

    # The crop centers are in micrometer, so the overlap check needs the crop size in micrometer.
    crop_size = [i * 2 * voxel_size for i in halo_size]
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


def _densify_path(path: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline at about 1 pixel steps. Returns the points and their unit tangents."""
    points = resample_path(path, 1.0)
    return points, path_tangents(points, 2.0, 1.0)


def _instance_features(
    segmentation: np.ndarray, path: np.ndarray, voxel_size: float,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, np.ndarray], np.ndarray]:
    """Measure the size, path distance, path tangent and principal axis of each instance.

    Returns the features and voxels of each instance, and the path resampled at about 1 pixel.
    """
    points, tangents = _densify_path(path)
    tree = cKDTree(points)
    features, coords = {}, {}
    for label_id, bbox in enumerate(find_objects(segmentation), start=1):
        if bbox is None:
            continue
        voxels = np.argwhere(segmentation[bbox] == label_id) + np.array([sl.start for sl in bbox])
        distance, index = tree.query(voxels.mean(axis=0))
        # The principal axis of an instance with less than 4 voxels is not defined.
        axis = np.linalg.eigh(np.cov(voxels.T))[1][:, -1] if len(voxels) > 3 else np.array([1.0, 0.0, 0.0])
        features[label_id] = {
            "n_pixels": len(voxels),
            "distance": distance * voxel_size,
            "tangent": tangents[index],
            "index": index,
            "axis": axis,
            "border": any(sl.start == 0 or sl.stop == size for sl, size in zip(bbox, segmentation.shape)),
        }
        coords[label_id] = voxels
    return features, coords, points


def _extent(voxels: np.ndarray, direction: np.ndarray, voxel_size: float) -> float:
    """Return the 2-98 percentile span of the voxels along a direction, in micrometer."""
    projection = voxels @ direction
    return float(np.percentile(projection, 98) - np.percentile(projection, 2)) * voxel_size


def _path_direction(tangent: np.ndarray, axis: np.ndarray) -> np.ndarray:
    direction = tangent - (tangent @ axis) * axis
    return direction / np.linalg.norm(direction)


def _round(value: float) -> int:
    return int(np.floor(value + 0.5))


def _at_border(voxels: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    return ((voxels == 0) | (voxels == np.array(shape) - 1)).any(axis=1)


def _extreme_voxels(key: np.ndarray, value: np.ndarray) -> np.ndarray:
    """Return the index of the voxel with the largest value for each distinct key."""
    order = np.lexsort((value, key))
    last = np.append(key[order][1:] != key[order][:-1], True)
    return order[last]


def _spaced(positions: np.ndarray, values: np.ndarray, min_distance: float) -> np.ndarray:
    """Keep positions in the order of decreasing value, without a kept position closer than min_distance."""
    keep = []
    for i in np.argsort(-values):
        if all(np.linalg.norm(positions[i] - positions[j]) >= min_distance for j in keep):
            keep.append(i)
    return np.array(keep, dtype=int)


def _tip_table(
    voxels: np.ndarray,
    labels: np.ndarray,
    tips: Dict[str, np.ndarray],
    positions: Dict[str, np.ndarray],
    max_pair_offset: float,
) -> pd.DataFrame:
    """Pair the tips of the two sides by the smallest total offset. A pair or an unpaired tip is one cell."""
    top, bottom = positions["top"], positions["bottom"]
    pairs = []
    if len(top) and len(bottom):
        cost = np.linalg.norm(top[:, None] - bottom[None], axis=-1)
        cost[cost > max_pair_offset] = 1e9
        pairs = [(i, j) for i, j in zip(*linear_sum_assignment(cost)) if cost[i, j] < 1e9]
    pairs += [(i, None) for i in range(len(top)) if i not in {i for i, _ in pairs}]
    pairs += [(None, j) for j in range(len(bottom)) if j not in {j for _, j in pairs}]
    rows = []
    for cell, pair in enumerate(pairs):
        for side, i in zip(("top", "bottom"), pair):
            if i is not None:
                voxel = tips[side][i]
                rows.append(dict(zip(TIP_COLUMNS, [side, *voxels[voxel].tolist(), labels[voxel], cell])))
    return pd.DataFrame(rows, columns=TIP_COLUMNS)


def find_ihc_tips_height(
    voxels: np.ndarray,
    labels: np.ndarray,
    axis: np.ndarray,
    shape: Tuple[int, ...],
    voxel_size: float = 0.38,
    min_prominence: float = 1.0,
    min_spacing: float = 4.5,
    max_pair_offset: float = 12.0,
) -> pd.DataFrame:
    """Find the IHC tips as peaks of two height maps along one long axis.

    Each IHC has a tip at both ends of its long axis. The voxels are projected onto the plane perpendicular
    to the axis. The height map of a side holds the extreme axis coordinate of each column. The tips are its
    maxima with a prominence of at least `min_prominence`. The tips of the two sides are paired by their
    distance in the plane.

    Args:
        voxels: Voxel coordinates in pixels, in (z, y, x) order, for example of all IHC row instances.
        labels: The label of each voxel.
        axis: The unit long axis in (z, y, x) order.
        shape: The volume shape. A tip on the volume border is a cut face and is dropped.
        voxel_size: Isotropic voxel size in micrometer.
        min_prominence: Minimal height of a tip over the saddle to a higher tip, in micrometer.
        min_spacing: Minimal distance between two tips of one side, in micrometer.
        max_pair_offset: Maximal distance between the two tips of one cell in the plane, in micrometer.

    Returns:
        One row per tip: 'side' ('top' or 'bottom'), the tip voxel 'z', 'y', 'x', its 'label_id', and 'cell'.
        The tips of one cell share the cell id.
    """
    plane = np.linalg.svd(axis[None])[2][1:]
    columns = np.round(voxels @ plane.T).astype(int)
    columns -= columns.min(axis=0)
    grid = tuple(columns.max(axis=0) + 1)
    key = np.ravel_multi_index(tuple(columns.T), grid)
    height = voxels @ axis
    tips, positions = {}, {}
    for side, sign in (("top", 1.0), ("bottom", -1.0)):
        value = sign * height
        extreme = _extreme_voxels(key, value)
        image = np.full(grid, value.min() - 1.0)
        image.flat[key[extreme]] = value[extreme]
        voxel_of_column = np.full(grid, -1)
        voxel_of_column.flat[key[extreme]] = extreme
        image = gaussian_filter(image, 1.0)
        maxima, n_maxima = label_structures(h_maxima(image, min_prominence / voxel_size))
        peaks = np.array(maximum_position(image, maxima, range(1, n_maxima + 1)), dtype=int).reshape(-1, 2)
        candidates = voxel_of_column[tuple(peaks.T)]
        valid = candidates >= 0
        valid[valid] = ~_at_border(voxels[candidates[valid]], shape)
        peaks, candidates = peaks[valid], candidates[valid]
        keep = _spaced(peaks, value[candidates], min_spacing / voxel_size)
        tips[side], positions[side] = candidates[keep], peaks[keep]
    return _tip_table(voxels, labels, tips, positions, max_pair_offset / voxel_size)


def foot_points(
    voxels: np.ndarray, points: np.ndarray, axes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move each voxel along the local long axis onto the central line.

    Args:
        voxels: Voxel coordinates in pixels, in (z, y, x) order.
        points: The central path in pixels, in (z, y, x) order, resampled at 1 pixel.
        axes: The unit long axis at each path point, in (z, y, x) order.

    Returns:
        The path index of the foot point of each voxel, the offset of the voxel from its foot point, and its
        height, the signed distance from the central line along the axis, in pixels.
    """
    tree = cKDTree(points)
    index = tree.query(voxels)[1]
    height = np.sum((voxels - points[index]) * axes[index], axis=1)
    index = tree.query(voxels - height[:, None] * axes[index])[1]
    offset = voxels - points[index]
    return index, offset, np.sum(offset * axes[index], axis=1)


def find_ihc_tips_path(
    voxels: np.ndarray,
    labels: np.ndarray,
    points: np.ndarray,
    axes: np.ndarray,
    shape: Tuple[int, ...],
    voxel_size: float = 0.38,
    min_prominence: float = 1.0,
    min_spacing: float = 4.5,
    max_pair_offset: float = 4.5,
    max_tip_angle: float = 45.0,
    tip_distance: Tuple[float, float] = (0.0, np.inf),
) -> pd.DataFrame:
    """Find the IHC tips as protrusions along the central path.

    The height of a voxel is its signed distance from the central line along the local long axis (the axis at
    its nearest path point). The voxel is moved along the axis by its height to its foot point, which falls
    on the central line. So the two tips of a cell have the same foot point, also if the cell leans along the
    row. The profile of a side holds the extreme height at each foot point. The tips are the peaks of the
    profile with a prominence of at least `min_prominence` over the labeled pixels next to them along the row.
    A tip must point along the local axis:
    - the angle between the axis and the line from its foot point to the tip is at most `max_tip_angle`;
    - the height of the tip is in `tip_distance`.
    The tips of the two sides are paired by the distance of their foot points along the path.

    Args:
        voxels: Voxel coordinates in pixels, in (z, y, x) order, for example of all IHC row instances.
        labels: The label of each voxel.
        points: The central path in pixels, in (z, y, x) order, resampled at 1 pixel.
        axes: The unit long axis at each path point, in (z, y, x) order.
        shape: The volume shape. A tip on the volume border is a cut face and is dropped.
        voxel_size: Isotropic voxel size in micrometer.
        min_prominence: Minimal height of a tip over the saddle to a higher tip, in micrometer.
        min_spacing: Minimal distance along the path between the foot points of two tips of one side, in
            micrometer.
        max_pair_offset: Maximal distance along the path between the foot points of the two tips of one cell,
            in micrometer.
        max_tip_angle: Maximal angle in degree between the local axis and the line from the foot point to the tip.
        tip_distance: Range of the distance between a tip and the central line along the axis, in micrometer.

    Returns:
        One row per tip, see `find_ihc_tips_height`.
    """
    index, offset, height = foot_points(voxels, points, axes)
    radius = np.linalg.norm(offset, axis=1)
    tips, positions = {}, {}
    for side, sign in (("top", 1.0), ("bottom", -1.0)):
        value = sign * height
        extreme = _extreme_voxels(index, value)
        # One empty point at each end, so that a tip next to the path end has its full prominence.
        profile = np.full(len(points) + 2, value.min() - 1.0)
        profile[index[extreme] + 1] = value[extreme]
        voxel_of_point = np.full(len(points) + 2, -1)
        voxel_of_point[index[extreme] + 1] = extreme
        peaks = find_peaks(gaussian_filter1d(profile, 1.0), prominence=min_prominence / voxel_size)[0]
        candidates = voxel_of_point[peaks]
        peaks, candidates = peaks[candidates >= 0], candidates[candidates >= 0]
        distance = value[candidates] * voxel_size
        angle = np.degrees(np.arccos(np.clip(value[candidates] / np.maximum(radius[candidates], 1e-6), -1.0, 1.0)))
        valid = (
            (angle <= max_tip_angle) & (distance >= tip_distance[0]) & (distance <= tip_distance[1])
            & ~_at_border(voxels[candidates], shape)
        )
        peaks, candidates = peaks[valid, None], candidates[valid]
        keep = _spaced(peaks, value[candidates], min_spacing / voxel_size)
        tips[side], positions[side] = candidates[keep], peaks[keep]
    return _tip_table(voxels, labels, tips, positions, max_pair_offset / voxel_size)


def _axis_field(
    features: Dict[int, Dict[str, Any]],
    points: np.ndarray,
    reference: Dict[str, Any],
    window: int,
) -> np.ndarray:
    """Estimate the long axis at each path point from the typical row instances within `window` path points.

    The axes of the instances are summed as size-weighted tensors. Points without a typical instance in the
    window get the reference axis.
    """
    tensor = np.zeros((len(points) + 1, 3, 3))
    for feature in features.values():
        if feature["typical"]:
            tensor[feature["index"] + 1] += feature["n_pixels"] * np.outer(feature["axis"], feature["axis"])
    cumulative = np.cumsum(tensor, axis=0)
    i = np.arange(len(points))
    tensor = cumulative[np.minimum(i + window + 1, len(points))] - cumulative[np.maximum(i - window, 0)]
    axes = np.linalg.eigh(tensor)[1][..., -1]
    axes[np.trace(tensor, axis1=1, axis2=2) == 0] = reference["axis"]
    return axes * np.where(axes @ reference["axis"] < 0, -1.0, 1.0)[:, None]


def _partition_along_path(
    index: np.ndarray, height: np.ndarray, centers: np.ndarray, min_size: float, min_depth: float,
) -> np.ndarray:
    """Split voxels into the cells that sit side by side along the path.

    The boundary between two neighbouring cells is at the minimum of the column length (the extent of the
    voxels along the axis at one foot point) between their centers: the neck between two IHCs. While more
    than one cell is left, a cell loses its center if:
    - it is the smallest cell and has less than `min_size` voxels;
    - it is the smaller cell at the shallowest neck, and the neck is less than `min_depth` below the longest
      column of both cells. A false tip on a single cell gives no neck.

    Args:
        index: The path index of the foot point of each voxel, see `foot_points`.
        height: The height of each voxel, see `foot_points`.
        centers: The path index of the center of each cell.
        min_size: Minimal number of voxels of a cell.
        min_depth: Minimal depth of a neck, in pixels.

    Returns:
        The cell of each voxel, as index into `centers`.
    """
    start = index.min()
    index = index - start
    top, bottom = np.full(index.max() + 1, -np.inf), np.full(index.max() + 1, np.inf)
    np.maximum.at(top, index, height)
    np.minimum.at(bottom, index, height)
    extent = gaussian_filter1d(np.where(np.isfinite(top), top - bottom, 0.0), 1.0)
    active = list(np.argsort(centers))
    while True:
        positions = np.clip(np.round(centers[active]).astype(int) - start, 0, len(extent) - 1)
        bounds = [a + int(np.argmin(extent[a:b + 1])) for a, b in zip(positions[:-1], positions[1:])]
        assignment = np.searchsorted(bounds, index, side="right")
        sizes = np.bincount(assignment, minlength=len(active))
        if len(active) == 1:
            break
        if sizes.min() < min_size:
            active.pop(int(np.argmin(sizes)))
            continue
        edges = [0, *bounds, len(extent) - 1]
        longest = [extent[a:b + 1].max() for a, b in zip(edges[:-1], edges[1:])]
        depths = [min(longest[k], longest[k + 1]) - extent[bound] for k, bound in enumerate(bounds)]
        k = int(np.argmin(depths))
        if depths[k] >= min_depth:
            break
        active.pop(k if sizes[k] < sizes[k + 1] else k + 1)
    return np.array(active)[assignment]


def _touching_pairs(segmentation: np.ndarray) -> np.ndarray:
    """Return the label pairs (smaller id first) that share a voxel face."""
    pairs = []
    for dim in range(segmentation.ndim):
        lower = segmentation[(slice(None),) * dim + (slice(None, -1),)]
        upper = segmentation[(slice(None),) * dim + (slice(1, None),)]
        touching = (lower != upper) & (lower > 0) & (upper > 0)
        a, b = lower[touching], upper[touching]
        pairs.append(np.stack([np.minimum(a, b), np.maximum(a, b)], axis=1))
    return np.unique(np.concatenate(pairs), axis=0)


def _merge_parts(
    segmentation: np.ndarray,
    features: Dict[int, Dict[str, Any]],
    coords: Dict[int, np.ndarray],
    points: np.ndarray,
    axes: np.ndarray,
    max_distance: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Merge touching row instances whose foot points are less than `max_distance` path points apart.

    The cells sit side by side along the path, so two touching instances at the same position are parts of one
    cell, for example its upper and lower half. The larger part keeps its id. Short instances are left to the
    tips: the short fragments of two neighbouring cells can form a chain at about the same position (126_R
    `1086`).

    Returns:
        The merged segmentation, and the id that each merged instance joins.
    """
    candidates = [i for i, f in features.items() if f["row"] and not f["border"] and not f["short"]]
    if not candidates:
        return segmentation, {}
    index = foot_points(np.concatenate([coords[i] for i in candidates]).astype(float), points, axes)[0]
    sizes = np.cumsum([len(coords[i]) for i in candidates])[:-1]
    position = {i: np.median(part) for i, part in zip(candidates, np.split(index, sizes))}
    target = {i: i for i in candidates}

    def root(i):
        while target[i] != i:
            i = target[i]
        return i

    for a, b in _touching_pairs(segmentation).tolist():
        if a in position and b in position and abs(position[a] - position[b]) < max_distance:
            small, large = sorted((root(a), root(b)), key=lambda i: features[i]["n_pixels"])
            if small != large:
                target[small] = large
    part_of = {i: root(i) for i in candidates if root(i) != i}
    if not part_of:
        return segmentation, {}
    merged = segmentation.copy()
    for i, j in part_of.items():
        merged[tuple(coords[i].T)] = j
    return merged, part_of


def _largest_contact(labels: np.ndarray, voxels: np.ndarray) -> int:
    """Return the label with the most face contacts to the voxels, or 0 if they touch no label."""
    neighbours = []
    for dim in range(3):
        for step in (-1, 1):
            shifted = voxels.copy()
            shifted[:, dim] += step
            inside = (shifted[:, dim] >= 0) & (shifted[:, dim] < labels.shape[dim])
            neighbours.append(labels[tuple(shifted[inside].T)])
    neighbours = np.concatenate(neighbours)
    neighbours = neighbours[neighbours > 0]
    if len(neighbours) == 0:
        return 0
    values, counts = np.unique(neighbours, return_counts=True)
    return values[np.argmax(counts)]


def _row_frame(
    segmentation: np.ndarray,
    path: np.ndarray,
    reference: Dict[str, Any],
    voxel_size: float,
    max_row_distance: float,
    path_smoothing: float,
    axis_window: float,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Measure the instances along the smoothed path, and the local long axis at each path point.

    Returns the instance features and voxels, the smoothed path resampled at 1 pixel, and the local axes.
    """
    path = moving_average_3d(_densify_path(path)[0], _round(path_smoothing / voxel_size))
    features, coords, points = _instance_features(segmentation, path, voxel_size)
    for feature, voxels in zip(features.values(), coords.values()):
        feature["n_rel"] = feature["n_pixels"] / reference["n_pixels"]
        feature["length_axis"] = _extent(voxels, reference["axis"], voxel_size)
        feature["length_path"] = _extent(voxels, _path_direction(feature["tangent"], reference["axis"]), voxel_size)
        feature["short"] = feature["n_rel"] < 0.6 or feature["length_axis"] < 0.7 * reference["length_axis"]
        feature["row"] = feature["distance"] < max_row_distance
        feature["typical"] = feature["row"] and not feature["border"] and 0.6 <= feature["n_rel"] <= 1.4
    axes = _axis_field(features, points, reference, _round(axis_window / voxel_size))
    return features, coords, points, axes


def ihc_row_reference(
    segmentations: List[np.ndarray],
    path: np.ndarray,
    voxel_size: float = 0.38,
    max_row_distance: float = 15.0,
) -> Dict[str, Any]:
    """Estimate the size and the long axis of a typical IHC in one crop.

    Pass several segmentations of the same crop, for example from two models. The pooled estimate stays
    stable when one segmentation is fragmented or merges many cells.

    Args:
        segmentations: IHC segmentations of the same crop, in (z, y, x) order.
        path: Central path through the IHC row in pixels, in (z, y, x) order, ordered along the row.
        voxel_size: Isotropic voxel size in micrometer.
        max_row_distance: Maximal distance in micrometer between the center of a row IHC and the path.

    Returns:
        The typical IHC: 'n_pixels', 'length_axis' and 'length_path' (extents in micrometer along the
        long axis and along the path), and 'axis', the unit long axis in (z, y, x) order.
    """
    row = []
    for segmentation in segmentations:
        features, coords, _ = _instance_features(segmentation, path, voxel_size)
        row.extend((feature, coords[label_id]) for label_id, feature in features.items()
                   if feature["distance"] < max_row_distance and not feature["border"])
    if len(row) == 0:
        raise ValueError("No instance is near the path.")

    sizes = np.array([feature["n_pixels"] for feature, _ in row])
    # The volume-weighted median is the start, so many small fragments do not pull the estimate down.
    sorted_sizes = np.sort(sizes)
    cumulative = np.cumsum(sorted_sizes)
    n_ref = sorted_sizes[np.searchsorted(cumulative, cumulative[-1] / 2)]
    for _ in range(10):
        n_ref = np.median(sizes[(sizes > 0.5 * n_ref) & (sizes < 1.5 * n_ref)])

    confident = [(feature, voxels) for feature, voxels in row if 0.7 * n_ref < feature["n_pixels"] < 1.3 * n_ref]
    # ponytail: one long axis per crop. Use a local axis along the path if a crop spans a strong turn.
    tensor = sum(feature["n_pixels"] * np.outer(feature["axis"], feature["axis"]) for feature, _ in confident)
    axis = np.linalg.eigh(tensor)[1][:, -1]
    aligned = [(feature, voxels) for feature, voxels in confident if abs(feature["axis"] @ axis) > 0.9]
    return {
        "n_pixels": float(n_ref),
        "length_axis": float(np.median([_extent(voxels, axis, voxel_size) for _, voxels in aligned])),
        "length_path": float(np.median([
            _extent(voxels, _path_direction(feature["tangent"], axis), voxel_size) for feature, voxels in aligned
        ])),
        "axis": axis,
    }


def correct_ihc_row(
    segmentation: np.ndarray,
    path: np.ndarray,
    reference: Dict[str, Any],
    voxel_size: float = 0.38,
    max_row_distance: float = 15.0,
    max_radius_factor: float = 0.8,
    tip_method: str = "path",
    path_smoothing: float = 10.0,
    axis_window: float = 15.0,
    max_axis_angle: float = 30.0,
    min_cell_size: float = 0.4,
    min_neck_depth: float = 1.0,
    max_part_distance: float = 0.4,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Propose a correction of an IHC segmentation, for a manual check before training.

    The correction uses the morphology of the IHC row: each IHC has a tip at both ends of its long axis, and
    the IHCs sit side by side along the central path. Instances and split-off pieces of an instance with less
    than 100 voxels are removed first. Then touching row instances that are not short and at the same position
    along the path (their foot points are less than `max_part_distance` times the reference path length apart)
    are merged: they are parts of one cell, for example its upper and lower half. Each instance gets one class:
    - off_row: the instance center is `max_row_distance` or more from the path. It is removed if it is short:
      smaller than 0.6 of the reference size, or shorter than 0.7 of the reference length along the long axis.
    - normal: the instance has at most one tip on each side, 0.6-1.6 of the reference size (the 1st-99th
      percentile of the annotated IHCs), and its long axis deviates by `max_axis_angle` or less from the local
      axis. It stays unchanged.
    - border: the instance touches the crop border and has at most one tip on each side. It stays unchanged.
    - uncertain: all other row instances.

    The tips come from `find_ihc_tips_path` (`tip_method` 'path') or from `find_ihc_tips_height` ('height',
    with the reference axis). The local long axis comes from the typical row instances within `axis_window`
    along the path. The voxels of the uncertain instances form connected components. In a component, the
    tip cells sit side by side along the path. Each voxel is moved along the local axis onto the central line
    (`foot_points`), and the boundary between two neighbouring cells is at the neck between them
    (`_partition_along_path`). A cell with less than `min_cell_size` of the reference size, or a cell
    without a neck of `min_neck_depth` to its neighbour, loses its tips.
    A component without a tip cell stays unchanged if it touches the crop border or has 0.6 or more of the
    reference size. Else it goes to the label that it touches most, or is removed. The output cell with the
    largest overlap of an input instance keeps its id. Other output cells get new ids.
    Finally, the voxels of each normal cell that are farther than `max_radius_factor` times the reference
    path length from its center in the plane perpendicular to the reference axis are removed.

    Args:
        segmentation: IHC segmentation in (z, y, x) order.
        path: Central path through the IHC row in pixels, in (z, y, x) order, ordered along the row.
        reference: The typical IHC of the crop, as returned by `ihc_row_reference`.
        voxel_size: Isotropic voxel size in micrometer.
        max_row_distance: Maximal distance in micrometer between the center of a row IHC and the path.
        max_radius_factor: Maximal distance of a cell voxel from the cell center in the projection, relative to
            the reference length along the path. The default 0.8 keeps 99% of the voxels of correct cells in
            the MWfLS crops.
        tip_method: 'path' or 'height', see above.
        path_smoothing: Half width in micrometer of the moving average of the path. A path through the
            instance centers zigzags at the scale of the IHC spacing.
        axis_window: Half width in micrometer of the path window for the local axis.
        max_axis_angle: Maximal angle in degree between the long axis of a normal instance and the local axis.
            97% of the annotated IHCs have less than 19 degree, and the next 1% have more than 42 degree.
        min_cell_size: Minimal size of an output cell of an uncertain component, relative to the reference size.
        min_neck_depth: Minimal depth in micrometer of the neck between two output cells of an uncertain
            component. The default 1.0 is about the 5th percentile of the annotated neighbours.
        max_part_distance: Maximal distance along the path between two parts of one cell, relative to the
            reference path length. Neighbouring cells in the MWfLS crops are 0.6 or more apart, and parts of one
            cell 0.35 or less.

    Returns:
        The corrected segmentation.
        A table with one row per input label that is not removed first. 'tips' is the largest number of tips
        of the instance on one side. 'output_ids' lists the output labels that hold 10% or more of its voxels.
        'trimmed' is the fraction of its voxels that the trim removes. 'review' marks an output label with a
        size outside 0.6-1.4 of the reference, a kept off_row instance, and a trimmed fraction above 0.05.
        The tips of the row instances, see `find_ihc_tips_height`.
    """
    if tip_method not in ("path", "height"):
        raise ValueError(f"Invalid tip_method {tip_method}. Choose 'path' or 'height'.")
    segmentation = filter_segmentation_3d(segmentation)
    frame = (reference, voxel_size, max_row_distance, path_smoothing, axis_window)
    input_features, input_coords, points, axes = _row_frame(segmentation, path, *frame)
    original = segmentation
    segmentation, part_of = _merge_parts(
        segmentation, input_features, input_coords, points, axes,
        max_part_distance * reference["length_path"] / voxel_size,
    )
    features, coords = input_features, input_coords
    if part_of:
        features, coords, points, axes = _row_frame(segmentation, path, *frame)

    ids = list(features)
    row_ids = [label_id for label_id in ids if features[label_id]["row"]]
    tips = pd.DataFrame(columns=TIP_COLUMNS)
    if row_ids:
        voxels = np.concatenate([coords[m] for m in row_ids])
        sources = np.concatenate([np.full(len(coords[m]), m) for m in row_ids])
        if tip_method == "path":
            tip_distance = (TIP_DISTANCE[0] * reference["length_axis"], TIP_DISTANCE[1] * reference["length_axis"])
            tips = find_ihc_tips_path(voxels, sources, points, axes, segmentation.shape, voxel_size,
                                      tip_distance=tip_distance)
        else:
            tips = find_ihc_tips_height(voxels, sources, reference["axis"], segmentation.shape, voxel_size)
    # The most tips on one side of an instance is the number of cells that it holds at least.
    tips_per_side = tips.groupby(["label_id", "side"]).size().groupby(level="label_id").max()
    for label_id in ids:
        feature = features[label_id]
        feature["tips"] = int(tips_per_side.get(label_id, 0))
        angle = np.degrees(np.arccos(min(1.0, abs(feature["axis"] @ axes[feature["index"]]))))
        if not feature["row"]:
            feature["class"] = "off_row"
        elif feature["border"]:
            feature["class"] = "uncertain" if feature["tips"] > 1 else "border"
        elif feature["tips"] <= 1 and 0.6 <= feature["n_rel"] <= 1.6 and angle <= max_axis_angle:
            feature["class"] = "normal"
        else:
            feature["class"] = "uncertain"

    labels = segmentation.copy()
    next_id = int(segmentation.max()) + 1
    uncertain = [label_id for label_id in ids if features[label_id]["class"] == "uncertain"]
    if uncertain:
        voxels = np.concatenate([coords[m] for m in uncertain])
        sources = np.concatenate([np.full(len(coords[m]), m) for m in uncertain])
        mask = np.zeros(segmentation.shape, dtype=bool)
        mask[tuple(voxels.T)] = True
        component_map = label_structures(mask)[0]
        components = component_map[tuple(voxels.T)]
        tip_voxels = tips[["z", "y", "x"]].values.astype(int)
        tip_components = component_map[tuple(tip_voxels.T)]
        tip_index = foot_points(tip_voxels.astype(float), points, axes)[0]
        foot_index, _, foot_height = foot_points(voxels.astype(float), points, axes)
        labels[tuple(voxels.T)] = 0
        claimed, short_parts = set(), []
        for component in np.unique(components):
            selection = np.flatnonzero(components == component)
            in_component = tip_components == component
            centers = pd.Series(tip_index[in_component]).groupby(tips["cell"].values[in_component]).mean()
            if len(centers) == 0:
                if (len(selection) < 0.6 * reference["n_pixels"]
                        and not _at_border(voxels[selection], segmentation.shape).any()):
                    short_parts.append(voxels[selection])
                else:
                    labels[tuple(voxels[selection].T)] = sources[selection]
                    claimed.update(np.unique(sources[selection]).tolist())
                continue
            assignment = _partition_along_path(
                foot_index[selection], foot_height[selection], centers.values,
                min_cell_size * reference["n_pixels"], min_neck_depth / voxel_size,
            )
            # The output cell with the largest overlap of an input instance keeps its id.
            pairs, counts = np.unique(np.stack([assignment, sources[selection]], axis=1), axis=0, return_counts=True)
            cell_ids = np.zeros(len(centers), dtype=segmentation.dtype)
            for cell, source in pairs[np.argsort(-counts, kind="stable")].tolist():
                if cell_ids[cell] == 0 and source not in claimed:
                    cell_ids[cell] = source
                    claimed.add(source)
            for cell in np.unique(assignment):
                if cell_ids[cell] == 0:
                    cell_ids[cell] = next_id
                    next_id += 1
            labels[tuple(voxels[selection].T)] = cell_ids[assignment]
        for part in short_parts:
            labels[tuple(part.T)] = _largest_contact(labels, part)

    for label_id in ids:
        if features[label_id]["class"] == "off_row" and features[label_id]["short"]:
            labels[tuple(coords[label_id].T)] = 0

    # Trim each normal cell to a disk around its center in the plane perpendicular to the long axis. This removes
    # the protrusions at the side of a cell. The tips already set the boundaries of the cells of the uncertain
    # instances. A trim with the reference length cuts them where the reference is too small (127_L 0826).
    trimmed = {}
    normal_ids = [label_id for label_id in ids if features[label_id]["class"] == "normal"]
    if normal_ids:
        plane = np.linalg.svd(reference["axis"][None])[2][1:]
        voxels = np.concatenate([coords[m] for m in normal_ids])
        owners = labels[tuple(voxels.T)]
        projection = voxels @ plane.T
        trim = np.zeros(len(voxels), dtype=bool)
        for owner in np.setdiff1d(owners, [0]):
            own = np.flatnonzero(owners == owner)
            radius = np.linalg.norm(projection[own] - np.median(projection[own], axis=0), axis=1) * voxel_size
            trim[own[radius > max_radius_factor * reference["length_path"]]] = True
        labels[tuple(voxels[trim].T)] = 0
        trimmed = dict(zip(*np.unique(original[tuple(voxels[trim].T)], return_counts=True)))

    output_ids, output_sizes = np.unique(labels[labels > 0], return_counts=True)
    output_rel = dict(zip(output_ids.tolist(), (output_sizes / reference["n_pixels"]).tolist()))
    rows = []
    for label_id, feature in input_features.items():
        cell = features[part_of.get(label_id, label_id)]
        assigned = labels[tuple(input_coords[label_id].T)]
        outputs, counts = np.unique(assigned[assigned > 0], return_counts=True)
        outputs = outputs[counts >= 0.1 * len(assigned)].tolist()
        if not outputs:
            # An instance that the tips split into many small cells has no output with 10% of its voxels.
            action = "split" if np.mean(assigned > 0) >= 0.5 else "remove"
        elif len(outputs) > 1:
            action = "split"
        elif outputs[0] != label_id:
            action = "merge"
        else:
            action = "repartition" if cell["class"] == "uncertain" else "keep"
        trimmed_fraction = trimmed.get(label_id, 0) / len(input_coords[label_id])
        review = cell["class"] != "border" and (
            (cell["class"] == "off_row" and action == "keep")
            or any(not 0.6 <= output_rel[o] <= 1.4 for o in outputs)
            or trimmed_fraction > 0.05
        )
        rows.append({
            "label_id": label_id,
            "class": cell["class"],
            "border": feature["border"],
            "n_rel": round(feature["n_rel"], 2),
            "length_axis": round(feature["length_axis"], 1),
            "length_path": round(feature["length_path"], 1),
            "distance": round(feature["distance"], 1),
            "tips": cell["tips"],
            "action": action,
            "output_ids": ",".join(str(o) for o in outputs),
            "trimmed": round(trimmed_fraction, 2),
            "review": review,
        })
    return labels, pd.DataFrame(rows), tips
