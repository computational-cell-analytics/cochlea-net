import math
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import gaussian_filter, grey_closing
from skimage.draw import line_nd

from flamingo_tools.postprocessing.cochlea_mapping import CENTRAL_PATH_METHODS, _centroids_per_component
from flamingo_tools.postprocessing.label_components import downscaled_centroids
from flamingo_tools.s3_utils import read_table


def insert_coordinate(
    df: pd.DataFrame,
    optimal_pos: int,
    new_point: Tuple[int],
) -> pd.DataFrame:
    print(f"Inserting {new_point} at {optimal_pos}.")
    # start of DataFrame
    if optimal_pos == 0:
        entry_dic = {
            "spot_id": len(df),
            "x": new_point[0],
            "y": new_point[1],
            "z": new_point[2],
        }
        insert_df = pd.DataFrame([entry_dic])
        result = pd.concat([
            insert_df,
            df.iloc[:]
        ], ignore_index=True)
    # end of DataFrame
    elif optimal_pos == len(df):
        insert_df = pd.DataFrame([entry_dic])
        result = pd.concat([
            df.iloc[:],
            insert_df
        ], ignore_index=True)

    else:
        # Insert the point at the optimal position
        coord_columns = ["x", "y", "z"]

        extrapolate_columns = [c for c in list(df.columns)[1:] if c not in coord_columns]
        entry_dic = {
            "spot_id": len(df),
            "x": new_point[0],
            "y": new_point[1],
            "z": new_point[2],
        }
        coord_before = [df.iloc[optimal_pos - 1]["x"], df.iloc[optimal_pos - 1]["y"], df.iloc[optimal_pos - 1]["z"]]
        coord_after = [df.iloc[optimal_pos + 1]["x"], df.iloc[optimal_pos + 1]["y"], df.iloc[optimal_pos + 1]["z"]]
        dist_before = math.dist(new_point, coord_before)
        dist_after = math.dist(new_point, coord_after)
        factor_before = dist_before / (dist_before + dist_after)
        factor_after = dist_after / (dist_before + dist_after)
        print(factor_before, factor_after)
        for col in extrapolate_columns:
            value_before = df.iloc[optimal_pos - 1][col]
            value_after = df.iloc[optimal_pos + 1][col]
            entry_dic[col] = factor_before * value_before + factor_after * value_after

        insert_df = pd.DataFrame([entry_dic])
        print(insert_df)

        result = pd.concat([
            df.iloc[:optimal_pos],
            insert_df,
            df.iloc[optimal_pos:]
        ], ignore_index=True)

    return result


def find_optimal_point_for_insert(
    df: pd.DataFrame,
    new_point: Tuple[int],
    tolerance: str = 1e-10,
) -> Tuple[int]:
    """Insert a single point into the path to minimize total path length.

    Args:
        df: pandas DataFrame with 'x' and 'y' columns (ordered path)
        new_point: tuple or array-like (x, y) of the point to insert

    Returns:
        Optimal position for point insertion
    """
    # Convert new point to numpy array
    new_point = np.array(new_point)
    coord_columns = ["x", "y", "z"]
    # Check if point already exists in the dataframe (with tolerance)
    # Check for existing point using numeric columns only
    df_numeric = df[coord_columns]
    distances = np.sqrt(((df_numeric.values - new_point) ** 2).sum(axis=1))
    if (distances < tolerance).any():
        print(f"Point {new_point} already exists in the path. Skipping insertion.")
        return None

    # Calculate distances between consecutive points in the original path
    coords = df[coord_columns].values
    n = len(coords)

    # If path has only one point, insert after it
    if n == 1:
        return 1

    # Calculate the cost of inserting at each position
    # Cost = distance from previous point to new point + distance from new point to next point
    # minus the original distance between previous and next point
    costs = []
    for i in range(n + 1):
        if i == 0:
            # Insert at beginning
            cost = math.dist(new_point, coords[0])
        elif i == n:
            # Insert at end
            cost = math.dist(new_point, coords[n - 1])
        else:
            # Insert between i and i+1
            cost = (math.dist(coords[i - 1], new_point) +
                    math.dist(new_point, coords[i]) -
                    math.dist(coords[i - 1], coords[i]))

        costs.append(cost)

    # Find the position with minimum cost (most reduction in total length)
    optimal_pos = np.argmin(costs)
    print(optimal_pos)

    return int(optimal_pos)


def insert_coordinates_to_central_path(
    table_path: str,
    output_path: str,
    coordinates: List[List[int]],
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Insert coordinates into a central path of an SGN or IHC segmentation.

    Args:
        table_path: Path to table featuring central path as spots.
        output_path: Output path.
        coordinates: List of a single or multiple coordinates to insert into the path.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """
    table = read_table(
        table_path, s3, bucket_name=s3_bucket_name,
        service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
    )

    for coord in coordinates:
        optimal_pos = find_optimal_point_for_insert(table, coord)
        if optimal_pos is not None:
            table = insert_coordinate(table, optimal_pos, coord)

    table.to_csv(output_path, sep="\t", index=False)


def mean_by_offset(
    table: pd.DataFrame,
    value_columns: List[str],
    n_length_bins: int = 20,
    offset_step: float = 10.0,
    max_offset: float = 100.0,
) -> pd.DataFrame:
    """Average columns of a mapped table per window along the central path and per offset bin.

    The windows split 'length_fraction' into `n_length_bins` equal parts. The offset bins are
    `offset_step` micrometers wide. The last bin holds every offset of `max_offset` or more.
    Instances without a mapping have an offset of -1 and are dropped.

    Args:
        table: Segmentation table with the columns of `tonotopic_mapping`.
        value_columns: Columns to average.
        n_length_bins: Number of windows along the central path.
        offset_step: Width of an offset bin in micrometers.
        max_offset: Lower edge of the last, open offset bin in micrometers.

    Returns:
        One row per window and offset bin, with the columns 'length_bin', 'offset_bin', 'n' and
        the mean of each value column. 'length_bin' and 'offset_bin' hold the lower edge of the
        bin. Empty cells have n = 0.
    """
    table = table[table["offset"] >= 0]
    length_edges = np.arange(n_length_bins + 1) / n_length_bins
    length_bin = pd.cut(
        table["length_fraction"], length_edges, labels=length_edges[:-1], include_lowest=True,
    ).rename("length_bin")
    offset_edges = np.append(np.arange(0, max_offset + offset_step, offset_step), np.inf)
    offset_bin = pd.cut(
        table["offset"], offset_edges, labels=offset_edges[:-1], right=False,
    ).rename("offset_bin")

    grouped = table.groupby([length_bin, offset_bin], observed=False)
    result = grouped[value_columns].mean()
    result.insert(0, "n", grouped.size())
    return result.reset_index()


def _output_prefix(table_path: str, cochlea: Optional[str] = None) -> str:
    parts = Path(table_path).parts
    if parts[-1] == "default.tsv" and len(parts) >= 4 and parts[-3] == "tables":
        name, seg_name = parts[-4], parts[-2]
    else:
        stem = Path(table_path).stem
        match = re.fullmatch(r"(.+)[_-]((?:SGN|IHC)[_-]v\d+[a-z]?)", stem)
        if cochlea is not None and stem.startswith(f"{cochlea}_"):
            name, seg_name = cochlea, stem[len(cochlea) + 1:]
        elif match:
            name, seg_name = match.groups()
        else:
            name, seg_name = stem, ""
    prefix = (cochlea or name).replace("_", "-")
    if re.search(r"[_-]v\d", seg_name):
        prefix += "_" + seg_name.replace("_", "-")
    return prefix


def export_central_paths(
    table_path: str,
    output_dir: str,
    component_list: List[int] = [1],
    cochlea: Optional[str] = None,
    scale_factor: int = 10,
    use_napari: bool = False,
    force_overwrite: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Export the central paths v1 and v2 through Rosenthal's canal as TIF files for napari.

    Path v1 uses the method 'edt' of the CochleaNet paper. Path v2 uses 'edt_refined'.
    The files `<prefix>_seg.tif`, `<prefix>_intensity.tif`, `<prefix>_path-v1.tif` and
    `<prefix>_path-v2.tif` share one downscaled grid in (Z, Y, X) order:

    - The segmentation holds the component label of each instance at its centroid. A closing with
      a 3x3x3 window fills the gaps between the centroid voxels.
    - The intensity holds the number of instances per voxel, smoothed with a Gaussian of one voxel.
      It sums to the number of instances.
    - A path holds 1 on each voxel that it passes through.

    The prefix is `<cochlea>_<segmentation>`. Each part has dashes in place of underscores, for
    example `M-AMD-000126-L_SGN-v2_path-v1.tif`. The parts come from the table path:

    - A table path `<cochlea>/tables/<segmentation>/default.tsv` (S3 and MoBIE layout) gives both
      parts. The segmentation is the full folder name, so `PV_SGN_v2` gives `PV-SGN-v2`.
    - A local file `<cochlea>_<segmentation>.tsv` gives both parts if the segmentation is `SGN` or
      `IHC` with a version, for example `M_AMD_000126_L_SGN_v2.tsv`. For a segmentation with a
      prefix, such as `M_LR_000184_L_PV_SGN_v2.tsv`, give `cochlea` to split the file name.
      Other file names give the cochlea only.
    - A segmentation without a version, such as `SGN_merged`, is left out. The prefix is then
      `<cochlea>`, for example `M-AMD-000126-L_path-v1.tif`.

    Args:
        table_path: Path to the SGN segmentation table, with the column 'component_labels'.
        output_dir: Output directory for the TIF files.
        component_list: Components to find the central paths for.
        cochlea: Cochlea part of the output file names. If None, it comes from the table path.
        scale_factor: Downscaling factor of the output grid. One voxel spans this many µm. The paths
            always use the 10 µm default of the path methods.
        use_napari: Show the segmentation and the paths in the napari viewer.
        force_overwrite: Overwrite existing TIF files. If False, skip the export if a TIF file exists.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """
    prefix = _output_prefix(table_path, cochlea)
    out_paths = {name: os.path.join(output_dir, f"{prefix}_{name}.tif")
                 for name in ("seg", "intensity", "path-v1", "path-v2")}
    existing = [path for path in out_paths.values() if os.path.isfile(path)]
    if existing and not force_overwrite:
        print(f"Skipping {prefix}. Output already exists: {', '.join(existing)}")
        return

    table = read_table(
        table_path, s3, bucket_name=s3_bucket_name,
        service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
    )
    centroids_components = _centroids_per_component(table, component_list)
    paths = {name: CENTRAL_PATH_METHODS[method](centroids_components)
             for name, method in [("path-v1", "edt"), ("path-v2", "edt_refined")]}

    centroids = table[["anchor_x", "anchor_y", "anchor_z"]].values
    # An 'edt_refined' path can run past the outermost centroid.
    ref_dimensions = np.max([centroids.max(axis=0)] + [p.max(axis=0) for ps in paths.values() for p in ps], axis=0)
    seg = downscaled_centroids(centroids, scale_factor, ref_dimensions=ref_dimensions,
                               component_labels=table["component_labels"], downsample_mode="components")
    counts = downscaled_centroids(centroids, scale_factor, ref_dimensions=ref_dimensions,
                                  downsample_mode="accumulated")
    volumes = {
        "seg": grey_closing(seg, size=3).astype("uint16"),
        # At 10 µm a voxel holds about one instance, so only the smoothed counts show the density.
        "intensity": gaussian_filter(counts.astype("float32"), sigma=1),
    }
    for name, component_paths in paths.items():
        volume = np.zeros(seg.shape, dtype="uint8")
        for path in component_paths:
            # Consecutive nodes can be two voxels apart: 'edt' nodes at 10 µm, and the 10 µm nodes of
            # 'edt_refined' below 10 µm. So connect them to a continuous line.
            # A refined node close to the origin can be negative, and a negative index wraps around.
            voxels = np.clip(np.asarray(path) // scale_factor, 0, None).astype(int)
            for start, stop in zip(voxels[:-1], voxels[1:]):
                volume[line_nd(start, stop, endpoint=True)] = 1
        volumes[name] = volume

    # Table order (x, y, z) to array order (Z, Y, X).
    volumes = {name: volume.T for name, volume in volumes.items()}
    os.makedirs(output_dir, exist_ok=True)
    for name, volume in volumes.items():
        tifffile.imwrite(out_paths[name], volume, compression="zlib")

    if use_napari:
        import napari
        viewer = napari.Viewer()
        # The 3D view writes a depth value at the surface of a layer, whatever its opacity. Without
        # the depth test, the intensity and the segmentation do not hide the paths inside them.
        viewer.add_image(volumes["intensity"], name=f"{prefix}_intensity", blending="additive")
        viewer.add_labels(volumes["seg"], name=f"{prefix}_seg", opacity=0.3, blending="translucent_no_depth")
        for name, color in [("path-v1", "magenta"), ("path-v2", "yellow")]:
            viewer.add_labels(volumes[name], name=f"{prefix}_{name}", opacity=1.0,
                              colormap={None: "transparent", 1: color})
        viewer.dims.ndisplay = 3
        napari.run()
