import math
import multiprocessing as mp
import os
from concurrent import futures
from typing import Callable, List, Optional, Sequence, Tuple, Union

import elf.parallel as parallel
import numpy as np
import networkx as nx
import pandas as pd

from bioimage_cpp.utils import Blocking

from elf.io import open_file
from flamingo_tools.json_util import archive_processing_params, load_processing_params
from flamingo_tools.s3_utils import (default_table_path, read_table, MOBIE_FOLDER,
                                     table_name_prefix)
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
from scipy.sparse import csr_matrix
from scipy.spatial import distance
from scipy.spatial import cKDTree, ConvexHull
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


#
# Spatial statistics:
# Three different spatial statistics implementations that
# can be used as the basis of a filtering criterion.
#


def nearest_neighbor_distance(table: pd.DataFrame, n_neighbors: int = 10) -> np.ndarray:
    """Compute the average distance to the n nearest neighbors.

    Args:
        table: The table with the centroid coordinates.
        n_neighbors: The number of neighbors to take into account for the distance computation.

    Returns:
        The average distances to the n nearest neighbors.
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    centroids = np.array(centroids)

    # Nearest neighbor is always itself, so n_neighbors+=1.
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(centroids)
    distances, indices = nbrs.kneighbors(centroids)

    # Average distance to nearest neighbors
    distance_avg = np.array([sum(d) / len(d) for d in distances[:, 1:]])
    return distance_avg


def local_ripleys_k(table: pd.DataFrame, radius: float = 15, volume: Optional[float] = None) -> np.ndarray:
    """Compute the local Ripley's K function for each point in a 2D / 3D.

    Args:
        table: The table with the centroid coordinates.
        radius: The radius within which to count neighboring points.
        volume: The area (2D) or volume (3D) of the study region. If None, it is estimated from the convex hull.

    Returns:
        An array containing the local K values for each point.
    """
    points = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    points = np.array(points)
    n_points, dim = points.shape

    if dim not in (2, 3):
        raise ValueError("Points array must be of shape (n_points, 2) or (n_points, 3).")

    # Estimate area/volume if not provided.
    if volume is None:
        hull = ConvexHull(points)
        volume = hull.volume  # For 2D, 'volume' is area; for 3D, it's volume.

    # Compute point density.
    density = n_points / volume

    # Build a KD-tree for efficient neighbor search.
    tree = cKDTree(points)

    # Count neighbors within the specified radius for each point
    counts = tree.query_ball_point(points, r=radius)
    local_counts = np.array([len(c) - 1 for c in counts])  # Exclude the point itself

    # Normalize by density to get local K values
    local_k = local_counts / density
    return local_k


def neighbors_in_radius(table: pd.DataFrame, radius: float = 15) -> np.ndarray:
    """Compute the number of neighbors within a given radius.

    Args:
        table: The table with the centroid coordinates.
        radius: The radius within which to count neighboring points.

    Returns:
        An array containing the number of neighbors within the given radius.
    """
    points = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    points = np.array(points)

    dist_matrix = distance.pdist(points)
    dist_matrix = distance.squareform(dist_matrix)

    # Create sparse matrix of connections within the threshold distance.
    sparse_matrix = csr_matrix(dist_matrix < radius, dtype=int)

    # Sum each row to count neighbors.
    neighbor_counts = sparse_matrix.sum(axis=1)
    return np.array(neighbor_counts)


#
# Filtering function:
# Filter the segmentation based on a spatial statistics from above.
#


def compute_table_on_the_fly(
    segmentation: np.typing.ArrayLike,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
) -> pd.DataFrame:
    """Compute a segmentation table compatible with MoBIE.

    The table contains information about the number of pixels per object,
    the anchor (= centroid) and the bounding box. Anchor and bounding box are given in physical coordinates.

    Args:
        segmentation: The segmentation for which to compute the table.
        voxel_size: The physical voxel spacing of the data.

    Returns:
        The segmentation table.
    """
    if not isinstance(voxel_size, float):
        if len(voxel_size) == 1:
            voxel_size = voxel_size * 3
        assert len(voxel_size) == 3
    else:
        voxel_size = (voxel_size,) * 3

    props = measure.regionprops(segmentation)
    label_ids = np.array([prop.label for prop in props])
    coordinates = np.array([prop.centroid for prop in props]).astype("float32")
    # transform pixel distance to physical units.
    # regionprops returns ZYX coordinates, so the (x, y, z) voxel size has to be reversed.
    voxel_size_zyx = np.array(voxel_size)[::-1]
    coordinates = coordinates * voxel_size_zyx
    bb_min = np.array([prop.bbox[:3] for prop in props]).astype("float32") * voxel_size_zyx
    bb_max = np.array([prop.bbox[3:] for prop in props]).astype("float32") * voxel_size_zyx
    sizes = np.array([prop.area for prop in props])
    table = pd.DataFrame({
        "label_id": label_ids,
        "anchor_x": coordinates[:, 2],
        "anchor_y": coordinates[:, 1],
        "anchor_z": coordinates[:, 0],
        "bb_min_x": bb_min[:, 2],
        "bb_min_y": bb_min[:, 1],
        "bb_min_z": bb_min[:, 0],
        "bb_max_x": bb_max[:, 2],
        "bb_max_y": bb_max[:, 1],
        "bb_max_z": bb_max[:, 0],
        "n_pixels": sizes,
    })
    return table


def filter_segmentation(
    segmentation: np.typing.ArrayLike,
    output_path: str,
    spatial_statistics: Callable,
    threshold: float,
    min_size: int = 1000,
    table: Optional[pd.DataFrame] = None,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
    output_key: str = "segmentation_postprocessed",
    **spatial_statistics_kwargs,
) -> Tuple[int, int]:
    """Postprocessing step to filter isolated objects from a segmentation.

    Instance segmentations are filtered based on spatial statistics and a threshold.
    In addition, objects smaller than a given size are filtered out.

    Args:
        segmentation: Dataset containing the segmentation.
        output_path: Output path for postprocessed segmentation.
        spatial_statistics: Function to calculate density measure for elements of segmentation.
        threshold: Distance in micrometer to check for neighbors.
        min_size: Minimal number of pixels for filtering small instances.
        table: Dataframe of segmentation table.
        voxel_size: Voxel size of segmentation in micrometer.
        output_key: Output key for postprocessed segmentation.
        spatial_statistics_kwargs: Arguments for spatial statistics function.

    Returns:
        The number of objects before filtering.
        The number of objects after filtering.
    """
    # Compute the table on the fly. This doesn't work for large segmentations.
    if table is None:
        table = compute_table_on_the_fly(segmentation, voxel_size=voxel_size)
    n_ids = len(table)

    # First apply the size filter.
    table = table[table.n_pixels > min_size]
    stat_values = spatial_statistics(table, **spatial_statistics_kwargs)

    keep_mask = np.array(stat_values > threshold).squeeze()
    keep_ids = table.label_id.values[keep_mask]

    shape = segmentation.shape
    block_shape = (128, 128, 128)
    chunks = (128, 128, 128)

    blocking = Blocking([0] * len(shape), shape, block_shape)

    output = open_file(output_path, mode="a")
    output_dataset = output.create_dataset(
        output_key, shape=shape, dtype=segmentation.dtype,
        chunks=chunks, compression="gzip"
    )

    def filter_chunk(block_id):
        """Set all points within a chunk to zero if they match filter IDs.
        """
        block = blocking.get_block(block_id)
        volume_index = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = segmentation[volume_index]
        data[np.isin(data, keep_ids)] = 0
        output_dataset[volume_index] = data

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as filter_pool:
        list(tqdm(filter_pool.map(filter_chunk, range(blocking.number_of_blocks)), total=blocking.number_of_blocks))

    seg_filtered, n_ids_filtered, _ = parallel.relabel_consecutive(
        output_dataset, start_label=1, keep_zeros=True, block_shape=block_shape
    )

    return n_ids, n_ids_filtered


def downscaled_centroids(
    centroids: np.ndarray,
    scale_factor: Union[int, Sequence[int]],
    ref_dimensions: Optional[Tuple[float, float, float]] = None,
    component_labels: Optional[List[int]] = None,
    downsample_mode: str = "accumulated",
) -> np.typing.NDArray:
    """Downscale centroids in dataframe.

    Args:
        centroids: Centroids of SGN segmentation, ndarray of shape (N, 3)
        scale_factor: Factor for downscaling coordinates, either a single value or one value per
            axis, in the same axis order as the centroids.
        ref_dimensions: Reference dimensions for downscaling. Taken from centroids if not supplied.
        component_labels: List of component labels, which has to be supplied for the downsampling mode 'components'
        downsample_mode: Flag for downsampling, either 'accumulated', 'capped', or 'components'.

    Returns:
        The downscaled array
    """
    factors = np.broadcast_to(np.asarray(scale_factor), (3,))
    centroids_scaled = [(c[0] / factors[0], c[1] / factors[1], c[2] / factors[2]) for c in centroids]

    if ref_dimensions is None:
        ref_dimensions = (max([c[0] for c in centroids]),
                          max([c[1] for c in centroids]),
                          max([c[2] for c in centroids]))

    bounding_dimensions_scaled = tuple(round(b // f + 1) for b, f in zip(ref_dimensions, factors))
    new_array = np.zeros(bounding_dimensions_scaled)

    if downsample_mode == "accumulated":
        for c in centroids_scaled:
            new_array[int(c[0]), int(c[1]), int(c[2])] += 1

    elif downsample_mode == "capped":
        for c in centroids_scaled:
            new_array[int(c[0]), int(c[1]), int(c[2])] = 1

    elif downsample_mode == "components":
        if component_labels is None:
            raise KeyError("Component labels must be supplied for downsampling with mode 'components'.")
        for comp, centr in zip(component_labels, centroids_scaled):
            if comp != 0:
                new_array[int(centr[0]), int(centr[1]), int(centr[2])] = comp

    else:
        raise ValueError("Choose one of the downsampling modes 'accumulated', 'capped', or 'components'.")

    new_array = np.round(new_array).astype(int)

    return new_array


def graph_connected_components(coords: dict, max_edge_distance: float, min_component_length: int):
    """Create a list of IDs for each connected component of a graph.

    Args:
        coords: Dictionary containing label IDs as keys and their position as value.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        min_component_length: Minimal length of nodes of connected component. Filtered out if lower.

    Returns:
        List of dictionary keys of connected components.
        Graph of connected components.
    """
    graph = nx.Graph()
    for num, pos in coords.items():
        graph.add_node(num, pos=pos)

    # create edges between points whose distance is less than threshold max_edge_distance
    for num_i, pos_i in coords.items():
        for num_j, pos_j in coords.items():
            if num_i < num_j:
                dist = math.dist(pos_i, pos_j)
                if dist <= max_edge_distance:
                    graph.add_edge(num_i, num_j, weight=dist)

    components = list(nx.connected_components(graph))

    # remove connected components with less nodes than threshold min_component_length
    for component in components:
        if len(component) < min_component_length:
            for c in component:
                graph.remove_node(c)

    components = [list(s) for s in nx.connected_components(graph)]
    components = sorted(components, key=lambda component: (len(component), component), reverse=True)

    return components, graph


def graph_components(
    table: pd.DataFrame,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
) -> List[List[int]]:
    """Create connected components of the instances of a segmentation table.

    Args:
        table: Dataframe of segmentation table.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        Subgraph components as lists of label_ids of dataframe, the largest first.
    """
    coords = dict(zip(map(int, table["label_id"]), zip(table["anchor_x"], table["anchor_y"], table["anchor_z"])))
    components, _ = graph_connected_components(coords, max_edge_distance, min_component_length)
    return components


def label_components_ihc(
    table: pd.DataFrame,
    min_size: int = 1000,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
) -> pd.DataFrame:
    """Label components using graph connected components. SGNs and IHCs use the same labeling.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        The table with the column "component_labels": 0 for background, then 1, 2, ... in descending order of size.
        The instances smaller than min_size follow the other instances.
    """
    entries_filtered = table[table.n_pixels < min_size]
    table = table[table.n_pixels >= min_size]
    components = graph_components(table, min_component_length=min_component_length,
                                  max_edge_distance=max_edge_distance)

    table = pd.concat([table, entries_filtered], ignore_index=True)
    table.loc[:, "component_labels"] = 0
    for lab, comp in enumerate(components):
        table.loc[table["label_id"].isin(comp), "component_labels"] = lab + 1

    return table


label_components_sgn = label_components_ihc


def dilate_and_trim(
    arr_orig: np.ndarray,
    edt: np.ndarray,
    iterations: int = 15,
    offset: float = 0.4,
) -> np.ndarray:
    """Dilate and trim original binary array according to a
    Euclidean Distance Trasform computed for a separate target array.

    A newly dilated voxel is kept only if it is closer to the target than the nearest face
    neighbour it grew from, by more than `offset`. The dilation therefore only moves downhill in
    `edt`, towards the target structure.

    Args:
        arr_orig: Original 3D binary array
        edt: 3D array containing Euclidean Distance transform for guiding dilation
        iterations: Number of iterations for dilations
        offset: Offset for regulating dilation. value should be in range(0, 0.45)

    Returns:
        Dilated binary array
    """
    # Face neighbours as (axis, shift) pairs. Neighbours are read with np.roll, so an index at the
    # array border wraps around, as it does for the negative indices of a plain coordinate offset.
    border_coords = [(0, 1), (0, -1), (1, 1), (1, -1), (2, 1), (2, -1)]
    for _ in range(iterations):
        arr_dilated = binary_dilation(arr_orig)
        candidates = arr_dilated & (arr_orig == 0)

        # Distance of the source voxels only. A value above 1 dilates, but does not act as a source.
        source_edt = np.where(arr_orig == 1, edt, np.inf)
        min_dist = np.full(edt.shape, np.inf)
        for axis, shift in border_coords:
            np.minimum(min_dist, np.roll(source_edt, shift, axis=axis), out=min_dist)

        trim = candidates & (edt >= min_dist - offset)
        arr_dilated[trim] = 0

        # The seed set only grows, so once an iteration keeps nothing the result cannot change.
        if not (candidates & ~trim).any():
            return arr_dilated
        arr_orig = arr_dilated
    return arr_dilated


def filter_cochlea_volume_single(
    table: pd.DataFrame,
    components: Optional[List[int]] = [1],
    scale_factor: Union[int, Sequence[int]] = 48,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    dilation_iterations: int = 12,
    padding_um: float = 456.0,
) -> np.ndarray:
    """Filter cochlea volume based on a segmentation table.
    Centroids contained in the segmentation table are used to create a down-scaled binary array.
    The array can be dilated.

    Args:
        table: Segmentation table.
        components: Component labels for filtering segmentation table.
        scale_factor: Down-sampling factor for filtering, in pixel. Either a single value or one
            value per axis in (x, y, z) order.
        voxel_size: Voxel size of data in micrometer, in (x, y, z) order.
        dilation_iterations: Iterations for dilating binary segmentation mask. A negative value omits binary closing.
        padding_um: Padding in micrometer to apply to guessed dimensions based on centroid coordinates.

    Returns:
        Binary 3D array of filtered cochlea.
    """
    # filter components
    if components is not None:
        table = table[table["component_labels"].isin(components)]

    # identify approximate input dimensions for down-scaling
    centroids = list(zip(table["anchor_x"] / voxel_size[0],
                         table["anchor_y"] / voxel_size[1],
                         table["anchor_z"] / voxel_size[2]))

    # padding the array allows for dilation without worrying about array borders
    ref_dimensions = tuple(
        (table[f"anchor_{axis}"].max() + padding_um) / voxel_size[num]
        for num, axis in enumerate("xyz")
    )

    # down-scale arrays
    array_downscaled = downscaled_centroids(centroids, ref_dimensions=ref_dimensions,
                                            scale_factor=scale_factor, downsample_mode="capped")

    if dilation_iterations > 0:
        array_dilated = binary_dilation(array_downscaled, np.ones((3, 3, 3)), iterations=dilation_iterations)
        return binary_closing(array_dilated, np.ones((3, 3, 3)), iterations=1)

    elif dilation_iterations == 0:
        return binary_closing(array_downscaled, np.ones((3, 3, 3)), iterations=1)

    else:
        return array_downscaled


def filter_cochlea_volume(
    sgn_table: pd.DataFrame,
    ihc_table: pd.DataFrame,
    sgn_components: Optional[List[int]] = [1],
    ihc_components: Optional[List[int]] = [1],
    scale_factor: Union[int, Sequence[int]] = 48,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    dilation_iterations: int = 12,
    padding_um: float = 456.0,
    dilation_method: str = "individual",
) -> np.ndarray:
    """Filter cochlea volume with SGN and IHC segmentation.
    Centroids contained in the segmentation tables are used to create down-scaled binary arrays.
    The arrays are then dilated using guided dilation to fill the section inbetween SGNs and IHCs.

    Args:
        sgn_table: SGN segmentation table.
        ihc_table: IHC segmentation table.
        sgn_components: Component labels for filtering SGN segmentation table.
        ihc_components: Component labels for filtering IHC segmentation table.
        scale_factor: Down-sampling factor for filtering, in pixel. Either a single value or one
            value per axis in (x, y, z) order.
        voxel_size: voxel_size of pixel in µm, in (x, y, z) order.
        dilation_iterations: Iterations for dilating binary segmentation mask.
        padding_um: Padding in micrometer to apply to guessed dimensions based on centroid coordinates.
        dilation_method: Dilation style for SGN and IHC segmentation, either 'individual', 'combined' or no dilation.

    Returns:
        Binary 3D array of filtered cochlea.
    """
    # filter components
    if sgn_components is not None:
        sgn_table = sgn_table[sgn_table["component_labels"].isin(sgn_components)]
    if ihc_components is not None:
        ihc_table = ihc_table[ihc_table["component_labels"].isin(ihc_components)]

    # identify approximate input dimensions for down-scaling
    centroids_sgn = list(zip(sgn_table["anchor_x"] / voxel_size[0],
                             sgn_table["anchor_y"] / voxel_size[1],
                             sgn_table["anchor_z"] / voxel_size[2]))
    centroids_ihc = list(zip(ihc_table["anchor_x"] / voxel_size[0],
                             ihc_table["anchor_y"] / voxel_size[1],
                             ihc_table["anchor_z"] / voxel_size[2]))

    # padding the array allows for dilation without worrying about array borders
    ref_dimensions = tuple(
        (max([sgn_table[f"anchor_{axis}"].max(), ihc_table[f"anchor_{axis}"].max()]) + padding_um) / voxel_size[num]
        for num, axis in enumerate("xyz")
    )

    # down-scale arrays
    array_downscaled_sgn = downscaled_centroids(centroids_sgn, ref_dimensions=ref_dimensions,
                                                scale_factor=scale_factor, downsample_mode="capped")

    array_downscaled_ihc = downscaled_centroids(centroids_ihc, ref_dimensions=ref_dimensions,
                                                scale_factor=scale_factor, downsample_mode="capped")

    # dilate down-scaled SGN array in direction of IHC segmentation
    distance_from_sgn = distance_transform_edt(~array_downscaled_sgn.astype(bool))
    iterations = 20
    arr_dilated = dilate_and_trim(array_downscaled_ihc.copy(), distance_from_sgn, iterations=iterations, offset=0.4)

    # dilate single structures first
    if dilation_method == "individual":
        ihc_dilated = binary_dilation(array_downscaled_ihc, np.ones((3, 3, 3)), iterations=dilation_iterations)
        sgn_dilated = binary_dilation(array_downscaled_sgn, np.ones((3, 3, 3)), iterations=dilation_iterations)
        combined_dilated = arr_dilated + ihc_dilated + sgn_dilated
        combined_dilated[combined_dilated > 0] = 1
        combined_dilated = binary_dilation(combined_dilated, np.ones((3, 3, 3)), iterations=1)

    # dilate combined structure
    elif dilation_method == "combined":
        # combine SGN, IHC, and region between both to form output mask
        combined_structure = arr_dilated + array_downscaled_ihc + array_downscaled_sgn
        combined_structure[combined_structure > 0] = 1
        combined_dilated = binary_dilation(combined_structure, np.ones((3, 3, 3)), iterations=dilation_iterations)

    # no dilation of combined structure
    else:
        combined_dilated = arr_dilated + array_downscaled_ihc + array_downscaled_sgn
        combined_dilated[combined_dilated > 0] = 1

    return combined_dilated


def _compute_path_from_table(
    table: pd.DataFrame,
    component_list: List[int],
) -> Tuple[float, dict]:
    """Compute the IHC central path from a labeled segmentation table.

    Args:
        table: Segmentation table with a 'component_labels' column.
        component_list: Component labels to include in the path computation.

    Returns:
        Total path length in µm and the raw path_dict from measure_run_length.
    """
    from flamingo_tools.postprocessing.cochlea_mapping import measure_run_length

    centroids_components = []
    for label in component_list:
        subset = table[table["component_labels"] == label]
        centroids_components.append(list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"])))

    total_distance, path_dict = measure_run_length(centroids_components, path_method="graph")
    return total_distance, path_dict


def _central_path_table(path_dict: dict, total_distance: float) -> pd.DataFrame:
    """Convert a path_dict without a frequency mapping to a central path table."""
    from flamingo_tools.postprocessing.cochlea_mapping import path_dict_to_central_path_table

    for item in path_dict.values():
        item["length[µm]"] = item["length_fraction"] * total_distance
        item["frequency[kHz]"] = 0.0
    return path_dict_to_central_path_table(path_dict)


def filter_ihc_by_path_deviation(
    table: pd.DataFrame,
    max_path_deviation: float,
    component_list: List[int] = [1],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter IHC instances by their deviation from the central path through the given components.

    The path comes from measure_run_length. Instances of the components whose centroid is farther
    than max_path_deviation µm from the nearest path node get component_labels = 0 (background).

    Args:
        table: Segmentation table with a 'component_labels' column.
        max_path_deviation: Maximum allowed distance in µm from the central path.
        component_list: Component labels whose instances form the path and are filtered.

    Returns:
        The table with outlier instances set to component_labels=0, and the central path table.
    """
    total_distance, path_dict = _compute_path_from_table(table, component_list)
    path_tree = cKDTree(np.array([v["pos"] for v in path_dict.values()]))
    subset = table[table["component_labels"].isin(component_list)]
    distances, _ = path_tree.query(subset[["anchor_x", "anchor_y", "anchor_z"]].values)

    far_ids = subset["label_id"].values[distances > max_path_deviation]
    table.loc[table["label_id"].isin(far_ids), "component_labels"] = 0
    print(f"Filtered {len(far_ids)} IHCs with path deviation > {max_path_deviation} µm.")

    return table, _central_path_table(path_dict, total_distance)


def ihc_row_path(centroids: np.ndarray) -> np.ndarray:
    """Compute the central path through the centroids of an IHC row.

    The path is the shortest path between the two most distant centroids. The edges are as long as the
    largest gap of the row, so the squared length is the edge weight: the path then follows the cells
    instead of long chords across the curved row. The path is resampled at 1 micrometer, because
    neighbouring path nodes can be a gap apart. The path ends are row centroids, so an off-row centroid
    past a row end becomes a path end.

    Args:
        centroids: The row centroids in micrometers, in (x, y, z) order.

    Returns:
        The path in micrometers, in (x, y, z) order, resampled at 1 micrometer.
    """
    from flamingo_tools.postprocessing.cochlea_mapping import (
        _auto_max_edge_distance, _component_graph, find_most_distant_nodes, resample_path,
    )

    graph = _component_graph(centroids, _auto_max_edge_distance([centroids]))
    nodes = nx.shortest_path(graph, *find_most_distant_nodes(graph), weight=lambda u, v, edge: edge["weight"] ** 2)
    return resample_path(centroids[nodes], 1.0)


def refine_ihc_row(
    table: pd.DataFrame,
    component_list: List[int],
    min_size: int = 10000,
    max_path_deviation: float = 15.0,
    max_edge_distance: float = 30,
    min_component_length: int = 50,
    merge_components: bool = False,
) -> Tuple[pd.DataFrame, np.ndarray, int]:
    """Refine the IHC row, guided by the central path through the given components.

    Each listed component gets its own path (`ihc_row_path`). The paths are linked end to start in the order
    of the list, as in the tonotopic mapping (`_order_components`), and straight lines bridge the gaps. The
    order is necessary: a component can pass closer to the middle of another component than to its end, for
    example on the next turn of the spiral. An instance joins the row if it has at least `min_size` pixels
    and its center is within `max_path_deviation` micrometers of the path, also if it was in no component.
    An off-row instance past a row end stays in the row, because it is a path end.

    The row instances then form new components, with an edge between two instances at most
    `max_edge_distance` micrometers apart. A row instance on a bridge can close a gap, so two listed
    components can merge, and a large gap keeps a broken row split. The components get the labels 1, 2, ...
    in the order of the path, so that this list is the component list for the tonotopic mapping. Components
    with less than `min_component_length` instances are dropped.

    Args:
        table: Segmentation table with a 'component_labels' column.
        component_list: The components that form the IHC row, in the order of the cochlea.
        min_size: Minimal number of pixels of a row instance.
        max_path_deviation: Maximal distance in micrometers between the center of a row instance and the path.
        max_edge_distance: Maximal distance in micrometers between two instances of the same row component.
        min_component_length: Minimal number of instances of a row component.
        merge_components: Make the whole row component 1 and drop no instance.

    Returns:
        The table, with component labels 1, 2, ... for the row and 0 for all other instances.
        The central path through the listed components in micrometers, in (x, y, z) order, resampled at
        1 micrometer.
        The number of row instances in dropped components.

    Raises:
        ValueError: If a listed component holds less than 2 instances.
    """
    from flamingo_tools.postprocessing.cochlea_mapping import _order_components, resample_path

    coords = table[["anchor_x", "anchor_y", "anchor_z"]].values
    labels = table["component_labels"].values
    for component in component_list:
        n_instances = (labels == component).sum()
        if n_instances < 2:
            raise ValueError(f"Component {component} holds {n_instances} instances. A path needs at least 2.")
    # ponytail: the path ends are row instances, so an off-row instance past a row end is never removed.
    # Upgrade: drop an end instance by its sideways deviation from the row, validated on the MWfLS cochleae.
    paths = _order_components([ihc_row_path(coords[labels == component]) for component in component_list])
    path = resample_path(np.concatenate(paths), 1.0)
    distances, position = cKDTree(path).query(coords)
    row = (table["n_pixels"].values >= min_size) & (distances <= max_path_deviation)
    if merge_components:
        table["component_labels"] = row.astype(int)
        print(f"The refined IHC row has {row.sum()} instances.")
        return table, path, 0

    pieces = graph_components(table[row], min_component_length=min_component_length,
                              max_edge_distance=max_edge_distance)
    masks = sorted((table["label_id"].isin(piece).values for piece in pieces), key=lambda mask: position[mask].mean())
    new_labels = np.zeros(len(table), dtype=int)
    for label, mask in enumerate(masks, 1):
        new_labels[mask] = label
    table["component_labels"] = new_labels
    n_dropped = int(row.sum() - (new_labels > 0).sum())
    print(f"The refined IHC row has {len(masks)} component(s) with {[int(mask.sum()) for mask in masks]} instances. "
          f"{n_dropped} instances in components with less than {min_component_length} instances were dropped.")
    return table, path, n_dropped


def label_custom_components(tsv_table, custom_dict):
    """Label IHC components using multiple post-processing configurations and combine the
    results into final components.
    The function applies successive post-processing steps defined in a `custom_dic`
    configuration. Each entry under `label_dicts` specifies:
    - `label_params`: a list of parameter sets. The segmentation is processed once for
    each parameter set (e.g., {"min_size": 500, "max_edge_distance": 65, "min_component_length": 5}).
    - `components`: lists of label IDs to extract from each corresponding post-processing run.
    Label IDs collected from all runs are merged to form the final component (e.g., key "1").
    Global filtering is applied using `min_size_global`, and any `missing_ids`
    (e.g., 4800 or 4832) are added explicitly to the final component.
    Example `custom_dic` structure:
    {
        "min_size_global": 500,
        "missing_ids": [4800, 4832],
        "label_dicts": {
            "1": {
                "label_params": [
                    {"min_size": 500, "max_edge_distance": 65, "min_component_length": 5},
                    {"min_size": 400, "max_edge_distance": 45, "min_component_length": 5}
                ],
                "components": [[18, 22], [1, 45, 83]]
            }
        }
    }

    Args:
        tsv_table: Pandas dataframe of the MoBIE segmentation table.
        custom_dict: Custom dictionary featuring post-processing parameters.

    Returns:
        Pandas dataframe featuring labeled components.
    """
    min_size = custom_dict["min_size_global"]
    component_labels = [0 for _ in range(len(tsv_table))]
    tsv_table.loc[:, "component_labels"] = component_labels
    for custom_comp, label_dict in custom_dict["label_dicts"].items():
        label_params = label_dict["label_params"]
        label_components = label_dict["components"]

        combined_label_ids = []
        for comp, other_kwargs in zip(label_components, label_params):
            tsv_table_tmp = label_components_ihc(tsv_table.copy(), **other_kwargs)
            label_ids = list(tsv_table_tmp.loc[tsv_table_tmp["component_labels"].isin(comp), "label_id"])
            combined_label_ids.extend(label_ids)
            print(f"{comp}", len(combined_label_ids))

        combined_label_ids = list(set(combined_label_ids))

        tsv_table.loc[tsv_table["label_id"].isin(combined_label_ids), "component_labels"] = int(custom_comp)

    tsv_table.loc[tsv_table["n_pixels"] < min_size, "component_labels"] = 0
    if "missing_ids" in list(custom_dict.keys()):
        for m in custom_dict["missing_ids"]:
            tsv_table.loc[tsv_table["label_id"] == m, "component_labels"] = 1

    return tsv_table


def _show_components(table: pd.DataFrame, scale_factor: int, layers: dict):
    """Show the down-scaled centroids in napari, with one label layer per entry of `layers`.

    Each entry of `layers` maps a layer name to one component label per row of the table.
    """
    import napari

    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    viewer = napari.Viewer()
    viewer.add_image(downscaled_centroids(centroids, scale_factor=scale_factor, downsample_mode="accumulated"),
                     name="3D Volume")
    for name, component_labels in layers.items():
        viewer.add_labels(downscaled_centroids(centroids, scale_factor=scale_factor,
                                               component_labels=list(component_labels),
                                               downsample_mode="components"), name=name)
    napari.run()


def _ask(prompt: str) -> str:
    """Read one answer from the terminal. Ctrl+C or Ctrl+D aborts the run before anything is written."""
    try:
        return input(prompt)
    except (KeyboardInterrupt, EOFError):
        raise SystemExit("\nAborted. Nothing was written.")


def _ask_components(prompt: str, existing: set) -> List[int]:
    """Ask for distinct components of `existing`. An empty answer gives an empty list."""
    while True:
        try:
            components = [int(c) for c in _ask(prompt).split()]
        except ValueError:
            components = [None]
        if len(set(components)) == len(components) and existing.issuperset(components):
            return components
        print(f"Enter distinct components of {sorted(existing)}, separated by spaces.")


def label_components_single(
    table_path: str,
    out_path: Optional[str] = None,
    force_overwrite: bool = False,
    cell_type: str = "sgn",
    component_list: List[int] = [1],
    component_list_path: Optional[List[int]] = None,
    max_edge_distance: float = 30,
    min_component_length: int = 50,
    min_size: int = 1000,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    custom_dic: Optional[dict] = None,
    use_napari: bool = False,
    scale_factor: int = 20,
    path_file: Optional[str] = None,
    max_path_deviation: Optional[float] = None,
    refine_row: bool = False,
    merge_components: bool = False,
    dataset_name: Optional[str] = None,
    segmentation_channel: Optional[str] = None,
    archive: Optional[str] = None,
    **_
):
    """Process a single cochlea using one set of parameters or a custom dictionary.
    The cochlea is analyzed using graph-connected components
    to label segmentation instances that are closer than a given maximal edge distance.
    This process acts on an input segmentation table to which a "component_labels" column is added.
    Each entry in this column refers to the index of a connected component.
    The largest connected component has an index of 1; the others follow in decreasing order.

    Args:
        table_path: File path to segmentation table.
        out_path: Output path to segmentation table with new column "component_labels".
        force_overwrite: Forcefully overwrite existing output path.
        cell_type: Cell type of the segmentation. Currently supports "sgn" and "ihc".
        component_list: The components that the later processing steps select.
        component_list_path: The components that form the central IHC path and that the instance counts are
            reported for. Default: component_list.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        min_component_length: Minimal length of nodes of connected component. Filtered out if lower.
        min_size: Minimal number of pixels for filtering small instances.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
        custom_dic: Custom dictionary which allows multiple post-processing configurations and combines the
            results into final components.
        use_napari: Visualize component labels with napari viewer. A prompt after the view asks for the
            components in the order of the cochlea: for IHCs the components of the row. Ctrl+C at a prompt
            aborts the run before anything is written.
        scale_factor: Scale factor for down-scaling data for visualization in Napari.
        path_file: Output path for the central IHC path that the run used: the path of the refined row, of the
            deviation filter, or else the path through the components. Only used when cell_type='ihc'.
        max_path_deviation: Maximum allowed deviation in µm from the central IHC path. Instances
            farther than this are set to component_labels=0. Requires cell_type='ihc'.
        refine_row: Refine the IHC row with `refine_ihc_row`. The components of `component_list_path`, in the
            order of the cochlea, guide the path, and `max_path_deviation` (15 micrometers if None) limits the
            distance to it. The row splits into components 1, 2, ... at gaps larger than `max_edge_distance`.
            Requires cell_type='ihc'. With use_napari, a prompt after each view asks for the row components,
            so a row can be chosen and checked without this flag.
        merge_components: Make the refined IHC row one component. Requires refine_row or use_napari. With
            use_napari, a prompt after the view asks for it when the refinement drops row instances in
            components with less than `min_component_length` instances.
        dataset_name: Name of the cochlea. Required for archive.
        segmentation_channel: Name of the segmentation channel. Required for archive.
        archive: Processing parameter file, or a directory for it, that receives the parameters of the run.
            See `archive_processing_params`.
    """
    if cell_type not in ("sgn", "ihc"):
        raise ValueError(f"Choose a supported cell type. Either 'sgn' or 'ihc', not '{cell_type}'.")
    if (refine_row or merge_components) and cell_type != "ihc":
        raise ValueError(f"refine_row and merge_components require cell_type='ihc', not '{cell_type}'.")
    if merge_components and not (refine_row or use_napari):
        raise ValueError("merge_components requires refine_row or use_napari.")
    if archive is not None and (dataset_name is None or segmentation_channel is None):
        raise ValueError("archive requires dataset_name and segmentation_channel.")

    # overwrite input segmentation table with labeled version
    if out_path is None:
        if s3:
            raise ValueError("Set an output path when accessing remote data.")
        out_path = table_path
        force_overwrite = True

    if os.path.isdir(out_path):
        raise ValueError(f"Output path {out_path} is a directory. Provide a path to a single output file.")

    table = read_table(
        table_path, s3, bucket_name=s3_bucket_name,
        service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
    )

    # overwrite input file
    if os.path.realpath(out_path) == os.path.realpath(table_path) and not s3:
        force_overwrite = True

    if os.path.isfile(out_path) and not force_overwrite:
        print(f"Skipping {out_path}. Table already exists.")
        return

    if custom_dic is not None:
        # use multiple post-processing configurations
        tsv_table = label_custom_components(table, custom_dic)
    else:
        tsv_table = label_components_ihc(table, min_size=min_size, min_component_length=min_component_length,
                                         max_edge_distance=max_edge_distance)

    labeled = tsv_table.copy()
    path_components = component_list_path or component_list
    row_deviation = 15.0 if max_path_deviation is None else max_path_deviation
    row_kwargs = dict(min_size=min_size, max_path_deviation=row_deviation,
                      max_edge_distance=max_edge_distance, min_component_length=min_component_length)
    path_df, n_dropped, merged = None, 0, merge_components

    if cell_type == "ihc" and refine_row:
        tsv_table, row_path, n_dropped = refine_ihc_row(labeled.copy(), path_components,
                                                        merge_components=merge_components, **row_kwargs)
    elif cell_type == "ihc" and max_path_deviation is not None:
        tsv_table, path_df = filter_ihc_by_path_deviation(tsv_table, max_path_deviation, path_components)
    elif cell_type == "ihc" and path_file is not None:
        total_distance, path_dict = _compute_path_from_table(tsv_table, path_components)
        path_df = _central_path_table(path_dict, total_distance)
    elif max_path_deviation is not None:
        print("Warning: --max_path_deviation is only supported for cell_type='ihc'. Skipping path filtering.")

    if use_napari:
        # The tonotopic mapping and the IHC row path need the components in the order of the cochlea.
        # A refined IHC row is checked in the next view.
        prompt = ("Enter IHC row components in order (space separated)" if cell_type == "ihc"
                  else "Enter SGN components in order (space separated)")
        prompt += ", Enter to accept the current mapping, Ctrl+C to abort: "
        counts = labeled["component_labels"].value_counts()
        existing = {int(label) for label, count in counts.items() if label != 0 and count >= 2}
        while True:
            if refine_row:
                layers = {"components": labeled["component_labels"], "IHC row": tsv_table["component_labels"]}
            else:
                layers = {"components": tsv_table["component_labels"]}
            _show_components(labeled, scale_factor, layers)
            merge_prompt = "Merge the components into one row instead? [Y]es/[N]o: "
            if n_dropped and _ask(merge_prompt).strip().lower().startswith("y"):
                tsv_table, row_path, n_dropped = refine_ihc_row(labeled.copy(), path_components,
                                                                merge_components=True, **row_kwargs)
                merged = True
                continue
            components = _ask_components(prompt, existing)
            if not components:
                break
            if cell_type == "sgn":
                component_list = path_components = components
                break
            path_components, refine_row, merged = components, True, merge_components
            tsv_table, row_path, n_dropped = refine_ihc_row(labeled.copy(), path_components,
                                                            merge_components=merge_components, **row_kwargs)
        if custom_dic is None:
            replay = f"--cell_type {cell_type}"
            if refine_row:
                replay += " --refine_row" + (" --merge_components" if merged else "")
            replay += " -c " + " ".join(str(comp) for comp in path_components)
            replay += (f" --min_size {min_size} --min_component_length {min_component_length}"
                       f" --max_edge_distance {max_edge_distance}")
            if refine_row or max_path_deviation is not None:
                replay += f" --max_path_deviation {row_deviation}"
            print("Replay with: " + replay)

    counted = path_components
    if refine_row:
        counted = list(range(1, int(tsv_table["component_labels"].max()) + 1))
        if not counted:
            raise ValueError(f"The refined IHC row has no component with at least {min_component_length} instances. "
                             "Lower min_component_length or use merge_components.")

    tsv_table.to_csv(out_path, sep="\t", index=False)

    if cell_type == "ihc" and path_file is not None:
        if refine_row:
            from flamingo_tools.postprocessing.cochlea_mapping import _path_dict_from_components, arc_length
            length = arc_length(row_path)[-1]
            path_df = _central_path_table(_path_dict_from_components([row_path], length), length)
        path_df.to_csv(path_file, sep="\t", index=False)
        print(f"Saved the central path to {path_file}.")

    if archive is not None:
        section = {"min_size": min_size, "min_component_length": min_component_length,
                   "max_edge_distance": max_edge_distance}
        if custom_dic is not None:
            # The row refinement uses the parameters next to custom_dic.
            section = {"custom_dic": custom_dic, **(section if refine_row else {})}
        if refine_row:
            section.update(component_list_path=path_components, max_path_deviation=row_deviation, refine_row=True)
            if merged:
                section["merge_components"] = True
        else:
            if component_list_path is not None:
                section["component_list_path"] = path_components
            if max_path_deviation is not None:
                section["max_path_deviation"] = max_path_deviation
        archive_processing_params({
            "dataset_name": dataset_name,
            "segmentation_channel": segmentation_channel,
            "cell_type": cell_type,
            "component_list": counted if refine_row else component_list,
            "label_components": section,
        }, archive)

    custom_comp = len(tsv_table[tsv_table["component_labels"].isin(counted)])
    print(f"Total {cell_type.upper()}s: {len(tsv_table)}")
    if counted == [1]:
        print(f"Largest component has {custom_comp} {cell_type.upper()}s.")
    else:
        for comp in counted:
            num_instances = len(tsv_table[tsv_table["component_labels"] == comp])
            print(f"Component {comp} has {num_instances} instances.")
        print(f"Custom component(s) have {custom_comp} {cell_type.upper()}s.")


def label_components_json_wrapper(
    json_file: str,
    out_path: Optional[str] = None,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    path_file: Optional[str] = None,
    **kwargs,
):
    """Label connected components for every entry of a parameter dictionary in a JSON file.

    A file may hold several entries, for example one per segmentation version of the cochlea.
    The segmentation table of an entry is derived from its "dataset_name" and
    "segmentation_channel", so one file can be replayed without naming any path.

    Args:
        json_file: JSON file with one parameter dictionary, or a list of them.
        out_path: Output path for the segmentation table. A directory receives one table per
            entry, named after the cochlea and the segmentation channel. Default: overwrite the
            input table, which requires local data.
        mobie_dir: Local MoBIE directory used for creating data paths. Ignored when s3 is set.
        s3: Flag for accessing data stored on S3 bucket.
        path_file: Output path for the central IHC path. Several entries or a directory out_path
            turn this into one file per entry.
        kwargs: Further arguments for label_components_single. An entry of the JSON file
            overrides them.
    """
    param_dicts = load_processing_params(json_file, "label_components")
    if not param_dicts:
        print(f"{json_file} has no 'label_components' section. Nothing to do.")
        return

    # An output path that is not a TSV file names a directory, which is created if needed. Testing
    # with os.path.isdir would classify a directory that does not exist yet as a single file.
    out_is_dir = out_path is not None and not out_path.endswith(".tsv")
    if out_is_dir:
        os.makedirs(out_path, exist_ok=True)
    if out_path is not None and not out_is_dir and len(param_dicts) > 1:
        raise ValueError(
            f"{json_file} holds {len(param_dicts)} entries, which cannot share the single output "
            f"file {out_path}. Pass an output directory instead."
        )

    for entry in param_dicts:
        cochlea = entry["dataset_name"]
        print(f"\n{cochlea}")
        seg_channel = entry["segmentation_channel"]
        table_path = default_table_path(cochlea, seg_channel, s3=s3, mobie_dir=mobie_dir)
        prefix = table_name_prefix(cochlea, seg_channel)

        save_path, entry_path_file = out_path, path_file
        if out_is_dir:
            save_path = os.path.join(out_path, f"{prefix}.tsv")
        # Several entries must not share one central path file, or each entry would overwrite the
        # path of the previous one.
        if path_file is not None and len(param_dicts) > 1:
            root = out_path if out_is_dir else os.path.dirname(path_file)
            entry_path_file = os.path.join(root, f"{prefix}_path.tsv")
        elif out_is_dir and path_file is not None:
            entry_path_file = os.path.join(out_path, f"{prefix}_path.tsv")

        label_components_single(
            table_path=table_path,
            out_path=save_path,
            s3=s3,
            path_file=entry_path_file,
            **{**kwargs, **entry},
        )
