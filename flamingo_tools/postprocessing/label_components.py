import math
import multiprocessing as mp
import os
from concurrent import futures
from typing import Callable, List, Optional, Tuple

import elf.parallel as parallel
import numpy as np
import nifty.tools as nt
import networkx as nx
import pandas as pd

from elf.io import open_file
from flamingo_tools.s3_utils import get_s3_path
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
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1).fit(centroids)
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


def compute_table_on_the_fly(segmentation: np.typing.ArrayLike, resolution: float) -> pd.DataFrame:
    """Compute a segmentation table compatible with MoBIE.

    The table contains information about the number of pixels per object,
    the anchor (= centroid) and the bounding box. Anchor and bounding box are given in physical coordinates.

    Args:
        segmentation: The segmentation for which to compute the table.
        resolution: The physical voxel spacing of the data.

    Returns:
        The segmentation table.
    """
    props = measure.regionprops(segmentation)
    label_ids = np.array([prop.label for prop in props])
    coordinates = np.array([prop.centroid for prop in props]).astype("float32")
    # transform pixel distance to physical units
    coordinates = coordinates * resolution
    bb_min = np.array([prop.bbox[:3] for prop in props]).astype("float32") * resolution
    bb_max = np.array([prop.bbox[3:] for prop in props]).astype("float32") * resolution
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
    resolution: float = 0.38,
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
        resolution: Resolution of segmentation in micrometer.
        output_key: Output key for postprocessed segmentation.
        spatial_statistics_kwargs: Arguments for spatial statistics function.

    Returns:
        The number of objects before filtering.
        The number of objects after filtering.
    """
    # Compute the table on the fly. This doesn't work for large segmentations.
    if table is None:
        table = compute_table_on_the_fly(segmentation, resolution=resolution)
    n_ids = len(table)

    # First apply the size filter.
    table = table[table.n_pixels > min_size]
    stat_values = spatial_statistics(table, **spatial_statistics_kwargs)

    keep_mask = np.array(stat_values > threshold).squeeze()
    keep_ids = table.label_id.values[keep_mask]

    shape = segmentation.shape
    block_shape = (128, 128, 128)
    chunks = (128, 128, 128)

    blocking = nt.blocking([0] * len(shape), shape, block_shape)

    output = open_file(output_path, mode="a")
    output_dataset = output.create_dataset(
        output_key, shape=shape, dtype=segmentation.dtype,
        chunks=chunks, compression="gzip"
    )

    def filter_chunk(block_id):
        """Set all points within a chunk to zero if they match filter IDs.
        """
        block = blocking.getBlock(block_id)
        volume_index = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = segmentation[volume_index]
        data[np.isin(data, keep_ids)] = 0
        output_dataset[volume_index] = data

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as filter_pool:
        list(tqdm(filter_pool.map(filter_chunk, range(blocking.numberOfBlocks)), total=blocking.numberOfBlocks))

    seg_filtered, n_ids_filtered, _ = parallel.relabel_consecutive(
        output_dataset, start_label=1, keep_zeros=True, block_shape=block_shape
    )

    return n_ids, n_ids_filtered


def erode_subset(
    table: pd.DataFrame,
    iterations: int = 1,
    min_cells: Optional[int] = None,
    threshold: int = 35,
    keyword: str = "distance_nn100",
) -> pd.DataFrame:
    """Erode coordinates of dataframe according to a keyword and a threshold.
    Use a copy of the dataframe as an input, if it should not be edited.

    Args:
        table: Dataframe of segmentation table.
        iterations: Number of steps for erosion process.
        min_cells: Minimal number of rows. The erosion is stopped after falling below this limit.
        threshold: Upper threshold for removing elements according to the given keyword.
        keyword: Keyword of dataframe for erosion.

    Returns:
        The dataframe containing elements left after the erosion.
    """
    print(f"Initial length: {len(table)}")
    n_neighbors = 100
    for i in range(iterations):
        table = table[table[keyword] < threshold]

        distance_avg = nearest_neighbor_distance(table, n_neighbors=n_neighbors)

        if min_cells is not None and len(distance_avg) < min_cells:
            print(f"{i}-th iteration, length of subset {len(table)}, stopping erosion")
            break

        table.loc[:, 'distance_nn'+str(n_neighbors)] = list(distance_avg)

        print(f"{i}-th iteration, length of subset {len(table)}")

    return table


def downscaled_centroids(
    centroids: np.ndarray,
    scale_factor: int,
    ref_dimensions: Optional[Tuple[float, float, float]] = None,
    component_labels: Optional[List[int]] = None,
    downsample_mode: str = "accumulated",
) -> np.typing.NDArray:
    """Downscale centroids in dataframe.

    Args:
        centroids: Centroids of SGN segmentation, ndarray of shape (N, 3)
        scale_factor: Factor for downscaling coordinates.
        ref_dimensions: Reference dimensions for downscaling. Taken from centroids if not supplied.
        component_labels: List of component labels, which has to be supplied for the downsampling mode 'components'
        downsample_mode: Flag for downsampling, either 'accumulated', 'capped', or 'components'.

    Returns:
        The downscaled array
    """
    centroids_scaled = [(c[0] / scale_factor, c[1] / scale_factor, c[2] / scale_factor) for c in centroids]

    if ref_dimensions is None:
        bounding_dimensions = (max([c[0] for c in centroids]),
                               max([c[1] for c in centroids]),
                               max([c[2] for c in centroids]))
        bounding_dimensions_scaled = tuple([round(b // scale_factor + 1) for b in bounding_dimensions])
        new_array = np.zeros(bounding_dimensions_scaled)

    else:
        bounding_dimensions_scaled = tuple([round(b // scale_factor + 1) for b in ref_dimensions])
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
    length_components = [len(c) for c in components]
    length_components, components = zip(*sorted(zip(length_components, components), reverse=True))

    return components, graph


def components_sgn(
    table: pd.DataFrame,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
) -> List[List[int]]:
    """Eroding the SGN segmentation.

    Args:
        table: Dataframe of segmentation table.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        Subgraph components as lists of label_ids of dataframe.
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    labels = [int(i) for i in list(table["label_id"])]
    coords = {}
    for index, element in zip(labels, centroids):
        coords[index] = element

    components, _ = graph_connected_components(coords, max_edge_distance, min_component_length)

    return components


def label_components_sgn(
    table: pd.DataFrame,
    min_size: int = 1000,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
) -> List[int]:
    """Label SGN components using graph connected components.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        List of component label for each point in dataframe. 0 - background, then in descending order of size
    """

    # First, apply the size filter.
    entries_filtered = table[table.n_pixels < min_size]
    table = table[table.n_pixels >= min_size]

    components = components_sgn(table, min_component_length=min_component_length,
                                max_edge_distance=max_edge_distance)

    # add size-filtered objects to have same initial length
    table = pd.concat([table, entries_filtered], ignore_index=True)
    table.sort_values("label_id")

    component_labels = [0 for _ in range(len(table))]
    table.loc[:, "component_labels"] = component_labels
    # be aware of 'label_id' of dataframe starting at 1
    for lab, comp in enumerate(components):
        table.loc[table["label_id"].isin(comp), "component_labels"] = lab + 1

    return table


def components_ihc(
    table: pd.DataFrame,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
):
    """Create connected components for IHC segmentation.

    Args:
        table: Dataframe of segmentation table.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.

    Returns:
        Subgraph components as lists of label_ids of dataframe.
    """
    centroids = list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))
    labels = [int(i) for i in list(table["label_id"])]
    coords = {}
    for index, element in zip(labels, centroids):
        coords[index] = element

    components, _ = graph_connected_components(coords, max_edge_distance, min_component_length)
    return components


def label_components_ihc(
    table: pd.DataFrame,
    min_size: int = 1000,
    min_component_length: int = 50,
    max_edge_distance: float = 30,
    min_nn_100_distance: float = None,
) -> List[int]:
    """Label components using graph connected components.

    Args:
        table: Dataframe of segmentation table.
        min_size: Minimal number of pixels for filtering small instances.
        min_component_length: Minimal length for filtering out connected components.
        max_edge_distance: Maximal distance in micrometer between points to create edges for connected components.
        min_nn_100_distance: Minimal value for average distance to 100 nearest neighbors.

    Returns:
        List of component label for each point in dataframe. 0 - background, then in descending order of size
    """

    # First, apply the size filter.
    entries_filtered = table[table.n_pixels < min_size]
    table = table[table.n_pixels >= min_size]

    keyword = "distance_nn100"
    if min_nn_100_distance is not None:
        distance_avg = nearest_neighbor_distance(table, n_neighbors=100)
        table.loc[:, keyword] = list(distance_avg)
        entries_filtered_01 = table[table[keyword] < min_nn_100_distance]
        table = table[table[keyword] >= min_nn_100_distance]
        entries_filtered = pd.concat([entries_filtered, entries_filtered_01], ignore_index=True)

    components = components_ihc(table, min_component_length=min_component_length,
                                max_edge_distance=max_edge_distance)

    # add size-filtered objects to have same initial length
    table = pd.concat([table, entries_filtered], ignore_index=True)
    table.sort_values("label_id")

    length_components = [len(c) for c in components]
    length_components, components = zip(*sorted(zip(length_components, components), reverse=True))

    component_labels = [0 for _ in range(len(table))]
    table.loc[:, "component_labels"] = component_labels
    # be aware of 'label_id' of dataframe starting at 1
    for lab, comp in enumerate(components):
        table.loc[table["label_id"].isin(comp), "component_labels"] = lab + 1

    return table


def dilate_and_trim(
    arr_orig: np.ndarray,
    edt: np.ndarray,
    iterations: int = 15,
    offset: float = 0.4,
) -> np.ndarray:
    """Dilate and trim original binary array according to a
    Euclidean Distance Trasform computed for a separate target array.

    Args:
        arr_orig: Original 3D binary array
        edt: 3D array containing Euclidean Distance transform for guiding dilation
        iterations: Number of iterations for dilations
        offset: Offset for regulating dilation. value should be in range(0, 0.45)

    Returns:
        Dilated binary array
    """
    border_coords = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(iterations):
        arr_dilated = binary_dilation(arr_orig)
        for x in range(arr_dilated.shape[0]):
            for y in range(arr_dilated.shape[1]):
                for z in range(arr_dilated.shape[2]):
                    if arr_dilated[x, y, z] != 0:
                        if arr_orig[x, y, z] == 0:
                            min_dist = float('inf')
                            for dx, dy, dz in border_coords:
                                nx, ny, nz = x+dx, y+dy, z+dz
                                if arr_orig[nx, ny, nz] == 1:
                                    min_dist = min([min_dist, edt[nx, ny, nz]])
                            if edt[x, y, z] >= min_dist - offset:
                                arr_dilated[x, y, z] = 0
        arr_orig = arr_dilated
    return arr_dilated


def filter_cochlea_volume_single(
    table: pd.DataFrame,
    components: Optional[List[int]] = [1],
    scale_factor: int = 48,
    resolution: float = 0.38,
    dilation_iterations: int = 12,
    padding: int = 1200,
) -> np.ndarray:
    """Filter cochlea volume based on a segmentation table.
    Centroids contained in the segmentation table are used to create a down-scaled binary array.
    The array can be dilated.

    Args:
        table: Segmentation table.
        components: Component labels for filtering segmentation table.
        scale_factor: Down-sampling factor for filtering.
        resolution: Resolution of pixel in µm.
        dilation_iterations: Iterations for dilating binary segmentation mask. A negative value omits binary closing.
        padding: Padding in pixel to apply to guessed dimensions based on centroid coordinates.

    Returns:
        Binary 3D array of filtered cochlea.
    """
    # filter components
    if components is not None:
        table = table[table["component_labels"].isin(components)]

    # identify approximate input dimensions for down-scaling
    centroids = list(zip(table["anchor_x"] / resolution,
                         table["anchor_y"] / resolution,
                         table["anchor_z"] / resolution))

    # padding the array allows for dilation without worrying about array borders
    max_x = table["anchor_x"].max() / resolution + padding
    max_y = table["anchor_y"].max() / resolution + padding
    max_z = table["anchor_z"].max() / resolution + padding
    ref_dimensions = (max_x, max_y, max_z)

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
    scale_factor: int = 48,
    resolution: float = 0.38,
    dilation_iterations: int = 12,
    padding: int = 1200,
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
        scale_factor: Down-sampling factor for filtering.
        resolution: Resolution of pixel in µm.
        dilation_iterations: Iterations for dilating binary segmentation mask.
        padding: Padding in pixel to apply to guessed dimensions based on centroid coordinates.
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
    centroids_sgn = list(zip(sgn_table["anchor_x"] / resolution,
                             sgn_table["anchor_y"] / resolution,
                             sgn_table["anchor_z"] / resolution))
    centroids_ihc = list(zip(ihc_table["anchor_x"] / resolution,
                             ihc_table["anchor_y"] / resolution,
                             ihc_table["anchor_z"] / resolution))

    # padding the array allows for dilation without worrying about array borders
    max_x = max([sgn_table["anchor_x"].max(), ihc_table["anchor_x"].max()]) / resolution + padding
    max_y = max([sgn_table["anchor_y"].max(), ihc_table["anchor_y"].max()]) / resolution + padding
    max_z = max([sgn_table["anchor_z"].max(), ihc_table["anchor_z"].max()]) / resolution + padding
    ref_dimensions = (max_x, max_y, max_z)

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
        combined_dilated = arr_dilated + ihc_dilated + sgn_dilated
        combined_dilated[combined_dilated > 0] = 1

    return combined_dilated


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


def label_components_single(
    table_path: str,
    out_path: Optional[str] = None,
    force_overwrite: bool = False,
    cell_type: str = "sgn",
    component_list: List[int] = [1],
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
        component_list: List of components. Can be passed to obtain the number of instances within the component list.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        min_component_length: Minimal length of nodes of connected component. Filtered out if lower.
        min_size: Minimal number of pixels for filtering small instances.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
        custom_dic: Custom dictionary which allows multiple post-processing configurations and combines the
            results into final components.
        use_napari: Visualize component labels with napari viewer.
        scale_factor: Scale factor for down-scaling data for visualization in Napari.
    """
    # overwrite input segmentation table with labeled version
    if out_path is None:
        if s3:
            raise ValueError("Set an output path when accessing remote data.")
        out_path = table_path
        force_overwrite = True

    if os.path.isdir(out_path):
        raise ValueError(f"Output path {out_path} is a directory. Provide a path to a single output file.")

    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table = pd.read_csv(table_path, sep="\t")

    # overwrite input file
    if os.path.realpath(out_path) == os.path.realpath(table_path) and not s3:
        force_overwrite = True

    if os.path.isfile(out_path) and not force_overwrite:
        print(f"Skipping {out_path}. Table already exists.")

    else:
        if custom_dic is not None:
            # use multiple post-processing configurations
            tsv_table = label_custom_components(table, custom_dic)
        else:
            if cell_type == "sgn":
                tsv_table = label_components_sgn(table, min_size=min_size,
                                                 min_component_length=min_component_length,
                                                 max_edge_distance=max_edge_distance)
            elif cell_type == "ihc":
                tsv_table = label_components_ihc(table, min_size=min_size,
                                                 min_component_length=min_component_length,
                                                 max_edge_distance=max_edge_distance)
            else:
                raise ValueError("Choose a supported cell type. Either 'sgn' or 'ihc'.")

        custom_comp = len(tsv_table[tsv_table["component_labels"].isin(component_list)])
        print(f"Total {cell_type.upper()}s: {len(tsv_table)}")
        if component_list == [1]:
            print(f"Largest component has {custom_comp} {cell_type.upper()}s.")
        else:
            for comp in component_list:
                num_instances = len(tsv_table[tsv_table["component_labels"] == comp])
                print(f"Component {comp} has {num_instances} instances.")
            print(f"Custom component(s) have {custom_comp} {cell_type.upper()}s.")

        tsv_table.to_csv(out_path, sep="\t", index=False)

        if use_napari:
            import napari
            centroids = list(zip(tsv_table["anchor_x"], tsv_table["anchor_y"], tsv_table["anchor_z"]))
            component_labels = list(tsv_table["component_labels"])
            array_downscaled = downscaled_centroids(centroids=centroids, scale_factor=scale_factor,
                                                    component_labels=component_labels, downsample_mode="components")
            image_downscaled = downscaled_centroids(centroids, scale_factor=scale_factor,
                                                    downsample_mode="accumulated")
            viewer = napari.Viewer()
            viewer.add_image(image_downscaled, name='3D Volume')
            viewer.add_labels(array_downscaled, name="components")
            napari.run()
