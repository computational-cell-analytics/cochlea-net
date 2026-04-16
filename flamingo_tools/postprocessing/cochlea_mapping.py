import json
import math
import os
from itertools import combinations
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
from scipy.interpolate import interp1d

from flamingo_tools.postprocessing.label_components import downscaled_centroids
from flamingo_tools.s3_utils import get_s3_path


def path_dict_to_central_path_table(path_dict):
    central_path_dict = []
    for key, item in path_dict.items():
        dict_tmp = {}
        dict_tmp["spot_id"] = key
        pos = item["pos"]
        dict_tmp["x"] = int(round(pos[0]))
        dict_tmp["y"] = int(round(pos[1]))
        dict_tmp["z"] = int(round(pos[2]))
        dict_tmp["length_fraction"] = item["length_fraction"]
        dict_tmp["length[µm]"] = item["length[µm]"]
        dict_tmp["frequency[kHz]"] = item["frequency[kHz]"]
        central_path_dict.append(dict_tmp)

    return pd.DataFrame(central_path_dict)


def central_path_table_to_path_dict(central_path_df):
    path_dict = {}
    for _, row in central_path_df.iterrows():
        ddict = {}
        ddict["pos"] = (row["x"], row["y"], row["z"])
        ddict["length_fraction"] = row["length_fraction"]
        ddict["length[µm]"] = row["length[µm]"]
        ddict["frequency[kHz]"] = row["frequency[kHz]"]
        spot_id = int(row["spot_id"])
        path_dict[spot_id] = ddict
    return path_dict


def find_most_distant_nodes(G: nx.classes.graph.Graph, weight: str = 'weight') -> Tuple[float, float]:
    """Find the most distant nodes in a graph.

    Args:
        G: Input graph.

    Returns:
        Node 1.
        Node 2.
    """
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight=weight))
    max_dist = 0
    farthest_pair = (None, None)

    for u, dist_dict in all_lengths.items():
        for v, d in dist_dict.items():
            if d > max_dist:
                max_dist = d
                farthest_pair = (u, v)

    u, v = farthest_pair
    return u, v


def central_path_edt_graph(
    mask: np.ndarray,
    start: Tuple[int],
    end: Tuple[int],
) -> Optional[np.ndarray]:
    """Find the central path within a binary mask between a start and an end coordinate.

    Args:
        mask: Binary mask of volume.
        start: Starting coordinate.
        end: End coordinate.

    Returns:
        Coordinates of central path or None if no path exists.
    """
    dt = distance_transform_edt(mask)
    G = nx.Graph()
    shape = mask.shape

    def idx_to_node(z, y, x):
        return z * shape[1] * shape[2] + y * shape[2] + x

    border_coords = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if not mask[z, y, x]:
                    continue
                u = idx_to_node(z, y, x)
                for dz, dy, dx in border_coords:
                    nz, ny, nx_ = z + dz, y + dy, x + dx
                    if nz >= 0 and nz < shape[0] and mask[nz, ny, nx_]:
                        v = idx_to_node(nz, ny, nx_)
                        w = 1.0 / (1e-3 + min(dt[z, y, x], dt[nz, ny, nx_]))
                        G.add_edge(u, v, weight=w)
    s = idx_to_node(*start)
    t = idx_to_node(*end)
    if not nx.has_path(G, source=s, target=t):
        return None
    path = nx.shortest_path(G, source=s, target=t, weight="weight")
    coords = [(p // (shape[1] * shape[2]), (p // shape[2]) % shape[1], p % shape[2]) for p in path]
    return np.array(coords)


def moving_average_3d(path: np.ndarray, window: int = 3) -> np.ndarray:
    """Smooth a 3D path with a simple moving average filter.

    Args:
        path: ndarray of shape (N, 3).
        window: half-window size; actual window = 2*window + 1.

    Returns:
        smoothed path: ndarray of same shape.
    """
    if not isinstance(path, np.ndarray):
        path = np.array(path)
    kernel_size = 2 * window + 1
    kernel = np.ones(kernel_size) / kernel_size

    smooth_path = np.zeros_like(path)

    for d in range(3):
        pad = np.pad(path[:, d], window, mode='edge')
        smooth_path[:, d] = np.convolve(pad, kernel, mode='valid')

    return smooth_path


def extend_path(
    path_pos,
    extension_length: Optional[float] = None,
    max_edge_distance: float = 30,
):
    """Extend path of nodes by adding additional points inbetween.
    Additional nodes are added linearly between existing ones in regular intervals.
    The interval can be specified using the extension length.

    Args:
        path_pos: List of coordinates in 3D space.
        extension_length: Length for the extension between existing nodes
        max_edge_distance: Maximal distance between two nodes which are viable for extension.

    Returns:
        Extended coordinates
    """
    if extension_length is not None:
        print(f"Extending nodes every {extension_length}µm.")
        path_extended = []
        for num in range(len(path_pos) - 1):
            dist = math.dist(path_pos[num], path_pos[num + 1])
            # section between two components
            if dist > max_edge_distance:
                path_extended.append(path_pos[num])
                continue
            extended_space = np.linspace(path_pos[num], path_pos[num + 1], round(dist / extension_length))
            path_extended.extend(extended_space[:-1])
        path_extended.append(path_pos[-1])
        return np.array(path_extended)
    else:
        return np.array(path_pos)


def measure_run_length_sgns(
        centroids_components: List[np.ndarray],
        scale_factor: int = 10,
        apex_higher: bool = True,
        include_gap: bool = False,
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the SGN segmentation by finding a central path through Rosenthal's canal.
    This function handles cases with a single or multiple components.
    The List of centroids has to be in order of neighboring components.
    For each component:
    1) Process centroids of each component:
        a) Create a binary mask based on down-scaled centroids.
        b) Dilate the mask and close holes to ensure a filled structure.
        c) Determine the endpoints of the structure using the principal axis.
        d) Identify a central path based on the 3D Euclidean distance transform.
        e) The path is up-scaled and smoothed using a moving average filter.
    2) Order paths to have consistent start/end points, e.g.
        [[start_c1, ..., end_c1], [end_c2, ..., start_c2]] --> [[start_c1, ..., end_c1], [start_c2, ..., end_c2]]
    3) Assign base/apex position to path.
    4) Assign distance of nodes by skipping intermediate space between separate components.
        Points of path wit their position and fractional length are stored in a dictionary.
    5) Concatenate individual paths to form total path

    Args:
        centroids_components: List of centroids of the SGN segmentation, ndarray of shape (N, 3).
        scale_factor: Downscaling factor for finding the central path.
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.
        include_gap: Include the distance between different components for calculating the run length.

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    total_path = []
    print(f"Evaluating {len(centroids_components)} component(s).")

    def _find_central_path_through_downscaled_mask(centroids, downscaling_factor):
        # Check for the existence of and the shortest path between the most distant nodes in the downscaled volume.
        mask = downscaled_centroids(centroids, scale_factor=downscaling_factor, downsample_mode="capped")
        mask = binary_dilation(mask, np.ones((3, 3, 3)), iterations=1)
        mask = binary_closing(mask, np.ones((3, 3, 3)), iterations=1)
        pts = np.argwhere(mask == 1)

        # find two endpoints: min/max along principal axis
        c_mean = pts.mean(axis=0)
        cov = np.cov((pts - c_mean).T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, np.argmax(evals)]
        proj = (pts - c_mean) @ axis
        start_voxel = tuple(pts[proj.argmin()])
        end_voxel = tuple(pts[proj.argmax()])

        # get central path and total distance
        return central_path_edt_graph(mask, start_voxel, end_voxel)

    # 1) Process centroids for each component
    for centroids in centroids_components:
        # get central path and total distance
        path = _find_central_path_through_downscaled_mask(centroids, downscaling_factor=scale_factor)

        # Use larger downscaling factor to have connected components in downscaled volume.
        while path is None:
            scale_factor = 2 * scale_factor
            print(f"Not all components are fully connected in downscaled volume. Trying scale factor {scale_factor}.")
            if scale_factor > 100:
                raise ValueError("Downscaling for tonotopic mapping not possible.")
            path = _find_central_path_through_downscaled_mask(centroids, downscaling_factor=scale_factor)

        path = path * scale_factor
        path = moving_average_3d(path, window=3)
        total_path.append(path)

    # 2) Order paths to have consistent start/end points
    if len(total_path) > 1:
        # Find starting order of first two components
        c1a = total_path[0][0, :]
        c1b = total_path[0][-1, :]

        c2a = total_path[1][0, :]
        c2b = total_path[1][-1, :]

        distances = [math.dist(c1a, c2a), math.dist(c1a, c2b), math.dist(c1b, c2a), math.dist(c1b, c2b)]
        min_index = distances.index(min(distances))
        if min_index in [0, 1]:
            total_path[0] = np.flip(total_path[0], axis=0)

        # Order other components from start to end
        for num in range(0, len(total_path) - 1):
            dist_connecting_nodes_1 = math.dist(total_path[num][-1, :], total_path[num + 1][0, :])
            dist_connecting_nodes_2 = math.dist(total_path[num][-1, :], total_path[num + 1][-1, :])
            if dist_connecting_nodes_2 < dist_connecting_nodes_1:
                total_path[num + 1] = np.flip(total_path[num + 1], axis=0)

    # 3) Assign base/apex position to path
    # compare y-value to not get into confusion with MoBIE dimensions
    if total_path[0][0, 1] > total_path[-1][-1, 1]:
        if not apex_higher:
            total_path.reverse()
            total_path = [np.flip(t) for t in total_path]
    elif apex_higher:
        total_path.reverse()
        total_path = [np.flip(t) for t in total_path]

    # 4) Assign distance of nodes by skipping intermediate space between separate components
    if include_gap:
        # flatten the list of components
        total_path = [np.array([node for comp_path in total_path for node in comp_path])]
    total_distance = sum([math.dist(p[num + 1], p[num]) for p in total_path for num in range(len(p) - 1)])
    print(f"The total path has length {round(total_distance)} µm. Gaps between components included: {include_gap}.")

    path_dict = {}
    accumulated = 0
    index = 0
    for num, pa in enumerate(total_path):
        # first node of first path segment
        if num == 0:
            path_dict[0] = {"pos": total_path[0][0], "length_fraction": 0}
        # first node of other path segments
        else:
            path_dict[index] = {"pos": total_path[num][0], "length_fraction": path_dict[index - 1]["length_fraction"]}

        index += 1
        for enum, p in enumerate(pa[1:]):
            distance = math.dist(total_path[num][enum], p)
            accumulated += distance
            rel_dist = accumulated / total_distance
            path_dict[index] = {"pos": p, "length_fraction": rel_dist}
            index += 1
    # add last node of last path segment
    path_dict[index - 1] = {"pos": total_path[-1][-1, :], "length_fraction": 1}

    # 5) Concatenate individual paths to form total path
    path = np.concatenate(total_path, axis=0)

    return total_distance, path, path_dict


def minimal_connection_distance(points):
    """Find minimal distance threshold of a set of 3D coordinates.
    The function creates a minimal spanning tree using networkx and then checks for the maximal edge length.

    Args:
        points: List of (x, y, z) tuples.

    Returns:
        minimal distance threshold required to make the graph connected
    """
    G = nx.Graph()
    for i, p in enumerate(points):
        G.add_node(i, pos=p)

    # add all pairwise edges with euclidean distance
    for (i, p1), (j, p2) in combinations(enumerate(points), 2):
        dist = math.dist(p1, p2)
        G.add_edge(i, j, weight=dist)

    # compute minimum spanning tree
    mst = nx.minimum_spanning_tree(G, weight="weight")
    max_edge = max(data["weight"] for _, _, data in mst.edges(data=True))

    return max_edge


def measure_run_length_ihcs(
    centroids: np.ndarray,
    centroids_components: Optional[List[np.ndarray]] = None,
    max_edge_distance: float = 30,
    apex_higher: bool = True,
    component_label: List[int] = [1],
    include_gap: bool = False,
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the IHC segmentation
    by determining the shortest path between the most distant nodes of a graph.
    The graph is created based on the maximal edge distance between nodes.

    If the graph consists of more than one connected component, a list of component labels must be supplied.
    The components are then connected with edges between nodes of neighboring components which are closest together.
    Gaps between individual components are ignored and do not count towards the path length.

    Args:
        centroids: Centroids of IHC segmentation.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.
        component_label: List of component labels. Determines the order of components to connect.
        include_gap: Include the distance between different components for calculating the run length.

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    if centroids_components is None:
        centroids_components = [centroids]

    graph = nx.Graph()
    coords = {}
    labels = [int(i) for i in range(len(centroids))]
    for index, element in zip(labels, centroids):
        coords[index] = element

    for num, pos in coords.items():
        graph.add_node(num, pos=pos)

    max_edge_distance = 0
    for centroids_comp in centroids_components:
        min_connect_dist = minimal_connection_distance(centroids_comp)
        max_edge_distance = max([max_edge_distance, min_connect_dist])

    max_edge_distance = round(max_edge_distance + 0.5)
    print("Automatically determined max edge distance", max_edge_distance)
    # create edges between points whose distance is less than threshold max_edge_distance
    for num_i, pos_i in coords.items():
        for num_j, pos_j in coords.items():
            if num_i < num_j:
                dist = math.dist(pos_i, pos_j)
                if dist <= max_edge_distance:
                    graph.add_edge(num_i, num_j, weight=dist)

    components = [list(c) for c in nx.connected_components(graph)]
    len_c = [len(c) for c in components]
    len_c, components = zip(*sorted(zip(len_c, components), reverse=True))

    # combine separate connected components by adding edges between nodes which are closest together
    if len(components) > 1:
        print(f"Graph consists of {len(components)} connected components.")
        if len(component_label) != len(components):
            raise ValueError(f"Length of graph components {len(components)} "
                             f"does not match number of component labels {len(component_label)}.")

        # Order connected components in order of component labels
        # e.g. component_labels = [7, 4, 1, 11] and len_c = [600, 400, 300, 55]
        # get re-ordered to [300, 400, 600, 55]
        # TODO: Find more robust way to sort components
        components_sorted = [
            c[1] for _, c in sorted(zip(sorted(range(len(component_label)), key=lambda i: component_label[i]),
                                        sorted(zip(len_c, components), key=lambda x: x[0], reverse=True)))]

        # Connect nodes of neighboring components that are closest together
        for num in range(0, len(components_sorted) - 1):
            min_dist = float("inf")
            closest_pair = None

            # Compare only nodes between two neighboring components
            for node_a in components_sorted[num]:
                for node_b in components_sorted[num + 1]:
                    dist = math.dist(graph.nodes[node_a]["pos"], graph.nodes[node_b]["pos"])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (node_a, node_b)
            graph.add_edge(closest_pair[0], closest_pair[1], weight=min_dist)

        print("Connect components in order of component labels.")

    start_node, end_node = find_most_distant_nodes(graph)

    # compare y-value to not get into confusion with MoBIE dimensions
    if graph.nodes[start_node]["pos"][1] > graph.nodes[end_node]["pos"][1]:
        apex_node = start_node if apex_higher else end_node
        base_node = end_node if apex_higher else start_node
    else:
        apex_node = end_node if apex_higher else start_node
        base_node = start_node if apex_higher else end_node

    path = nx.shortest_path(graph, source=apex_node, target=base_node)
    total_distance = nx.path_weight(graph, path, weight="weight")
    path_pos = [graph.nodes[p]["pos"] for p in path]
    if not include_gap:
        # remove distances between components which are larger than the max edge distance
        total_distance = sum([math.dist(path_pos[num], path_pos[num + 1]) for num in range(len(path_pos) - 1) if
                              math.dist(path_pos[num], path_pos[num + 1]) <= max_edge_distance])
    else:
        total_distance = sum([math.dist(path_pos[num], path_pos[num + 1]) for num in range(len(path_pos) - 1)])

    print(f"The total path has length {round(total_distance)} µm. Gaps between components included: {include_gap}.")
    path = moving_average_3d(path_pos, window=3)

    # assign relative distance to points on path
    path_dict = {}
    path_dict[0] = {"pos": path_pos[0], "length_fraction": 0}
    accumulated = 0
    for num, p in enumerate(path_pos[1:-1]):
        distance = math.dist(path_pos[num], p)
        if include_gap or distance <= max_edge_distance:
            accumulated += distance
        rel_dist = accumulated / total_distance
        path_dict[num + 1] = {"pos": p, "length_fraction": rel_dist}
    path_dict[len(path_pos)] = {"pos": path_pos[-1], "length_fraction": 1}

    return total_distance, path, path_dict


def map_frequency(path_dict: dict, animal: str = "mouse", otof: bool = False) -> pd.DataFrame:
    """Map the frequency range of SGNs in the cochlea
    using Greenwood function f(x) = A * (10 **(ax) - K).
    Values for humans: a=2.1, k=0.88, A = 165.4 [kHz].
    For mice: fit values between minimal (1kHz) and maximal (80kHz) values

    Args:
        table: Dataframe containing the segmentation.
        animal: Select the Greenwood function parameters specific to a species. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.

    Returns:
        Dataframe containing frequency in an additional column 'frequency[kHz]'.
    """
    if otof and animal == "mouse":
        # freq_min = 4.84 kHz
        # freq_max = 78.8 kHz
        # Mueller, Hearing Research 202 (2005) 63-73, https://doi.org/10.1016/j.heares.2004.08.011
        # function has format f(x) = 10 ** (a * (k - (1-x)))
        var_a = 100 / 82.5
        var_k = 1.565
        var_A = 1
        # bring it into same format as previous equation:
        # f(x) = 10 ** (a * (k - (1-x)))
        # f(x) = 10 ** (ax - (-a * (k-1)))
        # f(x) = 10 ** (ax - c) with c = (-a * (k-1))
        var_a = 100 / 82.5
        var_k = -0.684848485
        var_A = 1

    elif animal == "mouse":
        # freq_min = 1.5 kHz
        # freq_max = 86 kHz
        # ou bohne 2000 Hear res, "EDGES"
        var_A = 1.46
        var_a = 1.77
        var_k = 0

    elif animal == "gerbil":
        # freq_min = 0.0105 kHz
        # freq_max = 43.82 kHz
        # values used by keppeler, PNAS 2021 Vol. 118 No. 18, https://doi.org/10.1073/pnas.2014472118
        var_A = 0.35
        var_a = 2.1
        var_k = 0.7

    else:
        raise ValueError("Animal not supported. Use either 'mouse' or 'gerbil'.")

    for key in path_dict.keys():
        path_dict[key]["frequency[kHz]"] = var_A * (10 ** (var_a * path_dict[key]["length_fraction"]) - var_k)

    return path_dict


def get_centers_from_path(
    path: np.ndarray,
    total_distance: float,
    n_blocks: int = 10,
    offset_blocks: bool = True,
) -> List[float]:
    """Get equidistant centers from the central path (not restricted to node location).

    Args:
        path: Central path through Rosenthal's canal.
        total_distance: Length of the path.
        n_blocks: Number of equidistant centers for block creation.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.

    Returns:
        Equidistant centers.
    """
    diffs = np.diff(path, axis=0)
    seg_lens = np.linalg.norm(diffs, axis=1)
    cum_len = np.insert(np.cumsum(seg_lens), 0, 0)
    if offset_blocks:
        target_s = np.linspace(0, total_distance, n_blocks * 2 + 1)
        target_s = [s for num, s in enumerate(target_s) if num % 2 == 1]
    else:
        target_s = np.linspace(0, total_distance, n_blocks)
    try:
        f = interp1d(cum_len, path, axis=0)  # fill_value="extrapolate"
        centers = f(target_s)
    except ValueError:
        print("Using extrapolation to fill values.")
        f = interp1d(cum_len, path, axis=0, fill_value="extrapolate")
        centers = f(target_s)
    # TODO figure out why exactly coordinates need to be flipped
    centers = [np.flip(c) for c in centers]
    return centers


def get_centers_from_path_dict(
    path_dict: dict,
    n_blocks: int = 10,
    offset_blocks: bool = True,
) -> List[float]:
    """Get equidistant centers from a dictionary of nodes on the central path.

    Args:
        path_dict: Dictionary containing position and length fraction of nodes on the central path.
        n_blocks: Number of equidistant centers for block creation.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.

    Returns:
        Equidistant centers.
    """
    if offset_blocks:
        target_s = np.linspace(0, 1, n_blocks * 2 + 1)
        target_s = [s for num, s in enumerate(target_s) if num % 2 == 1]
    else:
        target_s = np.linspace(0, 1, n_blocks)

    # find node on path with length fraction closest to target value
    centers = []
    for target in target_s:
        min_dist = float('inf')
        nearest_node = None
        for key in list(path_dict.keys()):
            dist = abs(target - path_dict[key]["length_fraction"])
            if dist < min_dist:
                min_dist = dist
                nearest_node = key
        centers.append(path_dict[nearest_node]["pos"])
    centers = [np.flip(c) for c in centers]
    return centers


def node_dict_from_path_dict(
    path_dict: dict,
    label_ids: List[int],
    centroids: np.ndarray,
) -> dict:
    """Get dictionary for all nodes from dictionary of nodes on the central path.

    Args:
        path_dict: Dictionary containing position and length fraction of nodes on the central path.
        label_ids: Label IDs of all nodes/instance segmentations.
        centroids: Position of nodes/instance segmentations.

    Returns:
        Dictionary containing all nodes from the graph.
    """
    # add missing nodes from component and compute distance to path
    node_dict = {}
    for num, c in enumerate(label_ids):
        min_dist = float('inf')
        nearest_node = None

        for key in path_dict.keys():
            dist = math.dist(centroids[num], path_dict[key]["pos"])
            if dist < min_dist:
                min_dist = dist
                nearest_node = key

        node_dict[c] = {
            "label_id": c,
            "length_fraction": path_dict[nearest_node]["length_fraction"],
            "length[µm]": path_dict[nearest_node]["length[µm]"],
            "pos": path_dict[nearest_node]["pos"],
            "frequency[kHz]": path_dict[nearest_node]["frequency[kHz]"],
            "offset": min_dist,
        }
    return node_dict


def equidistant_centers(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    cell_type: str = "sgn",
    n_blocks: int = 10,
    offset_blocks: bool = True,
    include_gap: bool = False,
) -> np.ndarray:
    """Find equidistant centers within the central path of the Rosenthal's canal.

    Args:
        table: Dataframe containing centroids of SGN segmentation.
        component_label: List of components for centroid subset.
        cell_type: Cell type of the segmentation.
        n_blocks: Number of equidistant centers for block creation.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.
        include_gap: Include the distance between different components for calculating the run length.

    Returns:
        Equidistant centers as float values
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))

    if cell_type == "ihc":
        total_distance, path, path_dict = measure_run_length_ihcs(
            centroids, component_label=component_label, include_gap=include_gap,
        )
        return get_centers_from_path_dict(path_dict, total_distance, n_blocks=n_blocks, offset_blocks=offset_blocks)

    else:
        centroids_components = []
        for label in component_label:
            subset = table[table["component_labels"] == label]
            subset_centroids = list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"]))
            centroids_components.append(subset_centroids)
        total_distance, path, path_dict = measure_run_length_sgns(centroids_components, include_gap=include_gap)
        return get_centers_from_path_dict(path_dict, n_blocks=n_blocks, offset_blocks=offset_blocks)


def tonotopic_mapping(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    component_mapping: Optional[List[int]] = None,
    cell_type: str = "ihc",
    animal: str = "mouse",
    apex_higher: bool = True,
    otof: bool = False,
    central_path_df: Optional[pd.DataFrame] = None,
    include_gap: bool = False,
) -> pd.DataFrame:
    """Tonotopic mapping of SGNs or IHCs by supplying a table with component labels.
    The mapping assigns a tonotopic label to each instance according to the position along the length of the cochlea.

    Args:
        table: Dataframe of segmentation table.
        component_label: List of component labels to evaluate.
        components_mapping: Components to use for tonotopic mapping. Ignore components torn parallel to main canal.
        cell_type: Cell type of segmentation.
        animal: Animal specifier for species specific frequency mapping. Either "mouse" or "gerbil".
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
        central_path_df: Dataframe featuring the spots for the central path through the segmentation.
        include_gap: Include the distance between different components for calculating the run length.

    Returns:
        Table with tonotopic label for cells.
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    label_ids = [int(i) for i in list(new_subset["label_id"])]

    if component_mapping is None:
        component_mapping = component_label

    if central_path_df is None:

        if cell_type in ["ihc", "IHC"]:
            centroids_components = []
            for label in component_mapping:
                subset = table[table["component_labels"] == label]
                subset_centroids = list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"]))
                centroids_components.append(subset_centroids)

            total_distance, _, path_dict = measure_run_length_ihcs(
                centroids, centroids_components, component_label=component_label, apex_higher=apex_higher,
                include_gap=include_gap,
            )

        elif cell_type in ["sgn", "SGN"]:
            centroids_components = []
            for label in component_mapping:
                subset = table[table["component_labels"] == label]
                subset_centroids = list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"]))
                centroids_components.append(subset_centroids)
            total_distance, _, path_dict = measure_run_length_sgns(
                centroids_components, apex_higher=apex_higher,
                include_gap=include_gap,
            )

        else:
            raise ValueError(f"Unrecognized cell type: {cell_type}. Choose either 'sgn' or 'ihc'.")

        for key, items in path_dict.items():
            path_dict[key]["length[µm]"] = items["length_fraction"] * total_distance

        path_dict = map_frequency(path_dict, animal=animal, otof=otof)
        node_dict = node_dict_from_path_dict(path_dict, label_ids, centroids)
        central_path_df = path_dict_to_central_path_table(path_dict)
    else:
        path_dict = central_path_table_to_path_dict(central_path_df)
        path_dict = map_frequency(path_dict, animal=animal, otof=otof)
        node_dict = node_dict_from_path_dict(path_dict, label_ids, centroids)

    offset = [-1 for _ in range(len(table))]
    offset = list(np.float64(offset))
    table.loc[:, "offset"] = offset

    length_fraction = [0 for _ in range(len(table))]
    length_fraction = list(np.float64(length_fraction))
    table.loc[:, "length_fraction"] = length_fraction

    length_abs = [0 for _ in range(len(table))]
    length_abs = list(np.float64(length_abs))
    table.loc[:, "length[µm]"] = length_abs

    frequency = [0 for _ in range(len(table))]
    frequency = list(np.float64(frequency))
    table.loc[:, "frequency[kHz]"] = frequency

    for key in list(node_dict.keys()):
        table.loc[table["label_id"] == key, "offset"] = node_dict[key]["offset"]
        table.loc[table["label_id"] == key, "length_fraction"] = node_dict[key]["length_fraction"]
        table.loc[table["label_id"] == key, "length[µm]"] = node_dict[key]["length[µm]"]
        table.loc[table["label_id"] == key, "frequency[kHz]"] = node_dict[key]["frequency[kHz]"]

    return table, central_path_df


def tonotopic_mapping_single(
    table_path: str,
    out_path: str,
    force_overwrite: bool = False,
    cell_type: str = "sgn",
    animal: str = "mouse",
    otof: bool = False,
    apex_position: str = "apex_higher",
    component_list: List[int] = [1],
    component_mapping: Optional[List[int]] = None,
    central_spots_path: Optional[str] = None,
    include_gap: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **_,
):
    """Tonotopic mapping of a single cochlea.
    Each segmentation instance within a given component list is assigned a frequency[kHz], a run length and an offset.
    The components used for the mapping itself can be a subset of the component list to adapt to broken components
    along the Rosenthal's canal.
    If the cochlea is broken in the direction of the Rosenthal's canal, the components have to be provided in a
    continuous order which reflects the positioning within 3D.
    The frequency is calculated using the Greenwood function using animal specific parameters.
    The orientation of the mapping can be reversed using the apex position in reference to the y-coordinate.

    Args:
        table_path: File path to segmentation table.
        out_path: Output path to segmentation table with new column "component_labels".
        force_overwrite: Forcefully overwrite existing output path.
        cell_type: Cell type of the segmentation. Currently supports "sgn" and "ihc".
        animal: Animal for species specific frequency mapping. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
        apex_position: Identify position of apex and base. Apex is set to node with higher y-value per default.
        component_list: List of components. Can be passed to obtain the number of instances within the component list.
        component_mapping: Components to use for tonotopic mapping. Ignore components torn parallel to main canal.
        central_spots_path: Provide table featuring spots for central path through segmentation for tonotopic mapping.
        include_gap: Include the distance between different components for calculating the run length.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
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

    if central_spots_path is not None and os.path.isfile(central_spots_path):
        central_path_df = pd.read_csv(central_spots_path, sep="\t")
    else:
        central_path_df = None

    apex_higher = (apex_position == "apex_higher")

    # overwrite input file
    if os.path.realpath(out_path) == os.path.realpath(table_path) and not s3:
        force_overwrite = True

    if os.path.isfile(out_path) and not force_overwrite:
        print(f"Skipping {out_path}. Table already exists.")

    else:
        table, central_path_df = tonotopic_mapping(table, component_label=component_list, animal=animal,
                                                   cell_type=cell_type, component_mapping=component_mapping,
                                                   apex_higher=apex_higher,
                                                   central_path_df=central_path_df,
                                                   include_gap=include_gap,
                                                   otof=otof)

        table.to_csv(out_path, sep="\t", index=False)
        if central_spots_path is not None and not os.path.isfile(central_spots_path):
            print("Saving path", central_spots_path)
            central_path_df.to_csv(central_spots_path, sep="\t", index=False)


def equidistant_centers_single(
    table_path: str,
    output_path: str,
    n_blocks: int = 10,
    cell_type: str = "sgn",
    component_list: List[int] = [1],
    offset_blocks: bool = True,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **_,
):
    """Find equidistant centers within the central path of the Rosenthal's canal.

    Args:
        table_path: File path to segmentation table.
        output_path: Output path to JSON file with center coordinates.
        cell_type: Cell type of the segmentation. Currently supports "sgn" and "ihc".
        force_overwrite: Forcefully overwrite existing output path.
        component_list: List of components. Can be passed to obtain the number of instances within the component list.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """
    # overwrite input segmentation table with labeled version
    if output_path is None:
        raise ValueError("Set an output path for the JSON dictionary.")

    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table_path = os.path.realpath(table_path)
        table = pd.read_csv(table_path, sep="\t")

    if os.path.isfile(output_path):
        print(f"Updating parameters in {output_path}.")
        with open(output_path, "r") as f:
            dic = json.load(f)
        dic["n_blocks"] = n_blocks
        dic["cell_type"] = cell_type
        dic["component_list"] = component_list

    else:
        dic = {}
        dic["seg_table"] = table_path
        dic["n_blocks"] = n_blocks
        dic["cell_type"] = cell_type
        dic["component_list"] = component_list

    centers = equidistant_centers(
        table, component_label=component_list, cell_type=cell_type,
        n_blocks=n_blocks, offset_blocks=offset_blocks,
    )
    centers = [[round(c) for c in center] for center in centers]

    dic["crop_centers"] = centers

    with open(output_path, "w") as f:
        json.dump(dic, f, indent='\t', separators=(',', ': '))


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def tonotopic_mapping_json_wrapper(
    out_path: str,
    table_path: Optional[str] = None,
    json_file: Optional[str] = None,
    central_spots_path: Optional[str] = None,
    force_overwrite: bool = False,
    animal: str = "mouse",
    otof: bool = False,
    s3: bool = False,
    **kwargs
):
    """Wrapper function for tonotopic mapping using a segmentation table.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.

    Args:
        output_path: Output path to segmentation table with new column "component_labels".
        table_path: File path to segmentation table.
        json_file: JSON file containing parameters for tonotopic mapping.
        central_spots_path: Provide table featuring spots for central path through segmentation for tonotopic mapping.
        force: Forcefully overwrite existing output path.
        animal: Animal specifier for species specific frequency mapping. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
        s3: Use data path of S3 bucket for segmentation table.
    """
    if json_file is None:
        tonotopic_mapping_single(table_path, out_path=out_path, animal=animal, ototf=otof,
                                 central_spots_path=central_spots_path,
                                 force_overwrite=force_overwrite, s3=s3, **kwargs)
    else:
        if out_path is None:
            raise ValueError("Specify an output path when supplying a JSON dictionary.")
        param_dicts = _load_json_as_list(json_file)
        for params in param_dicts:

            cochlea = params["dataset_name"]
            print(f"\n{cochlea}")
            seg_channel = params["segmentation_channel"]
            table_path = os.path.join(f"{cochlea}", "tables", seg_channel, "default.tsv")

            if "OTOF" in cochlea:
                otof = True
            else:
                otof = False

            if cochlea[0] in ["M", "m"]:
                animal = "mouse"
            elif cochlea[0] in ["G", "g"]:
                animal = "gerbil"
            else:
                animal = "mouse"

            if os.path.isdir(out_path):
                cochlea_str = cochlea.replace('_', '-')
                table_str = seg_channel.replace('_', '-')
                save_path = os.path.join(out_path, "_".join([cochlea_str, f"{table_str}.tsv"]))
                # central_spots_path = os.path.join(out_path, "_".join([cochlea_str, f"{table_str}_path.tsv"]))
            else:
                save_path = out_path

            tonotopic_mapping_single(table_path=table_path, out_path=save_path, animal=animal, otof=otof,
                                     force_overwrite=force_overwrite, central_spots_path=central_spots_path,
                                     s3=s3, **params)
