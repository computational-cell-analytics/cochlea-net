import math
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_closing
from scipy.interpolate import interp1d

from flamingo_tools.segmentation.postprocessing import downscaled_centroids


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


def central_path_edt_graph(mask: np.ndarray, start: Tuple[int], end: Tuple[int]) -> np.ndarray:
    """Find the central path within a binary mask between a start and an end coordinate.

    Args:
        mask: Binary mask of volume.
        start: Starting coordinate.
        end: End coordinate.

    Returns:
        Coordinates of central path.
    """
    dt = distance_transform_edt(mask)
    G = nx.Graph()
    shape = mask.shape
    def idx_to_node(z, y, x): return z*shape[1]*shape[2] + y*shape[2] + x
    border_coords = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for z in range(shape[0]):
        for y in range(shape[1]):
            for x in range(shape[2]):
                if not mask[z, y, x]:
                    continue
                u = idx_to_node(z, y, x)
                for dz, dy, dx in border_coords:
                    nz, ny, nx_ = z+dz, y+dy, x+dx
                    if nz >= 0 and nz < shape[0] and mask[nz, ny, nx_]:
                        v = idx_to_node(nz, ny, nx_)
                        w = 1.0 / (1e-3 + min(dt[z, y, x], dt[nz, ny, nx_]))
                        G.add_edge(u, v, weight=w)
    s = idx_to_node(*start)
    t = idx_to_node(*end)
    path = nx.shortest_path(G, source=s, target=t, weight="weight")
    coords = [(p//(shape[1]*shape[2]),
               (p//shape[2]) % shape[1],
               p % shape[2]) for p in path]
    return np.array(coords)


def moving_average_3d(path: np.ndarray, window: int = 5) -> np.ndarray:
    """Smooth a 3D path with a simple moving average filter.

    Args:
        path: ndarray of shape (N, 3).
        window: half-window size; actual window = 2*window + 1.

    Returns:
        smoothed path: ndarray of same shape.
    """
    kernel_size = 2 * window + 1
    kernel = np.ones(kernel_size) / kernel_size

    smooth_path = np.zeros_like(path)

    for d in range(3):
        pad = np.pad(path[:, d], window, mode='edge')
        smooth_path[:, d] = np.convolve(pad, kernel, mode='valid')

    return smooth_path


def measure_run_length_sgns_multi_component(
        centroids_components: List[np.ndarray],
        scale_factor: int = 10,
        apex_higher: bool = True,
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the SGN segmentation by finding a central path through Rosenthal's canal.
    This function handles the case were the cochlea has been torn into multiple components.
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

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    total_path = []
    print(f"Evaluating {len(centroids_components)} components.")
    # 1) Process centroids for each component
    for centroids in centroids_components:
        mask = downscaled_centroids(centroids, scale_factor=scale_factor, downsample_mode="capped")
        mask = binary_dilation(mask, np.ones((3, 3, 3)), iterations=1)
        mask = binary_closing(mask, np.ones((3, 3, 3)), iterations=1)
        pts = np.argwhere(mask == 1)

        # find two endpoints: min/max along principal axis
        c_mean = pts.mean(axis=0)
        cov = np.cov((pts-c_mean).T)
        evals, evecs = np.linalg.eigh(cov)
        axis = evecs[:, np.argmax(evals)]
        proj = (pts - c_mean) @ axis
        start_voxel = tuple(pts[proj.argmin()])
        end_voxel = tuple(pts[proj.argmax()])

        # get central path and total distance
        path = central_path_edt_graph(mask, start_voxel, end_voxel)
        path = path * scale_factor
        path = moving_average_3d(path, window=5)
        total_path.append(path)

    # 2) Order paths to have consistent start/end points
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
        dist_connecting_nodes_1 = math.dist(total_path[num][-1, :], total_path[num+1][0, :])
        dist_connecting_nodes_2 = math.dist(total_path[num][-1, :], total_path[num+1][-1, :])
        if dist_connecting_nodes_2 < dist_connecting_nodes_1:
            total_path[num+1] = np.flip(total_path[num+1], axis=0)

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
    total_distance = sum([math.dist(p[num + 1], p[num]) for p in total_path for num in range(len(p) - 1)])
    path_dict = {}
    accumulated = 0
    index = 0
    for num, pa in enumerate(total_path):
        if num == 0:
            path_dict[0] = {"pos": total_path[0][0], "length_fraction": 0}
        else:
            path_dict[index] = {"pos": total_path[num][0], "length_fraction": path_dict[index-1]["length_fraction"]}

        index += 1
        for enum, p in enumerate(pa[1:]):
            distance = math.dist(total_path[num][enum], p)
            accumulated += distance
            rel_dist = accumulated / total_distance
            path_dict[index] = {"pos": p, "length_fraction": rel_dist}
            index += 1
    path_dict[index-1] = {"pos": total_path[-1][-1, :], "length_fraction": 1}

    # 5) Concatenate individual paths to form total path
    path = np.concatenate(total_path, axis=0)

    return total_distance, path, path_dict


def measure_run_length_sgns(
        centroids: np.ndarray,
        scale_factor: int = 10,
        apex_higher: bool = True,
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the SGN segmentation by finding a central path through Rosenthal's canal.
    1) Create a binary mask based on down-scaled centroids.
    2) Dilate the mask and close holes to ensure a filled structure.
    3) Determine the endpoints of the structure using the principal axis.
    4) Assign base/apex position to path.
    5) Identify a central path based on the 3D Euclidean distance transform.
    6) The path is up-scaled and smoothed using a moving average filter.
    7) The points of the path are fed into a dictionary along with the fractional length.

    Args:
        centroids: Centroids of the SGN segmentation, ndarray of shape (N, 3).
        scale_factor: Downscaling factor for finding the central path.
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    # 1) Create a binary mask based on down-scaled centroids.
    mask = downscaled_centroids(centroids, scale_factor=scale_factor, downsample_mode="capped")
    # 2) Dilate the mask and close holes to ensure a filled structure.
    mask = binary_dilation(mask, np.ones((3, 3, 3)), iterations=1)
    mask = binary_closing(mask, np.ones((3, 3, 3)), iterations=1)
    pts = np.argwhere(mask == 1)

    # 3) Find two endpoints: min/max along principal axis.
    c_mean = pts.mean(axis=0)
    cov = np.cov((pts-c_mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    proj = (pts - c_mean) @ axis
    start_voxel = tuple(pts[proj.argmin()])
    end_voxel = tuple(pts[proj.argmax()])

    # 4) Assign base/apex position to path.
    # compare y-value to not get into confusion with MoBIE dimensions
    if start_voxel[1] > end_voxel[1]:
        apex = start_voxel if apex_higher else end_voxel
        base = end_voxel if apex_higher else start_voxel
    else:
        apex = end_voxel if apex_higher else start_voxel
        base = start_voxel if apex_higher else end_voxel

    # 5) Identify a central path based on the 3D Euclidean distance transform.
    path = central_path_edt_graph(mask, apex, base)
    # 6) The path is up-scaled and smoothed using a moving average filter.
    path = path * scale_factor
    path = moving_average_3d(path, window=5)
    total_distance = sum([math.dist(path[num + 1], path[num]) for num in range(len(path) - 1)])

    # 7) The points of the path are fed into a dictionary along with the fractional length.
    path_dict = {}
    path_dict[0] = {"pos": path[0], "length_fraction": 0}
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = math.dist(path[num], p)
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_dict[num + 1] = {"pos": p, "length_fraction": rel_dist}
    path_dict[len(path)] = {"pos": path[-1], "length_fraction": 1}

    return total_distance, path, path_dict


def measure_run_length_ihcs_multi_component(
    centroids_components: List[np.ndarray],
    max_edge_distance: float = 30,
    apex_higher: bool = True,
    component_label: List[int] = [1],
) -> Tuple[float, np.ndarray, dict]:
    """Adaptation of measure_run_length_sgns_multi_component to IHCs.

    """
    total_path = []
    print(f"Evaluating {len(centroids_components)} components.")
    # 1) Process centroids for each component
    for centroids in centroids_components:
        graph = nx.Graph()
        coords = {}
        labels = [int(i) for i in range(len(centroids))]
        for index, element in zip(labels, centroids):
            coords[index] = element

        for num, pos in coords.items():
            graph.add_node(num, pos=pos)

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
                                 f"does not match number of component labels {len(component_label)}. "
                                 "Check max_edge_distance and post-processing.")

            # Order connected components in order of component labels
            # e.g. component_labels = [7, 4, 1, 11] and len_c = [600, 400, 300, 55]
            # get re-ordered to [300, 400, 600, 55]
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
        path_pos = np.array([graph.nodes[p]["pos"] for p in path])
        path = moving_average_3d(path_pos, window=5)
        total_path.append(path)

    # 2) Order paths to have consistent start/end points
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
        dist_connecting_nodes_1 = math.dist(total_path[num][-1, :], total_path[num+1][0, :])
        dist_connecting_nodes_2 = math.dist(total_path[num][-1, :], total_path[num+1][-1, :])
        if dist_connecting_nodes_2 < dist_connecting_nodes_1:
            total_path[num+1] = np.flip(total_path[num+1], axis=0)

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
    total_distance = sum([math.dist(p[num + 1], p[num]) for p in total_path for num in range(len(p) - 1)])
    path_dict = {}
    accumulated = 0
    index = 0
    for num, pa in enumerate(total_path):
        if num == 0:
            path_dict[0] = {"pos": total_path[0][0], "length_fraction": 0}
        else:
            path_dict[index] = {"pos": total_path[num][0], "length_fraction": path_dict[index-1]["length_fraction"]}

        index += 1
        for enum, p in enumerate(pa[1:]):
            distance = math.dist(total_path[num][enum], p)
            accumulated += distance
            rel_dist = accumulated / total_distance
            path_dict[index] = {"pos": p, "length_fraction": rel_dist}
            index += 1
    path_dict[index-1] = {"pos": total_path[-1][-1, :], "length_fraction": 1}

    # 5) Concatenate individual paths to form total path
    path = np.concatenate(total_path, axis=0)

    return total_distance, path, path_dict


def measure_run_length_ihcs(
    centroids: np.ndarray,
    max_edge_distance: float = 30,
    apex_higher: bool = True,
    component_label: List[int] = [1],
) -> Tuple[float, np.ndarray, dict]:
    """Measure the run lengths of the IHC segmentation
    by determining the shortest path between the most distant nodes of a graph.
    The graph is created based on a maximal edge distance between nodes.
    Take care, that this value should be identical to the one used to initially process the IHC segmentation.

    If the graph consists of more than one connected components, a list of component labels must be supplied.
    The components are then connected with edges between nodes of neighboring components which are closest together.
    Gaps between individual components are ignored and do not count towards the path length.

    Args:
        centroids: Centroids of IHC segmentation.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.
        component_label: List of component labels. Determines the order of components to connect.

    Returns:
        Total distance of the path.
        Path as an nd.array of positions.
        A dictionary containing the position and the length fraction of each point in the path.
    """
    graph = nx.Graph()
    coords = {}
    labels = [int(i) for i in range(len(centroids))]
    for index, element in zip(labels, centroids):
        coords[index] = element

    for num, pos in coords.items():
        graph.add_node(num, pos=pos)

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
                             f"does not match number of component labels {len(component_label)}. "
                             "Check max_edge_distance and post-processing.")

        # Order connected components in order of component labels
        # e.g. component_labels = [7, 4, 1, 11] and len_c = [600, 400, 300, 55]
        # get re-ordered to [300, 400, 600, 55]
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

    # assign relative distance to points on path
    path_dict = {}
    path_dict[0] = {"pos": graph.nodes[path[0]]["pos"], "length_fraction": 0}
    accumulated = 0
    for num, p in enumerate(path[1:-1]):
        distance = math.dist(graph.nodes[path[num]]["pos"], graph.nodes[p]["pos"])
        accumulated += distance
        rel_dist = accumulated / total_distance
        path_dict[num + 1] = {"pos": graph.nodes[p]["pos"], "length_fraction": rel_dist}
    path_dict[len(path)] = {"pos": graph.nodes[path[-1]]["pos"], "length_fraction": 1}

    path_pos = np.array([graph.nodes[p]["pos"] for p in path])
    path = moving_average_3d(path_pos, window=5)

    return total_distance, path, path_dict


def map_frequency(table: pd.DataFrame, animal: str = "mouse", otof: bool = False) -> pd.DataFrame:
    """Map the frequency range of SGNs in the cochlea
    using Greenwood function f(x) = A * (10 **(ax) - K).
    Values for humans: a=2.1, k=0.88, A = 165.4 [kHz].
    For mice: fit values between minimal (1kHz) and maximal (80kHz) values

    Args:
        table: Dataframe containing the segmentation.
        animal: Select the Greenwood function parameters specific to a species. Either "mouse" or "gerbil".

    Returns:
        Dataframe containing frequency in an additional column 'frequency[kHz]'.
    """
    if animal == "mouse":
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

    table.loc[table['offset'] >= 0, 'frequency[kHz]'] = var_A * (10 ** (var_a * table["length_fraction"]) - var_k)
    table.loc[table['offset'] < 0, 'frequency[kHz]'] = 0

    if otof and animal == "mouse":
        # freq_min = 4.84 kHz
        # freq_max = 78.8 kHz
        # Mueller, Hearing Research 202 (2005) 63–73, https://doi.org/10.1016/j.heares.2004.08.011
        # function has format f(x) = 10 ** (a * (k - (1-x)))
        var_a = 100 / 82.5
        var_k = 1.565
        var_A = 1
        table.loc[table['offset'] >= 0, 'frequency-mueller[kHz]'] = var_A * (10 ** (var_a * (var_k - (1 - table["length_fraction"])))) # noqa
        table.loc[table['offset'] < 0, 'frequency-mueller[kHz]'] = 0

    return table


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
            "pos": path_dict[nearest_node]["length_fraction"],
            "offset": min_dist,
            }
    return node_dict


def equidistant_centers(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    cell_type: str = "sgn",
    n_blocks: int = 10,
    max_edge_distance: float = 30,
    offset_blocks: bool = True,
) -> np.ndarray:
    """Find equidistant centers within the central path of the Rosenthal's canal.

    Args:
        table: Dataframe containing centroids of SGN segmentation.
        component_label: List of components for centroid subset.
        cell_type: Cell type of the segmentation.
        n_blocks: Number of equidistant centers for block creation.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.

    Returns:
        Equidistant centers as float values
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))

    if cell_type == "ihc":
        if len(component_label) == 1:
            total_distance, path, _ = measure_run_length_ihcs(
                centroids, component_label=component_label, max_edge_distance=max_edge_distance
            )
            return get_centers_from_path(path, total_distance, n_blocks=n_blocks, offset_blocks=offset_blocks)
        else:
            centroids_components = []
            for label in component_label:
                subset = table[table["component_labels"] == label]
                subset_centroids = list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"]))
                centroids_components.append(subset_centroids)
            total_distance, path, path_dict = measure_run_length_ihcs_multi_component(
                centroids_components, max_edge_distance=max_edge_distance
            )
            return get_centers_from_path_dict(path_dict, n_blocks=n_blocks, offset_blocks=offset_blocks)

    else:
        if len(component_label) == 1:
            total_distance, path, _ = measure_run_length_sgns(centroids)
            return get_centers_from_path(path,  total_distance, n_blocks=n_blocks, offset_blocks=offset_blocks)

        else:
            centroids_components = []
            for label in component_label:
                subset = table[table["component_labels"] == label]
                subset_centroids = list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"]))
                centroids_components.append(subset_centroids)
            total_distance, path, path_dict = measure_run_length_sgns_multi_component(centroids_components)
            return get_centers_from_path_dict(path_dict, n_blocks=n_blocks, offset_blocks=offset_blocks)


def tonotopic_mapping(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    component_mapping: Optional[List[int]] = None,
    cell_type: str = "ihc",
    animal: str = "mouse",
    max_edge_distance: float = 30,
    apex_higher: bool = True,
    otof: bool = False,
) -> pd.DataFrame:
    """Tonotopic mapping of IHCs by supplying a table with component labels.
    The mapping assigns a tonotopic label to each IHC according to the position along the length of the cochlea.

    Args:
        table: Dataframe of segmentation table.
        component_label: List of component labels to evaluate.
        components_mapping: Components to use for tonotopic mapping. Ignore components torn parallel to main canal.
        cell_type: Cell type of segmentation.

    Returns:
        Table with tonotopic label for cells.
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    label_ids = [int(i) for i in list(new_subset["label_id"])]

    if component_mapping is None:
        component_mapping = component_label

    if cell_type == "ihc":
        total_distance, _, path_dict = measure_run_length_ihcs(
            centroids, component_label=component_label, apex_higher=apex_higher,
            max_edge_distance=max_edge_distance
        )

    else:
        if len(component_mapping) == 1:
            total_distance, _, path_dict = measure_run_length_sgns(
                centroids, apex_higher=apex_higher,
            )

        else:
            centroids_components = []
            for label in component_mapping:
                subset = table[table["component_labels"] == label]
                subset_centroids = list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"]))
                centroids_components.append(subset_centroids)
            total_distance, _, path_dict = measure_run_length_sgns_multi_component(
                centroids_components, apex_higher=apex_higher,
            )

    node_dict = node_dict_from_path_dict(path_dict, label_ids, centroids)

    offset = [-1 for _ in range(len(table))]
    offset = list(np.float64(offset))
    table.loc[:, "offset"] = offset
    # 'label_id' of dataframe starting at 1
    for key in list(node_dict.keys()):
        table.loc[table["label_id"] == key, "offset"] = node_dict[key]["offset"]

    length_fraction = [0 for _ in range(len(table))]
    length_fraction = list(np.float64(length_fraction))
    table.loc[:, "length_fraction"] = length_fraction
    for num, key in enumerate(list(node_dict.keys())):
        table.loc[table["label_id"] == key, "length_fraction"] = node_dict[key]["length_fraction"]

    table.loc[:, "length[µm]"] = table["length_fraction"] * total_distance

    table = map_frequency(table, animal=animal, otof=otof)

    return table
