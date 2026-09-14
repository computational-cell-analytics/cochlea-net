"""Tonotopic mapping of a cochlea along the central path through its segmentation.

Every instance of a segmentation is assigned a position along the cochlea, from which the
Greenwood function gives a frequency. The position comes from a central path that runs through
the segmented structure: Rosenthal's canal for SGNs, the row of inner hair cells for IHCs.

`measure_run_length` is the single entry point. It runs one central path method per connected
component, then links, orients and measures the components with code that is shared by every
method:

    CENTRAL_PATH_METHODS[path_method]  ->  _order_components  ->  _orient_apex_base
                                       ->  _total_distance    ->  _path_dict_from_components

Adding a central path method
----------------------------
A method is one function plus one entry in `CENTRAL_PATH_METHODS`. The contract is:

    def component_paths_<name>(centroids_components, apex_higher=True, **kwargs)
            -> List[np.ndarray]

- `centroids_components` holds the cell centroids of one component per entry, in µm, with the
  coordinates in `(x, y, z)` table order — the order of the `anchor_x/y/z` columns, which is the
  reverse of the `(Z, Y, X)` order used for image arrays. Everything in this module works in
  table order, including the downscaled volumes of the methods that rasterize.
- Return one ordered path per component, **in the order of the input**, in the same units and
  axis order. The shared steps assume that neighboring entries of the list are neighbors in the
  cochlea; ordering them is the caller's job.
- The returned path is final. `measure_run_length` never smooths, so any smoothing a method wants
  is its own responsibility.
- A method receives the whole list, not one component at a time, because a method may carry state
  across components: `component_paths_edt` raises its downscaling factor for every component that
  follows a disconnected one, and `component_paths_graph` derives one edge threshold from all
  components together.
- `apex_higher` is passed to every method and may be ignored. `component_paths_graph` needs it
  because an unweighted shortest path is not symmetric in its source and target.

`DEFAULT_PATH_METHOD` picks the method for a cell type when the caller names none.
"""
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
from scipy.spatial import cKDTree, ConvexHull

from flamingo_tools.postprocessing.label_components import downscaled_centroids
from flamingo_tools.json_util import load_processing_params, STEP_KEYS
from flamingo_tools.s3_utils import (default_table_path, get_s3_path, MOBIE_FOLDER,
                                     table_name_prefix)


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


def find_most_distant_nodes(G: nx.classes.graph.Graph, weight: str = 'weight') -> Tuple[int, int]:
    """Find the two nodes of a graph that are farthest apart.

    Args:
        G: Input graph.
        weight: Edge attribute used as the distance. Pass None to count hops instead.

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
                    in_bounds = (0 <= nz < shape[0]) and (0 <= ny < shape[1]) and (0 <= nx_ < shape[2])
                    if in_bounds and mask[nz, ny, nx_]:
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
        Smoothed path of the same shape, always float64. Callers that need whole µm have to cast,
        which `component_paths_edt` does deliberately.
    """
    if not isinstance(path, np.ndarray):
        path = np.array(path)
    kernel_size = 2 * window + 1
    kernel = np.ones(kernel_size) / kernel_size

    smooth_path = np.zeros(path.shape, dtype=np.float64)

    for d in range(3):
        pad = np.pad(path[:, d], window, mode='edge')
        smooth_path[:, d] = np.convolve(pad, kernel, mode='valid')

    return smooth_path


def _outward_tangent(path: np.ndarray, end: str, lookback_distance: float = 50.0) -> np.ndarray:
    """Unit vector at one end of a component path, pointing outward.

    The vector points in the direction the path was already heading as it approaches
    `end` - i.e. the direction a continuation (e.g. a missing/broken segment) would extend.

    Args:
        path: Ordered array of 3D positions, shape (N, 3).
        end: Either "start" (path[0]) or "end" (path[-1]).
        lookback_distance: Physical distance (µm) to walk inward from the endpoint before
            measuring the direction, so the estimate is robust to varying point density.

    Returns:
        Unit vector (zero vector if the path has fewer than 2 points).
    """
    ordered = path if end == "end" else path[::-1]
    accumulated = 0.0
    ref_idx = 0
    for i in range(1, len(ordered)):
        accumulated += math.dist(ordered[i - 1], ordered[i])
        ref_idx = i
        if accumulated >= lookback_distance:
            break
    vec = np.asarray(ordered[0]) - np.asarray(ordered[ref_idx])
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _path_length(path: np.ndarray) -> float:
    """Total arc length of an ordered path (sum of consecutive point-to-point distances)."""
    return float(sum(math.dist(path[i], path[i + 1]) for i in range(len(path) - 1)))


def _flow_alignment_score(
    path_a: np.ndarray, end_a: str, path_b: np.ndarray, end_b: str, lookback_distance: float = 50.0,
) -> float:
    """Score how plausibly a missing/broken segment connects `path_a`'s `end_a` to `path_b`'s `end_b`.

    Compares each component's local directional trend (see `_outward_tangent`) against the
    direction straight across the gap between the two candidate endpoints. A real continuation
    should extend roughly in a straight line, so both components' outward tangents should point
    toward each other across the gap. Higher is better (straighter continuation).
    """
    p_a = path_a[0] if end_a == "start" else path_a[-1]
    p_b = path_b[0] if end_b == "start" else path_b[-1]
    gap = np.asarray(p_b) - np.asarray(p_a)
    gap_norm = np.linalg.norm(gap)
    if gap_norm == 0:
        return 0.0
    gap_dir = gap / gap_norm

    tangent_a = _outward_tangent(path_a, end_a, lookback_distance=lookback_distance)
    tangent_b = _outward_tangent(path_b, end_b, lookback_distance=lookback_distance)
    return float(np.dot(tangent_a, gap_dir) - np.dot(tangent_b, gap_dir))


def _combined_score(flow_score: float, distance: float, min_distance: float) -> float:
    """Discount a flow-alignment score by how much farther this candidate is than the closest one.

    A candidate exactly at `min_distance` keeps its full flow score; a farther candidate needs a
    proportionally better flow score to still win, so distance still has a say even when flow-based
    matching is used.
    """
    return flow_score * (min_distance / distance)


def _principal_axis_endpoints(mask: np.ndarray) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Find the two extreme voxels of a binary mask along its principal axis.

    Args:
        mask: Binary mask of the volume, with its axes in the order of the centroids it was
            built from, i.e. (x, y, z).

    Returns:
        Voxel at the lower end of the principal axis.
        Voxel at the upper end. Which of the two is the apex is decided later, by
        `_orient_apex_base`.
    """
    pts = np.argwhere(mask == 1)
    c_mean = pts.mean(axis=0)
    cov = np.cov((pts - c_mean).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, np.argmax(evals)]
    proj = (pts - c_mean) @ axis
    return tuple(pts[proj.argmin()]), tuple(pts[proj.argmax()])


def _central_path_downscaled(centroids: np.ndarray, scale_factor: int) -> Optional[np.ndarray]:
    """Find the central path of one component in a downscaled binary volume.

    Args:
        centroids: Centroids of one component in µm, shape (N, 3), in (x, y, z) order.
        scale_factor: Downscaling factor. One voxel of the mask spans this many µm.

    Returns:
        Path in voxel coordinates, with the axes in the same (x, y, z) order as the centroids,
        or None if the component is not connected at this factor. Note that
        `central_path_edt_graph`, which produces it, names its axes z, y, x; the names do not
        match this module's order, but both sides use the same axes, so the result is consistent.

    Raises:
        ValueError: If a coordinate is negative, or if the downscaled volume comes out empty.
    """
    # downscaled_centroids places a centroid at int(coordinate / scale_factor), which truncates
    # toward zero, and sizes the volume from the maximum. A negative coordinate would therefore
    # index from the end of the array and silently land at the far side of the volume.
    if np.asarray(centroids).min() < 0:
        raise ValueError("Centroids must have non-negative coordinates for the volumetric path methods.")

    mask = downscaled_centroids(centroids, scale_factor=scale_factor, downsample_mode="capped")
    mask = binary_dilation(mask, np.ones((3, 3, 3)), iterations=1)
    mask = binary_closing(mask, np.ones((3, 3, 3)), iterations=1)
    if not mask.any():
        # The closing erodes with a 3x3x3 element and a zero border value, so it empties any
        # volume whose downscaled extent is a single voxel along one axis. Centroids that lie in
        # an axis-aligned plane hit this; an oblique plane usually survives the dilation.
        raise ValueError(
            f"The downscaled volume is empty at a scale factor of {scale_factor} µm per voxel. "
            f"The centroids are probably flat along one axis, which the volumetric path methods "
            f"cannot handle."
        )
    start_voxel, end_voxel = _principal_axis_endpoints(mask)
    return central_path_edt_graph(mask, start_voxel, end_voxel)


def component_paths_edt(
    centroids_components: List[np.ndarray],
    apex_higher: bool = True,
    scale_factor: int = 10,
    smooth_window: int = 3,
    quantize: bool = True,
) -> List[np.ndarray]:
    """Find one central path per component with the 3D Euclidean distance transform.

    For each component the centroids are rasterized into a binary volume of `scale_factor` µm
    voxels, the volume is dilated and closed, and a path is traced between the two extremes of
    the principal axis. The path is pulled toward the medial axis because the weight of an edge
    is the inverse of the smaller of the two distance transform values it connects, so a step
    deep inside the structure is cheaper than one near its surface.

    This method reproduces the central path used for the CochleaNet paper. Two details are kept
    for that reason and must not be "cleaned up":

    - `scale_factor` is deliberately not reset between components. A component that is not
      connected at the current factor raises the factor for every component after it.
    - The smoothed path is truncated to whole µm. The original implementation allocated the
      output of the moving average with the integer dtype of the up-scaled path, so every
      coordinate was truncated. Removing the truncation shortens the run length by about 0.3 %
      (measured 0.29 to 0.33 % on four local cochleae).

    Args:
        centroids_components: List of centroids per component, each of shape (N, 3).
        apex_higher: Unused. Part of the shared method signature.
        scale_factor: Downscaling factor for the binary volume.
        smooth_window: Half-window of the moving average filter, which spans 2 * window + 1 nodes.
        quantize: Truncate the smoothed path to whole µm. Set to False for a refined path.

    Returns:
        One ordered path per component, in the order of the input, in µm and in (x, y, z) order.
        The node spacing is set by `scale_factor`. The dtype is int64 when `quantize` is set and
        float64 otherwise.

    Raises:
        ValueError: If no downscaling factor up to 100 µm per voxel connects a component.
    """
    paths = []
    for centroids in centroids_components:
        path = _central_path_downscaled(centroids, scale_factor)

        # Use a larger downscaling factor to have connected components in the downscaled volume.
        while path is None:
            scale_factor = 2 * scale_factor
            print(f"Not all components are fully connected in downscaled volume. Trying scale factor {scale_factor}.")
            if scale_factor > 100:
                raise ValueError("Downscaling for tonotopic mapping not possible.")
            path = _central_path_downscaled(centroids, scale_factor)

        path = moving_average_3d(path * scale_factor, window=smooth_window)
        paths.append(path.astype(np.int64) if quantize else path)
    return paths


def arc_length(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length of an ordered path, shape (N,)."""
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segments)])


def resample_path(path: np.ndarray, spacing: float) -> np.ndarray:
    """Resample an ordered path to a uniform arc-length spacing. Both endpoints are preserved.

    Args:
        path: Ordered array of 3D positions, shape (N, 3).
        spacing: Target distance between two consecutive samples in µm. The samples are evenly
            spaced, but the spacing that comes out is the path length divided by a whole number
            of intervals, so it only approximates this value.

    Returns:
        Resampled path of shape (M, 3). A path with fewer than two distinct positions is returned
        unchanged, without resampling.
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return path
    cum_len = arc_length(path)
    # interp1d needs a strictly increasing x, and a quantized path can hold repeated nodes.
    keep = np.concatenate([[True], np.diff(cum_len) > 1e-9])
    path, cum_len = path[keep], cum_len[keep]
    if len(path) < 2 or cum_len[-1] <= 0:
        return path
    n_samples = max(int(round(cum_len[-1] / spacing)) + 1, 2)
    return interp1d(cum_len, path, axis=0)(np.linspace(0.0, cum_len[-1], n_samples))


def path_tangents(samples: np.ndarray, window: float, spacing: float) -> np.ndarray:
    """Unit tangent at every sample, from a central difference over a fixed physical window.

    The indices are clipped at both ends, so the outermost `window / (2 * spacing)` samples use a
    shortened window and the two terminal samples use a one-sided difference.

    Args:
        samples: Path sampled at uniform arc length, shape (N, 3).
        window: Total length of the difference window in µm.
        spacing: Arc-length spacing of the samples in µm.

    Returns:
        Unit tangents of shape (N, 3). A sample whose window collapses to zero length falls back
        to the step toward the next sample, so the result is a unit vector for any path that
        `resample_path` produced.
    """
    half = max(int(round(window / (2 * spacing))), 1)
    index = np.arange(len(samples))
    forward = samples[np.clip(index + half, 0, len(samples) - 1)]
    backward = samples[np.clip(index - half, 0, len(samples) - 1)]
    tangents = forward - backward
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    # A path that doubles back on itself over the window gives a zero difference. Fall back to the
    # step to the next sample, which is non-zero for any path that resample_path produced.
    degenerate = (norms <= 0).ravel()
    if degenerate.any():
        step = np.diff(samples, axis=0, append=samples[-1:] * 2 - samples[-2:-1])
        tangents[degenerate] = step[degenerate]
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    return np.divide(tangents, norms, out=np.zeros_like(tangents), where=norms > 0)


def plane_basis(tangents: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Orthonormal basis of the plane perpendicular to every tangent.

    Args:
        tangents: Unit tangents of shape (N, 3), in (x, y, z) order like everything else in this
            module.

    Returns:
        First basis vector per tangent, shape (N, 3).
        Second basis vector per tangent, shape (N, 3). Both are unit length and perpendicular to
        each other and to the tangent. A zero tangent yields zero vectors rather than NaN.
    """
    helper = np.tile(np.array([0.0, 0.0, 1.0]), (len(tangents), 1))
    # A helper parallel to the tangent gives a zero cross product. Swap it well before that,
    # so the angle between tangent and helper always stays above 25 degrees.
    helper[np.abs(tangents[:, 2]) > 0.9] = np.array([1.0, 0.0, 0.0])
    first = np.cross(tangents, helper)
    # A degenerate tangent (see path_tangents) leaves a zero cross product, which would divide into
    # NaN and silently poison the cross-section of that sample.
    norms = np.linalg.norm(first, axis=1, keepdims=True)
    first = np.divide(first, norms, out=np.zeros_like(first), where=norms > 0)
    return first, np.cross(tangents, first)


def hull_area_centroid(points: np.ndarray) -> np.ndarray:
    """Area centroid of the convex hull of a set of 2D points.

    The area centroid is used instead of the mean of the points because the SGN density varies
    across the cross-section of Rosenthal's canal. The mean of the points follows the density,
    the area centroid follows the shape of the cross-section.

    Args:
        points: Points of shape (N, 2).

    Returns:
        Centroid of shape (2,). It falls back to the mean of the points for fewer than three
        points and when the hull is degenerate or cannot be built — that is, to the very quantity
        the area centroid exists to avoid. Callers that care should keep enough points in a
        cross-section that the fallback does not trigger; `refine_central_path` uses
        `min_cross_section_points` for that.
    """
    if len(points) < 3:
        return points.mean(axis=0)
    try:
        hull = ConvexHull(points)
    except Exception:
        # QhullError for collinear or duplicate input.
        return points.mean(axis=0)
    poly = points[hull.vertices]
    x, y = poly[:, 0], poly[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-9:
        return poly.mean(axis=0)
    centroid_x = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * area)
    centroid_y = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * area)
    return np.array([centroid_x, centroid_y])


def refine_central_path(
    path: np.ndarray,
    centroids: np.ndarray,
    sample_spacing: float = 20.0,
    tangent_window: float = 100.0,
    slab_half_thickness: float = 40.0,
    max_radius: float = 120.0,
    min_cross_section_points: int = 20,
    n_iterations: int = 3,
    convergence_tol: float = 2.0,
    smooth_length: float = 100.0,
    output_spacing: Optional[float] = 10.0,
) -> np.ndarray:
    """Recentre a central path on the area centroids of the cross-sections of the canal.

    Each sample of the path is moved, within the plane perpendicular to the local path direction,
    to the area centroid of the convex hull of the cells around it. Its position along the path is
    not changed. A sample with fewer than `min_cross_section_points` cells keeps the position of
    the initial guess.

    Every cell is assigned to its closest sample before a cross-section is built. That assignment
    is what keeps the neighboring turn of the spiral out of a cross-section, and it needs no
    threshold that would have to be tuned per cochlea; `max_radius` only trims stragglers.
    Measured on a synthetic pair of turns, the refinement is unaffected down to a separation of
    200 µm between the two axes and breaks down at 150 µm, where the path is pulled about 70 µm
    off course (see `test_neighboring_turn_of_the_spiral_is_kept_out`).

    Args:
        path: Initial guess of the central path, shape (N, 3), in µm.
        centroids: Centroids of the component, shape (M, 3), in µm.
        sample_spacing: Arc-length spacing of the path samples in µm.
        tangent_window: Length of the window used to estimate the local direction in µm.
        slab_half_thickness: Half thickness in µm of the cross-section slab along the tangent,
            which is what bounds the distance of a cell along the path.
        max_radius: Maximal distance in µm of a cell from the sample *within the cross-section
            plane*. The distance along the tangent is bounded by `slab_half_thickness` instead.
        min_cross_section_points: Minimal number of cells needed to move a sample. A sample with
            fewer keeps the position of the initial guess.
        n_iterations: Maximal number of refinement passes.
        convergence_tol: Stop once the 95th percentile of the correction is below this, in µm.
        smooth_length: Length of the moving average window along the path in µm.
        output_spacing: Arc-length spacing of the returned path in µm. None returns the path at
            `sample_spacing`. The default matches the node spacing of `component_paths_edt`, so
            the resulting `path_dict` keeps a comparable resolution.

    Returns:
        The refined path, resampled to `output_spacing`. It is shorter than the input on a real
        cochlea, by a few percent, because the initial guess wanders; see
        `component_paths_edt_refined` for what that means downstream.
    """
    centroids = np.asarray(centroids, dtype=float)
    samples = resample_path(path, sample_spacing)
    if len(samples) < 3:
        # Too short to estimate a direction. Return the initial guess at the requested spacing.
        return resample_path(samples, output_spacing) if output_spacing is not None else samples

    n_neighbor_bins = int(np.ceil(slab_half_thickness / sample_spacing)) + 1
    smooth_window = max(int(round(smooth_length / (2.0 * sample_spacing))), 1)

    for _ in range(n_iterations):
        tangents = path_tangents(samples, tangent_window, sample_spacing)
        first, second = plane_basis(tangents)

        _, assignment = cKDTree(samples).query(centroids, workers=-1)
        order = np.argsort(assignment, kind="stable")
        sorted_assignment = assignment[order]
        sample_index = np.arange(len(samples))
        bin_start = np.searchsorted(sorted_assignment, sample_index, "left")
        bin_end = np.searchsorted(sorted_assignment, sample_index, "right")

        refined = samples.copy()
        for i in range(len(samples)):
            low = max(0, i - n_neighbor_bins)
            high = min(len(samples) - 1, i + n_neighbor_bins)
            # The bins are contiguous in the sorted order, so one slice covers the whole window.
            selected = order[bin_start[low]:bin_end[high]]
            if selected.size < min_cross_section_points:
                continue

            relative = centroids[selected] - samples[i]
            along = relative @ tangents[i]
            in_plane = np.stack([relative @ first[i], relative @ second[i]], axis=1)
            keep = (np.abs(along) <= slab_half_thickness) & ((in_plane ** 2).sum(axis=1) <= max_radius ** 2)
            if int(keep.sum()) < min_cross_section_points:
                # Too few cells to trust the hull. Keep the initial guess for this sample.
                continue

            center = hull_area_centroid(in_plane[keep])
            refined[i] = samples[i] + center[0] * first[i] + center[1] * second[i]

        correction = np.linalg.norm(refined - samples, axis=1)
        smoothed = moving_average_3d(refined, window=smooth_window)
        # moving_average_3d pads with the edge value, which pulls both terminal nodes inward by
        # sample_spacing * w(w+1) / (2(2w+1)). resample_path then pins the shortened ends, so the
        # loss would accumulate over the passes and make the run length depend on the number of
        # passes. Keep the endpoints the refinement itself produced.
        smoothed[0], smoothed[-1] = refined[0], refined[-1]
        samples = resample_path(smoothed, sample_spacing)
        # Only the samples that actually moved say anything about convergence. A sample that was
        # skipped for too few cells contributes a correction of exactly 0.
        moved = correction[correction > 0]
        # The correction plateaus at the noise of the hull centroid itself, so the tolerance has
        # to sit above that floor and the number of passes has to stay capped.
        if moved.size == 0 or np.percentile(moved, 95) < convergence_tol:
            break

    if output_spacing is not None:
        samples = resample_path(samples, output_spacing)
    return samples


def component_paths_edt_refined(
    centroids_components: List[np.ndarray],
    apex_higher: bool = True,
    scale_factor: int = 10,
    smooth_window: int = 3,
    **refine_kwargs,
) -> List[np.ndarray]:
    """Find one central path per component, then recentre it on the cross-sections of the canal.

    The first step is `component_paths_edt`, without the truncation to whole µm. The second step
    is `refine_central_path`, which recentres every sample on the area centroid of the convex hull
    of the cells around it.

    Measured with `scripts/validation/central_path/compare_path_methods.py` over 171 components of
    155 local cochleae, building each path from one half of the cells and measuring on the other
    half, the offset of the path from the area centroid of its own cross-section drops from a
    median of 14.8 µm to 4.3 µm, and the 95th percentile of the curvature drops from 21.6 /mm to
    8.3 /mm. The refined path improved 169 of the 171 components; both exceptions come from the
    anisotropic LaVision acquisition, whose 3 µm z-sampling makes a cross-section sparse.

    The refined path is shorter than the `edt` path by a median of 3 % (range -8 % to +37 %; the
    outlier is a cochlea whose `edt` path was 60 µm off centre and too short). `length[µm]`,
    `length_fraction` and the mapped `frequency[kHz]` therefore all differ between the two
    methods, and results from the two must not be mixed within one analysis.

    Args:
        centroids_components: List of centroids per component, each of shape (N, 3).
        apex_higher: Unused. Part of the shared method signature.
        scale_factor: Downscaling factor of the initial guess.
        smooth_window: Half-window of the moving average filter of the initial guess.
        refine_kwargs: Passed to `refine_central_path`.

    Returns:
        One ordered path per component, in the order of the input.
    """
    initial = component_paths_edt(
        centroids_components, scale_factor=scale_factor, smooth_window=smooth_window, quantize=False,
    )
    return [refine_central_path(path, np.asarray(centroids, dtype=float), **refine_kwargs)
            for path, centroids in zip(initial, centroids_components)]


def _auto_max_edge_distance(centroids_components: List[np.ndarray]) -> int:
    """Edge distance that makes every component connected on its own.

    Each component needs a threshold of at least its largest minimum-spanning-tree edge to become
    one connected graph. The largest of those over all components is used for all of them, so that
    every component is built with the same rule. This says nothing about connecting the components
    to each other, which `_order_components` does geometrically instead.

    Args:
        centroids_components: List of centroids per component, each of shape (N, 3), in µm.

    Returns:
        The threshold in whole µm. `round(x + 0.5)` is kept from the original implementation: it
        is neither a ceiling nor plain rounding, but changing it would move the IHC run lengths.
    """
    max_edge_distance = max(minimal_connection_distance(c) for c in centroids_components)
    return round(max_edge_distance + 0.5)


def _component_graph(centroids: np.ndarray, max_edge_distance: float) -> nx.Graph:
    """Build a graph of one component, with an edge between every pair closer than a threshold.

    Args:
        centroids: Centroids of one component, shape (N, 3), in µm.
        max_edge_distance: Two nodes get an edge when they are this far apart or closer, in µm.

    Returns:
        The graph. Each node is keyed by its index into `centroids` and carries its position as
        the node attribute 'pos'. Each edge carries its length as the attribute 'weight', which
        `find_most_distant_nodes` uses to pick the endpoints of the path.
    """
    graph = nx.Graph()
    for index, position in enumerate(centroids):
        graph.add_node(index, pos=position)
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            distance = math.dist(centroids[i], centroids[j])
            if distance <= max_edge_distance:
                graph.add_edge(i, j, weight=distance)
    return graph


def component_paths_graph(
    centroids_components: List[np.ndarray],
    apex_higher: bool = True,
    max_edge_distance: Optional[float] = None,
) -> List[np.ndarray]:
    """Find one path per component through the centroids themselves.

    This is the method for IHCs, which form a single row of cells. The centroids are therefore
    already the central path and no volumetric estimate is needed. Each component is turned into a
    graph of its centroids, and its path runs between the two nodes that are farthest apart.

    Two different metrics meet here, and the difference matters. The endpoints come from
    `find_most_distant_nodes`, which uses a **weighted** all-pairs Dijkstra, i.e. geometric
    distance. The path between them is then an **unweighted** shortest path, i.e. the one with the
    fewest hops. The unweighted path is deliberate: it reproduces the paths used for the
    CochleaNet paper. Passing `weight="weight"` would give the geometric shortest path and lower
    the reported IHC run lengths by 0.4 to 1.7 % on the local cochleae.

    Each component is traced over its own full extent. The implementation this replaced merged all
    components into one graph and bridged them with a single edge, so a component only contributed
    the section between the two nodes that bridged it to its neighbors. Run lengths of cochleae
    with several components are therefore larger than they were before.

    Args:
        centroids_components: List of centroids per component, each of shape (N, 3).
        apex_higher: Apex is the node with the higher y-value if True. Only used to pick the
            source and the target of the shortest path, which are not interchangeable for an
            unweighted path.
        max_edge_distance: Two nodes get an edge when they are this far apart or closer, in µm.
            Derived from all components with `_auto_max_edge_distance` if None.

    Returns:
        One ordered path per component, in the order of the input, in µm and in (x, y, z) order.
        The nodes are cell centroids, so the spacing follows the spacing of the cells.

    Raises:
        ValueError: If a component is not connected at `max_edge_distance`. The implementation
            this replaced bridged such a split silently; name the pieces as separate entries of
            the component list instead.
    """
    if max_edge_distance is None:
        max_edge_distance = _auto_max_edge_distance(centroids_components)
        print("Automatically determined max edge distance", max_edge_distance)

    paths = []
    for centroids in centroids_components:
        graph = _component_graph(centroids, max_edge_distance)
        if not nx.is_connected(graph):
            raise ValueError(
                f"A component is not connected at a maximal edge distance of {max_edge_distance} µm. "
                f"It splits into {nx.number_connected_components(graph)} parts. Either the component "
                f"labels do not match the connected components of the segmentation, or the component "
                f"has to be split into several entries of the component list."
            )
        start_node, end_node = find_most_distant_nodes(graph)

        # Compare the y-value to not get into confusion with MoBIE dimensions.
        if graph.nodes[start_node]["pos"][1] > graph.nodes[end_node]["pos"][1]:
            apex_node = start_node if apex_higher else end_node
            base_node = end_node if apex_higher else start_node
        else:
            apex_node = end_node if apex_higher else start_node
            base_node = start_node if apex_higher else end_node

        path = nx.shortest_path(graph, source=apex_node, target=base_node)
        paths.append(np.array([graph.nodes[node]["pos"] for node in path], dtype=float))
    return paths


# The central path methods, keyed by the name that reaches the CLI and the parameter files.
# See the module docstring for the contract a method has to satisfy.
CENTRAL_PATH_METHODS = {
    "edt": component_paths_edt,
    "edt_refined": component_paths_edt_refined,
    "graph": component_paths_graph,
}

# Used when a caller names no method. 'edt' stays available to reproduce the CochleaNet paper.
DEFAULT_PATH_METHOD = {"sgn": "edt_refined", "ihc": "graph"}

# Transitional notice. The default central path method for SGNs changed from 'edt' to
# 'edt_refined'. Remove this message once the transition period is over.
_DEFAULT_METHOD_NOTICE = {
    "edt_refined": (
        "Using the refined central path method 'edt_refined', which is the new default. It "
        "recentres the path on the cross-sections of Rosenthal's canal, so the run length and "
        "the mapped frequency differ from earlier results. Pass path_method='edt' to reproduce "
        "the mapping used for the CochleaNet paper."
    ),
}


def _resolve_path_method(cell_type: str, path_method: Optional[str] = None) -> str:
    """Resolve the central path method for a cell type.

    The cell type is validated whether or not a method is given. Resolving to the default prints
    a one-off notice, because the default for SGNs changed from 'edt' to 'edt_refined'.

    Args:
        cell_type: Cell type of the segmentation. Either 'sgn' or 'ihc', in any case.
        path_method: Explicit method, a key of `CENTRAL_PATH_METHODS`. The default of the cell
            type is used if None.

    Returns:
        Name of the method, a key of `CENTRAL_PATH_METHODS`.

    Raises:
        ValueError: If the cell type or the method is not recognized.
    """
    # The cell type is validated even when a method is given, so that a typo is caught rather
    # than silently mapping a cochlea with the wrong kind of segmentation.
    if str(cell_type).lower() not in DEFAULT_PATH_METHOD:
        raise ValueError(f"Unrecognized cell type: {cell_type}. Choose either 'sgn' or 'ihc'.")

    if path_method is None:
        path_method = DEFAULT_PATH_METHOD[str(cell_type).lower()]
        if path_method in _DEFAULT_METHOD_NOTICE:
            print(_DEFAULT_METHOD_NOTICE[path_method])
    elif path_method not in CENTRAL_PATH_METHODS:
        raise ValueError(f"Unrecognized path method: {path_method}. "
                         f"Choose one of {sorted(CENTRAL_PATH_METHODS)}.")
    return path_method


def _resolve_ambiguous_junction(
    path_a: np.ndarray,
    path_b: np.ndarray,
    candidates: List[Tuple[str, str]],
    distances: List[float],
) -> int:
    """Pick between two candidate endpoint pairs of a junction by flow alignment.

    The flow score is discounted by how much farther a candidate is than the closer of the two,
    so distance still has a say. See `_combined_score`.

    Args:
        path_a: Path of the first component.
        path_b: Path of the second component.
        candidates: The two candidate (end of a, end of b) pairs.
        distances: Distance of each candidate pair in µm, in the same order.

    Returns:
        Index into `candidates` of the winning pair, 0 or 1. A tie goes to the first.
    """
    min_distance = min(distances)
    scores = [
        _combined_score(_flow_alignment_score(path_a, end_a, path_b, end_b), distance, min_distance)
        for (end_a, end_b), distance in zip(candidates, distances)
    ]
    return 0 if scores[0] >= scores[1] else 1


def _order_components(
    paths: List[np.ndarray],
    ambiguous_margin: float = 200.0,
    min_flow_length: float = 600.0,
) -> List[np.ndarray]:
    """Flip the component paths so that consecutive components connect end to start.

    The paths are expected in the order of neighboring components, e.g.
    [[start_c1, ..., end_c1], [end_c2, ..., start_c2]] --> [[start_c1, ..., end_c1], [start_c2, ..., end_c2]]

    Which endpoints to connect is normally decided by closest distance. If the two closest
    candidate distances are within `ambiguous_margin` of each other, that criterion is unreliable,
    because two different turns of the spiral can pass closer to each other than the true
    anatomical continuation. A flow-based fallback is used instead. It picks whichever candidate
    keeps the local directional trend of each component pointing toward the other, discounted by
    how much farther it is than the closest candidate, so distance still has a say.

    The fallback only engages when both components are at least `min_flow_length` long. Small
    torn-off fragments do not have a reliable directional trend, because their own shape is
    unrelated to the true direction of the canal, and closest distance is the better default.

    Args:
        paths: One ordered path per component. The list is modified in place.
        ambiguous_margin: Margin in µm below which a junction counts as ambiguous.
        min_flow_length: Minimal arc length in µm that both components of a junction need for the
            flow-based fallback to be used.

    Returns:
        The same list object, with the paths flipped where needed.
    """
    if len(paths) <= 1:
        return paths

    path_lengths = [_path_length(p) for p in paths]

    # Find the starting order of the first two components.
    ends_1 = (paths[0][0, :], paths[0][-1, :])
    ends_2 = (paths[1][0, :], paths[1][-1, :])
    end_combos = [("start", "start"), ("start", "end"), ("end", "start"), ("end", "end")]
    distances = [math.dist(ends_1[0], ends_2[0]), math.dist(ends_1[0], ends_2[1]),
                 math.dist(ends_1[1], ends_2[0]), math.dist(ends_1[1], ends_2[1])]
    order = sorted(range(len(distances)), key=lambda i: distances[i])
    min_index = order[0]
    margin = distances[order[1]] - distances[order[0]]
    long_enough = path_lengths[0] >= min_flow_length and path_lengths[1] >= min_flow_length
    if margin < ambiguous_margin and long_enough:
        print(f"Endpoint distances for linking components 0 and 1 are ambiguous "
              f"({distances[order[0]]:.1f} vs {distances[order[1]]:.1f} µm, difference {margin:.1f} µm "
              f"< {ambiguous_margin} µm). Falling back to flow-based matching.")
        candidates = [end_combos[order[0]], end_combos[order[1]]]
        winner = _resolve_ambiguous_junction(paths[0], paths[1], candidates,
                                             [distances[order[0]], distances[order[1]]])
        min_index = order[winner]
    if min_index in [0, 1]:
        paths[0] = np.flip(paths[0], axis=0)

    # Order the other components from start to end.
    for num in range(0, len(paths) - 1):
        distance_to_start = math.dist(paths[num][-1, :], paths[num + 1][0, :])
        distance_to_end = math.dist(paths[num][-1, :], paths[num + 1][-1, :])
        margin = abs(distance_to_end - distance_to_start)
        long_enough = path_lengths[num] >= min_flow_length and path_lengths[num + 1] >= min_flow_length
        if margin < ambiguous_margin and long_enough:
            print(f"Endpoint distances for linking components {num} and {num + 1} are ambiguous "
                  f"({min(distance_to_start, distance_to_end):.1f} vs "
                  f"{max(distance_to_start, distance_to_end):.1f} µm, difference {margin:.1f} µm "
                  f"< {ambiguous_margin} µm). Falling back to flow-based matching.")
            winner = _resolve_ambiguous_junction(paths[num], paths[num + 1],
                                                 [("end", "start"), ("end", "end")],
                                                 [distance_to_start, distance_to_end])
            flip = winner == 1
        else:
            flip = distance_to_end < distance_to_start
        if flip:
            paths[num + 1] = np.flip(paths[num + 1], axis=0)

    return paths


def _orient_apex_base(paths: List[np.ndarray], apex_higher: bool = True) -> List[np.ndarray]:
    """Order the paths so that the first node is the apex.

    Args:
        paths: One ordered path per component, already linked end to start.
        apex_higher: Apex is the node with the higher y-value if True.

    Returns:
        The paths, reversed as a whole if needed.
    """
    # Compare the y-value to not get into confusion with MoBIE dimensions.
    if paths[0][0, 1] > paths[-1][-1, 1]:
        if apex_higher:
            return paths
    elif not apex_higher:
        return paths
    paths = list(reversed(paths))
    return [np.flip(p, axis=0) for p in paths]


def _total_distance(paths: List[np.ndarray]) -> float:
    """Sum of the arc lengths of the paths. The space between components does not count."""
    return sum([math.dist(p[num + 1], p[num]) for p in paths for num in range(len(p) - 1)])


def _path_dict_from_components(paths: List[np.ndarray], total_distance: float) -> dict:
    """Collect the nodes of all components with their position and fractional run length.

    The fractional run length continues across a component boundary without a jump, so the space
    between two components does not count towards the run length.

    Args:
        paths: One ordered path per component, ordered from apex to base or the other way round.
        total_distance: Total arc length of all components in µm. Must be greater than zero.

    Returns:
        Dictionary of the nodes, keyed by a consecutive index starting at 0. Each value holds
        'pos', the position as an array of shape (3,), and 'length_fraction', the position along
        the cochlea in [0, 1]. `path_dict_to_central_path_table` additionally expects 'length[µm]'
        and 'frequency[kHz]', which the caller adds.
    """
    path_dict = {}
    accumulated = 0
    index = 0
    for num, component_path in enumerate(paths):
        if num == 0:
            path_dict[0] = {"pos": paths[0][0], "length_fraction": 0}
        else:
            path_dict[index] = {"pos": paths[num][0], "length_fraction": path_dict[index - 1]["length_fraction"]}

        index += 1
        for enum, position in enumerate(component_path[1:]):
            accumulated += math.dist(paths[num][enum], position)
            path_dict[index] = {"pos": position, "length_fraction": accumulated / total_distance}
            index += 1
    # The last node closes the path, so its fraction is exactly 1.
    path_dict[index - 1] = {"pos": paths[-1][-1, :], "length_fraction": 1}
    return path_dict


def measure_run_length(
    centroids_components: List[np.ndarray],
    path_method: str,
    apex_higher: bool = True,
    include_gap: bool = False,
    ambiguous_margin: float = 200.0,
    min_flow_length: float = 600.0,
    method_kwargs: Optional[dict] = None,
) -> Tuple[float, dict]:
    """Measure the run length of a segmentation along the central path through the cochlea.

    The list of centroids has to be in the order of neighboring components. The steps are:

    1) Find one central path per component. This is the only cell type specific step and is
       selected with `path_method`. See `CENTRAL_PATH_METHODS`.
    2) Flip the paths so that consecutive components connect end to start, see `_order_components`.
    3) Assign the apex and the base position, see `_orient_apex_base`.
    4) Measure the run length of every node. The space between two components is skipped unless
       `include_gap` is set, in which case the components are joined into one path first.

    Args:
        centroids_components: List of centroids per component, each of shape (N, 3), in µm and in
            (x, y, z) order. Neighboring entries have to be neighbors in the cochlea.
        path_method: Key of `CENTRAL_PATH_METHODS`. There is deliberately no default here, because
            the default depends on the cell type; `_resolve_path_method` applies
            `DEFAULT_PATH_METHOD` for callers that work from a cell type.
        apex_higher: Apex is set to the node with the higher y-value if True.
        include_gap: Include the distance between different components in the run length.
        ambiguous_margin: If the two smallest candidate endpoint distances of a junction are
            within this many µm of each other, fall back to flow-based matching.
        min_flow_length: Minimal arc length in µm that both components of a junction must have
            for the flow-based fallback. Below this, closest distance is used.
        method_kwargs: Extra arguments for the central path method. What is accepted depends on
            the method: `scale_factor` and `smooth_window` for 'edt', those plus every argument of
            `refine_central_path` for 'edt_refined', and `max_edge_distance` for 'graph'.

    Returns:
        Total distance of the path in µm.
        A dictionary of the nodes of the path, keyed by a consecutive index, each holding 'pos'
        and 'length_fraction'. `tonotopic_mapping` adds 'length[µm]' and 'frequency[kHz]'.
    """
    if path_method not in CENTRAL_PATH_METHODS:
        raise ValueError(f"Unrecognized path method: {path_method}. "
                         f"Choose one of {sorted(CENTRAL_PATH_METHODS)}.")

    print(f"Evaluating {len(centroids_components)} component(s).")
    paths = CENTRAL_PATH_METHODS[path_method](
        centroids_components, apex_higher=apex_higher, **(method_kwargs or {}),
    )

    paths = _order_components(paths, ambiguous_margin=ambiguous_margin, min_flow_length=min_flow_length)
    paths = _orient_apex_base(paths, apex_higher=apex_higher)

    if include_gap:
        # Flatten the list of components, so the junctions count towards the run length.
        paths = [np.array([node for component_path in paths for node in component_path])]
    total_distance = _total_distance(paths)
    print(f"The total path has length {round(total_distance)} µm. Gaps between components included: {include_gap}.")

    return total_distance, _path_dict_from_components(paths, total_distance)


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


def map_frequency(path_dict: dict, animal: str = "mouse", otof: bool = False) -> dict:
    """Map the frequency range of SGNs in the cochlea
    using Greenwood function f(x) = A * (10 **(ax) - K).
    Values for humans: a=2.1, k=0.88, A = 165.4 [kHz].
    For mice: fit values between minimal (1kHz) and maximal (80kHz) values

    Args:
        path_dict: Dictionary of the nodes of the central path, each holding 'length_fraction'.
        animal: Select the Greenwood function parameters specific to a species. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
            Only has an effect together with animal="mouse".

    Returns:
        The same dictionary, with 'frequency[kHz]' added to every node.
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


def get_centers_from_path_dict(
    path_dict: dict,
    n_blocks: int = 10,
    offset_blocks: bool = True,
) -> List[np.ndarray]:
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

    n_unique = len({tuple(c) for c in centers})
    if n_unique != len(centers):
        print(f"Warning: only {n_unique} of {len(centers)} centers are unique. "
              f"The central path has {len(path_dict)} nodes, which is too few to separate the requested centers.")

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


def _centroids_per_component(table: pd.DataFrame, component_label: List[int]) -> List[List[tuple]]:
    """Split the centroids of a segmentation table into one list of (x, y, z) per component label.

    The order of `component_label` is kept, because it is the order in which the components are
    linked along the cochlea.
    """
    centroids_components = []
    for label in component_label:
        subset = table[table["component_labels"] == label]
        centroids_components.append(list(zip(subset["anchor_x"], subset["anchor_y"], subset["anchor_z"])))
    return centroids_components


def equidistant_centers(
    table: pd.DataFrame,
    component_label: List[int] = [1],
    cell_type: str = "sgn",
    n_blocks: int = 10,
    offset_blocks: bool = True,
    include_gap: bool = False,
    path_method: Optional[str] = None,
) -> List[np.ndarray]:
    """Find equidistant centers within the central path of the Rosenthal's canal.

    Args:
        table: Dataframe containing centroids of SGN segmentation.
        component_label: List of components for centroid subset.
        cell_type: Cell type of the segmentation.
        n_blocks: Number of equidistant centers for block creation.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.
        include_gap: Include the distance between different components for calculating the run length.
        path_method: Method used to find the central path. The default of the cell type is used if None.

    Returns:
        One position per block, taken from the nodes of the central path. The dtype follows the
        path method, so it is int64 for 'edt' and float64 for the others.
    """
    centroids_components = _centroids_per_component(table, component_label)
    _, path_dict = measure_run_length(
        centroids_components, path_method=_resolve_path_method(cell_type, path_method), include_gap=include_gap,
    )
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
    ambiguous_margin: float = 200.0,
    min_flow_length: float = 600.0,
    path_method: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Tonotopic mapping of SGNs or IHCs by supplying a table with component labels.
    The mapping assigns a tonotopic label to each instance according to the position along the length of the cochlea.

    Args:
        table: Dataframe of segmentation table.
        component_label: List of component labels to evaluate.
        component_mapping: Components to use for tonotopic mapping. Ignore components torn parallel to main canal.
        cell_type: Cell type of segmentation.
        animal: Animal specifier for species specific frequency mapping. Either "mouse" or "gerbil".
        apex_higher: Flag for identifying apex and base. Apex is set to node with higher y-value if True.
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
        central_path_df: Dataframe featuring the spots for the central path through the segmentation.
        include_gap: Include the distance between different components for calculating the run length.
        ambiguous_margin: Fall back to flow-based component linking (instead of closest endpoint
            distance) when the two closest candidate distances for a junction are within this many µm of
            each other. See `_order_components`.
        min_flow_length: Minimal arc length (µm) both components of a junction must have for the
            flow-based fallback to be used. See `_order_components`.
        path_method: Method used to find the central path. The default of the cell type is used if None.
            See `CENTRAL_PATH_METHODS`.

    Returns:
        The table, with the columns 'offset', 'length_fraction', 'length[µm]' and 'frequency[kHz]'
        added for every instance in `component_label`.
        The central path as a table of spots, which can be stored and passed back in.
    """
    # subset of centroids for given component label(s)
    new_subset = table[table["component_labels"].isin(component_label)]
    centroids = list(zip(new_subset["anchor_x"], new_subset["anchor_y"], new_subset["anchor_z"]))
    label_ids = [int(i) for i in list(new_subset["label_id"])]

    if component_mapping is None:
        component_mapping = component_label

    if central_path_df is None:
        centroids_components = _centroids_per_component(table, component_mapping)
        total_distance, path_dict = measure_run_length(
            centroids_components, path_method=_resolve_path_method(cell_type, path_method),
            apex_higher=apex_higher, include_gap=include_gap,
            ambiguous_margin=ambiguous_margin, min_flow_length=min_flow_length,
        )

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
    path_method: Optional[str] = None,
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
        out_path: Output path for the segmentation table, which gains the columns "offset",
            "length_fraction", "length[µm]" and "frequency[kHz]".
        force_overwrite: Forcefully overwrite existing output path.
        cell_type: Cell type of the segmentation. Currently supports "sgn" and "ihc".
        animal: Animal for species specific frequency mapping. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
        apex_position: Identify position of apex and base. Apex is set to node with higher y-value per default.
        component_list: List of components. Can be passed to obtain the number of instances within the component list.
        component_mapping: Components to use for tonotopic mapping. Ignore components torn parallel to main canal.
        central_spots_path: Provide table featuring spots for central path through segmentation for tonotopic mapping.
        include_gap: Include the distance between different components for calculating the run length.
        path_method: Method used to find the central path. The default of the cell type is used if None.
            Use "edt" to reproduce the mapping of the CochleaNet paper. See `CENTRAL_PATH_METHODS`.
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
                                                   path_method=path_method,
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
    include_gap: bool = False,
    path_method: Optional[str] = None,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    dict_index: Optional[int] = None,
    **_,
):
    """Find equidistant centers within the central path of the Rosenthal's canal.

    Args:
        table_path: File path to segmentation table.
        output_path: Output path to JSON file with center coordinates. An existing file is updated
            in place rather than overwritten.
        n_blocks: Number of equidistant centers to compute.
        cell_type: Cell type of the segmentation. Currently supports "sgn" and "ihc".
        component_list: List of components. Can be passed to obtain the number of instances within the component list.
        offset_blocks: Centers are shifted by half a length if True. Avoid centers at the start/end of the path.
        include_gap: Include the distance between different components for calculating the run length.
            Use the same value as for the tonotopic mapping to keep the centers consistent with the table.
        path_method: Method used to find the central path. The default of the cell type is used if
            None. The resolved name is written into the parameter file, because the centers depend
            on it just as they depend on include_gap.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
        dict_index: Index of the entry to update, if output_path already holds a JSON list of parameter
            dictionaries instead of a single dictionary.
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

    # Record the resolved method rather than None: the crop centers depend on it, exactly as they
    # depend on include_gap, so the file has to say which method produced them.
    path_method = _resolve_path_method(cell_type, path_method)

    if os.path.isfile(output_path):
        print(f"Updating parameters in {output_path}.")
        with open(output_path, "r") as f:
            dic = json.load(f)
        if isinstance(dic, list):
            if dict_index is None:
                raise ValueError(
                    f"{output_path} holds a list of parameter dictionaries. "
                    "Pass 'dict_index' to select which entry to update."
                )
            target = dic[dict_index]
        else:
            target = dic
        target["n_blocks"] = n_blocks
        target["cell_type"] = cell_type
        target["component_list"] = component_list
        target["include_gap"] = include_gap
        target["path_method"] = path_method

    else:
        dic = {}
        target = dic
        target["seg_table"] = table_path
        target["n_blocks"] = n_blocks
        target["cell_type"] = cell_type
        target["component_list"] = component_list
        target["include_gap"] = include_gap
        target["path_method"] = path_method

    centers = equidistant_centers(
        table, component_label=component_list, cell_type=cell_type,
        n_blocks=n_blocks, offset_blocks=offset_blocks, include_gap=include_gap, path_method=path_method,
    )
    centers = [[round(c) for c in center] for center in centers]

    target["crop_centers"] = centers

    with open(output_path, "w") as f:
        json.dump(dic, f, indent='\t', separators=(',', ': '))


def equidistant_centers_json_wrapper(
    json_file: str,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    overrides: Optional[dict] = None,
    **kwargs,
):
    """Recompute the crop centers of every entry of a JSON file and write them back in place.

    The segmentation table of an entry is derived from its "dataset_name" and
    "segmentation_channel", so the crop centers of a whole parameter file can be refreshed
    without naming any path.

    Args:
        json_file: JSON file with one parameter dictionary, or a list of them.
        mobie_dir: Local MoBIE directory used for creating data paths. Ignored when s3 is set.
        s3: Flag for accessing data stored on S3 bucket.
        overrides: Parameters that win over the entry, for the flags the caller set explicitly.
        kwargs: Further arguments for equidistant_centers_single. An entry of the JSON file
            overrides them, and overrides wins over both.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    is_list = isinstance(data, list)
    param_dicts = data if is_list else [data]

    # equidistant_centers_single writes n_blocks, include_gap and crop_centers back into the
    # entry, none of which a processing file may hold at the top level, so it would make that
    # file unreadable for the three processing steps.
    for entry in param_dicts:
        sections = sorted(set(entry) & set(STEP_KEYS))
        if sections:
            raise ValueError(
                f"{json_file} is a processing parameter file, holding the section(s) {sections}. "
                "The crop centers of a block extraction file are updated in place, which would "
                "add keys that a processing file must not have. Point --json_info at a file in "
                "reproducibility/block_extraction instead."
            )

    for index, entry in enumerate(param_dicts):
        cochlea = entry["dataset_name"]
        print(f"\n{cochlea}")
        seg_channel = entry["segmentation_channel"]

        table_path = default_table_path(cochlea, seg_channel, s3=s3, mobie_dir=mobie_dir)

        equidistant_centers_single(
            table_path=table_path,
            output_path=json_file,
            dict_index=index if is_list else None,
            s3=s3,
            **{**kwargs, **entry, **(overrides or {})},
        )


def tonotopic_mapping_json_wrapper(
    out_path: str,
    table_path: Optional[str] = None,
    json_file: Optional[str] = None,
    central_spots_path: Optional[str] = None,
    force_overwrite: bool = False,
    animal: str = "mouse",
    otof: bool = False,
    s3: bool = False,
    mobie_dir: str = MOBIE_FOLDER,
    **kwargs
):
    """Wrapper function for tonotopic mapping using a segmentation table.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.

    Args:
        out_path: Output path for the mapped segmentation table, or a directory when the JSON file
            holds several entries.
        table_path: File path to segmentation table. Ignored when json_file is given.
        json_file: JSON file containing parameters for tonotopic mapping.
        central_spots_path: Provide table featuring spots for central path through segmentation for tonotopic mapping.
        force_overwrite: Forcefully overwrite existing output path.
        animal: Animal specifier for species specific frequency mapping. Either "mouse" or "gerbil".
            Derived from the cochlea name for every entry of a JSON file.
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
            Derived from the cochlea name for every entry of a JSON file.
        s3: Use data path of S3 bucket for segmentation table.
        mobie_dir: Local MoBIE directory used for creating data paths. Ignored when s3 is set.
        kwargs: Passed to `tonotopic_mapping_single`. A value recorded in the JSON file wins.
    """
    if json_file is None:
        tonotopic_mapping_single(table_path, out_path=out_path, animal=animal, otof=otof,
                                 central_spots_path=central_spots_path,
                                 force_overwrite=force_overwrite, s3=s3, **kwargs)
    else:
        if out_path is None:
            raise ValueError("Specify an output path when supplying a JSON dictionary.")
        param_dicts = load_processing_params(json_file, "tonotopic_mapping")
        if not param_dicts:
            print(f"{json_file} has no 'tonotopic_mapping' section. Nothing to do.")
            return

        # An output path that is not a TSV file names a directory, which is created if needed.
        out_is_dir = not out_path.endswith(".tsv")
        if out_is_dir:
            os.makedirs(out_path, exist_ok=True)
        elif len(param_dicts) > 1:
            raise ValueError(
                f"{json_file} holds {len(param_dicts)} entries, which cannot share the single "
                f"output file {out_path}. Pass an output directory instead."
            )

        for params in param_dicts:

            cochlea = params["dataset_name"]
            print(f"\n{cochlea}")
            seg_channel = params["segmentation_channel"]
            table_path = default_table_path(cochlea, seg_channel, s3=s3, mobie_dir=mobie_dir)

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

            prefix = table_name_prefix(cochlea, seg_channel)
            save_path, entry_spots_path = out_path, central_spots_path
            if out_is_dir:
                save_path = os.path.join(out_path, f"{prefix}.tsv")
            # Several entries must not share one central path file.
            if central_spots_path is not None and (out_is_dir or len(param_dicts) > 1):
                root = out_path if out_is_dir else os.path.dirname(central_spots_path)
                entry_spots_path = os.path.join(root, f"{prefix}_path.tsv")

            tonotopic_mapping_single(table_path=table_path, out_path=save_path,
                                     force_overwrite=force_overwrite, central_spots_path=entry_spots_path,
                                     s3=s3, **{**kwargs, "animal": animal, "otof": otof, **params})
