"""Recognize the IHCs of an IHC row by their morphology, and correct the row instances with it.

Each IHC has a tip at both ends of its long axis, and the IHCs sit side by side along the central path of the
row. `find_ihc_tips_path` finds the tips, `ihc_row_reference` estimates the typical IHC of a crop, and
`correct_ihc_row` merges, splits and trims the row instances. doc/ihc_postprocessing.md summarizes the analysis
behind the rules and the defaults.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import find_objects, gaussian_filter, gaussian_filter1d, maximum_position
from scipy.ndimage import label as label_structures
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.spatial import cKDTree
from skimage.morphology import h_maxima

from flamingo_tools.analysis.training_data_utils import filter_segmentation_3d
from flamingo_tools.postprocessing.cochlea_mapping import moving_average_3d, path_tangents, resample_path

# The columns of the IHC tip tables of find_ihc_tips_height and find_ihc_tips_path.
TIP_COLUMNS = ["side", "z", "y", "x", "label_id", "cell"]
# Range of the distance between an IHC tip and the central line, relative to the reference IHC length.
# The annotated IHCs of the IHC_v11 training crops have 0.28-0.81 (1st-99th percentile). The upper limit has a
# margin, because the row path of a cochlea is not always at the cell centers: an upper limit of 0.85 misses
# tips in the MWfLS test crops.
TIP_DISTANCE = (0.25, 1.0)
# The shape of a normal IHC instance, from the annotated IHCs of the IHC_v11 training crops (1st-99th percentile;
# `correction_settings` in scripts/ihc_postprocessing/shape_analysis.py): the length along the long axis is at most
# 1.18 times the reference length (the 0.7 lower limit is the rule for a short instance), and the elongation (main
# semi-axis / second semi-axis) is 1.19-3.02.
MAX_LENGTH_AXIS = 1.18
ELONGATION = (1.19, 3.02)
# Half widths in micrometer of the path smoothing and of the path window for the local axis, see correct_ihc_row.
PATH_SMOOTHING = 10.0
AXIS_WINDOW = 15.0


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
        # The principal axes of an instance with less than 4 voxels are not defined.
        values, vectors = np.linalg.eigh(np.cov(voxels.T)) if len(voxels) > 3 else (np.ones(3), np.eye(3))
        features[label_id] = {
            "n_pixels": len(voxels),
            "distance": distance * voxel_size,
            "tangent": tangents[index],
            "index": index,
            "axis": vectors[:, -1],
            "elongation": float(np.sqrt(values[-1] / max(values[-2], 1e-6))),
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


def _valley(extent: np.ndarray, lo: int, hi: int) -> Optional[int]:
    """Return the position of the smallest column length in [lo, hi], or None if it is at the window edge.

    Of equal minima, the one nearest the middle is taken. A minimum at the window edge is a slope, not a neck.
    """
    window = extent[lo:hi + 1]
    minima = np.flatnonzero(window == window.min())
    best = int(minima[np.argmin(np.abs(minima - (hi - lo) / 2))])
    return lo + best if 0 < best < hi - lo else None


def _neck(extent: np.ndarray, a: int, b: int) -> int:
    """Return the boundary between two cells with the centers a and b.

    The boundary is the valley of the column length in the middle half between the centers, so that a minimum
    next to a center cannot leave a tiny cell. Without a valley there, it is the valley between the centers.
    Without any valley, the column length is a slope, for example at the end of a component, and the boundary is
    in the middle.
    """
    for lo, hi in ((a + (b - a) // 4, b - (b - a) // 4), (a, b)):
        valley = _valley(extent, lo, hi)
        if valley is not None:
            return valley
    return (a + b) // 2


def _partition_along_path(
    index: np.ndarray,
    height: np.ndarray,
    centers: np.ndarray,
    cell_size: float,
    cell_width: float,
    min_size: float,
    max_size: float,
    min_depth: float,
) -> np.ndarray:
    """Split voxels into the cells that sit side by side along the path.

    The boundary between two neighbouring cells is at the minimum of the column length (the extent of the
    voxels along the axis at one foot point) in the middle half between their centers: the neck between two
    IHCs. While more than one cell is left, a cell loses its center if:
    - it is the smallest cell and has less than `min_size` voxels;
    - it is the smaller cell at the shallowest neck, the neck is less than `min_depth` below the longest
      column of both cells, and both cells together have `max_size` voxels or less. A false tip on a single
      cell gives no neck. Two full cells without a neck stay two cells.
    Then each cell with more than `max_size` voxels is split by its size and its extent along the path: into
    the smaller of round(size / `cell_size`) and round(extent / `cell_width`) cells. The boundaries are at the
    necks next to the equal-size positions. This adds the cells without a tip. The extent keeps one cell whose
    upper and lower part come from two cells, because they sit at one position.

    Args:
        index: The path index of the foot point of each voxel, see `foot_points`.
        height: The height of each voxel, see `foot_points`.
        centers: The path index of the center of each cell. Can be empty.
        cell_size: Typical number of voxels of a cell.
        cell_width: Typical extent of a cell along the path (2-98 percentile span of the foot points), in pixels.
        min_size: Minimal number of voxels of a cell.
        max_size: Maximal number of voxels of a cell.
        min_depth: Minimal depth of a neck, in pixels.

    Returns:
        The cell of each voxel, numbered from 0 along the path.
    """
    start = index.min()
    index = index - start
    top, bottom = np.full(index.max() + 1, -np.inf), np.full(index.max() + 1, np.inf)
    np.maximum.at(top, index, height)
    np.minimum.at(bottom, index, height)
    extent = gaussian_filter1d(np.where(np.isfinite(top), top - bottom, 0.0), 1.0)
    positions = sorted(np.clip(np.round(centers).astype(int) - start, 0, len(extent) - 1).tolist()) or [0]
    while True:
        bounds = [_neck(extent, a, b) for a, b in zip(positions[:-1], positions[1:])]
        assignment = np.searchsorted(bounds, index, side="right")
        sizes = np.bincount(assignment, minlength=len(positions))
        if len(positions) == 1:
            break
        if sizes.min() < min_size:
            positions.pop(int(np.argmin(sizes)))
            continue
        edges = [0, *bounds, len(extent) - 1]
        longest = [extent[a:b + 1].max() for a, b in zip(edges[:-1], edges[1:])]
        depths = [min(longest[k], longest[k + 1]) - extent[bound] for k, bound in enumerate(bounds)]
        mergeable = [k for k in range(len(bounds)) if sizes[k] + sizes[k + 1] <= max_size]
        k = min(mergeable, key=depths.__getitem__) if mergeable else None
        if k is None or depths[k] >= min_depth:
            break
        positions.pop(k if sizes[k] < sizes[k + 1] else k + 1)

    for cell in np.flatnonzero(sizes > max_size):
        in_cell = index[assignment == cell]
        n_cells = min(_round(sizes[cell] / cell_size), _round(np.ptp(np.percentile(in_cell, [2, 98])) / cell_width))
        cumulative = np.cumsum(np.bincount(in_cell, minlength=len(extent)))
        # The valley is searched within a quarter cell of each equal-size position, so no part is tiny.
        fractions = (np.arange(1, n_cells)[:, None] + np.array([-0.25, 0.25])) / n_cells
        windows = np.searchsorted(cumulative, cumulative[-1] * fractions)
        bounds.extend(v if (v := _valley(extent, lo, hi)) is not None else (lo + hi) // 2 for lo, hi in windows)
    return np.searchsorted(sorted(bounds), index, side="right")


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
    max_size: float,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Merge touching row instances whose foot points are less than `max_distance` path points apart.

    The cells sit side by side along the path, so two touching instances at the same position are parts of one
    cell, for example its upper and lower half. The larger part keeps its id. A merged instance has at most
    `max_size` voxels. Two short instances are merged only if they are stacked along the axis: their height
    ranges overlap by less than half of the shorter range. So the short fragments of two neighbouring cells at
    about the same position stay apart, and the upper and lower part of one cell are merged.

    Returns:
        The merged segmentation, and the id that each merged instance joins.
    """
    candidates = [i for i, f in features.items() if f["row"]]
    if not candidates:
        return segmentation, {}
    index, _, height = foot_points(np.concatenate([coords[i] for i in candidates]).astype(float), points, axes)
    sizes = np.cumsum([len(coords[i]) for i in candidates])[:-1]
    position = {i: np.median(part) for i, part in zip(candidates, np.split(index, sizes))}
    heights = {i: np.percentile(part, [5, 95]) for i, part in zip(candidates, np.split(height, sizes))}

    def stacked(a, b):
        overlap = min(heights[a][1], heights[b][1]) - max(heights[a][0], heights[b][0])
        return overlap < 0.5 * min(np.ptp(heights[a]), np.ptp(heights[b]))
    target = {i: i for i in candidates}
    size = {i: features[i]["n_pixels"] for i in candidates}

    def root(i):
        while target[i] != i:
            i = target[i]
        return i

    for a, b in _touching_pairs(segmentation).tolist():
        if (a in position and b in position and abs(position[a] - position[b]) < max_distance
                and (not features[a]["short"] and not features[b]["short"] or stacked(a, b))):
            small, large = sorted((root(a), root(b)), key=size.get)
            if small != large and size[small] + size[large] <= max_size:
                target[small] = large
                size[large] += size[small]
    part_of = {i: root(i) for i in candidates if root(i) != i}
    if not part_of:
        return segmentation, {}
    merged = segmentation.copy()
    for i, j in part_of.items():
        merged[tuple(coords[i].T)] = j
    return merged, part_of


def _contacts(labels: np.ndarray, voxels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return the labels with face contacts to the voxels and the number of contacts."""
    neighbours = []
    for dim in range(3):
        for step in (-1, 1):
            shifted = voxels.copy()
            shifted[:, dim] += step
            inside = (shifted[:, dim] >= 0) & (shifted[:, dim] < labels.shape[dim])
            neighbours.append(labels[tuple(shifted[inside].T)])
    neighbours = np.concatenate(neighbours)
    return np.unique(neighbours[neighbours > 0], return_counts=True)


def _largest_contact(labels: np.ndarray, voxels: np.ndarray) -> int:
    """Return the label with the most face contacts to the voxels, or 0 if they touch no label."""
    values, counts = _contacts(labels, voxels)
    return values[np.argmax(counts)] if len(values) else 0


def _contact_along_path(
    labels: np.ndarray, voxels: np.ndarray, points: np.ndarray, axes: np.ndarray, margin: int,
) -> int:
    """Return the touching label whose foot points are nearest to the foot points of the voxels, or 0.

    A part of a cell sits at the position of its cell along the path, also if it touches the neighbouring
    cell more. The foot points of a touching label come from its voxels within `margin` pixels of the voxels.
    """
    values, _ = _contacts(labels, voxels)
    if len(values) < 2:
        return values[0] if len(values) else 0
    box = tuple(slice(max(lo - margin, 0), hi + margin + 1) for lo, hi in zip(voxels.min(axis=0), voxels.max(axis=0)))
    start = np.array([sl.start for sl in box])
    position = np.median(foot_points(voxels.astype(float), points, axes)[0])
    distances = [
        abs(np.median(foot_points((np.argwhere(labels[box] == value) + start).astype(float), points, axes)[0])
            - position)
        for value in values
    ]
    return values[int(np.argmin(distances))]


def _reassign_pieces(labels: np.ndarray, min_size: float, next_id: int) -> Dict[int, int]:
    """Keep the largest connected piece of each label, in place.

    A smaller piece goes to the label that it touches most. A piece without a contact gets a new label if it
    has `min_size` voxels or more, else it is removed.

    Returns:
        The number of moved pieces of each label.
    """
    moved = {}
    for label_id, bbox in enumerate(find_objects(labels), start=1):
        if bbox is None:
            continue
        pieces, n_pieces = label_structures(labels[bbox] == label_id)
        if n_pieces < 2:
            continue
        largest = np.argmax(np.bincount(pieces.ravel())[1:]) + 1
        start = np.array([sl.start for sl in bbox])
        for piece in set(range(1, n_pieces + 1)) - {largest}:
            voxels = np.argwhere(pieces == piece) + start
            labels[tuple(voxels.T)] = 0
            target = _largest_contact(labels, voxels)
            if target == 0 and len(voxels) >= min_size:
                target, next_id = next_id, next_id + 1
            labels[tuple(voxels.T)] = target
        moved[label_id] = n_pieces - 1
    return moved


def row_voxels(coords: Dict[int, np.ndarray], ids: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Return the voxels of the instances `ids` and the label of each voxel.

    Args:
        coords: The voxels of each instance, as returned by `row_frame`.
        ids: The instances.
    """
    return np.concatenate([coords[i] for i in ids]), np.concatenate([np.full(len(coords[i]), i) for i in ids])


def _path_tips(
    voxels: np.ndarray,
    labels: np.ndarray,
    points: np.ndarray,
    axes: np.ndarray,
    reference: Dict[str, Any],
    shape: Tuple[int, ...],
    voxel_size: float,
) -> pd.DataFrame:
    """Find the tips with `find_ihc_tips_path` and the tip distance `TIP_DISTANCE` of the reference IHC."""
    tip_distance = (TIP_DISTANCE[0] * reference["length_axis"], TIP_DISTANCE[1] * reference["length_axis"])
    return find_ihc_tips_path(voxels, labels, points, axes, shape, voxel_size, tip_distance=tip_distance)


def row_frame(
    segmentation: np.ndarray,
    path: np.ndarray,
    reference: Dict[str, Any],
    voxel_size: float = 0.38,
    max_row_distance: float = 15.0,
    path_smoothing: float = PATH_SMOOTHING,
    axis_window: float = AXIS_WINDOW,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Measure the instances along the smoothed path, and the local long axis at each path point.

    The local long axis comes from the typical row instances within `axis_window` along the path. This is the
    frame of `correct_ihc_row`.

    Args:
        segmentation: IHC segmentation in (z, y, x) order, with consecutive label ids.
        path: Central path through the IHC row in pixels, in (z, y, x) order, ordered along the row.
        reference: The typical IHC of the crop, as returned by `ihc_row_reference`.
        voxel_size: Isotropic voxel size in micrometer.
        max_row_distance: Maximal distance in micrometer between the center of a row IHC and the path.
        path_smoothing: Half width in micrometer of the moving average of the path.
        axis_window: Half width in micrometer of the path window for the local axis.

    Returns:
        The features of each instance, among them 'n_rel' (size relative to the reference), 'row', 'border',
        'short' and 'typical'. The voxels of each instance. The smoothed path resampled at 1 pixel. The unit
        local axis at each path point, in (z, y, x) order.
    """
    path = moving_average_3d(_densify_path(path)[0], _round(path_smoothing / voxel_size))
    features, coords, points = _instance_features(segmentation, path, voxel_size)
    for feature, voxels in zip(features.values(), coords.values()):
        feature["n_rel"] = feature["n_pixels"] / reference["n_pixels"]
        feature["length_axis"] = _extent(voxels, reference["axis"], voxel_size)
        feature["length_path"] = _extent(voxels, _path_direction(feature["tangent"], reference["axis"]), voxel_size)
        feature["short"] = feature["n_rel"] < 0.6 or feature["length_axis"] < 0.7 * reference["length_axis"]
        feature["row"] = feature["distance"] < max_row_distance
        # A short fragment can have the size of a cell but not its axis. A merge of cells
        # side by side can have the size of a cell, if the reference is a merge, but its axis is along the path.
        feature["typical"] = (feature["row"] and not feature["border"] and not feature["short"]
                              and 0.6 <= feature["n_rel"] <= 1.4 and abs(feature["axis"] @ feature["tangent"]) < 0.7)
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
    stable when one segmentation is fragmented or merges many cells. The estimate has two steps:
    - A preliminary IHC from the instance sizes. If all segmentations split the IHCs into an upper and a lower
      part, the typical instance is a part. Then the columns of the row (the extent of the row
      voxels along the axis at one path point) are longer than the typical instance: their 75th percentile is
      more than 1.5 times its length. In the annotated IHCs, it is 1.02-1.30 times the length (median 1.15).
      The size and the length along the axis are then scaled by ratio / 1.15.
    - The tips mark the cells independently of the instances. In a crop with few cells, the instance sizes
      alone often give a merge of 2 or 3 cells, or a half cell. So the size and the path length of the IHC
      are the medians of the single cells: the inner row instances with one top and one bottom tip of the same
      cell (`find_ihc_tips_path` in the frame of the preliminary IHC). With less than 2 single cells, the size
      is the volume of the inner row instances per tip cell in them.

    Args:
        segmentations: IHC segmentations of the same crop, in (z, y, x) order.
        path: Central path through the IHC row in pixels, in (z, y, x) order, ordered along the row.
        voxel_size: Isotropic voxel size in micrometer.
        max_row_distance: Maximal distance in micrometer between the center of a row IHC and the path.

    Returns:
        The typical IHC: 'n_pixels', 'length_axis' and 'length_path' (extents in micrometer along the
        long axis and along the path), and 'axis', the unit long axis in (z, y, x) order.
    """
    row, member_voxels = [], []
    for segmentation in segmentations:
        features, coords, points = _instance_features(segmentation, path, voxel_size)
        members = [(feature, coords[label_id]) for label_id, feature in features.items()
                   if feature["distance"] < max_row_distance and not feature["border"]]
        row.extend(members)
        if members:
            member_voxels.append(np.concatenate([voxels for _, voxels in members]).astype(float))
    if len(row) == 0:
        raise ValueError("No instance is near the path.")

    sizes = np.array([feature["n_pixels"] for feature, _ in row])
    # The volume-weighted median is the start, so many small fragments do not pull the estimate down.
    sorted_sizes = np.sort(sizes)
    cumulative = np.cumsum(sorted_sizes)
    n_ref = sorted_sizes[np.searchsorted(cumulative, cumulative[-1] / 2)]
    for _ in range(10):
        n_ref = np.median(sizes[(sizes > 0.5 * n_ref) & (sizes < 1.5 * n_ref)])

    # Bimodal sizes, for example halves and merged pairs, can leave no instance near n_ref.
    confident = [(feature, voxels) for feature, voxels in row if 0.7 * n_ref < feature["n_pixels"] < 1.3 * n_ref] or row
    # ponytail: one long axis per crop. Use a local axis along the path if a crop spans a strong turn.
    # The axis of each instance is taken across the path: a merge of cells side by side is longest along the path.
    tensor = np.zeros((3, 3))
    for feature, voxels in confident:
        across = np.eye(3) - np.outer(feature["tangent"], feature["tangent"])
        main = np.linalg.eigh(across @ np.cov(voxels.T) @ across)[1][:, -1]
        tensor += feature["n_pixels"] * np.outer(main, main)
    axis = np.linalg.eigh(tensor)[1][:, -1]
    # Few cells that lean in different directions can leave no aligned instance.
    aligned = [(feature, voxels) for feature, voxels in confident if abs(feature["axis"] @ axis) > 0.9] or confident
    length_axis = np.median([_extent(voxels, axis, voxel_size) for _, voxels in aligned])

    columns = []
    for voxels in member_voxels:
        index, _, height = foot_points(voxels, points, np.tile(axis, (len(points), 1)))
        top, bottom = np.full(len(points), -np.inf), np.full(len(points), np.inf)
        np.maximum.at(top, index, height)
        np.minimum.at(bottom, index, height)
        columns.append(np.percentile((top - bottom)[np.isfinite(top)], 75) * voxel_size)
    ratio = min(columns) / length_axis
    scale = ratio / 1.15 if ratio > 1.5 else 1.0
    preliminary = {
        "n_pixels": float(scale * n_ref),
        "length_axis": float(scale * length_axis),
        "length_path": float(np.median([
            _extent(voxels, _path_direction(feature["tangent"], axis), voxel_size) for feature, voxels in aligned
        ])),
        "axis": axis,
    }

    singles, volume, n_cells = [], 0, 0
    for segmentation in segmentations:
        features, coords, points, axes = row_frame(segmentation, path, preliminary, voxel_size, max_row_distance)
        row_ids = [label_id for label_id, feature in features.items() if feature["row"]]
        if not row_ids:
            continue
        tips = _path_tips(*row_voxels(coords, row_ids), points, axes, preliminary, segmentation.shape, voxel_size)
        inner = {label_id for label_id in row_ids if not features[label_id]["border"]}
        volume += sum(features[label_id]["n_pixels"] for label_id in inner)
        n_cells += sum(set(labels) <= inner for _, labels in tips.groupby("cell")["label_id"])
        for label_id, own in tips.groupby("label_id"):
            if label_id in inner and sorted(own["side"]) == ["bottom", "top"] and own["cell"].nunique() == 1:
                singles.append(features[label_id])
    if len(singles) >= 2:
        return {**preliminary, "n_pixels": float(np.median([feature["n_pixels"] for feature in singles])),
                "length_path": float(np.median([feature["length_path"] for feature in singles]))}
    if n_cells == 0:
        return preliminary
    n_pixels = volume / n_cells
    return {**preliminary, "n_pixels": float(n_pixels),
            "length_path": preliminary["length_path"] * n_pixels / preliminary["n_pixels"]}


def correct_ihc_row(
    segmentation: np.ndarray,
    path: np.ndarray,
    reference: Dict[str, Any],
    voxel_size: float = 0.38,
    max_row_distance: float = 15.0,
    max_radius_factor: float = 1.0,
    tip_method: str = "path",
    path_smoothing: float = PATH_SMOOTHING,
    axis_window: float = AXIS_WINDOW,
    max_axis_angle: float = 30.0,
    min_cell_size: float = 0.4,
    min_neck_depth: float = 1.0,
    max_part_distance: float = 0.4,
    max_cell_size: float = 1.37,
    max_cell_width: float = 1.39,
    min_cell_width: float = 0.7,
    max_length_axis: float = MAX_LENGTH_AXIS,
    elongation: Tuple[float, float] = ELONGATION,
) -> Tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Correct the instances of an IHC row with the morphology of the IHC row.

    Use it as post-processing of an IHC segmentation, or to propose a correction of annotations for a manual
    check ('review' in the returned table).

    The correction uses the morphology of the IHC row: each IHC has a tip at both ends of its long axis, and
    the IHCs sit side by side along the central path. Instances and split-off pieces of an instance with less
    than 100 voxels are removed first. Then touching row instances at the same position along the path (their
    foot points are less than `max_part_distance` times the reference path length apart) are merged, up to
    `max_cell_size`: they are parts of one cell, for example its upper and lower half. Two short instances
    are merged only if they are stacked along the axis (`_merge_parts`). Each instance gets one class:
    - off_row: the instance center is `max_row_distance` or more from the path. It is removed if it is short:
      smaller than 0.6 of the reference size, or shorter than 0.7 of the reference length along the long axis.
    - normal: the instance has the shape of an annotated IHC (1st-99th percentile) and stays unchanged:
      at most one tip on each side, 0.6 to `max_cell_size` of the reference size, `min_cell_width` to
      `max_cell_width` of the reference path width, 0.7 to `max_length_axis` of the reference length along the
      long axis, an elongation in `elongation`, and a long axis `max_axis_angle` or less from the local axis.
    - border: the instance touches the crop border, has at most one tip on each side, at most `max_cell_size`
      of the reference size and at most `max_cell_width` of the reference path width. It stays unchanged.
    - uncertain: all other row instances.

    The tips come from `find_ihc_tips_path` (`tip_method` 'path') or from `find_ihc_tips_height` ('height',
    with the reference axis). The local long axis comes from the typical row instances within `axis_window`
    along the path. The voxels of the uncertain instances form connected components:
    - A component with less than `min_cell_size` of the reference size is a part of a cell. It goes to the
      touching label at the nearest position along the path (`_contact_along_path`), or is removed.
    - A component without a tip cell and with at most `max_cell_size` of the reference size stays unchanged if
      it touches the crop border or has 0.6 or more of the reference size. Else it is a part of a cell, see above.
    - In the other components, the tip cells sit side by side along the path. Each voxel is moved along the
      local axis onto the central line (`foot_points`), and the boundary between two neighbouring cells is at
      the neck between them (`_partition_along_path`). A cell with less than `min_cell_size` of the reference
      size loses its tips. A cell without a neck of `min_neck_depth` to its neighbour loses its tips if both
      cells together fit in `max_cell_size`. A cell larger than `max_cell_size` is split by its size and its
      extent along the path. This adds the cells whose tips are missing.
    The output cell with the largest overlap of an input instance keeps its id. Other output cells get new ids.
    Then the voxels of each normal cell that are farther than `max_radius_factor` times the reference path
    length from its center in the plane perpendicular to the local axis are removed. Finally, each output
    label keeps its largest connected piece. The other pieces go to the label that they touch most
    (`_reassign_pieces`).

    Args:
        segmentation: IHC segmentation in (z, y, x) order.
        path: Central path through the IHC row in pixels, in (z, y, x) order, ordered along the row.
        reference: The typical IHC of the crop, as returned by `ihc_row_reference`.
        voxel_size: Isotropic voxel size in micrometer.
        max_row_distance: Maximal distance in micrometer between the center of a row IHC and the path.
        max_radius_factor: Maximal distance of a cell voxel from the cell center in the projection, relative to
            the reference length along the path. With the default, 99% of the annotated IHCs keep 98% of their
            voxels.
        tip_method: 'path' or 'height', see above.
        path_smoothing: Half width in micrometer of the moving average of the path. A path through the
            instance centers zigzags at the scale of the IHC spacing.
        axis_window: Half width in micrometer of the path window for the local axis.
        max_axis_angle: Maximal angle in degree between the long axis of a normal instance and the local axis.
            97.8% of the annotated IHCs have 20 degree or less, and 1.6% have more than 30 degree.
        min_cell_size: Minimal size of an output cell of an uncertain component, relative to the reference size.
        min_neck_depth: Minimal depth in micrometer of the neck between two output cells of an uncertain
            component. The default 1.0 is about the 5th percentile of the annotated neighbours.
        max_part_distance: Maximal distance along the path between two parts of one cell, relative to the
            reference path length. Neighbouring cells in the MWfLS crops are 0.6 or more apart, and parts of one
            cell 0.35 or less.
        max_cell_size: Maximal size of one cell, relative to the reference size. The default is the 99th
            percentile of the annotated IHCs.
        max_cell_width: Maximal extent of one cell along the path, relative to the reference. The default is the
            98.5th percentile of the annotated IHCs.
        min_cell_width: Minimal extent of a normal instance along the path, relative to the reference. 99% of
            the annotated IHCs have 0.78 (mouse) or 0.85 (gerbil) or more. A half cell that is cut along the
            path has the full length, but half of the width.
        max_length_axis: Maximal length of a normal instance along the long axis, relative to the reference.
        elongation: Range of the elongation (main semi-axis / second semi-axis) of a normal instance.

    Returns:
        The corrected segmentation.
        A table with one row per input label that is not removed first. 'tips' is the largest number of tips
        of the instance on one side. 'output_ids' lists the output labels that hold 10% or more of its voxels.
        'trimmed' is the fraction of its voxels that the trim removes. 'pieces' is the number of pieces that
        `_reassign_pieces` moves from its output labels. 'review' marks an output label with a size outside
        0.6-1.4 of the reference, a kept off_row instance, a trimmed fraction above 0.05, and moved pieces.
        The tips of the row instances, see `find_ihc_tips_height`.
    """
    if tip_method not in ("path", "height"):
        raise ValueError(f"Invalid tip_method {tip_method}. Choose 'path' or 'height'.")
    segmentation = filter_segmentation_3d(segmentation)
    frame = (reference, voxel_size, max_row_distance, path_smoothing, axis_window)
    input_features, input_coords, points, axes = row_frame(segmentation, path, *frame)
    original = segmentation
    segmentation, part_of = _merge_parts(
        segmentation, input_features, input_coords, points, axes,
        max_part_distance * reference["length_path"] / voxel_size, max_cell_size * reference["n_pixels"],
    )
    features, coords = input_features, input_coords
    if part_of:
        features, coords, points, axes = row_frame(segmentation, path, *frame)

    ids = list(features)
    row_ids = [label_id for label_id in ids if features[label_id]["row"]]
    tips = pd.DataFrame(columns=TIP_COLUMNS)
    if row_ids:
        voxels, sources = row_voxels(coords, row_ids)
        if tip_method == "path":
            tips = _path_tips(voxels, sources, points, axes, reference, segmentation.shape, voxel_size)
        else:
            tips = find_ihc_tips_height(voxels, sources, reference["axis"], segmentation.shape, voxel_size)
    # The most tips on one side of an instance is the number of cells that it holds at least.
    tips_per_side = tips.groupby(["label_id", "side"]).size().groupby(level="label_id").max()
    for label_id in ids:
        feature = features[label_id]
        feature["tips"] = int(tips_per_side.get(label_id, 0))
        angle = np.degrees(np.arccos(min(1.0, abs(feature["axis"] @ axes[feature["index"]]))))
        single = (feature["tips"] <= 1 and feature["n_rel"] <= max_cell_size
                  and feature["length_path"] <= max_cell_width * reference["length_path"])
        if not feature["row"]:
            feature["class"] = "off_row"
        elif feature["border"]:
            feature["class"] = "border" if single else "uncertain"
        elif (single and feature["n_rel"] >= 0.6 and angle <= max_axis_angle and not feature["short"]
              and feature["length_axis"] <= max_length_axis * reference["length_axis"]
              and feature["length_path"] >= min_cell_width * reference["length_path"]
              and elongation[0] <= feature["elongation"] <= elongation[1]):
            feature["class"] = "normal"
        else:
            feature["class"] = "uncertain"

    labels = segmentation.copy()
    next_id = int(segmentation.max()) + 1
    # The short off-row instances go first, so that no part of a cell joins them.
    for label_id in ids:
        if features[label_id]["class"] == "off_row" and features[label_id]["short"]:
            labels[tuple(coords[label_id].T)] = 0
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
        tip_cells = tips["cell"].values
        foot_index, _, foot_height = foot_points(voxels.astype(float), points, axes)
        labels[tuple(voxels.T)] = 0
        claimed, short_parts = set(), []
        for component in np.unique(components):
            selection = np.flatnonzero(components == component)
            in_component = tip_components == component
            centers = pd.Series(tip_index[in_component]).groupby(tip_cells[in_component]).mean()
            at_border = _at_border(voxels[selection], segmentation.shape).any()
            # A component with less than the minimal cell size is a part of a cell, also if it has a tip.
            if len(selection) < min_cell_size * reference["n_pixels"] and not at_border:
                short_parts.append(voxels[selection])
                continue
            if len(centers) == 0 and len(selection) <= max_cell_size * reference["n_pixels"]:
                if len(selection) < 0.6 * reference["n_pixels"] and not at_border:
                    short_parts.append(voxels[selection])
                else:
                    labels[tuple(voxels[selection].T)] = sources[selection]
                    claimed.update(np.unique(sources[selection]).tolist())
                continue
            assignment = _partition_along_path(
                foot_index[selection], foot_height[selection], centers.values, reference["n_pixels"],
                reference["length_path"] / voxel_size, min_cell_size * reference["n_pixels"],
                max_cell_size * reference["n_pixels"], min_neck_depth / voxel_size,
            )
            # The output cell with the largest overlap of an input instance keeps its id.
            pairs, counts = np.unique(np.stack([assignment, sources[selection]], axis=1), axis=0, return_counts=True)
            cell_ids = np.zeros(assignment.max() + 1, dtype=segmentation.dtype)
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
            labels[tuple(part.T)] = _contact_along_path(labels, part, points, axes, _round(10.0 / voxel_size))

    # Trim each normal cell to a disk around its center in the plane perpendicular to the local axis. This removes
    # the protrusions at the side of a cell. The tips already set the boundaries of the cells of the uncertain
    # instances. A trim with the reference length cuts them where the reference is too small.
    # The plane of the crop axis cut the tips of correct leaning cells.
    cut = [np.zeros((0, 3), dtype=int)]
    for label_id in ids:
        if features[label_id]["class"] == "normal":
            voxels = coords[label_id]
            projection = voxels @ np.linalg.svd(axes[features[label_id]["index"]][None])[2][1:].T
            radius = np.linalg.norm(projection - np.median(projection, axis=0), axis=1) * voxel_size
            cut.append(voxels[radius > max_radius_factor * reference["length_path"]])
    cut = np.concatenate(cut)
    labels[tuple(cut.T)] = 0
    trimmed = dict(zip(*np.unique(original[tuple(cut.T)], return_counts=True)))

    moved = _reassign_pieces(labels, 0.6 * reference["n_pixels"], next_id)
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
        pieces = sum(moved.get(o, 0) for o in outputs)
        review = pieces > 0 or cell["class"] != "border" and (
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
            "pieces": pieces,
            "review": review,
        })
    return labels, pd.DataFrame(rows), tips
