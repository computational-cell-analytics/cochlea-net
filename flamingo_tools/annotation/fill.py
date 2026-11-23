"""Complete the volume of corrected cells from a few edited slices.

The user corrects the labels on sparse z slices. `fill_sparse` then sets the voxels between the slices:
- On the edited slices, the result is the edit.
- Between two slices, the 2D signed distances of each label and of the union of the labels are interpolated
  linearly in z. The union gives the foreground; in it, the label with the largest distance wins. So two cells
  that share a border leave no gap where the border moves between the slices. A label that is on only one of the
  two slices shrinks and vanishes at `vanish` of the way to the other slice.
- The current labels on the same slices are interpolated in the same way. Only the voxels where the two
  interpolations differ change, so the current labels keep their detail where the edit does not reach.
- Only the labels that the edit changes take part. The other labels stay fixed and act as walls.
"""
from typing import Optional, Sequence, Set, Tuple

import numpy as np
from scipy import ndimage


def select_slices(
    mask: np.ndarray,
    voxel_size: float,
    spacing: float = 2.0,
    cap: float = 2.0,
    inset: float = 0.5,
    max_slices: int = 16,
) -> np.ndarray:
    """Choose the sparse z slices to edit a cell.

    The slices are:
    - a cap slice `cap` beyond each z end of the cell. The fill does not go past the cap slices, so the user
      paints a tip that reaches a cap. At the volume border, the border slice is the cap.
    - an end slice `inset` inside each z end, and slices in between at most `spacing` apart. If this gives more
      than `max_slices` slices, the spacing is larger.

    Args:
        mask: The cell, in (z, y, x) order. It can be the current cell together with a proposal.
        voxel_size: Isotropic voxel size in micrometer.
        spacing: Maximal distance in micrometer between neighbouring inner slices.
        cap: Distance in micrometer of the cap slices from the cell ends.
        inset: Distance in micrometer of the end slices from the cell ends.
        max_slices: Maximal number of slices.

    Returns:
        The sorted z indices of the slices.
    """
    z_cell = np.flatnonzero(mask.any(axis=(1, 2)))
    if len(z_cell) == 0:
        raise ValueError("The cell mask is empty.")
    n_z = mask.shape[0]
    z0, z1 = int(z_cell[0]), int(z_cell[-1])
    cap_px, inset_px = max(1, round(cap / voxel_size)), round(inset / voxel_size)

    caps = []
    if z0 > 0:
        caps.append(max(0, z0 - cap_px))
    if z1 < n_z - 1:
        caps.append(min(n_z - 1, z1 + cap_px))
    low = z0 + inset_px if z0 > 0 else 0
    high = z1 - inset_px if z1 < n_z - 1 else n_z - 1
    if high < low:
        low = high = (z0 + z1) // 2

    n_inner = max(2, max_slices - len(caps))
    n_intervals = min(int(np.ceil((high - low) * voxel_size / spacing)), n_inner - 1)
    inner = np.round(np.linspace(low, high, max(n_intervals, 0) + 1)).astype(int)
    return np.unique(np.concatenate([inner, caps]).astype(int))


def visit_region(
    labels: np.ndarray,
    mask: np.ndarray,
    voxel_size: float,
    slice_axis: int = 0,
    margin: float = 5.0,
    slice_margin: float = 3.0,
    max_margin: float = 15.0,
) -> Tuple[slice, slice, slice]:
    """The region of a visit: the cell with its complete neighbours.

    The region starts as the bounding box of the cell, `margin` larger, and at least `slice_margin` larger along
    the slice axis, so that it holds the cap slices of `select_slices`. Then it grows to the full bounding box of
    each other label in it, but at most `max_margin` beyond the cell.

    Args:
        labels: The labels of the crop, in (z, y, x) order.
        mask: The cell, for example the cell together with a proposal.
        voxel_size: Isotropic voxel size in micrometer.
        slice_axis: The axis of the edit slices.
        margin: Margin in micrometer around the cell.
        slice_margin: Minimal margin in micrometer along the slice axis.
        max_margin: Maximal margin in micrometer around the cell.

    Returns:
        The region as slices, clipped to the volume.
    """
    bbox = ndimage.find_objects(mask.astype("uint8"))
    if not bbox:
        raise ValueError("The cell mask is empty.")
    shape = np.array(mask.shape)
    low, high = np.array([sl.start for sl in bbox[0]]), np.array([sl.stop for sl in bbox[0]])
    margins = np.full(3, round(margin / voxel_size))
    margins[slice_axis] = max(margins[slice_axis], round(slice_margin / voxel_size))
    start_low, start_high = np.maximum(low - margins, 0), np.minimum(high + margins, shape)
    limit = round(max_margin / voxel_size)
    limit_low, limit_high = np.maximum(low - limit, 0), np.minimum(high + limit, shape)

    region_low, region_high = start_low.copy(), start_high.copy()
    box = tuple(slice(lo, hi) for lo, hi in zip(start_low, start_high))
    neighbours = np.unique(labels[box])
    boxes = ndimage.find_objects(labels)
    for label_id in neighbours[neighbours > 0]:
        neighbour_box = boxes[label_id - 1]
        if neighbour_box is None:
            continue
        region_low = np.minimum(region_low, [sl.start for sl in neighbour_box])
        region_high = np.maximum(region_high, [sl.stop for sl in neighbour_box])
    region_low, region_high = np.maximum(region_low, limit_low), np.minimum(region_high, limit_high)
    return tuple(slice(int(lo), int(hi)) for lo, hi in zip(region_low, region_high))


def _signed_distance(mask: np.ndarray, voxel_size: float) -> Optional[np.ndarray]:
    """The 2D signed distance to the mask boundary in micrometer, positive inside. None for an empty mask."""
    if not mask.any():
        return None
    if mask.all():
        return np.full(mask.shape, np.inf)
    inside = ndimage.distance_transform_edt(mask)
    outside = ndimage.distance_transform_edt(~mask)
    return np.where(mask, inside - 0.5, 0.5 - outside) * voxel_size


def _interval_distances(
    edit_a: np.ndarray, edit_b: np.ndarray, n: int, ids: Sequence[int], voxel_size: float, vanish: float,
) -> np.ndarray:
    """The interpolated signed distance of each label on the n - 1 slices between two edited slices."""
    t = (np.arange(1, n) / n)[:, None, None]
    distances = np.full((len(ids), n - 1) + edit_a.shape, -np.inf, dtype="float32")
    for index, label_id in enumerate(ids):
        phi_a = _signed_distance(edit_a == label_id, voxel_size)
        phi_b = _signed_distance(edit_b == label_id, voxel_size)
        if phi_a is None and phi_b is None:
            continue
        # A label on one slice only vanishes at the fraction `vanish` of the way to the other slice.
        if phi_a is None:
            phi_a = phi_b - np.max(phi_b) / vanish
        if phi_b is None:
            phi_b = phi_a - np.max(phi_a) / vanish
        distances[index] = (1 - t) * phi_a + t * phi_b
    return distances


def _interpolate_labels(
    slice_a: np.ndarray, slice_b: np.ndarray, n: int, ids: Sequence[int], voxel_size: float, vanish: float,
) -> np.ndarray:
    """The labels on the n - 1 slices between two slices.

    Each label alone would leave a gap where one cell recedes and its neighbour advances: midway, the distances of
    both are negative. So the interpolated union of the labels sets the foreground, and in it the label with the
    largest distance wins, also if its distance is negative.
    """
    distances = _interval_distances(slice_a, slice_b, n, ids, voxel_size, vanish)
    union_a, union_b = (np.isin(s, ids).astype("uint8") for s in (slice_a, slice_b))
    union = _interval_distances(union_a, union_b, n, [1], voxel_size, vanish)[0]
    return np.where(union > 0, np.asarray(ids)[distances.argmax(axis=0)], 0)


def _fill_interval(
    labels: np.ndarray,
    edit: np.ndarray,
    result: np.ndarray,
    a: int,
    b: int,
    changed_ids: Set[int],
    voxel_size: float,
    vanish: float,
) -> Optional[np.ndarray]:
    """Fill the slices between the edited slices a and b in place. Returns the mask of the filled voxels."""
    n = b - a
    # A changed label that is on neither slice keeps its voxels between them, for example the end of a neighbour.
    ids = sorted(
        label_id for label_id in changed_ids
        if any((slice_ == label_id).any() for slice_ in (edit[a], edit[b], labels[a], labels[b]))
    )
    if n < 2 or not ids:
        return None
    interior = labels[a + 1:b]
    domain = (interior == 0) | np.isin(interior, ids)
    filled = _interpolate_labels(edit[a], edit[b], n, ids, voxel_size, vanish)
    # The interpolation of the current labels on the same slices. Only the voxels where the edit changes the
    # interpolation change, so the current labels keep their detail where the edit does not reach.
    current = _interpolate_labels(labels[a], labels[b], n, ids, voxel_size, vanish)
    region = filled != current
    erased = [label_id for label_id in ids if not ((edit[a] == label_id).any() or (edit[b] == label_id).any())]
    region |= np.isin(interior, erased)

    update = domain & region
    result[a + 1:b][update] = filled[update]
    return update


def fill_sparse(
    labels: np.ndarray,
    edit: np.ndarray,
    z: Sequence[int],
    voxel_size: float,
    vanish: float = 0.7,
) -> Tuple[np.ndarray, Set[int]]:
    """Complete the labels from the edited sparse z slices.

    Only the z intervals next to a changed slice are filled again. In them, only the voxels of the changed
    labels and the background change. After the fill, a piece of a changed label that lies in the filled voxels
    and does not touch the label on an edited slice is removed.

    Args:
        labels: The current labels, in (z, y, x) order.
        edit: The edited labels. Only the slices in `z` are read.
        z: The z indices of the edited slices.
        voxel_size: Isotropic voxel size in micrometer.
        vanish: Fraction of the slice interval where a label that is on one slice only vanishes.

    Returns:
        The completed labels.
        The labels that lost voxels.
    """
    z = np.unique(np.asarray(z, dtype=int))
    result = labels.copy()
    changed = [index for index, slice_z in enumerate(z) if not np.array_equal(edit[slice_z], labels[slice_z])]
    if not changed:
        return result, set()

    difference = edit[z[changed]] != labels[z[changed]]
    changed_ids = set(np.unique(edit[z[changed]][difference]).tolist())
    changed_ids |= set(np.unique(labels[z[changed]][difference]).tolist())
    changed_ids.discard(0)
    result[z] = edit[z]

    intervals = sorted({i for index in changed for i in (index - 1, index) if 0 <= i < len(z) - 1})
    filled = np.zeros(labels.shape, dtype=bool)
    for i in intervals:
        a, b = int(z[i]), int(z[i + 1])
        domain = _fill_interval(labels, edit, result, a, b, changed_ids, voxel_size, vanish)
        if domain is not None:
            filled[a + 1:b] = domain

    on_slices = np.zeros(labels.shape, dtype=bool)
    on_slices[z] = True
    for label_id in changed_ids:
        pieces, n_pieces = ndimage.label(result == label_id)
        if n_pieces < 2:
            continue
        anchored = np.unique(pieces[(pieces > 0) & (on_slices | ~filled)])
        result[(pieces > 0) & ~np.isin(pieces, anchored)] = 0

    ids = np.asarray(sorted(changed_ids))
    before = ndimage.sum_labels(np.ones(labels.shape), labels, ids) if len(ids) else []
    after = ndimage.sum_labels(np.ones(labels.shape), result, ids) if len(ids) else []
    shrunk = {int(label_id) for label_id, n0, n1 in zip(ids, before, after) if n1 < n0}
    return result, shrunk
