"""Clean the labels after an edit: stray pieces, holes and interleaved borders.

Only the editable labels change, and only near the voxels that the edit changed. All other labels stay fixed. The
border smoothing works on the borders between editable cells only: the outer surface of a cell, for example a thin
tip, stays as the fill made it. The fixed labels and the background compete in the smoothing, so a cell does not
take the voxels of another cell at a junction with a third one.
"""
from typing import Callable, Collection, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


FACES_2D = ndimage.generate_binary_structure(2, 1)
FACES_3D = ndimage.generate_binary_structure(3, 1)


def _changed_ids(labels: np.ndarray, before: np.ndarray, editable: Collection[int]) -> np.ndarray:
    changed = labels != before
    ids = np.union1d(np.unique(labels[changed]), np.unique(before[changed]))
    return ids[(ids > 0) & np.isin(ids, list(editable))]


def _contact_label(labels: np.ndarray, piece: np.ndarray, candidates: Collection[int], structure) -> int:
    """The candidate label with the most face contacts to a piece, or 0."""
    shell = ndimage.binary_dilation(piece, structure) & ~piece
    values, counts = np.unique(labels[shell], return_counts=True)
    keep = (values > 0) & np.isin(values, list(candidates))
    return int(values[keep][np.argmax(counts[keep])]) if keep.any() else 0


def _fill_holes(labels: np.ndarray, label_id: int, zone: Optional[np.ndarray] = None) -> None:
    """Give the holes of a label to the label, in place.

    A hole is a region that the label encloses and that holds only background. A region that holds another label,
    for example where a cell wraps around the tip of its neighbour, stays. With `zone`, only holes that touch the
    zone count. A cell body has no lumen, so a hole of any size is an error of the edit or the fill.
    """
    mask = labels == label_id
    enclosed = ndimage.binary_fill_holes(mask) & ~mask
    if not enclosed.any():
        return
    pieces, n_pieces = ndimage.label(enclosed)
    keep = np.bincount(pieces.ravel(), weights=labels.ravel() != 0, minlength=n_pieces + 1) == 0
    keep[0] = False
    if zone is not None:
        keep &= np.bincount(pieces.ravel(), weights=zone.ravel(), minlength=n_pieces + 1) > 0
    labels[keep[pieces]] = label_id


def clean_slice(
    slice_: np.ndarray,
    before: np.ndarray,
    editable: Collection[int],
    voxel_size: float,
    min_area: float = 2.0,
) -> np.ndarray:
    """Clean an edited 2D slice: remove specks and fill the holes of the labels that the edit changed.

    - A piece of a changed label with less than `min_area` um2 that is not its largest piece goes to the
      touching editable label with the most contacts, or to the background.
    - A hole of a changed label goes to the label, see `_fill_holes`.

    Args:
        slice_: The edited slice.
        before: The slice before the edit.
        editable: The labels that may change.
        voxel_size: Isotropic voxel size in micrometer.
        min_area: Minimal area in um2 of a second piece of a label.

    Returns:
        The cleaned slice.
    """
    out = slice_.copy()
    ids = _changed_ids(out, before, editable)
    candidates = set(editable)
    min_pixels = min_area / voxel_size ** 2
    for label_id in ids:
        pieces, n_pieces = ndimage.label(out == label_id, np.ones((3, 3), dtype=bool))
        if n_pieces < 2:
            continue
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        for piece_id in np.flatnonzero((sizes > 0) & (sizes < min_pixels)):
            if piece_id == largest:
                continue
            piece = pieces == piece_id
            out[piece] = _contact_label(out, piece, candidates - {label_id}, FACES_2D)
    for label_id in ids:
        _fill_holes(out, label_id)
    return out


def _pieces(
    mask: np.ndarray, structure: np.ndarray, border: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The connected pieces of a mask, their sizes (index 0 is zero), and the pieces at the region border."""
    pieces, _ = ndimage.label(mask, structure)
    sizes = np.bincount(pieces.ravel())
    sizes[0] = 0
    return pieces, sizes, np.unique(pieces[border & (pieces > 0)])


def _smooth(labels: np.ndarray, movable: np.ndarray, targets: Collection[int], sigma: float) -> None:
    """Set the movable voxels to the label with the largest smoothed mask, if it is a target label, in place."""
    pad = int(np.ceil(3 * sigma)) + 1
    voxels = np.argwhere(movable)
    box = tuple(slice(max(lo - pad, 0), min(hi + pad + 1, size))
                for lo, hi, size in zip(voxels.min(axis=0), voxels.max(axis=0), labels.shape))
    region, inside = labels[box], movable[box]
    competitors = np.unique(region)
    best = np.full(region.shape, -1.0, dtype="float32")
    winner = region.copy()
    for label_id in competitors:
        smoothed = ndimage.gaussian_filter((region == label_id).astype("float32"), sigma)
        better = smoothed > best
        best[better], winner[better] = smoothed[better], label_id
    update = inside & np.isin(winner, list(targets))
    region[update] = winner[update]


def clean_labels(
    labels: np.ndarray,
    before: np.ndarray,
    editable: Collection[int],
    voxel_size: float,
    sigma: float = 1.0,
    band: float = 1.0,
    split_size: int = 1000,
    new_label: Optional[Callable[[], int]] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Clean a filled region after an edit.

    1. Border smoothing: within `band` um of the changed voxels, a voxel of an editable label takes the editable
       label with the largest Gaussian-smoothed mask (`sigma` in pixels, 0 switches it off). It keeps its label if a
       fixed label or the background wins, so only the borders between editable cells move.
    2. Pieces: each changed label keeps its largest piece and the pieces at the region border. A piece with
       `split_size` voxels or more and no contact to the cell is a second cell with the same id, so it gets a new
       label from `new_label`. A smaller piece without any contact to the cell, or with an edge or corner contact
       only near the change, goes to the touching editable label with the most contacts, or to the background.
    3. Holes: the holes of a changed label near the change go to the label (`_fill_holes`): in 3D, and on the
       planes along each axis.

    Args:
        labels: The filled region.
        before: The region before the edit.
        editable: The labels that may change.
        voxel_size: Isotropic voxel size in micrometer.
        sigma: Smoothing of the label masks in pixels.
        band: Width in micrometer of the smoothed band around the changed voxels.
        split_size: Minimal size in voxels of a piece that becomes a new cell.
        new_label: Gives a new label id. Without it, a large piece is kept.

    Returns:
        The cleaned region.
        The label that each new label of a split piece came from.
    """
    out = labels.copy()
    changed = out != before
    if not changed.any():
        return out, {}
    zone = ndimage.binary_dilation(changed, FACES_3D, iterations=max(1, round(band / voxel_size)))
    editable = sorted(set(editable) - {0})
    if sigma > 0:
        movable = zone & np.isin(out, editable)
        if movable.any():
            _smooth(out, movable, set(editable), sigma)
    ids = _changed_ids(out, before, editable)

    splits = {}
    border = np.ones(out.shape, dtype=bool)
    border[(slice(1, -1),) * out.ndim] = False
    candidates = set(editable)
    for label_id in ids:
        # A second cell with the same id: a large piece without any contact, also at an edge or corner.
        pieces, sizes, at_border = _pieces(out == label_id, np.ones((3, 3, 3), dtype=bool), border)
        for piece_id in np.flatnonzero(sizes >= split_size):
            if piece_id == np.argmax(sizes) or piece_id in at_border or new_label is None:
                continue
            split_id = new_label()
            out[pieces == piece_id] = split_id
            splits[split_id] = int(label_id)
        # A stray piece: a small piece without any contact to the cell, or one that touches the cell only at an edge
        # or a corner near the change. Edge contacts away from the change are part of the cell as annotated.
        loose, loose_sizes, _ = _pieces(out == label_id, np.ones((3, 3, 3), dtype=bool), border)
        apart = (loose > 0) & (loose != np.argmax(loose_sizes))
        pieces, sizes, at_border = _pieces(out == label_id, FACES_3D, border)
        stray = np.unique(pieces[(zone | apart) & (pieces > 0)])
        for piece_id in np.flatnonzero((sizes > 0) & (sizes < split_size)):
            if piece_id == np.argmax(sizes) or piece_id in at_border or piece_id not in stray:
                continue
            piece = pieces == piece_id
            out[piece] = _contact_label(out, piece, candidates - {label_id}, FACES_3D)

    # Holes on the planes of all axes: a seam along y or x is a hole only in the planes across it.
    for label_id in list(ids) + list(splits):
        _fill_holes(out, label_id, zone)
        for plane_axis in range(out.ndim):
            moved, moved_zone = np.moveaxis(out, plane_axis, 0), np.moveaxis(zone, plane_axis, 0)
            present = (moved == label_id).any(axis=(1, 2)) & moved_zone.any(axis=(1, 2))
            for index in np.flatnonzero(present):
                _fill_holes(moved[index], label_id, moved_zone[index])
    return out, splits
