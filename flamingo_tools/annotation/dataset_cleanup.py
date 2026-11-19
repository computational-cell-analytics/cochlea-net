"""Clean a complete annotation before it goes into a training set.

`remove_particles`, `fill_enclosed_background` and `fill_plane_holes` reassign small pieces and fill holes in the
whole volume. `remap_high_ids` gives label ids above a limit new ids. They are for the finalization of a training
data set, not for the correction loop (`cleanup` cleans the changes of one visit).
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .cleanup import FACES_2D, _contact_label
from .ids import compact


def remap_high_ids(labels: np.ndarray, limit: int = 65535) -> Tuple[np.ndarray, Dict[int, int]]:
    """Give the label ids above a limit the next free ids after the largest id up to the limit.

    The other ids stay, so that they still match the comments and tables of the annotation.

    Args:
        labels: The segmentation.
        limit: The largest id that stays.

    Returns:
        The segmentation as uint32.
        The new id of each remapped id, in the order of the old ids.
    """
    compact_labels, original = compact(labels)
    high = original > limit
    if not high.any():
        return labels.astype("uint32"), {}
    start = int(original[~high].max()) + 1
    table = original.copy()
    table[high] = start + np.arange(high.sum())
    if table.max() > limit:
        raise ValueError(f"The remapped label ids exceed the limit {limit}.")
    mapping = {int(old): int(new) for old, new in zip(original[high], table[high])}
    return table.astype("uint32")[compact_labels], mapping


def _padded_box(box: Tuple[slice, ...], shape: Sequence[int], pad: int = 1) -> Tuple[slice, ...]:
    return tuple(slice(max(sl.start - pad, 0), min(sl.stop + pad, size)) for sl, size in zip(box, shape))


def _at_border(box: Tuple[slice, ...], shape: Sequence[int]) -> bool:
    return any(sl.start == 0 or sl.stop == size for sl, size in zip(box, shape))


def _center(mask: np.ndarray, box: Tuple[slice, ...]) -> Tuple[int, ...]:
    """The rounded center of a mask in a box, in the coordinates of the full array."""
    return tuple(int(round(c)) + sl.start for c, sl in zip(ndimage.center_of_mass(mask), box))


def remove_particles(
    labels: np.ndarray, min_size: int = 1000, min_border_size: int = 100,
) -> Tuple[np.ndarray, List[dict]]:
    """Reassign the small isolated pieces of a complete annotation.

    A particle is:
    - a label with less than `min_size` voxels that does not touch the volume border;
    - a label at the volume border with less than `min_border_size` voxels. A larger one is a cell that the crop
      cuts;
    - a face-connected piece of a label besides its largest piece, with less than `min_size` voxels.

    A particle goes to the touching label with the most face contacts, or to the background. Touching particle
    labels go together to one label, so the result does not depend on the label ids.

    Args:
        labels: The annotation.
        min_size: The minimal size in voxels of a label inside the volume and of a second piece.
        min_border_size: The minimal size in voxels of a label at the volume border.

    Returns:
        The cleaned annotation.
        One record per particle: kind ('label' or 'piece'), at_border, label_id, target (0 is the background),
        n_voxels and center.
    """
    compact_labels, original = compact(labels)
    structure = ndimage.generate_binary_structure(labels.ndim, 1)
    records = []

    def reassign(box, piece, kind, at_border):
        region = compact_labels[box]
        target = _contact_label(region, piece, np.unique(region), structure)
        for label_id in np.unique(region[piece]):
            part = piece & (region == label_id)
            records.append({"kind": kind, "at_border": bool(at_border[label_id]), "label_id": int(original[label_id]),
                            "target": int(original[target]), "n_voxels": int(part.sum()), "center": _center(part, box)})
        region[piece] = target

    label_sizes = np.bincount(compact_labels.ravel())
    at_border = np.array([False] + [_at_border(box, labels.shape) for box in ndimage.find_objects(compact_labels)])
    is_particle = (label_sizes < min_border_size) | ((label_sizes < min_size) & ~at_border)
    is_particle[0] = False
    groups, _ = ndimage.label(is_particle[compact_labels], structure)
    for group_id, box in enumerate(ndimage.find_objects(groups), start=1):
        padded = _padded_box(box, labels.shape)
        reassign(padded, groups[padded] == group_id, "label", at_border)

    # The boxes again: a label can have grown by a particle.
    for label_id, box in enumerate(ndimage.find_objects(compact_labels), start=1):
        if box is None:
            continue
        padded = _padded_box(box, labels.shape)
        pieces, n_pieces = ndimage.label(compact_labels[padded] == label_id, structure)
        if n_pieces < 2:
            continue
        sizes = np.bincount(pieces.ravel())
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        for piece_id in np.flatnonzero((sizes > 0) & (sizes < min_size)):
            if piece_id != largest:
                piece = pieces == piece_id
                local = ndimage.find_objects(piece.astype("uint8"))[0]
                box = tuple(slice(p.start + sl.start, p.start + sl.stop) for p, sl in zip(padded, local))
                reassign(padded, piece, "piece", {label_id: _at_border(box, labels.shape)})
    return original.astype(labels.dtype)[compact_labels], records


def fill_enclosed_background(labels: np.ndarray, max_island: int = 100) -> Tuple[np.ndarray, List[dict]]:
    """Fill the background that the labels of a complete annotation enclose in 3D.

    An island is a background region without a face path to the volume border.
    - An island that one label encloses goes to that label, at any size: a cell body has no lumen.
    - An island between several labels with at most `max_island` voxels: each voxel takes the nearest label.
    - A larger island between several labels stays.

    Args:
        labels: The annotation.
        max_island: The maximal size in voxels of a filled island between several labels.

    Returns:
        The filled annotation.
        One record per island: n_voxels, center, label_ids (the enclosing labels) and filled.
    """
    out = labels.copy()
    structure = ndimage.generate_binary_structure(labels.ndim, 1)
    background = out == 0
    islands, _ = ndimage.label(ndimage.binary_fill_holes(~background, structure) & background, structure)
    records = []
    for island_id, box in enumerate(ndimage.find_objects(islands), start=1):
        if box is None:
            continue
        # An island does not touch the volume border, so the padded box holds all labels that enclose it.
        padded = _padded_box(box, out.shape)
        region = out[padded]
        island = islands[padded] == island_id
        shell = ndimage.binary_dilation(island, structure) & ~island
        enclosing = np.unique(region[shell])
        enclosing = enclosing[enclosing > 0]
        n_voxels = int(island.sum())
        filled = len(enclosing) == 1 or n_voxels <= max_island
        if len(enclosing) == 1:
            region[island] = enclosing[0]
        elif filled:
            indices = ndimage.distance_transform_edt(region == 0, return_distances=False, return_indices=True)
            region[island] = region[tuple(indices)][island]
        records.append({"n_voxels": n_voxels, "center": _center(island, padded),
                        "label_ids": [int(i) for i in enclosing], "filled": bool(filled)})
    return out, records


def fill_plane_holes(labels: np.ndarray, max_area: int = 20) -> Tuple[np.ndarray, List[dict]]:
    """Fill the small holes that one label of a 3D annotation encloses on the planes along each axis.

    A plane hole is a background region of a plane without a face path to the plane border, for example a gap of
    the brush in one slice. A hole with at most `max_area` pixels that one label encloses goes to that label. A
    hole between several labels is a border between cells and stays. The planes of each axis are filled in turn,
    so a hole that a z plane fills is no hole in the y and x planes.

    Args:
        labels: The 3D annotation.
        max_area: The maximal size in pixels of a filled hole.

    Returns:
        The filled annotation.
        One record per hole that one label encloses: axis, n_voxels, center, label_id and filled.
    """
    out = labels.copy()
    records = []
    for axis in range(out.ndim):
        moved = np.moveaxis(out, axis, 0)
        for index in range(moved.shape[0]):
            plane = moved[index]
            holes = ndimage.binary_fill_holes(plane > 0) & (plane == 0)
            if not holes.any():
                continue
            pieces, _ = ndimage.label(holes, FACES_2D)
            for hole_id, box in enumerate(ndimage.find_objects(pieces), start=1):
                padded = _padded_box(box, plane.shape)
                region = plane[padded]
                hole = pieces[padded] == hole_id
                enclosing = np.unique(region[ndimage.binary_dilation(hole, FACES_2D) & ~hole])
                if len(enclosing) != 1:
                    continue
                n_voxels = int(hole.sum())
                filled = n_voxels <= max_area
                if filled:
                    region[hole] = enclosing[0]
                center = list(_center(hole, padded))
                center.insert(axis, index)
                records.append({"axis": axis, "n_voxels": n_voxels, "center": tuple(center),
                                "label_id": int(enclosing[0]), "filled": filled})
    return out, records
