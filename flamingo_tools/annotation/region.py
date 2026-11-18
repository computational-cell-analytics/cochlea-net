"""Propose a new segmentation for a stretch of the IHC row between two cells.

The user names two cells A and B of the row. The stretch between them gets one cell per expected position:
- The row spacing of the crop gives the number of cells, round(distance / spacing) - 1, evenly between A and B.
- The candidates are the row instances of the S3 model segmentations of the cochlea, the row correction of each
  (`correct_ihc_row`, the post-processing of IHC_v11b), the union of two overlapping candidates of two models at
  one position, and the annotated cells of the stretch.
- A candidate loses its voxels on the end cells (of the annotation and of the base). It is dropped if this leaves
  less than half of it, else it is eligible if the rest has the shape of a typical IHC (`ELIGIBLE`) and sits within
  half a position interval of its position.
- At each position, the eligible candidate of the newest model wins, and a tie goes to the better score: the
  shape error and the offset from the position, minus half the agreement with the other models. Without an
  eligible candidate, the best candidate with `MIN_FALLBACK_SIZE` of the typical size wins, else the typical
  IHC shape (`issues.ihc_template`).

All positions along the row are path indices of the foot points (`foot_points`), in pixels.
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial import cKDTree

from .cleanup import clean_labels
from .crop_reference import model_rank
from .issues import RowContext, annotation_row_path, cell_voxels, ihc_template, inner_point

# The shape of a typical IHC relative to the typical IHC of the crop: size, length along the local axis and width
# along the path (the limits of the class 'normal' of `correct_ihc_row`).
ELIGIBLE = {"n_rel": (0.6, 1.4), "length_axis": (0.7, 1.18), "length_path": (0.7, 1.39)}
MIN_FALLBACK_SIZE = 0.4
MAX_ROW_DISTANCE = 15.0
RULES = ("newest", "score")
ANNOTATION = "annotation"
CORRECTED = " row-corrected"
UNION = " + "


@dataclass
class Candidate:
    """A candidate cell at one position of the stretch. `rank` orders the models, a larger rank is newer."""
    source: str
    group: str
    rank: Tuple[int, str, int]
    voxels: np.ndarray = field(repr=False)
    position: float
    n_rel: float
    length_axis: float
    length_path: float
    distance: float
    base_label: Optional[int] = None
    agreement: float = 0.0

    @property
    def eligible(self) -> bool:
        return self.distance < MAX_ROW_DISTANCE and all(
            low <= getattr(self, name) <= high for name, (low, high) in ELIGIBLE.items())

    def score(self, slot: float, interval: float) -> float:
        shape = abs(np.log(self.n_rel)) + abs(self.length_axis - 1) + 0.5 * abs(self.length_path - 1)
        return shape + abs(self.position - slot) / interval - 0.5 * self.agreement

    def describe(self) -> str:
        return (f"{self.source}: size {self.n_rel:.2f}, length {self.length_axis:.2f}, width {self.length_path:.2f}, "
                f"agreement {self.agreement:.2f}")


@dataclass
class RegionCell:
    """A proposed cell. `base_label` is the label of the base that the cell keeps, or None for a new label.

    `foot` holds the position of each voxel along the row.
    """
    slot: float
    voxels: np.ndarray = field(repr=False)
    foot: np.ndarray = field(repr=False)
    comment: str
    base_label: Optional[int] = None


@dataclass
class RegionProposal:
    """The proposal of a stretch: the cells, the base labels that it replaces, and the base labels at the ends."""
    cells: List[RegionCell]
    removed: List[int]
    end_labels: List[int]
    slots: List[float]
    spacing: float


def source_rank(name: str) -> Tuple[int, str, int]:
    """The selection rank of a candidate source: the row correction of a model ranks just above the model.

    A union of two candidates ranks below all models, and the annotation ranks last.
    """
    if name == ANNOTATION:
        return (-2, "", 0)
    if UNION in name:
        return (-1, "", 0)
    corrected = name.endswith(CORRECTED)
    return (*model_rank(name[:-len(CORRECTED)] if corrected else name), int(corrected))


def source_group(name: str) -> str:
    """The model of a source, for the agreement: IHC_v11, IHC_v11b and their row corrections are one model."""
    if name == ANNOTATION:
        return ANNOTATION
    if UNION in name:
        return "union"
    return f"v{source_rank(name)[0]}"


class _Frame:
    """The row frame of the crop: path points, local axes and the typical IHC."""

    def __init__(self, row: RowContext, shape: Sequence[int], voxel_size: float):
        self.row, self.shape, self.voxel_size = row, tuple(shape), voxel_size
        self.tree = cKDTree(row.points)

    def foot_index(self, voxels: np.ndarray) -> np.ndarray:
        from flamingo_tools.postprocessing.ihc_morphology import foot_points

        return foot_points(voxels.astype(float), self.row.points, self.row.axes)[0]

    def position(self, voxels: np.ndarray) -> float:
        return float(np.median(self.foot_index(voxels)))

    def candidate(self, source: str, voxels: np.ndarray, base_label: Optional[int] = None) -> Candidate:
        reference, points = self.row.reference, self.row.points
        position = self.position(voxels)
        index = int(round(position))
        axis = self.row.axes[index]
        tangent = points[min(index + 1, len(points) - 1)] - points[max(index - 1, 0)]
        tangent = tangent - (tangent @ axis) * axis
        tangent /= max(np.linalg.norm(tangent), 1e-6)
        distance = self.tree.query(voxels.mean(axis=0))[0] * self.voxel_size
        return Candidate(
            source, source_group(source), source_rank(source), voxels, position,
            n_rel=len(voxels) / reference["n_pixels"],
            length_axis=_extent(voxels, axis) * self.voxel_size / reference["length_axis"],
            length_path=_extent(voxels, tangent) * self.voxel_size / reference["length_path"],
            distance=distance, base_label=base_label,
        )


def _extent(voxels: np.ndarray, direction: np.ndarray) -> float:
    projection = voxels @ direction
    return float(np.percentile(projection, 98) - np.percentile(projection, 2))


def _iou(a: np.ndarray, b: np.ndarray, shape: Sequence[int]) -> float:
    a_index, b_index = np.ravel_multi_index(a.T, shape), np.ravel_multi_index(b.T, shape)
    intersection = len(np.intersect1d(a_index, b_index, assume_unique=True))
    return intersection / (len(a_index) + len(b_index) - intersection)


def row_correction(labels: np.ndarray, path: np.ndarray, reference: dict, voxel_size: float) -> np.ndarray:
    """The row correction of a segmentation (`correct_ihc_row`), as for IHC_v11b."""
    from flamingo_tools.postprocessing.ihc_morphology import correct_ihc_row

    return correct_ihc_row(labels, path, reference, voxel_size)[0]


def row_context(
    id_labels: np.ndarray, sources: Dict[str, np.ndarray], voxel_size: float,
) -> Tuple[np.ndarray, RowContext]:
    """The row path of the labels and the row frame, with the typical IHC pooled from the labels and the sources."""
    from flamingo_tools.postprocessing.ihc_morphology import ihc_row_reference, row_frame

    path = annotation_row_path(id_labels, cell_voxels(id_labels), voxel_size)
    if path is None:
        raise ValueError("The crop needs 3 cells for the row path of the region proposal.")
    reference = ihc_row_reference([id_labels] + list(sources.values()), path, voxel_size)
    _, _, points, axes = row_frame(id_labels, path, reference, voxel_size)
    return path, RowContext(points, axes, reference)


def _row_spacing(frame: _Frame, coords: Dict[int, np.ndarray], low: float, high: float) -> float:
    """The median distance between neighbouring typical cells outside the stretch, in path pixels."""
    positions = []
    for voxels in coords.values():
        candidate = frame.candidate(ANNOTATION, voxels)
        if candidate.eligible and not low < candidate.position < high:
            positions.append(candidate.position)
    gaps = np.diff(np.sort(positions))
    # A gap of more than 1.6 spacings holds missing cells (the gap rule of `issues.ihc_findings`).
    if len(gaps) >= 2:
        median = np.median(gaps)
        return float(np.median(gaps[gaps <= 1.6 * median]))
    return frame.row.reference["length_path"] / frame.voxel_size


def propose_region(
    base: np.ndarray,
    id_labels: np.ndarray,
    ends: Tuple[int, int],
    sources: Dict[str, np.ndarray],
    voxel_size: float,
    base_source: str = ANNOTATION,
    rule: str = "newest",
    path_row: Optional[Tuple[np.ndarray, RowContext]] = None,
    correct: Optional[Callable[[np.ndarray, np.ndarray, dict, float], np.ndarray]] = row_correction,
) -> RegionProposal:
    """Propose the cells of the row stretch between two cells.

    Args:
        base: The labels that the proposal changes (compact ids).
        id_labels: The labels that name the end cells, the annotation or the base (compact ids). They give the
            row path, and their cells in the stretch are candidates.
        ends: The two end cells, ids of `id_labels`.
        sources: The S3 model segmentations of the crop by source name (compact ids), for example IHC_v11. A
            name that ends with ' row-corrected' is a row correction (`source_variants`).
        voxel_size: Isotropic voxel size in micrometer.
        base_source: The source of the base: 'annotation' or a name in `sources`. Its candidates keep their id.
        rule: 'newest' (the newest eligible model wins) or 'score' (the best eligible score wins).
        path_row: The row path and frame, if they are known (`row_context`).
        correct: The row correction of the sources. None takes the sources as they are.

    Returns:
        The proposal.
    """
    if rule not in RULES:
        raise ValueError(f"Invalid rule {rule}. Choose one of {RULES}.")
    path, row = path_row or row_context(id_labels, sources, voxel_size)
    frame = _Frame(row, base.shape, voxel_size)
    id_coords = cell_voxels(id_labels)
    for end in ends:
        if end not in id_coords:
            raise ValueError(f"The end cell {end} is not in the labels.")
    low, high = sorted(frame.position(id_coords[end]) for end in ends)
    spacing = _row_spacing(frame, id_coords, low, high)
    n_cells = int(round((high - low) / spacing)) - 1
    if n_cells < 1:
        raise ValueError(f"The cells {ends} are neighbours ({(high - low) * voxel_size:.1f} um apart, spacing "
                         f"{spacing * voxel_size:.1f} um), so there is no cell to propose between them.")
    interval = (high - low) / (n_cells + 1)
    slots = [low + k * interval for k in range(1, n_cells + 1)]

    # The end cells stay. A candidate loses its voxels on them, in the labels that name them and in the base: the
    # end cell of an S3 base can be larger than the annotated one.
    end_labels = end_base_labels(base, id_labels, ends)
    end_mask = np.isin(id_labels, list(ends)) | np.isin(base, end_labels)
    variants = source_variants(sources, path, row.reference, voxel_size, correct) if correct else sources
    candidates: Dict[int, List[Candidate]] = {k: [] for k in range(n_cells)}

    def add(source: str, voxels: np.ndarray, base_label: Optional[int] = None) -> None:
        free = ~end_mask[tuple(voxels.T)]
        if free.mean() < 0.5:
            return
        candidate = frame.candidate(source, voxels[free], base_label)
        slot = int(np.argmin([abs(candidate.position - s) for s in slots]))
        if abs(candidate.position - slots[slot]) < 0.5 * interval:
            candidates[slot].append(candidate)

    box = _stretch_box(frame, low, high, spacing)
    for name, labels in variants.items():
        for label_id, voxels in _coords_in(labels, box).items():
            add(name, voxels, label_id if name == base_source else None)
    for label_id, voxels in _coords_in(id_labels, box).items():
        if label_id not in ends:
            add(ANNOTATION, voxels, label_id if base_source == ANNOTATION else None)

    cells = []
    for k, slot in enumerate(slots):
        pool = candidates[k]
        for candidate in pool:
            others = [_iou(candidate.voxels, o.voxels, base.shape) for o in pool if o.group != candidate.group]
            candidate.agreement = max(others, default=0.0)
        _add_unions(pool, frame, base.shape)
        chosen = _select(pool, slot, interval, rule)
        if chosen is None:
            center = row.points[int(round(slot))] + _end_offset(frame, id_coords, ends)
            voxels = ihc_template(row, center, base.shape, voxel_size)
            cells.append(RegionCell(slot, voxels, frame.foot_index(voxels), "typical IHC shape (no candidate)"))
        else:
            comment = chosen.describe() + ("" if chosen.eligible else " (not typical)")
            cells.append(RegionCell(slot, chosen.voxels, frame.foot_index(chosen.voxels), comment,
                                    chosen.base_label))

    kept = {cell.base_label for cell in cells if cell.base_label is not None}
    removed = _stretch_labels(base, frame, low, high, box, set(end_labels) | kept)
    return RegionProposal(cells, removed, end_labels, slots, spacing)


def source_variants(
    sources: Dict[str, np.ndarray], path: np.ndarray, reference: dict, voxel_size: float,
    correct: Callable[[np.ndarray, np.ndarray, dict, float], np.ndarray] = row_correction,
) -> Dict[str, np.ndarray]:
    """The sources and the row correction of each, named '<source> row-corrected'."""
    variants = {}
    for name, labels in sources.items():
        variants[name] = labels
        variants[name + CORRECTED] = correct(labels, path, reference, voxel_size)
    return variants


def end_base_labels(base: np.ndarray, id_labels: np.ndarray, ends: Sequence[int]) -> List[int]:
    """The base label with the most voxels of each end cell."""
    labels = []
    for end in ends:
        values, counts = np.unique(base[id_labels == end], return_counts=True)
        counts[values == 0] = 0
        if len(counts) and counts.max() > 0:
            labels.append(int(values[np.argmax(counts)]))
    return labels


def _stretch_box(frame: _Frame, low: float, high: float, spacing: float) -> Tuple[slice, ...]:
    """The bounding box of the path points of the stretch, with a margin of the IHC length."""
    points = frame.row.points[int(np.floor(low)):int(np.ceil(high)) + 1]
    margin = frame.row.reference["length_axis"] / frame.voxel_size + spacing
    lo = np.maximum(np.floor(points.min(axis=0) - margin).astype(int), 0)
    hi = np.minimum(np.ceil(points.max(axis=0) + margin).astype(int) + 1, frame.shape)
    return tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))


def _coords_in(labels: np.ndarray, box: Tuple[slice, ...]) -> Dict[int, np.ndarray]:
    """The complete voxels of the labels that reach into the box."""
    ids = np.unique(labels[box])
    coords = {}
    for label_id in ids[ids > 0]:
        coords[int(label_id)] = np.argwhere(labels == label_id)
    return coords


def _add_unions(pool: List[Candidate], frame: _Frame, shape: Sequence[int]) -> None:
    """Add the union of each overlapping pair of candidates of two models. Its agreement is their overlap."""
    singles = [c for c in pool if c.group not in (ANNOTATION, "union")]
    for i, a in enumerate(singles):
        for b in singles[i + 1:]:
            overlap = _iou(a.voxels, b.voxels, shape) if a.group != b.group else 0.0
            if overlap == 0:
                continue
            voxels = np.unique(np.concatenate([a.voxels, b.voxels]), axis=0)
            union = frame.candidate(f"{a.source}{UNION}{b.source}", voxels)
            union.agreement = overlap
            pool.append(union)


def _select(pool: List[Candidate], slot: float, interval: float, rule: str) -> Optional[Candidate]:
    # Newer models first, so that a tie of the score goes to the newer model.
    pool = sorted(pool, key=lambda c: c.rank, reverse=True)
    eligible = [c for c in pool if c.eligible]
    if eligible:
        if rule == "newest":
            eligible = [c for c in eligible if c.rank == eligible[0].rank]
        return min(eligible, key=lambda c: c.score(slot, interval))
    fallback = [c for c in pool if c.n_rel >= MIN_FALLBACK_SIZE and c.distance < MAX_ROW_DISTANCE]
    return min(fallback, key=lambda c: c.score(slot, interval)) if fallback else None


def _end_offset(frame: _Frame, coords: Dict[int, np.ndarray], ends: Sequence[int]) -> np.ndarray:
    """The mean offset of the end cell centers from the path, for the position of a typical IHC shape."""
    offsets = []
    for end in ends:
        center = coords[end].mean(axis=0)
        offsets.append(center - frame.row.points[frame.tree.query(center)[1]])
    return np.mean(offsets, axis=0)


def _stretch_labels(
    base: np.ndarray, frame: _Frame, low: float, high: float, box: Tuple[slice, ...], kept: set,
) -> List[int]:
    """The row labels of the base between the end cells, without the kept labels."""
    removed = []
    for label_id, voxels in _coords_in(base, box).items():
        if label_id in kept:
            continue
        position = frame.position(voxels)
        distance = frame.tree.query(voxels.mean(axis=0))[0] * frame.voxel_size
        if low < position < high and distance < MAX_ROW_DISTANCE:
            removed.append(label_id)
    return sorted(removed)


def apply_region(
    labels: np.ndarray,
    proposal: RegionProposal,
    new_label: Callable[[], int],
    voxel_size: float,
    sigma: float = 1.0,
    band: float = 1.0,
) -> Tuple[Tuple[slice, ...], np.ndarray, List[Tuple[int, RegionCell]]]:
    """The labels of a region proposal, in the bounding box of the change.

    The removed labels are cleared. Each cell takes its free voxels; a voxel that two cells claim goes to the
    cell with the nearer position along the row. The cells come from different sources, so their borders are
    cleaned (`cleanup.clean_labels` with the smoothing `sigma` in pixels within `band` um, without splits).

    Returns:
        The box, the new labels in the box, and the label of each proposed cell.
    """
    removed = np.argwhere(np.isin(labels, proposal.removed)) if proposal.removed else np.zeros((0, 3), dtype=int)
    voxels = np.concatenate([cell.voxels for cell in proposal.cells] + [removed])
    box = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(voxels.min(axis=0), voxels.max(axis=0)))
    start = np.array([sl.start for sl in box])
    after = labels[box].copy()
    after[np.isin(after, proposal.removed)] = 0

    owner = np.full(after.shape, -1, dtype=int)
    offset_of_owner = np.full(after.shape, np.inf)
    for index, cell in enumerate(proposal.cells):
        if not len(cell.voxels):
            continue
        local = tuple((cell.voxels - start).T)
        free = (after[local] == 0) | (after[local] == (cell.base_label or 0))
        offset = np.abs(cell.foot - cell.slot)
        nearer = free & (offset < offset_of_owner[local])
        target = tuple(axis[nearer] for axis in local)
        owner[target], offset_of_owner[target] = index, offset[nearer]

    result = []
    for index, cell in enumerate(proposal.cells):
        label_id = cell.base_label if cell.base_label is not None else new_label()
        after[owner == index] = label_id
        result.append((label_id, cell))
    after, _ = clean_labels(after, labels[box], [label_id for label_id, _ in result], voxel_size, sigma, band)
    return box, after, result


def region_items(
    labels: np.ndarray, cells: List[Tuple[int, RegionCell]], end_labels: Sequence[int], first_id: int = 0,
):
    """The items of a region proposal: one per proposed cell, and the end cells."""
    from .issues import Item

    items = []
    for label_id, cell in cells:
        voxels = np.argwhere(labels == label_id)
        if len(voxels) == 0:
            continue
        items.append(Item(0, "cell", inner_point(voxels), int(label_id), ["region"], order=cell.slot,
                          comment=cell.comment))
    for order, label_id in zip((-1.0, np.inf), end_labels):
        voxels = np.argwhere(labels == label_id)
        if len(voxels):
            items.append(Item(0, "cell", inner_point(voxels), int(label_id), ["region_border"], order=order))
    items.sort(key=lambda item: item.order)
    for offset, item in enumerate(items):
        item.item_id = first_id + offset
        item.order = float(offset)
    return items
