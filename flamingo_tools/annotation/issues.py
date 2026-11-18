"""Find the cells of an annotation crop that need a correction.

The detectors return findings. `collect_items` groups them into the items of the correction loop: one item per
cell with all its reasons, and one item per missing cell. The reasons:
- pieces: the label has several connected pieces.
- indent: closing the cell adds a cavity or an indent, for example at a nucleus with low signal.
- size: the cell is much smaller or larger than the median cell of the crop.
- missing_<tag>: an object of the comparison segmentation has no counterpart. The tag is 's3' if the comparison
  is the S3 segmentation, and 'annotation' if it is the local annotation.
- extent_<tag>: the comparison object of the cell reaches past the cell ends.
- row_<class>_<action>: `correct_ihc_row` does not classify the IHC as normal, or changes it.
- short: the IHC is short along its local long axis.
- gap: the IHC row has a gap of one or more cells.
- table_<reason>: a row of an issue table, for example the known issues of a reference annotation.

All positions are in pixels in (z, y, x) order, in the coordinates of the crop.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.morphology import ball

# The 5th percentile of the IHC length along the local axis in um, of the refined annotations of the IHC_v11
# training crops (scripts/ihc_postprocessing/results/standard_ihc.tsv, feature 'length', tonotopy 'all').
STANDARD_LENGTH_P5 = {"mouse": 19.5, "gerbil": 23.5}
# Smaller annotated objects are partial cells at the crop border or debris (MIN_IHC_SIZE of the IHC analysis).
MIN_CELL_SIZE = 5000


@dataclass
class Finding:
    kind: str
    reason: str
    anchor: Tuple[int, int, int]
    label_id: int = 0
    proposal: Optional[np.ndarray] = field(default=None, repr=False)
    # Proposals with a lower rank win when findings are merged: 0 = S3 reference, 1 = template.
    proposal_rank: int = 2
    comment: str = ""


@dataclass
class Item:
    """One step of the correction loop: a cell ('cell') or a missing cell ('missing').

    The anchor is a voxel of the cell, so the item finds its cell again after the label id changes.
    `proposal` holds voxels (n x 3) that prefill the edit; it is not part of `to_dict`.
    """
    item_id: int
    kind: str
    anchor: Tuple[int, int, int]
    label_id: int
    reasons: List[str]
    order: float = 0.0
    comment: str = ""
    status: str = "pending"
    visits: int = 0
    done_at: int = -1
    proposal: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data.pop("proposal")
        data["anchor"] = [int(a) for a in self.anchor]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any], proposal: Optional[np.ndarray] = None) -> "Item":
        return cls(**{**data, "anchor": tuple(data["anchor"])}, proposal=proposal)


def cell_voxels(labels: np.ndarray) -> Dict[int, np.ndarray]:
    """The voxels of each label, as (n x 3) arrays."""
    coords = {}
    for label_id, bbox in enumerate(ndimage.find_objects(labels), start=1):
        if bbox is not None:
            coords[label_id] = np.argwhere(labels[bbox] == label_id) + np.array([sl.start for sl in bbox])
    return coords


def inner_point(voxels: np.ndarray) -> Tuple[int, int, int]:
    """The voxel nearest to the center of the voxels."""
    return tuple(int(v) for v in voxels[np.argmin(((voxels - voxels.mean(axis=0)) ** 2).sum(axis=1))])


def _at_border(voxels: np.ndarray, shape: Sequence[int]) -> bool:
    return bool((voxels.min(axis=0) == 0).any() or (voxels.max(axis=0) == np.array(shape) - 1).any())


def _bbox(voxels: np.ndarray, shape: Sequence[int], pad: int) -> Tuple[slice, ...]:
    low, high = voxels.min(axis=0) - pad, voxels.max(axis=0) + pad + 1
    return tuple(slice(max(0, lo), min(size, hi)) for lo, hi, size in zip(low, high, shape))


def piece_findings(labels: np.ndarray, coords: Dict[int, np.ndarray], min_piece: int = 50) -> List[Finding]:
    """Labels with a second connected piece (26-connectivity) of `min_piece` voxels or more."""
    findings = []
    structure = np.ones((3, 3, 3), dtype=bool)
    for label_id, voxels in coords.items():
        box = _bbox(voxels, labels.shape, 0)
        pieces, n_pieces = ndimage.label(labels[box] == label_id, structure)
        if n_pieces < 2:
            continue
        sizes = np.sort(np.bincount(pieces.ravel())[1:])
        if sizes[-2] >= min_piece:
            findings.append(Finding("cell", "pieces", inner_point(voxels), label_id))
    return findings


def indent_findings(
    labels: np.ndarray, coords: Dict[int, np.ndarray], voxel_size: float, radius: float = 1.5,
    min_fraction: float = 0.07,
) -> List[Finding]:
    """Cells that a closing with a ball of `radius` um grows by `min_fraction` or more into the background.

    The proposal is the added background voxels.
    """
    radius_px = max(1, round(radius / voxel_size))
    structure = ball(radius_px).astype(bool)
    findings = []
    for label_id, voxels in coords.items():
        box = _bbox(voxels, labels.shape, 2 * radius_px)
        region = labels[box]
        cell = region == label_id
        padded = np.pad(cell, radius_px)
        closed = ndimage.binary_closing(padded, structure)[(slice(radius_px, -radius_px),) * 3]
        indent = closed & (region == 0)
        if indent.sum() >= min_fraction * len(voxels):
            proposal = np.argwhere(indent) + np.array([sl.start for sl in box])
            findings.append(Finding("cell", "indent", inner_point(voxels), label_id, proposal, 1))
    return findings


def median_cell_size(labels: np.ndarray, coords: Optional[Dict[int, np.ndarray]] = None) -> Optional[float]:
    """The median size of the complete cells (not at the crop border, MIN_CELL_SIZE voxels or more), or None.

    A crop needs 3 complete cells.
    """
    if coords is not None:
        sizes = [len(v) for v in coords.values() if not _at_border(v, labels.shape) and len(v) >= MIN_CELL_SIZE]
    else:
        counts = np.bincount(labels.ravel())
        sizes = [counts[label_id] for label_id, bbox in enumerate(ndimage.find_objects(labels), start=1)
                 if bbox is not None and counts[label_id] >= MIN_CELL_SIZE
                 and not any(sl.start == 0 or sl.stop == size for sl, size in zip(bbox, labels.shape))]
    return float(np.median(sizes)) if len(sizes) >= 3 else None


def size_findings(
    labels: np.ndarray, coords: Dict[int, np.ndarray], size_range: Tuple[float, float] = (0.5, 1.6),
    median: Optional[float] = None,
) -> List[Finding]:
    """Complete cells (not at the crop border) outside `size_range` times the median complete cell.

    The median comes from `coords`, unless it is given.
    """
    median = median or median_cell_size(labels, coords)
    if median is None:
        return []
    return [
        Finding("cell", "size", inner_point(v), label_id) for label_id, v in coords.items()
        if not _at_border(v, labels.shape) and len(v) >= MIN_CELL_SIZE
        and not size_range[0] <= len(v) / median <= size_range[1]
    ]


def shape_findings(
    labels: np.ndarray, label_ids: Sequence[int], voxel_size: float, median: Optional[float] = None,
) -> List[Finding]:
    """The fast detectors (pieces, indent, size) for a few cells, for example the neighbours of an edit."""
    coords = {}
    for label_id in label_ids:
        voxels = np.argwhere(labels == label_id)
        if len(voxels):
            coords[int(label_id)] = voxels
    return (piece_findings(labels, coords) + indent_findings(labels, coords, voxel_size)
            + size_findings(labels, coords, median=median))


def _long_axis(voxels: np.ndarray) -> np.ndarray:
    return np.linalg.eigh(np.cov(voxels.T.astype(float)))[1][:, -1]


def reference_findings(
    labels: np.ndarray,
    coords: Dict[int, np.ndarray],
    reference: np.ndarray,
    min_size: float = 0.4,
    max_annotated: float = 0.2,
    min_extent: float = 0.05,
    tips_only: bool = True,
    tag: str = "s3",
) -> List[Finding]:
    """Compare the labels with a comparison segmentation of the crop (the reference).

    - missing_<tag>: a reference object with less than `max_annotated` of its voxels labeled. Its unlabeled
      voxels (largest piece) must have `min_size` of the median cell size. They are the proposal.
    - extent_<tag>: the unlabeled voxels of the reference object of a cell that touch the cell, with
      `min_extent` of the cell size or more. With `tips_only`, only the voxels past the cell ends along its long
      axis count, because a smooth model boundary adds a thin shell around every cell.
    """
    sizes = [len(v) for v in coords.values() if len(v) >= MIN_CELL_SIZE]
    median = np.median(sizes) if sizes else MIN_CELL_SIZE
    findings = []
    ref_coords = cell_voxels(reference)
    for ref_id, ref_voxels in ref_coords.items():
        annotated = labels[tuple(ref_voxels.T)] > 0
        if annotated.mean() >= max_annotated:
            continue
        free = ref_voxels[~annotated]
        mask = np.zeros(labels.shape, dtype=bool)
        mask[tuple(free.T)] = True
        box = _bbox(free, labels.shape, 0)
        pieces, n_pieces = ndimage.label(mask[box])
        if n_pieces == 0:
            continue
        largest = np.argmax(np.bincount(pieces.ravel())[1:]) + 1
        proposal = np.argwhere(pieces == largest) + np.array([sl.start for sl in box])
        if len(proposal) >= min_size * median:
            findings.append(Finding("missing", f"missing_{tag}", inner_point(proposal), 0, proposal, 0))

    for label_id, voxels in coords.items():
        if len(voxels) < 4:
            continue
        ref_ids, counts = np.unique(reference[tuple(voxels.T)], return_counts=True)
        counts[ref_ids == 0] = 0
        if counts.max() < 0.5 * len(voxels):
            continue
        ref_id = ref_ids[np.argmax(counts)]
        box = _bbox(np.concatenate([voxels, ref_coords[ref_id]]), labels.shape, 1)
        cell = labels[box] == label_id
        free = (reference[box] == ref_id) & (labels[box] == 0)
        # The free voxels that are connected to the cell.
        pieces, _ = ndimage.label(free)
        touching = np.unique(pieces[ndimage.binary_dilation(cell) & free])
        extent = np.argwhere(np.isin(pieces, touching[touching > 0])) + np.array([sl.start for sl in box])
        if tips_only and len(extent):
            axis = _long_axis(voxels)
            projection, cell_projection = extent @ axis, voxels @ axis
            extent = extent[(projection < cell_projection.min()) | (projection > cell_projection.max())]
        if len(extent) >= min_extent * len(voxels):
            findings.append(Finding("cell", f"extent_{tag}", inner_point(voxels), label_id, extent, 0))
    return findings


def annotation_row_path(
    labels: np.ndarray, coords: Dict[int, np.ndarray], voxel_size: float, extension: float = 20.0,
) -> Optional[np.ndarray]:
    """The central path through the annotated IHCs, in pixels in (z, y, x) order, or None.

    The path goes through the centers of the cells with MIN_CELL_SIZE voxels or more (`ihc_row_path`) and
    `extension` um beyond the outer cells, so their tips are not at the path end. A crop needs 3 such cells.
    """
    from flamingo_tools.postprocessing.label_components import ihc_row_path

    centers = np.array([v.mean(axis=0) for v in coords.values() if len(v) >= MIN_CELL_SIZE])
    if len(centers) < 3:
        return None
    path = ihc_row_path(centers[:, ::-1] * voxel_size)
    # ihc_row_path samples at 1 um.
    steps = round(extension)
    path = np.concatenate([path[:1] - steps * (path[1] - path[0]), path, path[-1:] + steps * (path[-1] - path[-2])])
    return path[:, ::-1] / voxel_size


@dataclass
class RowContext:
    """The IHC row of a crop: the path points, the local axis at each point and the typical IHC."""
    points: np.ndarray
    axes: np.ndarray
    reference: Dict[str, Any]
    spacing: Optional[float] = None


def template_cell(
    shape: Sequence[int], center: np.ndarray, axes: np.ndarray, semi_axes: Sequence[float],
) -> np.ndarray:
    """The voxels (n x 3) of an ellipsoid. `axes` holds the unit axes as rows, `semi_axes` are in pixels."""
    extent = int(np.ceil(max(semi_axes)))
    low = np.maximum(np.round(center).astype(int) - extent, 0)
    high = np.minimum(np.round(center).astype(int) + extent + 1, shape)
    if (high <= low).any():
        return np.zeros((0, 3), dtype=int)
    grid = np.stack(np.meshgrid(*[np.arange(lo, hi) for lo, hi in zip(low, high)], indexing="ij"), axis=-1)
    offsets = (grid - center) @ np.asarray(axes).T
    inside = ((offsets / np.asarray(semi_axes)) ** 2).sum(axis=-1) <= 1
    return grid[inside]


def ihc_template(row: RowContext, center: np.ndarray, shape: Sequence[int], voxel_size: float) -> np.ndarray:
    """The voxels of a typical IHC at a position, oriented by the local axis and the path tangent."""
    index = cKDTree(row.points).query(center)[1]
    axis = row.axes[index]
    tangent = row.points[min(index + 1, len(row.points) - 1)] - row.points[max(index - 1, 0)]
    tangent = tangent - (tangent @ axis) * axis
    tangent /= max(np.linalg.norm(tangent), 1e-6)
    a = row.reference["length_axis"] / 2 / voxel_size
    b = row.reference["length_path"] / 2 / voxel_size
    c = 3 * row.reference["n_pixels"] / (4 * np.pi * a * b)
    return template_cell(shape, center, np.stack([axis, tangent, np.cross(axis, tangent)]), (a, b, c))


def ihc_findings(
    labels: np.ndarray,
    coords: Dict[int, np.ndarray],
    voxel_size: float,
    species: str = "mouse",
    short_factor: float = 0.85,
    gap_factor: float = 1.6,
) -> Tuple[List[Finding], Optional[RowContext]]:
    """The IHC row findings of a crop: row classes, short cells and row gaps.

    - row_*: the report of `correct_ihc_row` on the annotation with its own row path. Rows with the class
      'uncertain' or 'off_row', the flag 'review', or an action other than 'keep'.
    - short: a complete row IHC whose length along the local axis is below the species 5th percentile, or
      below `short_factor` times the 75th percentile of the complete row IHCs of the crop. The second rule finds
      a row section of cut cells.
    - gap: two neighbouring row IHCs are more than `gap_factor` times the median spacing apart. The missing
      cells sit evenly in the gap, and the proposal is a typical IHC.

    Returns:
        The findings and the row context, or no findings and None if the crop has fewer than 3 cells.
    """
    from flamingo_tools.postprocessing.ihc_morphology import (
        correct_ihc_row, foot_points, ihc_row_reference, row_frame,
    )

    path = annotation_row_path(labels, coords, voxel_size)
    if path is None:
        return [], None
    try:
        reference = ihc_row_reference([labels], path, voxel_size)
    except ValueError:
        return [], None
    features, frame_coords, points, axes = row_frame(labels, path, reference, voxel_size)
    row = RowContext(points, axes, reference)
    findings = []

    _, report, _ = correct_ihc_row(labels, path, reference, voxel_size)
    for record in report.to_dict("records"):
        label_id = int(record["label_id"])
        flagged = record["class"] in ("uncertain", "off_row") or record["review"] or record["action"] != "keep"
        if flagged and label_id in coords:
            reason = f"row_{record['class']}_{record['action']}"
            findings.append(Finding("cell", reason, inner_point(coords[label_id]), label_id))

    complete = {i: f for i, f in features.items() if f["row"] and not f["border"] and f["n_pixels"] >= MIN_CELL_SIZE}
    lengths = {}
    for label_id in complete:
        _, _, height = foot_points(frame_coords[label_id].astype(float), points, axes)
        lengths[label_id] = (height.max() - height.min()) * voxel_size
    if lengths:
        limit = STANDARD_LENGTH_P5.get(species, 0.0)
        if len(lengths) >= 4:
            limit = max(limit, short_factor * np.percentile(list(lengths.values()), 75))
        for label_id, length in lengths.items():
            if length < limit:
                findings.append(Finding("cell", "short", inner_point(coords[label_id]), label_id,
                                        comment=f"length {length:.1f} um < {limit:.1f} um"))

    row_ids = sorted((f["index"], i) for i, f in features.items() if f["row"] and f["n_rel"] >= 0.4)
    if len(row_ids) >= 3:
        positions = np.array([index for index, _ in row_ids])
        spacing = np.median(np.diff(positions))
        row.spacing = float(spacing * voxel_size)
        offsets = {i: coords[i].mean(axis=0) - points[f["index"]] for i, f in features.items() if i in coords}
        for (start, id_a), (stop, id_b) in zip(row_ids[:-1], row_ids[1:]):
            gap = stop - start
            if spacing <= 0 or gap <= gap_factor * spacing:
                continue
            n_missing = max(1, int(round(gap / spacing)) - 1)
            offset = (offsets[id_a] + offsets[id_b]) / 2
            for k in range(1, n_missing + 1):
                index = int(round(start + k * gap / (n_missing + 1)))
                center = points[index] + offset
                proposal = ihc_template(row, center, labels.shape, voxel_size)
                proposal = proposal[labels[tuple(proposal.T)] == 0] if len(proposal) else proposal
                anchor = tuple(int(c) for c in np.clip(np.round(center), 0, np.array(labels.shape) - 1))
                findings.append(Finding("missing", "gap", anchor, 0, proposal, 1,
                                        comment=f"gap of {gap * voxel_size:.0f} um between {id_a} and {id_b}"))
    return findings, row


def _table_label(
    label_id: int, labels: np.ndarray, coords: Dict[int, np.ndarray], source_labels: np.ndarray,
    min_overlap: float = 0.3,
) -> Tuple[str, int, Tuple[int, int, int], Optional[np.ndarray]]:
    """Map a label of the annotation to the labels: the label with most of its voxels, or a missing cell."""
    voxels = np.argwhere(source_labels == label_id)
    if len(voxels) == 0:
        raise ValueError(f"The label {label_id} of the issue table is not in the annotation.")
    targets = labels[tuple(voxels.T)]
    ids, counts = np.unique(targets[targets > 0], return_counts=True)
    if len(ids) and counts.max() >= min_overlap * len(voxels):
        target = int(ids[np.argmax(counts)])
        return "cell", target, inner_point(voxels[targets == target]), None
    free = voxels[targets == 0]
    return "missing", 0, inner_point(free if len(free) else voxels), free


def table_findings(
    rows: pd.DataFrame,
    labels: np.ndarray,
    coords: Dict[int, np.ndarray],
    original: Optional[np.ndarray] = None,
    source_labels: Optional[np.ndarray] = None,
) -> List[Finding]:
    """The findings of an issue table, see `read_issue_table`.

    A row with a label id gives a cell finding ('all' gives one for each cell). A row with a position only gives
    a missing cell. The label ids of the table are ids of the annotation:
    - If the labels are the annotation, `original` maps the compact ids to the table ids (see `ids.compact`).
    - If the labels are another segmentation, `source_labels` is the annotation with the table ids. A table
      label becomes the label that holds most of its voxels (at least 30%), or else a missing cell with its
      unlabeled voxels as the proposal.
    - Without both, a label row uses the label at its position, and a row without a position is skipped.
    """
    findings = []
    for record in rows.itertuples():
        reason, comment = f"table_{record.reason}", record.comment if isinstance(record.comment, str) else ""
        position = None
        if not any(pd.isna(getattr(record, axis)) for axis in "zyx"):
            position = tuple(int(getattr(record, axis)) for axis in "zyx")
        label = str(record.label_id) if not pd.isna(record.label_id) else ""
        if label == "all":
            findings += [Finding("cell", reason, inner_point(v), i, comment=comment) for i, v in coords.items()]
            continue
        if label and source_labels is not None:
            kind, label_id, anchor, proposal = _table_label(int(float(label)), labels, coords, source_labels)
            findings.append(Finding(kind, reason, anchor, label_id, proposal, 0, comment=comment))
        elif label and original is not None:
            label_id = int(float(label))
            index = np.searchsorted(original, label_id)
            label_id = int(index) if index < len(original) and original[index] == label_id else -1
            if label_id not in coords:
                raise ValueError(f"The label {label} of the issue table is not in the annotation.")
            anchor = position if position is not None and labels[position] == label_id else \
                inner_point(coords[label_id])
            findings.append(Finding("cell", reason, anchor, label_id, comment=comment))
        elif label:
            if position is None or labels[position] == 0:
                print(f"Issue table: label {label} cannot be found without the annotation; the row is skipped.")
                continue
            findings.append(Finding("cell", reason, position, int(labels[position]), comment=comment))
        elif position is not None:
            findings.append(Finding("missing", reason, position, 0, comment=comment))
    return findings


def read_issue_table(path: str, crop: str) -> pd.DataFrame:
    """The rows of an issue table for one crop.

    The table is tab-separated with the columns crop, label_id, z, y, x, reason and comment. label_id is an id
    of the annotation, 'all', or empty for a missing cell at (z, y, x). The position is in napari coordinates of
    the crop.
    """
    table = pd.read_csv(path, sep="\t", dtype={"label_id": str})
    missing = {"crop", "label_id", "z", "y", "x", "reason", "comment"} - set(table.columns)
    if missing:
        raise ValueError(f"The issue table {path} misses the columns {sorted(missing)}.")
    return table[table["crop"] == crop].reset_index(drop=True)


def collect_items(
    findings: List[Finding], row: Optional[RowContext] = None, shape: Optional[Sequence[int]] = None,
    min_distance: float = 10.0,
) -> List[Item]:
    """Group the findings into items, ordered along the IHC row or by position.

    Findings of one cell form one item. A missing cell closer than `min_distance` pixels to an earlier one joins
    its item, and the proposal with the lowest rank wins. Gap findings come first, so a table row between two
    gap cells joins one of them.
    """
    findings = sorted(findings, key=lambda f: f.kind == "missing" and not f.reason == "gap")
    groups: List[List[Finding]] = []
    by_label: Dict[int, List[Finding]] = {}
    for finding in findings:
        if finding.kind == "cell":
            if finding.label_id not in by_label:
                by_label[finding.label_id] = []
                groups.append(by_label[finding.label_id])
            by_label[finding.label_id].append(finding)
            continue
        for group in groups:
            distance = np.linalg.norm(np.subtract(group[0].anchor, finding.anchor))
            if group[0].kind == "missing" and distance < min_distance:
                group.append(finding)
                break
        else:
            groups.append([finding])

    tree = cKDTree(row.points) if row is not None else None
    items = []
    for group in groups:
        first = group[0]
        ranked = sorted((f for f in group if f.proposal is not None and len(f.proposal)), key=lambda f: f.proposal_rank)
        if first.kind == "cell":
            proposals = [f.proposal for f in group if f.proposal is not None and len(f.proposal)]
            proposal = np.unique(np.concatenate(proposals), axis=0) if proposals else None
        else:
            proposal = ranked[0].proposal if ranked else None
        if tree is not None:
            order = float(tree.query(first.anchor)[1])
        else:
            order = float(np.ravel_multi_index(first.anchor, shape)) if shape is not None else 0.0
        reasons = list(dict.fromkeys(f.reason for f in group))
        comment = "; ".join(dict.fromkeys(f.comment for f in group if f.comment))
        items.append(Item(0, first.kind, first.anchor, first.label_id, reasons, order, comment, proposal=proposal))
    items.sort(key=lambda item: item.order)
    for item_id, item in enumerate(items):
        item.item_id = item_id
    return items


def detect(
    labels: np.ndarray,
    voxel_size: float,
    cell_type: str = "ihc",
    reference: Optional[np.ndarray] = None,
    issue_rows: Optional[pd.DataFrame] = None,
    original: Optional[np.ndarray] = None,
    species: str = "mouse",
    label_ids: Optional[Sequence[int]] = None,
    reference_tag: str = "s3",
    source_labels: Optional[np.ndarray] = None,
) -> Tuple[List[Item], Optional[RowContext]]:
    """Run the detectors of a cell type and group the findings into items.

    Args:
        labels: The annotation with compact ids, see `ids.compact`.
        voxel_size: Isotropic voxel size in micrometer.
        cell_type: 'ihc' or 'sgn'. SGN crops use the generic detectors only.
        reference: The comparison segmentation of the crop (compact ids), or None.
        issue_rows: The rows of an issue table for the crop, see `read_issue_table`.
        original: The original ids of the compact ids, if the labels are the annotation of the issue table.
        species: 'mouse' or 'gerbil', for the IHC length limit.
        label_ids: Only return the findings of these cells, without missing cells. For the check of the
            changed cells at the end of the loop.
        reference_tag: The tag of the comparison in the reasons, 's3' or 'annotation'.
        source_labels: The annotation with the ids of the issue table, if the labels are another segmentation.

    Returns:
        The items, and the IHC row context or None.
    """
    if cell_type not in ("ihc", "sgn"):
        raise ValueError(f"Invalid cell type {cell_type}. Choose 'ihc' or 'sgn'.")
    coords = cell_voxels(labels)
    findings = piece_findings(labels, coords) + indent_findings(labels, coords, voxel_size)
    findings += size_findings(labels, coords)
    if reference is not None:
        findings += reference_findings(labels, coords, reference, tips_only=cell_type == "ihc", tag=reference_tag)
    row = None
    if cell_type == "ihc":
        row_findings, row = ihc_findings(labels, coords, voxel_size, species)
        findings += row_findings
    if issue_rows is not None and len(issue_rows):
        findings += table_findings(issue_rows, labels, coords, original, source_labels)

    for finding in findings:
        if finding.kind == "missing" and finding.proposal is None:
            finding.proposal, finding.proposal_rank = missing_proposal(
                finding.anchor, labels, coords, reference, row, voxel_size)
    if label_ids is not None:
        findings = [f for f in findings if f.kind == "cell" and f.label_id in set(label_ids)]
    # Missing cells closer than about half the cell spacing are one cell.
    spacing = row.spacing if row is not None and row.spacing else 8.0
    return collect_items(findings, row, labels.shape, min_distance=0.6 * spacing / voxel_size), row


def missing_proposal(
    anchor: Tuple[int, int, int], labels: np.ndarray, coords: Dict[int, np.ndarray],
    reference: Optional[np.ndarray], row: Optional[RowContext], voxel_size: float,
) -> Tuple[Optional[np.ndarray], int]:
    """The proposal of a missing cell at a position: the S3 object there, a typical IHC, or a sphere."""
    if reference is not None and reference[anchor] > 0:
        voxels = np.argwhere(reference == reference[anchor])
        return voxels[labels[tuple(voxels.T)] == 0], 0
    center = np.asarray(anchor, dtype=float)
    if row is not None:
        voxels = ihc_template(row, center, labels.shape, voxel_size)
    else:
        sizes = [len(v) for v in coords.values() if len(v) >= MIN_CELL_SIZE] or [MIN_CELL_SIZE]
        radius = (3 * np.median(sizes) / (4 * np.pi)) ** (1 / 3)
        voxels = template_cell(labels.shape, center, np.eye(3), (radius,) * 3)
    return (voxels[labels[tuple(voxels.T)] == 0] if len(voxels) else None), 1
