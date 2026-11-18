"""The correction loop over the items of one crop, without a user interface.

The loop:
- Each visit shows one item: a region around the cell with its complete neighbours, and sparse slices to edit.
  The slices are z planes, or planes along the array axis nearest to the long axis of the cell.
- A visit can change its target cell and the cells near it (the focus). All other cells are locked, until the
  user unlocks one (`Controller.unlock`).
- Next: the edit is filled (`fill_visit`) and committed. A cell that the visit changes goes to the end of the
  queue for another look. A cell that the visit does not change is done.
- Accept: as Next, but the cell is done in any case.
- Reject drops a proposed cell, Defer moves the item to the end of the queue, Delete removes the cell.
- A focus cell that the visit changes goes back into the queue only if the shape detectors flag it.
- Undo reverts the last action (also a Next or Accept without a change) and restores the loop as it was before
  it: the item is the current visit again, with the edit of that visit.
- When the queue is empty, the detectors check the cells that changed after they were done. If they find
  nothing, the loop is finished.

The labels have compact ids (`ids.compact`) and are changed in place. The session stores every commit.
"""
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

from .cleanup import clean_labels, clean_slice
from .fill import fill_sparse, select_slices, visit_region
from .issues import Item, inner_point, median_cell_size, shape_findings, template_cell
from .session import Session


SLICE_AXES = ("z", "long")


@dataclass
class Settings:
    """The settings of the slices, the visit region, the fill and the cleanup. Distances are in micrometer.

    `slice_axis` 'z' places the slices `spacing` apart in z. 'long' places `n_slices` slices along the array axis
    nearest to the long axis of the cell. A cell with `min_contact` voxels or more within `focus_gap` of the target
    is in the focus of the visit (0: touching cells only). The slices also cover a focus cell smaller than
    `fragment_size` times the median cell size, so that the user can remove or merge it. The cleanup settings are
    explained in
    `cleanup.clean_slice` and `cleanup.clean_labels` (`boundary_sigma` in pixels, 0 switches the smoothing off;
    `min_area` in um2).
    """
    voxel_size: float = 0.38
    slice_axis: str = "z"
    spacing: float = 2.0
    n_slices: int = 10
    cap: float = 2.0
    inset: float = 0.5
    max_slices: int = 16
    margin: float = 5.0
    max_margin: float = 15.0
    vanish: float = 0.7
    min_contact: int = 20
    focus_gap: float = 1.0
    fragment_size: float = 0.5
    min_area: float = 2.0
    boundary_sigma: float = 1.0
    boundary_band: float = 1.0
    split_size: int = 1000


@dataclass
class Visit:
    """The current item, its label, its region and its sparse slices.

    The slices are positions along the array axis `axis`, in crop coordinates. `focus` holds the target label and
    the labels near it and the unlocked labels, `new_ids` the labels that the visit added. Only these labels can
    change.
    """
    item: Item
    label_id: int
    roi: Tuple[slice, slice, slice]
    axis: int
    slices: List[int]
    focus: List[int] = field(default_factory=list)
    new_ids: List[int] = field(default_factory=list)
    proposal: Optional[np.ndarray] = field(default=None, repr=False)

    @property
    def editable(self) -> List[int]:
        return sorted(set(self.focus) | set(self.new_ids))

    @property
    def start(self) -> Tuple[int, int, int]:
        return tuple(sl.start for sl in self.roi)

    @property
    def local_slices(self) -> np.ndarray:
        return np.asarray(self.slices, dtype=int) - self.roi[self.axis].start


def slice_axis_of(mask: np.ndarray) -> int:
    """The array axis nearest to the long axis (first principal axis) of a mask."""
    voxels = np.argwhere(mask).astype(float)
    if len(voxels) < 4:
        return 0
    return int(np.argmax(np.abs(np.linalg.eigh(np.cov(voxels.T))[1][:, -1])))


def visit_geometry(
    labels: np.ndarray, mask: np.ndarray, settings: Settings, extra: Optional[np.ndarray] = None,
) -> Tuple[int, Tuple[slice, slice, slice], np.ndarray]:
    """The slice axis, the region and the edit slices (local positions along the axis) of a visit to a cell.

    The axis comes from the cell `mask`. The region and the slices also cover the `extra` voxels.
    """
    s = settings
    axis = slice_axis_of(mask) if s.slice_axis == "long" else 0
    cover = mask if extra is None else mask | extra
    roi = visit_region(labels, cover, s.voxel_size, axis, s.margin, s.cap + 1.0, s.max_margin)
    moved = np.moveaxis(cover[roi], axis, 0)
    spacing, max_slices = s.spacing, s.max_slices
    if s.slice_axis == "long":
        positions = np.flatnonzero(moved.any(axis=(1, 2)))
        extent = (positions[-1] - positions[0]) * s.voxel_size
        spacing, max_slices = max(extent / max(s.n_slices - 1, 1), s.voxel_size), s.n_slices + 2
    return axis, roi, select_slices(moved, s.voxel_size, spacing, s.cap, s.inset, max_slices)


def focus_ids(
    labels: np.ndarray, mask: np.ndarray, label_id: int, min_contact: int = 20, gap: int = 0,
) -> List[int]:
    """The target label and the labels with at least `min_contact` voxels in a shell around the target mask.

    The shell is `gap` + 1 voxels wide (face steps), so a cell that does not touch the target, but is at most
    `gap` voxels away, can be in the focus.
    """
    shell = ndimage.binary_dilation(mask, ndimage.generate_binary_structure(3, 1), iterations=gap + 1) & ~mask
    values, counts = np.unique(labels[shell], return_counts=True)
    near = values[(values > 0) & (values != label_id) & (counts >= min_contact)]
    return sorted({int(label_id)} | set(near.tolist()))


def filter_paint_atoms(
    atoms: list,
    allowed: Sequence[int],
    data: np.ndarray,
    axis: int = 0,
    locked: Optional[np.ndarray] = None,
    locked_value: Optional[Callable[[int], bool]] = None,
) -> str:
    """Revert the painted voxels that the visit must not change, in `data` and in the atoms.

    napari passes the atoms of one paint stroke as a list of (indices, old values, new value). The list is also
    the undo history entry, so the filter in place keeps the history consistent. Reverted are:
    - a stroke with a locked label ('label');
    - voxels on a locked cell ('locked');
    - voxels outside the allowed slices along `axis` ('slice').

    Args:
        atoms: The atoms of a paint event.
        allowed: The positions of the edit slices along `axis`.
        data: The layer data.
        axis: The slice axis.
        locked: The voxels of the locked cells.
        locked_value: Tells whether a label value is locked.

    Returns:
        The reason of the first revert, or '' if nothing was reverted.
    """
    kept, reason = [], ""
    allowed = np.asarray(allowed)
    for indices, old, new in atoms:
        if locked_value is not None and np.ndim(new) == 0 and new != 0 and locked_value(int(new)):
            data[indices] = old
            reason = reason or "label"
            continue
        inside = np.isin(indices[axis], allowed)
        if not inside.all():
            reason = reason or "slice"
        if locked is not None:
            on_locked = locked[indices]
            if (inside & on_locked).any():
                reason = reason or "locked"
            inside &= ~on_locked
        if not inside.all():
            data[tuple(i[~inside] for i in indices)] = old[~inside] if np.ndim(old) else old
        if inside.any():
            kept.append((tuple(i[inside] for i in indices), old[inside] if np.ndim(old) else old,
                         new[inside] if np.ndim(new) else new))
    atoms[:] = kept
    return reason


def fill_visit(
    current: np.ndarray,
    edit: np.ndarray,
    axis: int,
    local_slices: Sequence[int],
    editable: Sequence[int],
    settings: Settings,
    new_label: Optional[Callable[[], int]] = None,
) -> Tuple[np.ndarray, Dict[int, int]]:
    """Complete the region of a visit from its edited slices, and clean the result.

    On the edit slices, the voxels of locked labels (not in `editable`) keep their label, and a locked label in
    the edit is ignored. Each changed slice is cleaned (`clean_slice`), the slices are filled (`fill_sparse`), and
    the result is cleaned (`clean_labels`).

    Args:
        current: The labels of the region.
        edit: The edit of the region. Only the edit slices are read.
        axis: The slice axis.
        local_slices: The positions of the edit slices along the axis, in the region.
        editable: The labels that may change.
        settings: The settings.
        new_label: Gives a new label id for a split piece, see `clean_labels`.

    Returns:
        The completed region, and the label that each new label of a split piece came from.
    """
    s = settings
    editable = sorted(set(editable))
    full_edit = current.copy()
    moved_current, moved_edit, moved_full = (np.moveaxis(a, axis, 0) for a in (current, edit, full_edit))
    for index in local_slices:
        before = moved_current[index]
        painted = np.where(np.isin(moved_edit[index], editable) | (moved_edit[index] == 0), moved_edit[index], before)
        slice_ = np.where(np.isin(before, editable) | (before == 0), painted, before)
        if not np.array_equal(slice_, before):
            slice_ = clean_slice(slice_, before, editable, s.voxel_size, s.min_area)
        moved_full[index] = slice_
    # fill_sparse fills along the first axis, so the slice axis moves to the front and back.
    result, _ = fill_sparse(moved_current, moved_full, local_slices, s.voxel_size, vanish=s.vanish)
    return clean_labels(np.moveaxis(result, 0, axis), current, editable, s.voxel_size, s.boundary_sigma,
                        s.boundary_band, s.split_size, new_label)


class Controller:
    """The correction loop. Create it with `Controller.create` or `Controller.resume`.

    Args:
        image: The image of the crop, in (z, y, x) order.
        labels: The labels with compact ids. They are changed in place.
        items: The items of the loop.
        session: The session that stores the commits and the state.
        settings: The settings.
        propose: Gives the proposal voxels of a new cell at a position, or None.
        redetect: Gives the items of the listed cells, for the check at the end of the loop.
        proposals: The proposal of each item id, also of the items whose proposal is closed (for undo).
    """

    def __init__(
        self,
        image: np.ndarray,
        labels: np.ndarray,
        items: List[Item],
        session: Session,
        settings: Settings,
        propose: Optional[Callable[[Tuple[int, int, int]], Optional[np.ndarray]]] = None,
        redetect: Optional[Callable[[np.ndarray, List[int]], List[Item]]] = None,
        proposals: Optional[Dict[int, np.ndarray]] = None,
    ):
        if settings.slice_axis not in SLICE_AXES:
            raise ValueError(f"Invalid slice axis {settings.slice_axis}. Choose one of {SLICE_AXES}.")
        self.image, self.labels, self.session, self.settings = image, labels, session, settings
        self.propose, self.redetect = propose, redetect
        self.items: Dict[int, Item] = {item.item_id: item for item in items}
        self.proposals: Dict[int, np.ndarray] = dict(proposals or {})
        self.proposals.update({i: item.proposal for i, item in self.items.items() if item.proposal is not None})
        state = session.state
        self.queue: List[int] = state.get("queue", [i for i, item in self.items.items() if item.status == "pending"])
        self.next_label = int(state.get("next_label", int(labels.max()) + 1))
        self.step = int(state.get("step", 0))
        self.last_change: Dict[int, int] = {int(k): v for k, v in state.get("last_change", {}).items()}
        self.checked_at = int(state.get("checked_at", 0))
        self.finished = bool(state.get("finished", False))
        self.history: List[dict] = state.get("history", [])
        self.visit: Optional[Visit] = None
        # The edit of the visit that an undo reopened; `prefill` returns it once.
        self.restored_edit: Optional[np.ndarray] = None
        # Notes for the user interface, for example on split labels. The interface clears them and sets `name_of`
        # to show the file ids.
        self.messages: List[str] = []
        self.name_of: Callable[[int], str] = str
        self._median_size: Optional[float] = None

    @classmethod
    def create(cls, image, labels, items, session, settings, propose=None, redetect=None) -> "Controller":
        controller = cls(image, labels, items, session, settings, propose, redetect)
        controller._save(proposals=True)
        return controller

    @classmethod
    def resume(cls, image, labels, session, settings, propose=None, redetect=None) -> "Controller":
        """Continue a session. `labels` must be the input labels; the patches are replayed on them."""
        session.replay(labels)
        proposals = {int(key): value for key, value in session.load_arrays().items()}
        items = cls._items_of(session.state, proposals)
        return cls(image, labels, items, session, settings, propose, redetect, proposals)

    @staticmethod
    def _items_of(state: dict, proposals: Dict[int, np.ndarray]) -> List[Item]:
        """The items of a stored loop state, with the proposals that are still open."""
        open_ids = state.get("open_proposals")
        return [Item.from_dict(data, proposals.get(data["item_id"])
                               if open_ids is None or data["item_id"] in open_ids else None)
                for data in state["items"]]

    def _loop_state(self) -> dict:
        return {
            "items": [item.to_dict() for item in self.items.values()],
            "open_proposals": [i for i, item in self.items.items() if item.proposal is not None],
            "queue": list(self.queue), "next_label": self.next_label, "step": self.step,
            "checked_at": self.checked_at, "last_change": {str(k): v for k, v in self.last_change.items()},
            "finished": self.finished,
        }

    def _update_state(self) -> None:
        self.session.state.update({**self._loop_state(), "history": self.history, "settings": asdict(self.settings)})

    def _save(self, proposals: bool = False) -> None:
        self._update_state()
        self.session.save()
        if proposals:
            self.session.save_arrays({str(i): proposal for i, proposal in self.proposals.items()})

    @property
    def n_done(self) -> int:
        return sum(item.status != "pending" for item in self.items.values())

    # The visit.

    def _resolve_label(self, item: Item) -> int:
        """The label of an item now.

        This is its own label while it exists. A cell whose label is gone takes the label at its anchor. A
        missing cell keeps the new label that it got at its first visit.
        """
        if item.label_id and (self.labels == item.label_id).any():
            return item.label_id
        at_anchor = int(self.labels[item.anchor])
        if item.kind == "cell" and at_anchor:
            item.label_id = at_anchor
            return at_anchor
        if item.kind == "missing" and item.label_id:
            return item.label_id
        item.label_id = self.next_label
        self.next_label += 1
        return item.label_id

    def view(self) -> Optional[Visit]:
        """The current visit. Starts the next item, or returns None if the loop is finished."""
        if self.visit is not None:
            return self.visit
        while not self.queue:
            if self.finished or not self._check_changed():
                self.finished = True
                self._save()
                return None
        item = self.items[self.queue[0]]
        label_id = self._resolve_label(item)
        mask = self.labels == label_id
        proposal = self._open_proposal(item, label_id)
        if proposal is not None:
            mask[tuple(proposal.T)] = True
        if not mask.any():
            radius = 5.0 / self.settings.voxel_size
            voxels = template_cell(mask.shape, np.asarray(item.anchor, dtype=float), np.eye(3), (radius,) * 3)
            mask[tuple(voxels.T)] = True
        s = self.settings
        gap = round(s.focus_gap / s.voxel_size)
        axis, roi, local_slices = visit_geometry(self.labels, mask, s)
        focus = focus_ids(self.labels[roi], mask[roi], label_id, s.min_contact, gap)
        fragments = self._fragments([i for i in focus if i != label_id])
        if fragments:
            axis, roi, local_slices = visit_geometry(self.labels, mask, s, np.isin(self.labels, fragments))
            focus = focus_ids(self.labels[roi], mask[roi], label_id, s.min_contact, gap)
        slices = (local_slices + roi[axis].start).tolist()
        self.visit = Visit(item, label_id, roi, axis, slices, focus, proposal=self._local_proposal(proposal, roi))
        return self.visit

    def _median_cell_size(self) -> Optional[float]:
        if self._median_size is None:
            self._median_size = median_cell_size(self.labels) or 0.0
        return self._median_size or None

    def _fragments(self, label_ids: List[int]) -> List[int]:
        """The labels that are smaller than `fragment_size` times the median cell size."""
        median = self._median_cell_size()
        if not label_ids or median is None:
            return []
        sizes = ndimage.sum_labels(np.ones(self.labels.shape, dtype="uint8"), self.labels, label_ids)
        return [i for i, size in zip(label_ids, sizes) if size < self.settings.fragment_size * median]

    def _open_proposal(self, item: Item, label_id: int) -> Optional[np.ndarray]:
        """The proposal voxels of an item that are free or of its label, or None."""
        if item.proposal is None or not len(item.proposal):
            return None
        return item.proposal[np.isin(self.labels[tuple(item.proposal.T)], (0, label_id))]

    @staticmethod
    def _local_proposal(proposal: Optional[np.ndarray], roi) -> Optional[np.ndarray]:
        if proposal is None:
            return None
        local = proposal - np.array([sl.start for sl in roi])
        inside = np.all((local >= 0) & (local < [sl.stop - sl.start for sl in roi]), axis=1)
        return local[inside]

    def unlock(self, label_id: int) -> bool:
        """Make a locked cell in the region of the visit editable. Returns False if it is editable already.

        If the cell is a fragment (`Settings.fragment_size`), the slices also cover it.
        """
        visit = self.view()
        if label_id in visit.editable:
            return False
        if label_id == 0 or not (self.labels[visit.roi] == label_id).any():
            raise ValueError(f"The label {self.name_of(label_id)} is not in the region of the visit.")
        visit.focus = sorted(set(visit.focus) | {int(label_id)})
        if self._fragments([int(label_id)]):
            self._cover(int(label_id))
        return True

    def _cover(self, label_id: int) -> None:
        """Add slices beyond the current ones, so that the slices span the target and the cell."""
        visit, s = self.visit, self.settings
        cover = np.moveaxis(np.isin(self.labels[visit.roi], [visit.label_id, label_id]), visit.axis, 0)
        spacing = float(np.median(np.diff(visit.slices))) * s.voxel_size if len(visit.slices) > 1 else s.spacing
        start = visit.roi[visit.axis].start
        low, high = min(visit.slices), max(visit.slices)
        added = [int(p) + start for p in select_slices(cover, s.voxel_size, spacing, s.cap, s.inset, 2 * s.max_slices)
                 if not low <= int(p) + start <= high]
        visit.slices = sorted(set(visit.slices) | set(added))

    def new_label(self) -> int:
        """A new label id for the current visit, for example to split a cell."""
        label_id = self.next_label
        self.next_label += 1
        if self.visit is not None:
            self.visit.new_ids.append(label_id)
        self._save()
        return label_id

    def prefill(self) -> np.ndarray:
        """The edit of the current visit: the focus labels on the sparse slices, with the proposal; zero elsewhere.

        After an undo, it is the stored edit of the reopened visit (once).
        """
        visit = self.view()
        if self.restored_edit is not None:
            edit, self.restored_edit = self.restored_edit, None
            return edit
        current = self.labels[visit.roi]
        edit = np.zeros_like(current)
        local = visit.local_slices
        on_slices = np.moveaxis(current, visit.axis, 0)[local]
        np.moveaxis(edit, visit.axis, 0)[local] = np.where(np.isin(on_slices, visit.focus), on_slices, 0)
        if visit.proposal is not None and len(visit.proposal):
            on_slices = visit.proposal[np.isin(visit.proposal[:, visit.axis], local)]
            free = edit[tuple(on_slices.T)] == 0
            edit[tuple(on_slices[free].T)] = visit.label_id
        return edit

    def locked(self) -> np.ndarray:
        """The voxels of the locked cells in the region of the current visit."""
        visit = self.view()
        current = self.labels[visit.roi]
        return (current > 0) & ~np.isin(current, visit.editable)

    def add_slice(self, position: int) -> None:
        """Add an edit slice at a position along the slice axis of the visit, in crop coordinates."""
        visit = self.view()
        if not visit.roi[visit.axis].start <= position < visit.roi[visit.axis].stop:
            raise ValueError(f"The slice {position} is outside the region of the visit.")
        visit.slices = sorted(set(visit.slices) | {int(position)})

    def remove_slice(self, position: int) -> None:
        visit = self.view()
        visit.slices = [s for s in visit.slices if s != int(position)]

    def preview(self, edit: np.ndarray) -> np.ndarray:
        """The filled labels of the region for an edit, without a commit."""
        return self._fill(edit, allocate=False)[0]

    def _fill(self, edit: np.ndarray, allocate: bool = True) -> Tuple[np.ndarray, Dict[int, int]]:
        visit = self.view()
        if allocate:
            new_label = self.new_label
        else:
            # The preview does not use up label ids.
            counter = iter(range(self.next_label, 2 ** 32))
            new_label = counter.__next__
        return fill_visit(self.labels[visit.roi], edit, visit.axis, visit.local_slices, visit.editable, self.settings,
                          new_label)

    # The actions.

    def _history_entry(self, action: str) -> dict:
        """The record of an action for the undo: the loop state before it and the geometry of the visit."""
        visit, geometry = self.visit, None
        if visit is not None:
            geometry = {
                "item": visit.item.item_id, "label_id": int(visit.label_id),
                "roi": [[sl.start, sl.stop] for sl in visit.roi], "axis": visit.axis,
                "slices": [int(i) for i in visit.slices], "focus": [int(i) for i in visit.focus],
                "new_ids": [int(i) for i in visit.new_ids],
            }
        return {"action": action, "patch": None, "state": self._loop_state(), "visit": geometry}

    def _commit(
        self, roi, after: np.ndarray, action: str, item: Item, label_id: int, edit: Optional[np.ndarray] = None,
        record: Optional[dict] = None,
    ) -> Optional[dict]:
        """Store and apply a change of the labels in a region. Returns the patch entry, or None without a change.

        `record` is the history entry of the action. It gets the patch index, and it is stored with the patch.
        """
        before = self.labels[roi]
        meta = {"item": item.item_id, "action": action, "label_id": int(label_id), "reasons": ",".join(item.reasons)}
        if self.visit is not None and edit is not None:
            meta.update({"axis": self.visit.axis, "slices": self.visit.slices, "focus": self.visit.editable})
        if record is not None:
            record["patch"] = self.session.state["n_patches"]
            self._update_state()
        entry = self.session.commit(tuple(sl.start for sl in roi), before, after, meta, edit=edit)
        if entry is None:
            if record is not None:
                record["patch"] = None
            return None
        self.labels[roi] = after
        self.step += 1
        for label in entry["delta"]:
            self.last_change[int(label)] = self.step
        return entry

    def _close(self, status: str) -> None:
        item = self.visit.item
        item.status, item.visits, item.proposal = status, item.visits + 1, None
        self.queue.remove(item.item_id)
        if status == "pending":
            self.queue.append(item.item_id)
        else:
            item.done_at = self.step
        self.visit = None
        self._save(proposals=True)

    def next(self, edit: np.ndarray) -> bool:
        """Fill and commit the edit. Returns True if it changed the labels; then the item comes back later."""
        return self._finish_visit(edit, accept=False)

    def accept(self, edit: np.ndarray) -> bool:
        """Fill and commit the edit, and mark the item as done. Returns True if it changed the labels."""
        return self._finish_visit(edit, accept=True)

    def _finish_visit(self, edit: np.ndarray, accept: bool) -> bool:
        visit = self.view()
        action = "accept" if accept else "next"
        record = self._history_entry(action)
        result, splits = self._fill(edit)
        self.history.append(record)
        entry = self._commit(visit.roi, result, action, visit.item, visit.label_id, edit, record)
        reasons: Dict[int, List[str]] = {split_id: ["split"] for split_id in splits}
        for split_id, label_id in splits.items():
            self.messages.append(f"A separate piece of label {self.name_of(label_id)} became the new label "
                                 f"{self.name_of(split_id)}.")
        if entry is not None:
            neighbours = {int(i) for i in entry["delta"]} & set(visit.focus) - {visit.label_id}
            for finding in self._shape_findings(sorted(neighbours)):
                reasons.setdefault(finding.label_id, []).append(f"neighbour_{finding.reason}")
        self._queue_cells(reasons)
        self._close("done" if accept or entry is None else "pending")
        return entry is not None

    def _shape_findings(self, label_ids: List[int]) -> list:
        if not label_ids:
            return []
        return shape_findings(self.labels, label_ids, self.settings.voxel_size, self._median_cell_size())

    def reject(self) -> None:
        """Drop the item without a change, for example a wrong proposal."""
        self.view()
        self.history.append(self._history_entry("reject"))
        self._close("rejected")

    def defer(self) -> None:
        """Move the item to the end of the queue without a change."""
        visit = self.view()
        self.history.append(self._history_entry("defer"))
        self.queue.remove(visit.item.item_id)
        self.queue.append(visit.item.item_id)
        self.visit = None
        self._save()

    def delete_cell(self) -> None:
        """Remove the label of the visit from the crop, and mark the item as done."""
        visit = self.view()
        record = self._history_entry("delete")
        self.history.append(record)
        mask = self.labels == visit.label_id
        if mask.any():
            voxels = np.argwhere(mask)
            roi = tuple(slice(lo, hi + 1) for lo, hi in zip(voxels.min(axis=0), voxels.max(axis=0)))
            after = self.labels[roi].copy()
            after[after == visit.label_id] = 0
            self._commit(roi, after, "delete", visit.item, visit.label_id, record=record)
        self._close("done")

    def new_cell(self, anchor: Sequence[int]) -> Item:
        """Add a missing cell at a position before the current item, and start it."""
        anchor = tuple(int(a) for a in anchor)
        proposal = self.propose(anchor) if self.propose is not None else None
        self.history.append(self._history_entry("new cell"))
        item = Item(max(self.items, default=-1) + 1, "missing", anchor, 0, ["manual"], proposal=proposal)
        self.items[item.item_id] = item
        if proposal is not None:
            self.proposals[item.item_id] = proposal
        self.queue.insert(0, item.item_id)
        self.visit = None
        self._save(proposals=True)
        return item

    def reset(self) -> Visit:
        """Start the current item again, with new slices from the current labels."""
        self.visit = None
        return self.view()

    def undo_last(self) -> Optional[str]:
        """Revert the last action and restore the loop as it was before it. Returns the action, or None.

        The action can be a Next or Accept without a change. Its visit is the current visit again, with the stored
        edit of the user (see `prefill`). The commits of the region proposal at the start are not actions of the
        loop, so they stay.
        """
        if not self.history:
            return None
        record = self.history.pop()
        state = record["state"]
        self.items = {item.item_id: item for item in self._items_of(state, self.proposals)}
        self.queue = list(state["queue"])
        self.next_label, self.step = int(state["next_label"]), int(state["step"])
        self.checked_at = int(state["checked_at"])
        self.last_change = {int(k): v for k, v in state["last_change"].items()}
        self.finished = bool(state["finished"])
        self.visit, self.restored_edit, self._median_size = None, None, None
        self._update_state()
        edit = None
        if record["patch"] is None:
            self.session.save()
        else:
            patches = self.session.state["patches"]
            if not patches or patches[-1]["patch"] != record["patch"]:
                raise RuntimeError(f"The last patch of the session is not the patch {record['patch']} of the last "
                                   "action, so the undo stops.")
            _, edit = self.session.undo_last(self.labels)
        geometry = record["visit"]
        if geometry is not None:
            item = self.items[geometry["item"]]
            roi = tuple(slice(start, stop) for start, stop in geometry["roi"])
            proposal = self._local_proposal(self._open_proposal(item, geometry["label_id"]), roi)
            self.visit = Visit(item, geometry["label_id"], roi, geometry["axis"], list(geometry["slices"]),
                               list(geometry["focus"]), list(geometry["new_ids"]), proposal=proposal)
            self.restored_edit = edit
        return record["action"]

    def _queue_cells(self, reasons: Dict[int, List[str]]) -> None:
        """Queue cells again with new reasons, or add an item for them."""
        for label_id in sorted(reasons):
            item = next((i for i in self.items.values() if i.label_id == label_id), None)
            if item is None:
                voxels = np.argwhere(self.labels == label_id)
                if len(voxels) == 0:
                    continue
                item = Item(max(self.items) + 1, "cell", inner_point(voxels), label_id, [])
                self.items[item.item_id] = item
            item.reasons.extend(reason for reason in reasons[label_id] if reason not in item.reasons)
            if item.item_id not in self.queue:
                item.status = "pending"
                self.queue.append(item.item_id)
            self.messages.append(f"Label {self.name_of(label_id)} comes back: {', '.join(reasons[label_id])}.")

    def _check_changed(self) -> bool:
        """Run the detectors on the cells that changed after they were done. Returns True if items were added."""
        done_at = {}
        for item in self.items.values():
            if item.status != "pending" and item.label_id:
                done_at[item.label_id] = max(done_at.get(item.label_id, -1), item.done_at)
        changed = [label for label, step in self.last_change.items()
                   if step > done_at.get(label, self.checked_at) and step > self.checked_at]
        self.checked_at = self.step
        if not changed or self.redetect is None:
            return False
        present = set(np.unique(self.labels[np.isin(self.labels, changed)]).tolist())
        new_items = self.redetect(self.labels, [label for label in changed if label in present])
        for item in new_items:
            item.item_id = max(self.items, default=-1) + 1
            self.items[item.item_id] = item
            if item.proposal is not None:
                self.proposals[item.item_id] = item.proposal
            self.queue.append(item.item_id)
        self._save(proposals=bool(new_items))
        return bool(new_items)

    def export(self, original: np.ndarray, output_path: str, table_path: str, keep: bool = True) -> None:
        """Write the corrected annotation with the original ids, see `Session.export`."""
        self.session.export(self.labels, original, output_path, table_path, keep)
