"""The napari user interface of the correction loop, see `controller.Controller`.

napari and Qt are imported in `run_correction`, so the rest of the package works without them.
"""
from typing import Optional, Tuple

import numpy as np

from .controller import Controller, filter_paint_atoms
from .ids import IdMap

PAINT_MODES = ("paint", "fill", "erase", "polygon")
AXIS_NAMES = "zyx"
HELP = (
    "Paint on the edit slices of the edit layer (j / k: previous / next slice). Only the target and the cells "
    "near it are in the edit layer; the other cells are locked.\n"
    "o: unlock the cell under the cursor, so that it can change too.\n"
    "m: a new label, for example to split a cell. The label ids are the ids of the files.\n"
    "n: Next - fill and commit; a changed cell comes back once more, an unchanged cell is done.\n"
    "Shift-N: Accept - fill, commit, and mark the cell as done.\n"
    "g: Preview the fill in the segmentation layer. u: Reset the edit."
)
REVERT_MESSAGES = {
    "slice": "Paint outside the edit slices was undone. Use 'Add slice' first.",
    "locked": "Paint on a locked cell was undone. Press o on the cell to unlock it.",
    "label": "This label belongs to a locked cell, so the paint was undone. Press o on the cell to unlock it, or "
             "m for a new label.",
}


def run_correction(
    controller: Controller,
    crop_name: str,
    original: np.ndarray,
    output_path: str,
    table_path: str,
    comparison: Optional[np.ndarray] = None,
    comparison_name: str = "comparison",
    remove_session: bool = False,
    contrast_limits: Tuple[float, float] = (0.0, 3000.0),
    show: bool = True,
):
    """Open the correction loop in napari. The corrected segmentation is written when all items are done.

    The layers show the file ids: the ids of the base file, which the output keeps, and the ids of the
    comparison file.

    Args:
        controller: The correction loop.
        crop_name: The crop name for the window title.
        original: The original ids of the compact ids, for the display and the export.
        output_path: The path of the corrected segmentation.
        table_path: The path of the change table.
        comparison: The comparison segmentation of the crop (S3 or the annotation) with its file ids, or None.
        comparison_name: The layer name of the comparison.
        remove_session: Remove the session folder after the export.
        contrast_limits: The contrast limits of the image at the start.
        show: Run the napari event loop. False returns the viewer and the widget, for tests.

    Returns:
        The viewer and the dock widget if `show` is False.
    """
    import napari
    from qtpy.QtWidgets import QGridLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

    viewer = napari.Viewer(title=f"Annotation correction | {crop_name}", show=show)
    empty = np.zeros((1, 1, 1), dtype="uint32")
    image_layer = viewer.add_image(np.zeros((1, 1, 1), dtype=controller.image.dtype), name="image",
                                   colormap="gray")
    annotation_layer = viewer.add_labels(empty.copy(), name="segmentation", opacity=0.35)
    comparison_layer = None
    if comparison is not None:
        comparison_layer = viewer.add_labels(empty.copy(), name=comparison_name, opacity=0.35, visible=False)
    edit_layer = viewer.add_labels(empty.copy(), name="edit (sparse slices)", opacity=0.8)
    viewer.text_overlay.visible = True
    viewer.scale_bar.visible = True

    id_map = IdMap(original)

    def display_id(label_id: int) -> str:
        return str(id_map.file_id(label_id))

    controller.name_of = display_id

    def to_compact(file_ids: np.ndarray) -> Tuple[np.ndarray, list]:
        """The compact ids of an edit in file ids. An unknown id becomes a new label."""
        labels = np.zeros(file_ids.shape, dtype="uint32")
        renamed = []
        for file_id in np.unique(file_ids):
            if file_id == 0:
                continue
            label_id = id_map.compact_id(file_id, controller.next_label)
            if label_id is None:
                label_id = controller.new_label()
                renamed.append((int(file_id), id_map.file_id(label_id)))
            labels[file_ids == file_id] = label_id
        return labels, renamed

    def locked_value(file_id: int) -> bool:
        label_id = id_map.compact_id(file_id, controller.next_label)
        return controller.visit is not None and label_id is not None and label_id not in controller.visit.editable

    class CorrectionWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.item_text = QLabel("")
            self.item_text.setWordWrap(True)
            self.message_text = QLabel("")
            self.message_text.setWordWrap(True)
            help_text = QLabel(HELP)
            help_text.setWordWrap(True)
            self.buttons = {}
            grid = QGridLayout()
            actions = [
                ("Next [n]", self.next_item), ("Accept [Shift-N]", self.accept_item),
                ("Preview [g]", self.preview), ("Reset [u]", self.reset),
                ("Previous slice [j]", lambda: self.jump(-1)), ("Next slice [k]", lambda: self.jump(1)),
                ("Add slice", self.add_slice), ("Remove slice", self.remove_slice),
                ("New label [m]", self.new_label), ("Defer", self.defer),
                ("Reject proposal", self.reject), ("Delete cell", self.delete_cell),
                ("New cell at cursor", self.new_cell), ("Unlock cell at cursor [o]", self.unlock),
                ("Undo last visit", self.undo),
            ]
            for index, (text, callback) in enumerate(actions):
                button = QPushButton(text)
                button.clicked.connect(callback)
                grid.addWidget(button, index // 2, index % 2)
                self.buttons[text] = button
            layout = QVBoxLayout()
            layout.addWidget(self.item_text)
            layout.addLayout(grid)
            layout.addWidget(self.message_text)
            layout.addWidget(help_text)
            self.setLayout(layout)
            self.edit_mode = "paint"
            self.first_visit = True
            self.locked = None

        # The visit.

        def load_visit(self) -> None:
            visit = controller.view()
            if visit is None:
                self.finish()
                return
            start = visit.start
            image_layer.data = controller.image[visit.roi]
            image_layer.translate = start
            if self.first_visit:
                # The placeholder data of the layer fixed the contrast range to [0, 1].
                high = max(float(contrast_limits[1]), float(controller.image.max()))
                image_layer.contrast_limits_range = (min(0.0, float(contrast_limits[0])), high)
                image_layer.contrast_limits = tuple(float(c) for c in contrast_limits)
                self.first_visit = False
            self.set_labels(annotation_layer, id_map.to_file(controller.labels[visit.roi]), start)
            if comparison_layer is not None:
                self.set_labels(comparison_layer, comparison[visit.roi].astype("uint32"), start)
            edit_layer.data = id_map.to_file(controller.prefill())
            edit_layer.translate = start
            edit_layer._reset_history()
            edit_layer.selected_label = id_map.file_id(visit.label_id)
            self.locked = controller.locked()
            edit_layer.mode = self.edit_mode
            viewer.layers.selection.active = edit_layer
            # The slice axis is the slider axis, so the user paints in the plane perpendicular to it.
            viewer.dims.order = (visit.axis,) + tuple(a for a in range(3) if a != visit.axis)
            viewer.reset_view()
            viewer.dims.set_point(visit.axis, visit.slices[len(visit.slices) // 2])
            self.update_text()

        def set_labels(self, layer, data, start) -> None:
            layer.data = data
            layer.translate = start
            # The data setter makes the layer editable again.
            layer.editable = False

        def current_position(self) -> int:
            """The current position along the slice axis of the visit."""
            axis = controller.visit.axis if controller.visit is not None else 0
            return int(round(viewer.dims.point[axis]))

        def update_text(self) -> None:
            visit = controller.visit
            if visit is None:
                return
            item = visit.item
            position, name = self.current_position(), AXIS_NAMES[visit.axis]
            if position in visit.slices:
                slice_text = f"{name} {position} ({visit.slices.index(position) + 1}/{len(visit.slices)})"
            else:
                slice_text = f"{name} {position} (no edit slice)"
            done = len(controller.items) - len(controller.queue) + 1
            text = (f"item {done}/{len(controller.items)} - label {display_id(visit.label_id)} - "
                    f"{', '.join(item.reasons) or item.kind} - {slice_text}")
            viewer.text_overlay.text = text
            others = [display_id(i) for i in visit.editable if i != visit.label_id]
            focus = (f"{display_id(visit.label_id)} (target) | editable: {', '.join(others) or 'none'} | "
                     "other cells are locked (o: unlock)")
            details = f"{text}\n{focus}\nvisits {item.visits}, anchor {list(item.anchor)}"
            self.item_text.setText(details + (f"\n{item.comment}" if item.comment else ""))

        def on_dims(self, event=None) -> None:
            visit = controller.visit
            if visit is None:
                return
            on_slice = self.current_position() in visit.slices
            if on_slice and edit_layer.mode == "pan_zoom":
                edit_layer.mode = self.edit_mode
            elif not on_slice and edit_layer.mode in PAINT_MODES:
                self.edit_mode = edit_layer.mode
                edit_layer.mode = "pan_zoom"
            self.update_text()

        def on_paint(self, event) -> None:
            visit = controller.visit
            if visit is None:
                return
            reason = filter_paint_atoms(event.value, visit.local_slices, edit_layer.data, visit.axis,
                                        self.locked, locked_value)
            if reason:
                edit_layer.refresh()
                self.message_text.setText(REVERT_MESSAGES[reason])
                # Also on the canvas, next to the paint. The next slice change shows the item text again.
                viewer.text_overlay.text = f"{viewer.text_overlay.text.splitlines()[0]}\n{REVERT_MESSAGES[reason]}"

        def edit(self) -> np.ndarray:
            """The edit in compact ids. A label id that the tool does not know becomes a new label."""
            labels, renamed = to_compact(np.asarray(edit_layer.data))
            for file_id, new_id in renamed:
                controller.messages.append(f"The unknown label {file_id} became the new label {new_id}.")
                # The layer takes the new id, so a second preview or a commit does not add another label.
                edit_layer.data[edit_layer.data == file_id] = new_id
                if edit_layer.selected_label == file_id:
                    edit_layer.selected_label = new_id
            if renamed:
                edit_layer.refresh()
            return labels

        # The actions.

        def run(self, action, message=None) -> None:
            """Run an action of the controller, show its message, and load the next visit."""
            try:
                result = action()
            except Exception as error:  # Show the error and keep the loop alive.
                QMessageBox.critical(self, "Correction", str(error))
                return
            text = message(result) if callable(message) else (message or "").format(result=result)
            self.message_text.setText("\n".join([text] + controller.messages).strip())
            controller.messages.clear()
            self.load_visit()

        def next_item(self) -> None:
            self.run(lambda: controller.next(self.edit()), lambda changed: (
                "The cell changed. It comes back once more." if changed else "No change: the cell is done."))

        def accept_item(self) -> None:
            self.run(lambda: controller.accept(self.edit()), lambda changed: (
                "Accepted with the change." if changed else "Accepted without a change."))

        def preview(self) -> None:
            if controller.visit is None:
                return
            annotation_layer.data = id_map.to_file(controller.preview(self.edit()))
            annotation_layer.editable = False
            self.message_text.setText("Preview of the fill in the segmentation layer. Nothing is committed yet.")

        def new_label(self) -> None:
            if controller.visit is None:
                return
            edit_layer.selected_label = id_map.file_id(controller.new_label())
            self.message_text.setText(f"New label {edit_layer.selected_label}.")
            self.update_text()

        def reset(self) -> None:
            self.run(controller.reset, "The edit is reset.")

        def jump(self, step: int) -> None:
            visit = controller.visit
            if visit is None:
                return
            position = self.current_position()
            targets = [s for s in visit.slices if (s > position if step > 0 else s < position)]
            if targets:
                viewer.dims.set_point(visit.axis, targets[0] if step > 0 else targets[-1])

        def add_slice(self) -> None:
            visit = controller.visit
            if visit is None:
                return
            position = self.current_position()
            try:
                controller.add_slice(position)
            except ValueError as error:
                self.message_text.setText(str(error))
                return
            local = position - visit.roi[visit.axis].start
            current = np.moveaxis(controller.labels[visit.roi], visit.axis, 0)[local]
            np.moveaxis(edit_layer.data, visit.axis, 0)[local] = id_map.to_file(
                np.where(np.isin(current, visit.editable), current, 0))
            edit_layer.refresh()
            self.on_dims()

        def remove_slice(self) -> None:
            visit = controller.visit
            if visit is None:
                return
            position = self.current_position()
            controller.remove_slice(position)
            np.moveaxis(edit_layer.data, visit.axis, 0)[position - visit.roi[visit.axis].start] = 0
            edit_layer.refresh()
            self.on_dims()

        def defer(self) -> None:
            self.run(controller.defer, "Deferred to the end of the queue.")

        def reject(self) -> None:
            self.run(controller.reject, "Rejected.")

        def delete_cell(self) -> None:
            visit = controller.visit
            if visit is None:
                return
            answer = QMessageBox.question(self, "Delete cell", f"Delete label {display_id(visit.label_id)}?")
            if answer == QMessageBox.Yes:
                self.run(controller.delete_cell, "The cell is deleted.")

        def new_cell(self) -> None:
            position = np.round(viewer.cursor.position[-3:]).astype(int)
            self.run(lambda: controller.new_cell(position), "New cell at the cursor.")

        def unlock(self) -> None:
            visit = controller.visit
            if visit is None:
                return
            position = tuple(np.round(viewer.cursor.position[-3:]).astype(int))
            inside = all(sl.start <= p < sl.stop for sl, p in zip(visit.roi, position))
            label_id = int(controller.labels[position]) if inside else 0
            if label_id == 0:
                self.message_text.setText("No cell of the region under the cursor.")
                return
            before = set(visit.slices)
            if not controller.unlock(label_id):
                self.message_text.setText(f"Label {display_id(label_id)} is editable already.")
                return
            # A fragment can add slices. They get the editable cells, as in 'Add slice'.
            added = sorted(set(visit.slices) - before)
            current = np.moveaxis(controller.labels[visit.roi], visit.axis, 0)
            edit = np.moveaxis(edit_layer.data, visit.axis, 0)
            for slice_position in added:
                local = slice_position - visit.roi[visit.axis].start
                edit[local] = id_map.to_file(np.where(np.isin(current[local], visit.editable), current[local], 0))
            # The edit slices get the cell, as in the prefill.
            local = visit.local_slices
            current = np.moveaxis(controller.labels[visit.roi], visit.axis, 0)[local]
            edit = np.moveaxis(edit_layer.data, visit.axis, 0)
            on_cell = current == label_id
            edit[local] = np.where(on_cell & (edit[local] == 0), id_map.file_id(label_id), edit[local])
            edit_layer.refresh()
            self.locked = controller.locked()
            note = f" It is a fragment: {len(added)} edit slices added." if added else ""
            self.message_text.setText(f"Label {display_id(label_id)} is editable now.{note}")
            self.on_dims()

        def undo(self) -> None:
            def message(action):
                if action is None:
                    return "Nothing to undo."
                if controller.visit is None:
                    return f"Undo of '{action}'."
                return (f"Undo of '{action}': back to label {display_id(controller.visit.label_id)}, with the edit "
                        "of that visit.")

            self.run(controller.undo_last, message)

        def finish(self) -> None:
            for button in self.buttons.values():
                button.setEnabled(False)
            try:
                controller.export(original, output_path, table_path, keep=not remove_session)
            except Exception as error:
                QMessageBox.critical(self, "Export failed", f"{error}\nThe session is kept; start the tool again.")
                return
            viewer.text_overlay.text = "All items are done."
            self.message_text.setText(f"All items are done. The segmentation is in {output_path}, "
                                      f"the changes in {table_path}.")

    widget = CorrectionWidget()
    edit_layer.events.paint.connect(widget.on_paint)
    viewer.dims.events.point.connect(widget.on_dims)
    viewer.window.add_dock_widget(widget, name="Correction", area="right")
    for key, callback in (("n", widget.next_item), ("Shift-N", widget.accept_item), ("g", widget.preview),
                          ("u", widget.reset), ("j", lambda: widget.jump(-1)), ("k", lambda: widget.jump(1)),
                          ("o", widget.unlock)):
        viewer.bind_key(key, lambda _viewer, callback=callback: callback(), overwrite=True)
    # The napari key m of a labels layer takes the largest id of the layer + 1, which can be a cell elsewhere.
    edit_layer.bind_key("m", lambda _layer: widget.new_label(), overwrite=True)
    widget.load_visit()
    if not show:
        return viewer, widget
    napari.run()
