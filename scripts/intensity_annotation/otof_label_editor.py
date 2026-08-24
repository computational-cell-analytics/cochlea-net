"""Shared data preparation and napari UI for OTOF assignment editing."""

import os
import tempfile
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt


SEGMENTATION_NAME = "IHC_v11"
OTOF_NAME = "Otof"
MARKER_COLUMN = "marker_labels"
POSITIVE = 1
NEGATIVE = 2
BASE_VOXEL_SIZE_XYZ = (0.38, 0.38, 0.38)


@dataclass
class EditorData:
    table: pd.DataFrame
    ihcs: np.ndarray
    marker_mask: np.ndarray
    otof: np.ndarray
    scale_zyx: Tuple[float, float, float]


def validate_table(table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Validate the columns that define the editable marker assignments."""
    required = {"label_id", MARKER_COLUMN}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"The IHC table is missing required columns: {missing}.")

    label_values = pd.to_numeric(table["label_id"], errors="coerce").to_numpy()
    marker_values = pd.to_numeric(table[MARKER_COLUMN], errors="coerce").to_numpy()
    if not np.isfinite(label_values).all() or not np.equal(label_values, np.floor(label_values)).all():
        raise ValueError("The 'label_id' column must contain finite integer values.")
    if not np.isfinite(marker_values).all() or not np.equal(marker_values, np.floor(marker_values)).all():
        raise ValueError(f"The '{MARKER_COLUMN}' column must contain finite integer values.")

    label_ids = label_values.astype("int64")
    marker_labels_int = marker_values.astype("int64")
    if np.any(label_ids < 0):
        raise ValueError("The 'label_id' column must not contain negative values.")
    if pd.Index(label_ids).has_duplicates:
        duplicates = pd.Index(label_ids)[pd.Index(label_ids).duplicated()].unique().tolist()
        raise ValueError(f"The IHC table contains duplicate label IDs: {duplicates}.")

    invalid = sorted(set(np.unique(marker_labels_int)).difference({0, POSITIVE, NEGATIVE}))
    if invalid:
        raise ValueError(
            f"The '{MARKER_COLUMN}' column contains invalid values {invalid}; expected only 0, 1, or 2."
        )
    marker_labels = marker_labels_int.astype("uint8")
    return label_ids, marker_labels


def validate_volumes(ihcs: np.ndarray, otof: np.ndarray) -> None:
    """Validate the arrays before they are added to napari."""
    if ihcs.ndim != 3 or otof.ndim != 3:
        raise ValueError(f"Expected two 3D volumes, but received shapes {ihcs.shape} and {otof.shape}.")
    if ihcs.shape != otof.shape:
        raise ValueError(f"The IHC and Otof volumes have different shapes: {ihcs.shape} and {otof.shape}.")
    if not np.issubdtype(ihcs.dtype, np.integer):
        raise ValueError(f"The IHC segmentation must have an integer dtype, not {ihcs.dtype}.")


def build_editable_layers(segmentation: np.ndarray, table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Create the editable IHC ID layer and its positive/negative class mask."""
    label_ids, marker_labels = validate_table(table)
    if not np.issubdtype(segmentation.dtype, np.integer):
        raise ValueError(f"The IHC segmentation must have an integer dtype, not {segmentation.dtype}.")
    if np.issubdtype(segmentation.dtype, np.signedinteger) and segmentation.size and segmentation.min() < 0:
        raise ValueError("The IHC segmentation must not contain negative label IDs.")

    segmentation_max = int(segmentation.max()) if segmentation.size else 0
    table_max = int(label_ids.max()) if label_ids.size else 0
    max_label = max(segmentation_max, table_max)
    if max_label > np.iinfo("uint32").max:
        raise ValueError(f"Label ID {max_label} exceeds the supported uint32 range.")

    ihcs = segmentation.astype("uint32", copy=False)
    marker_lut = np.zeros(max_label + 1, dtype="uint8")
    marker_lut[label_ids] = marker_labels
    marker_mask = marker_lut[ihcs]

    ihcs = ihcs.copy()
    ihcs[marker_mask == 0] = 0
    return ihcs, marker_mask


def mask_signal_to_ihcs(
    signal: np.ndarray,
    ihcs: np.ndarray,
    radius: float = 0.0,
    scale_zyx: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Keep signal within a physical radius of the IHC mask."""
    if signal.shape != ihcs.shape:
        raise ValueError(f"The signal and IHC mask have different shapes: {signal.shape} and {ihcs.shape}.")

    radius = float(radius)
    if not np.isfinite(radius) or radius < 0:
        raise ValueError(f"The masking radius must be a finite, non-negative value, not {radius}.")
    scale = tuple(float(value) for value in scale_zyx)
    if len(scale) != 3 or not np.isfinite(scale).all() or any(value <= 0 for value in scale):
        raise ValueError(f"The layer scale must contain three finite, positive values, not {scale_zyx}.")

    if radius == 0:
        dilated_ihcs = ihcs != 0
    else:
        distance_to_ihcs = distance_transform_edt(ihcs == 0, sampling=scale)
        dilated_ihcs = distance_to_ihcs <= radius
    signal[~dilated_ihcs] = 0
    return signal


def prepare_editor_data(
    table: pd.DataFrame,
    segmentation: np.ndarray,
    otof: np.ndarray,
    scale_zyx: Sequence[float],
    scale_name: str,
    masking_radius: Optional[float] = None,
) -> EditorData:
    """Validate loaded data and build the layers used by the editor."""
    validate_volumes(segmentation, otof)
    ihcs, marker_mask = build_editable_layers(segmentation, table)
    if masking_radius is not None:
        mask_signal_to_ihcs(otof, ihcs, masking_radius, scale_zyx)

    label_ids, marker_labels = validate_table(table)
    assigned_ids = set(label_ids[np.isin(marker_labels, (POSITIVE, NEGATIVE))].tolist())
    visible_ids = set(np.unique(ihcs).tolist())
    visible_ids.discard(0)
    missing_ids = sorted(assigned_ids.difference(visible_ids))
    if missing_ids:
        warnings.warn(
            f"{len(missing_ids)} assigned IHC IDs are absent at {scale_name}. "
            "They will remain unchanged in the exported table."
        )

    scale_tuple = tuple(float(value) for value in scale_zyx)
    if len(scale_tuple) != 3 or any(value <= 0 for value in scale_tuple):
        raise ValueError(f"The layer scale must contain three positive values, not {scale_zyx}.")
    return EditorData(table=table, ihcs=ihcs, marker_mask=marker_mask, otof=otof, scale_zyx=scale_tuple)


def assignment_for_label(table: pd.DataFrame, label_id: int) -> Tuple[object, int]:
    """Return the table index and marker assignment for one label ID."""
    label_ids, marker_labels = validate_table(table)
    matches = np.flatnonzero(label_ids == int(label_id))
    if len(matches) != 1:
        raise ValueError(f"Label ID {label_id} is not present in the IHC table.")
    position = int(matches[0])
    return table.index[position], int(marker_labels[position])


def _label_roi(
    row: pd.Series,
    shape: Sequence[int],
    scale_zyx: Sequence[float],
) -> Optional[Tuple[slice, slice, slice]]:
    columns = ("bb_min_z", "bb_min_y", "bb_min_x", "bb_max_z", "bb_max_y", "bb_max_x")
    if any(column not in row.index for column in columns):
        return None

    bounds = pd.to_numeric(row[list(columns)], errors="coerce").to_numpy(dtype="float64")
    if not np.isfinite(bounds).all():
        return None

    minimum = bounds[:3]
    maximum = bounds[3:]
    scale = np.asarray(scale_zyx, dtype="float64")
    if np.any(scale <= 0) or np.any(maximum < minimum):
        return None

    start = np.floor(minimum / scale).astype(int) - 1
    stop = np.ceil(maximum / scale).astype(int) + 2
    start = np.maximum(start, 0)
    stop = np.minimum(stop, np.asarray(shape, dtype=int))
    return tuple(slice(int(begin), int(end)) for begin, end in zip(start, stop))


def _update_marker_mask(
    ihcs: np.ndarray,
    marker_mask: np.ndarray,
    label_id: int,
    marker_label: int,
    roi: Optional[Tuple[slice, slice, slice]],
) -> int:
    label_region = ihcs if roi is None else ihcs[roi]
    selected = label_region == label_id
    if selected.any():
        mask_region = marker_mask if roi is None else marker_mask[roi]
        mask_region[selected] = marker_label
        return int(selected.sum())

    if roi is not None:
        selected = ihcs == label_id
        if selected.any():
            marker_mask[selected] = marker_label
            return int(selected.sum())
    raise ValueError(f"Label ID {label_id} is not visible at the selected pyramid scale.")


def switch_assignment(
    table: pd.DataFrame,
    ihcs: np.ndarray,
    marker_mask: np.ndarray,
    label_id: int,
    scale_zyx: Sequence[float],
) -> Tuple[int, int]:
    """Switch one visible assigned IHC and update the table and class mask."""
    if int(label_id) == 0:
        raise ValueError("Select an IHC instead of the background.")

    row_index, old_label = assignment_for_label(table, int(label_id))
    if old_label not in (POSITIVE, NEGATIVE):
        raise ValueError(f"Label ID {label_id} is unassigned and cannot be edited.")

    new_label = NEGATIVE if old_label == POSITIVE else POSITIVE
    roi = _label_roi(table.loc[row_index], ihcs.shape, scale_zyx)
    _update_marker_mask(ihcs, marker_mask, int(label_id), new_label, roi)
    table.at[row_index, MARKER_COLUMN] = new_label
    return old_label, new_label


def export_table(table: pd.DataFrame, output_path: str) -> str:
    """Write the complete edited table atomically as a TSV file."""
    validate_table(table)
    output_path = os.path.abspath(output_path)
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_dir,
            prefix=f".{os.path.basename(output_path)}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = temporary.name
            table.to_csv(temporary, sep="\t", index=False)
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None and os.path.exists(temporary_path):
            os.unlink(temporary_path)
        raise
    return output_path


def _class_name(marker_label: int) -> str:
    return {POSITIVE: "positive", NEGATIVE: "negative"}.get(marker_label, "unassigned")


def run_editor(dataset_name: str, output_table: str, data: EditorData) -> None:
    """Open the napari editor and run its Qt event loop."""
    import napari
    from napari.utils.colormaps import DirectLabelColormap
    from qtpy.QtWidgets import QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

    viewer = napari.Viewer(title=f"OTOF assignment editor | {dataset_name}")
    viewer.add_image(
        data.otof,
        name=OTOF_NAME,
        colormap="gray",
        blending="additive",
        scale=data.scale_zyx,
    )
    ihc_layer = viewer.add_labels(
        data.ihcs,
        name=f"{SEGMENTATION_NAME} editable IDs",
        opacity=0.25,
        scale=data.scale_zyx,
    )
    marker_layer = viewer.add_labels(
        data.marker_mask,
        name="positive-negative",
        colormap=DirectLabelColormap(color_dict={
            None: np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            0: np.array([0.0, 0.0, 0.0, 0.0], dtype="float32"),
            POSITIVE: np.array([0.2, 1.0, 0.2, 1.0], dtype="float32"),
            NEGATIVE: np.array([1.0, 0.0, 1.0, 1.0], dtype="float32"),
        }),
        opacity=0.7,
        scale=data.scale_zyx,
    )
    viewer.layers.selection.active = ihc_layer
    ihc_layer.mode = "pick"
    ihc_layer.selected_label = 0
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "µm"

    initial_ids, initial_labels = validate_table(data.table)
    initial_assignments = dict(zip(initial_ids.tolist(), initial_labels.tolist()))
    changed_ids = set()

    class EditorWidget(QWidget):
        def __init__(self):
            super().__init__()
            self.selection_text = QLabel("Select an assigned IHC with the picker.")
            self.pending_text = QLabel("Pending changes: 0")
            self.message_text = QLabel("")
            self.switch_button = QPushButton("Switch positive / negative")
            self.switch_button.setEnabled(False)
            self.switch_button.clicked.connect(self.switch_selected)
            self.export_button = QPushButton("Export table")
            self.export_button.clicked.connect(self.write_table)

            layout = QVBoxLayout()
            layout.addWidget(self.selection_text)
            layout.addWidget(self.pending_text)
            layout.addWidget(self.switch_button)
            layout.addWidget(self.export_button)
            layout.addWidget(self.message_text)
            self.setLayout(layout)

        def selected_assignment(self) -> Tuple[int, Optional[int]]:
            label_id = int(ihc_layer.selected_label)
            if label_id == 0:
                return label_id, None
            try:
                _, marker_label = assignment_for_label(data.table, label_id)
            except ValueError:
                return label_id, None
            return label_id, marker_label

        def refresh_selection(self, event=None) -> None:
            label_id, marker_label = self.selected_assignment()
            editable = marker_label in (POSITIVE, NEGATIVE)
            if editable:
                self.selection_text.setText(f"Selected IHC: {label_id} ({_class_name(marker_label)})")
            elif label_id == 0:
                self.selection_text.setText("Select an assigned IHC with the picker.")
            else:
                self.selection_text.setText(f"Selected IHC: {label_id} (not editable)")
            self.switch_button.setEnabled(editable)
            self.pending_text.setText(f"Pending changes: {len(changed_ids)}")

        def switch_selected(self) -> None:
            label_id = int(ihc_layer.selected_label)
            try:
                _, new_label = switch_assignment(
                    data.table,
                    data.ihcs,
                    data.marker_mask,
                    label_id,
                    data.scale_zyx,
                )
            except ValueError as error:
                self.message_text.setText(str(error))
                self.refresh_selection()
                return

            if new_label == initial_assignments[label_id]:
                changed_ids.discard(label_id)
            else:
                changed_ids.add(label_id)
            marker_layer.refresh()
            self.message_text.setText(f"IHC {label_id} is now {_class_name(new_label)}.")
            self.refresh_selection()

        def write_table(self) -> None:
            if os.path.exists(output_table):
                answer = QMessageBox.question(
                    self,
                    "Replace table?",
                    f"Replace the existing file?\n{os.path.abspath(output_table)}",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if answer != QMessageBox.Yes:
                    return
            try:
                exported_path = export_table(data.table, output_table)
            except Exception as error:
                QMessageBox.critical(self, "Export failed", str(error))
                return
            self.message_text.setText(f"Exported {len(changed_ids)} changes to {exported_path}.")

    widget = EditorWidget()
    ihc_layer.events.selected_label.connect(widget.refresh_selection)
    viewer.window.add_dock_widget(widget, name="OTOF assignments", area="right")

    @viewer.bind_key("t")
    def switch_selected(viewer_instance):
        widget.switch_selected()

    napari.run()
