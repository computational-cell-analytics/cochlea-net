import os
import warnings
from typing import Dict, List, Optional

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QComboBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from magicgui import magicgui

from elf.parallel.distance_transform import distance_transform
from elf.parallel.seeded_watershed import seeded_watershed

from flamingo_tools.measurements import get_object_measures_from_table
from flamingo_tools.s3_utils import get_s3_path


class HistogramWidget(QWidget):
    """Qt widget that draws/updates a histogram for one napari layer."""
    def __init__(self, statistics, default_stat, bins: int = 32, parent=None):
        super().__init__(parent)
        self.bins = bins

        # --- layout ------------------------------------------------------
        self.fig, self.ax = plt.subplots(figsize=(4, 3), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)

        self.param_box = QComboBox()

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_hist)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Choose statistic:"))
        layout.addWidget(self.param_box)
        layout.addWidget(self.canvas)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)

        self.set_statistics(statistics, default_stat)

    def set_statistics(self, statistics, default_stat):
        """Swap in a new statistics table, e.g. after switching the active channel."""
        self.statistics = statistics
        # We exclude the label id and the volume / surface measurements.
        self.stat_names = statistics.columns[1:-2] if len(statistics.columns) > 2 else statistics.columns[1:]
        self.param_choices = self.stat_names

        self.param_box.clear()
        self.param_box.addItems(self.param_choices)
        default_stat = default_stat if default_stat in self.param_choices else self.param_choices[0]
        self.param_box.setCurrentText(default_stat)

        self.update_hist()

    def update_hist(self):
        """Redraw the histogram."""
        self.ax.clear()

        stat_name = self.param_box.currentText()

        data = self.statistics[stat_name]
        # Seaborn version (nicer aesthetics)
        sns.histplot(data, bins=self.bins, ax=self.ax, kde=False)

        self.ax.set_xlabel(f"{stat_name} Marker Intensity")
        self.ax.set_ylabel("Count")
        self.canvas.draw_idle()


def _create_stat_widget(statistics, default_stat):
    widget = HistogramWidget(statistics, default_stat)
    return widget


# Just dilate by 3 pixels.
def _extend_seg_simple(seg, dilation):
    block_shape = (128,) * 3
    halo = (dilation + 2,) * 3

    distances = distance_transform(seg == 0, block_shape=block_shape, halo=halo, n_threads=8)
    mask = distances < dilation

    seg_extended = np.zeros_like(seg)
    seg_extended = seeded_watershed(
        distances, seg, seg_extended, block_shape=block_shape, halo=halo, n_threads=8, mask=mask
    )

    return seg_extended


def find_channel_measurement_tables(
    table_dir: str,
    channels: List[str],
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> Dict[str, str]:
    """Find per-channel object-measures tables in a directory.

    Tables are expected to follow the naming convention
    "<channel>_<seg_name>_object-measures[-bg-mask].tsv", e.g. "Alphatag_IHC-v11_object-measures-bg-mask.tsv".
    Channels without a matching table are omitted from the result.

    Args:
        table_dir: Directory containing the object-measures tables, local or on an S3 bucket.
        channels: Channel names to look for, e.g. ["Alphatag", "Otof"].
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.

    Returns:
        Dictionary mapping channel name to table path.
    """
    if s3:
        dir_store, fs = get_s3_path(table_dir, bucket_name=s3_bucket_name,
                                    service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        file_names = [os.path.basename(p) for p in fs.ls(dir_store.path, detail=False)]
    else:
        file_names = [entry.name for entry in os.scandir(table_dir)]

    tables = {}
    for channel in channels:
        matches = sorted(name for name in file_names if name.startswith(f"{channel}_") and name.endswith(".tsv"))
        if not matches:
            continue
        if len(matches) > 1:
            bg_mask_matches = [name for name in matches if "bg-mask" in name]
            chosen = bg_mask_matches[0] if bg_mask_matches else matches[0]
            warnings.warn(f"Multiple measurement tables found for channel '{channel}' in {table_dir}: "
                          f"{matches}. Using '{chosen}'.")
        else:
            chosen = matches[0]
        tables[channel] = os.path.join(table_dir, chosen)
    return tables


def annotation_napari(
    stain_dict: dict,
    measurement_tables: Dict[str, str],
    seg_name: str,
    seg_file: str,
    default_channel: Optional[str] = None,
    statistics_keyword: str = "median",
    is_otof: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Visualize data in Napari for thresholding.

    Args:
        stain_dict: Dictionary containing stain names and file paths.
        measurement_tables: Dictionary mapping channel name to a table of object measures for that channel.
        seg_name: Segmentation name, e.g. SGN_v2.
        seg_file: File path to segmentation data.
        default_channel: Channel selected by default. Falls back to the first entry of `measurement_tables`
            if not given or not present in `measurement_tables`.
        statistics_keyword: Keyword for column in object measures dataframe.
        is_otof: Flag for analyzing OTOF cochleae.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    if not measurement_tables:
        raise ValueError("No measurement tables were given.")

    def _load_table(path):
        if s3:
            table_path_s3, fs = get_s3_path(path, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            with fs.open(table_path_s3, "r") as f:
                return pd.read_csv(f, sep="\t")
        return pd.read_csv(path, sep="\t")

    seg = imageio.imread(seg_file)
    all_statistics = {
        channel: get_object_measures_from_table(seg, table=_load_table(path), keyword=statistics_keyword)
        for channel, path in measurement_tables.items()
    }

    if default_channel is None or default_channel not in all_statistics:
        default_channel = next(iter(all_statistics))

    seg_extended = _extend_seg_simple(seg, dilation=4)
    if is_otof:
        seg_extended = seg.copy()

    # Open the napari viewer.
    v = napari.Viewer()

    for num, (stain_name, file_path) in enumerate(stain_dict.items()):
        stain = imageio.imread(file_path)
        if num == 0:
            stain_shape = stain.shape
        v.add_image(stain, visible=(stain_name == default_channel), name=stain_name)

    # Add the base layers.
    v.add_labels(seg, visible=False, name=f"{seg_name}s")
    v.add_labels(seg_extended, name=f"{seg_name}s-extended")

    # Add additional layers for intensity coloring and classification
    data_labels = np.zeros(stain_shape, dtype="uint8")
    v.add_labels(data_labels, name="positive-negative")

    # Add widgets:

    # 1.) The widget for selecting the statistics to be used and displaying the histogram.
    stat_widget = _create_stat_widget(all_statistics[default_channel], statistics_keyword)
    stat_widget.setWindowTitle(f"{default_channel} Histogram")

    # 2.) Precompute statistic ranges.
    all_values = all_statistics[default_channel][stat_widget.stat_names].values
    min_val = all_values.min()
    max_val = all_values.max()

    # 3.) The widget for printing the intensity of a selected cell.
    @magicgui(
        value={
            "label": "value", "enabled": False, "widget_type": "FloatSpinBox", "min": min(min_val, 0), "max": max_val
        },
        call_button="Pick Value"
    )
    def pick_widget(viewer: napari.Viewer, value: float = 0.0):
        statistics = all_statistics[channel_widget.channel.value]
        layer = viewer.layers[f"{seg_name}s-extended"]
        selected_id = layer.selected_label

        stat_name = stat_widget.param_box.currentText()
        label_ids = statistics.label_id.values
        if selected_id not in label_ids:
            return {"value": 0.0}

        vals = statistics[stat_name].values
        picked_value = vals[label_ids == selected_id][0]
        pick_widget.value.value = picked_value

    # 4.) The widget for setting the threshold and updating the positive / negative classification based on it.
    @magicgui(
        threshold={
            "widget_type": "FloatSlider",
            "label": "Threshold",
            "min": min_val,
            "max": max_val,
            "step": 1,
        },
        call_button="Apply",
    )
    def threshold_widget(viewer: napari.Viewer, threshold: float = (max_val + min_val) / 2):
        statistics = all_statistics[channel_widget.channel.value]
        label_ids = statistics.label_id.values
        stat_name = stat_widget.param_box.currentText()
        vals = statistics[stat_name].values
        pos_ids = label_ids[vals >= threshold]
        neg_ids = label_ids[vals <= threshold]
        data_labels = np.zeros(stain_shape, dtype="uint8")
        data_labels[np.isin(seg_extended, pos_ids)] = 2
        data_labels[np.isin(seg_extended, neg_ids)] = 1
        viewer.layers["positive-negative"].data = data_labels

    threshold_widget.viewer.value = v

    # 5.) The widget for selecting which channel's statistics drive the widgets above.
    @magicgui(
        channel={"widget_type": "ComboBox", "choices": list(all_statistics), "label": "Channel"},
        auto_call=True,
        call_button=False,
    )
    def channel_widget(channel: str = default_channel):
        for stain_name in stain_dict:
            v.layers[stain_name].visible = (stain_name == channel)

        stat_widget.set_statistics(all_statistics[channel], statistics_keyword)
        stat_widget.setWindowTitle(f"{channel} Histogram")

        vals = all_statistics[channel][stat_widget.stat_names].values
        new_min, new_max = vals.min(), vals.max()
        threshold_widget.threshold.min = new_min
        threshold_widget.threshold.max = new_max
        threshold_widget.threshold.value = (new_min + new_max) / 2
        pick_widget.value.min = min(new_min, 0)
        pick_widget.value.max = new_max

    # Bind the widgets. Registration order controls the top-to-bottom stacking in the dock area.
    if len(all_statistics) > 1:
        v.window.add_dock_widget(channel_widget, area="right")
    v.window.add_dock_widget(stat_widget, area="right")
    v.window.add_dock_widget(pick_widget, area="right")
    v.window.add_dock_widget(threshold_widget, area="right")

    napari.run()
