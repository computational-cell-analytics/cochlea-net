from typing import Optional

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

        # We exclude the label id and the volume / surface measurements.
        self.stat_names = statistics.columns[1:-2] if len(statistics.columns) > 2 else statistics.columns[1:]
        self.param_choices = self.stat_names

        self.param_box = QComboBox()
        self.param_box.addItems(self.param_choices)
        self.param_box.setCurrentText(default_stat)

        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.update_hist)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Choose statistic:"))
        layout.addWidget(self.param_box)
        layout.addWidget(self.canvas)
        layout.addWidget(self.refresh_btn)
        self.setLayout(layout)

        self.statistics = statistics
        self.update_hist()  # initial draw

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


def annotation_napari(
    stain_dict,
    measurement_table_path: str,
    seg_name: str,
    seg_file: str,
    statistics_keyword: str = "median",
    is_otof: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    if s3:
        table_path_s3, fs = get_s3_path(measurement_table_path, s3_bucket_name, s3_service_endpoint, s3_credentials)
        with fs.open(table_path_s3, "r") as f:
            measurement_table = pd.read_csv(f, sep="\t")
    else:
        measurement_table = pd.read_csv(measurement_table, sep="\t")

    seg = imageio.imread(seg_file)
    statistics = get_object_measures_from_table(seg, table=measurement_table, keyword=statistics_keyword)

    seg_extended = _extend_seg_simple(seg, dilation=4)
    if is_otof:
        seg_extended = seg.copy()

    # Open the napari viewer.
    v = napari.Viewer()

    for num, (stain_name, file_path) in enumerate(stain_dict.items()):
        stain = imageio.imread(file_path)

        if num == 0:
            stain_shape = stain.shape
            main_stain_name = stain_name
            v.add_image(stain, name=stain_name)
        else:
            v.add_image(stain, visible=False, name=stain_name)

    # Add the base layers.
    v.add_labels(seg, visible=False, name=f"{seg_name}s")
    v.add_labels(seg_extended, name=f"{seg_name}s-extended")

    # Add additional layers for intensity coloring and classification
    # data_numerical = np.zeros(gfp.shape, dtype="float32")
    data_labels = np.zeros(stain_shape, dtype="uint8")
    # v.add_image(data_numerical, name="gfp-intensity")
    v.add_labels(data_labels, name="positive-negative")

    # Add widgets:

    # 1.) The widget for selcting the statistics to be used and displaying the histogram.
    stat_widget = _create_stat_widget(statistics, statistics_keyword)

    # 2.) Precompute statistic ranges.
    stat_names = stat_widget.stat_names
    all_values = statistics[stat_names].values
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

    # Bind the widgets.
    v.window.add_dock_widget(stat_widget, area="right")
    v.window.add_dock_widget(pick_widget, area="right")
    v.window.add_dock_widget(threshold_widget, area="right")
    stat_widget.setWindowTitle(f"{main_stain_name} Histogram")

    napari.run()
