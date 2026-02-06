import argparse
import os
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

from flamingo_tools.measurements import compute_object_measures_impl, get_object_measures_from_table
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


def _create_mask(seg_extended, stain):
    from skimage.transform import downscale_local_mean, resize

    stain_averaged = downscale_local_mean(stain, factors=(16, 16, 16))
    # The 35th percentile seems to be a decent approximation for the background subtraction.
    threshold = np.percentile(stain_averaged, 35)
    mask = stain_averaged > threshold
    mask = resize(mask, seg_extended.shape, order=0, anti_aliasing=False, preserve_range=True).astype(bool)
    mask[seg_extended != 0] = 0

    # v = napari.Viewer()
    # v.add_image(stain)
    # v.add_image(stain_averaged, scale=(16, 16, 16))
    # v.add_labels(mask)
    # # v.add_labels(mask, scale=(16, 16, 16))
    # v.add_labels(seg_extended)
    # napari.run()

    return mask


def gfp_annotation(
    prefix: str,
    default_stat: str = "median",
    background_norm: Optional[str] = None,
    is_otof: bool = False,
    seg_version: Optional[str] = None,
):
    assert background_norm in (None, "division", "subtraction")

    direc = os.path.dirname(os.path.abspath(prefix))
    basename = os.path.basename(prefix)
    file_names = [entry.name for entry in os.scandir(direc)]
    if is_otof:  # OTOF cochlea with VGlut3, Alphatag and IHC segmentation.
        stain1_file = [name for name in file_names if basename in name and "pha" in name][0]
        stain2_file = [name for name in file_names if basename in name and "lut3" in name][0]
        seg_file = [name for name in file_names if basename in name and "IHC" in name][0]

        stain1_name = "Alphatag"
        stain2_name = "Vglut3"
        seg_name = "IHC"

    else:  # ChReef cochlea with PV, GFP and SGN segmentation
        stain1_file = [name for name in file_names if basename in name and "GFP" in name][0]
        stain2_file = [name for name in file_names if basename in name and "PV" in name][0]
        seg_file = [name for name in file_names if basename in name and "SGN" in name][0]

        stain1_name = "GFP"
        stain2_name = "PV"
        seg_name = "SGN"

    stain1 = imageio.imread(os.path.join(direc, stain1_file))
    stain2 = imageio.imread(os.path.join(direc, stain2_file))
    seg = imageio.imread(os.path.join(direc, seg_file))

    # Extend the sgns so that they cover the SGN boundaries.
    # sgns_extended = _extend_seg(gfp, sgns)
    # TODO we need to integrate this directly in the object measurement to efficiently do it at scale.
    seg_extended = _extend_seg_simple(seg, dilation=4)
    if is_otof:
        seg_extended = seg.copy()

    # Compute the intensity statistics.
    if background_norm is None:
        mask = None
        feature_set = "default"
    else:
        mask = _create_mask(seg_extended, stain1)
        assert mask.shape == seg_extended.shape
        feature_set = "default_background_norm" if background_norm == "division" else "default_background_subtract"

    if seg_version is not None:
        seg_string = "-".join(seg_version.split("_"))
        cochlea_str = os.path.basename(prefix).split("_crop_")[0]
        cochlea = "_".join(cochlea_str.split("-"))

        table_measurement_path = f"{cochlea}/tables/{seg_version}/{stain1_name}_{seg_string}_object-measures.tsv"
        table_path_s3, fs = get_s3_path(table_measurement_path)
        with fs.open(table_path_s3, "r") as f:
            table_measurement = pd.read_csv(f, sep="\t")

        statistics = get_object_measures_from_table(seg, table=table_measurement)

    else:
        statistics = compute_object_measures_impl(
            stain1, seg_extended, feature_set=feature_set, background_mask=mask, median_only=True
        )

    # Open the napari viewer.
    v = napari.Viewer()

    # Add the base layers.
    v.add_image(stain1, name=stain1_name)
    v.add_image(stain2, visible=False, name=stain2_name)
    v.add_labels(seg, visible=False, name=f"{seg_name}s")
    v.add_labels(seg_extended, name=f"{seg_name}s-extended")
    if mask is not None:
        v.add_labels(mask, name="mask-for-background", visible=False)

    # Add additional layers for intensity coloring and classification
    # data_numerical = np.zeros(gfp.shape, dtype="float32")
    data_labels = np.zeros(stain1.shape, dtype="uint8")

    # v.add_image(data_numerical, name="gfp-intensity")
    v.add_labels(data_labels, name="positive-negative")

    # Add widgets:

    # 1.) The widget for selcting the statistics to be used and displaying the histogram.
    stat_widget = _create_stat_widget(statistics, default_stat)

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
        data_labels = np.zeros(stain1.shape, dtype="uint8")
        data_labels[np.isin(seg_extended, pos_ids)] = 2
        data_labels[np.isin(seg_extended, neg_ids)] = 1
        viewer.layers["positive-negative"].data = data_labels

    threshold_widget.viewer.value = v

    # Bind the widgets.
    v.window.add_dock_widget(stat_widget, area="right")
    v.window.add_dock_widget(pick_widget, area="right")
    v.window.add_dock_widget(threshold_widget, area="right")
    stat_widget.setWindowTitle(f"{stain1_name} Histogram")

    napari.run()
