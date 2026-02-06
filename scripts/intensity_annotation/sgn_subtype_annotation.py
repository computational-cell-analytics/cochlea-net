import argparse
import os

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd

from magicgui import magicgui

import flamingo_tools.intensity_annotation.annotation_utils as annotation_utils
from flamingo_tools.measurements import get_object_measures_from_table
from flamingo_tools.s3_utils import get_s3_path


def sgn_subtype_annotation(prefix, subtype, seg_version="SGN_v2", default_stat="median"):

    direc = os.path.dirname(os.path.abspath(prefix))
    basename = os.path.basename(prefix)
    file_names = [entry.name for entry in os.scandir(direc)]

    stain1_file = [name for name in file_names if basename in name and subtype in name][0]
    pv_files = [name for name in file_names if basename in name and "PV.tif" in name]
    if len(pv_files) == 0:
        ratio = False
        stain2_name = ""
        stain2 = None
        seg_version = "SGN_merged"
    else:
        ratio = True
        stain2_name = "PV"
        stain2_file = [name for name in file_names if basename in name and "PV.tif" in name][0]
        stain2 = imageio.imread(os.path.join(direc, stain2_file))

    seg_file = [name for name in file_names if basename in name and "SGN" in name][0]

    if "PV" in seg_file:
        seg_version = f"PV_{seg_version}"

    stain1_name = subtype
    seg_name = "SGN"

    stain1 = imageio.imread(os.path.join(direc, stain1_file))
    seg = imageio.imread(os.path.join(direc, seg_file))

    # bb = np.s_[128:-128, 128:-128, 128:-128]
    # gfp, sgns, pv = gfp[bb], sgns[bb], pv[bb]
    # print(gfp.shape)

    # Extend the sgns so that they cover the SGN boundaries.
    # sgns_extended = _extend_seg(gfp, sgns)
    # TODO we need to integrate this directly in the object measurement to efficiently do it at scale.
    seg_extended = annotation_utils._extend_seg_simple(seg, dilation=4)
    # Compute the intensity statistics.
    mask = None

    cochlea = os.path.basename(stain1_file).split("_crop_")[0]

    if ratio:
        table_measurement_path = f"{cochlea}/tables/{seg_version}/subtype_ratio.tsv"
        print(table_measurement_path)
        table_path_s3, fs = get_s3_path(table_measurement_path)
        with fs.open(table_path_s3, "r") as f:
            table_measurement = pd.read_csv(f, sep="\t")

        subtype_ratio = f"{subtype}_ratio_PV"
        statistics = get_object_measures_from_table(seg, table=table_measurement, keyword=subtype_ratio)
    else:
        seg_string = "-".join(seg_version.split("_"))
        table_measurement_path = f"{cochlea}/tables/{seg_version}/{stain1_name}_{seg_string}_object-measures.tsv"
        print(table_measurement_path)
        table_path_s3, fs = get_s3_path(table_measurement_path)
        with fs.open(table_path_s3, "r") as f:
            table_measurement = pd.read_csv(f, sep="\t")
        statistics = get_object_measures_from_table(seg, table=table_measurement)

    # Open the napari viewer.
    v = napari.Viewer()

    # Add the base layers.
    v.add_image(stain1, name=stain1_name)
    if stain2 is not None:
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
    stat_widget = annotation_utils._create_stat_widget(statistics, default_stat)

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


def main():
    parser = argparse.ArgumentParser(
        description="Start a GUI for determining an intensity threshold for positive "
        "/ negative transduction in segmented cells.")
    parser.add_argument("prefix", help="The prefix of the files to open with the annotation tool.")
    parser.add_argument("--subtype", type=str, default=None,
                        help="Supply SGN subtype, e.g. Calb1, Prph, Lypd1, ...")
    parser.add_argument("--seg_version", type=str, default="SGN_v2",
                        help="Supply segmentation version, e.g. SGN_v2, to use intensities from object measure table.")
    args = parser.parse_args()

    sgn_subtype_annotation(args.prefix, args.subtype, seg_version=args.seg_version)


if __name__ == "__main__":
    main()
