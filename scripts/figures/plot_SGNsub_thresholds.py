import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.sgn_subtype_utils import CUSTOM_THRESHOLDS, COCHLEAE, ALIAS

png_dpi = 300


def plot_intensity_thresholds(input_dir, output_dir, cochlea, plot=False, sharex=True, custom_only=False,
                              title_type="generic"):
    """Plot histograms for positive and negative populations of subtype markers based on thresholding.
    """
    os.makedirs(output_dir, exist_ok=True)

    om_paths = [entry.path for entry in os.scandir(input_dir) if "_om.json" in entry.name]
    cochlea_str = "-".join(cochlea.split("_"))
    om_paths = [p for p in om_paths if cochlea_str in p]
    stains = [os.path.basename(p).split(f"{cochlea_str}_")[1].split("_om")[0] for p in om_paths]

    data_name = COCHLEAE[cochlea]["seg_data"]
    stain_str = " and ".join(stains)
    print(f"Evaluating {cochlea} with stains {stain_str}.")
    for stain, json_file in zip(stains, om_paths):
        save_path = os.path.join(output_dir, f"{cochlea}_{stain}_thresholds.png")

        try:
            with open(json_file, "r") as myfile:
                data = myfile.read()
        except PermissionError as e:
            print(e)
            continue
        param_dicts = json.loads(data)
        center_strs = param_dicts["center_strings"]
        center_strs.sort()
        intensity_mode = param_dicts["intensity_mode"]
        number_plots = len(center_strs)
        if number_plots == 0:
            print(f"No columns for cochlea {cochlea}.")
            continue

        seg_str = "-".join(data_name.split("_"))
        if intensity_mode == "ratio":
            table_measurement_path = f"{cochlea}/tables/{data_name}/subtype_ratio.tsv"
            column = f"{stain}_ratio_PV"
        elif intensity_mode == "absolute":
            table_measurement_path = f"{cochlea}/tables/{data_name}/{stain}_{seg_str}_object-measures.tsv"
            column = "median"

        # check for custom threshold
        if cochlea in CUSTOM_THRESHOLDS and stain in CUSTOM_THRESHOLDS[cochlea]:
            rows = 2
            row_types = ["threshold", "custom"]
            columns = number_plots
            if custom_only:
                rows = 2
                row_types = ["custom"]
                columns = number_plots // 2

        else:
            rows = 1
            row_types = ["threshold"]
            columns = number_plots

        main_label_size = 20
        main_tick_size = 16
        fig, axes = plt.subplots(rows, columns, figsize=(columns*2.5, rows*2.5), sharex=sharex)
        ax = axes.flatten()
        table_path_s3, fs = get_s3_path(table_measurement_path)
        with fs.open(table_path_s3, "r") as f:
            table_measurement = pd.read_csv(f, sep="\t")

        for enum, row_type in enumerate(row_types):
            if row_type == "threshold":
                for num, center_str in enumerate(center_strs):
                    seg_ids = param_dicts[center_str]["seg_ids"]
                    threshold = float(param_dicts[center_str]["median_intensity"][0])

                    subset = table_measurement[table_measurement["label_id"].isin(seg_ids)]
                    subset_neg = subset[(subset[column] < threshold)]
                    subset_pos = subset[(subset[column] > threshold)]
                    neg_values = list(subset_neg[column])
                    pos_values = list(subset_pos[column])

                    bins = np.linspace(min(neg_values + pos_values), max(neg_values + pos_values), 30)

                    ax[enum*columns + num].hist(pos_values, bins=bins, alpha=0.6, label='Positive', color='tab:blue')
                    ax[enum*columns + num].hist(neg_values, bins=bins, alpha=0.6, label='Negative', color='tab:orange')
                    ax[enum*columns + num].set_ylabel('Count', fontsize=main_label_size)
                    ax[enum*columns + num].set_xlabel('Intensity', fontsize=main_label_size)
                    ax[enum*columns + num].legend()
                    if title_type == "center_str":
                        ax[enum*columns + num].set_title(center_str)
                    else:
                        ax[enum*columns + num].set_title(f"Crop {str(num+1).zfill(1)}")
            else:
                threshold_dic = CUSTOM_THRESHOLDS[cochlea][stain]
                for num, center_str in enumerate(center_strs):
                    seg_ids = param_dicts[center_str]["seg_ids"]
                    if isinstance(threshold_dic, (int, float)):
                        threshold = threshold_dic
                    else:
                        threshold = threshold_dic[center_str]["manual"]

                    subset = table_measurement[table_measurement["label_id"].isin(seg_ids)]
                    subset_neg = subset[(subset[column] < threshold)]
                    subset_pos = subset[(subset[column] > threshold)]
                    neg_values = list(subset_neg[column])
                    pos_values = list(subset_pos[column])

                    bins = np.linspace(min(neg_values + pos_values), max(neg_values + pos_values), 30)

                    ax[enum*columns + num].hist(pos_values, bins=bins, alpha=0.6,
                                                label='Positive', color='tab:blue')
                    ax[enum*columns + num].hist(neg_values, bins=bins, alpha=0.6,
                                                label='Negative', color='tab:orange')
                    ax[enum*columns + num].set_ylabel('Count', fontsize=main_label_size)
                    ax[enum*columns + num].set_xlabel('Intensity', fontsize=main_label_size)
                    ax[enum*columns + num].tick_params(axis='x', labelsize=main_tick_size)
                    ax[enum*columns + num].tick_params(axis='y', labelsize=main_tick_size)
                    ax[enum*columns + num].legend()
                    if title_type == "center_str":
                        ax[enum*columns + num].set_title(center_str, fontsize=main_label_size)
                    else:
                        ax[enum*columns + num].set_title(f"Crop {str(num+1).zfill(1)}")

        alias = ALIAS[cochlea]
        fig.suptitle(f"{alias} - {stain}", fontsize=30)
        plt.tight_layout()

        if ".png" in save_path:
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
        else:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        if plot:
            plt.show()
        else:
            plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker based on annotation thresholds."
    )
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to process.")
    parser.add_argument("--figure_dir", "-f", type=str, required=True, help="Output directory for plots.")
    parser.add_argument("--sharex", action="store_true", help="Shared x-axis of subplots.")
    parser.add_argument("--custom", action="store_true", help="Plot only custom thresholds.")
    parser.add_argument("-i", "--input_dir", type=str,
                        default="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/Subtype_marker",  # noqa
                        help="Directory containing object measures of the cochleae.")

    args = parser.parse_args()
    for cochlea in args.cochlea:
        plot_intensity_thresholds(args.input_dir, args.figure_dir, cochlea, sharex=args.sharex, custom_only=args.custom)


if __name__ == "__main__":
    main()
