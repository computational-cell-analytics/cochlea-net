import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path

png_dpi = 300

COCHLEAE = {
    "M_LR_000098_L": {"seg_data": "SGN_v2", "subtype": ["CR", "Ntng1"], "intensity": "ratio"},
    "M_LR_000099_L": {"seg_data": "PV_SGN_v2", "subtype": ["Calb1", "Lypd1"], "intensity": "ratio"},
    "M_LR_000184_L": {"seg_data": "SGN_v2", "subtype": ["Prph"], "output_seg": "SGN_v2b", "intensity": "ratio"},
    "M_LR_000184_R": {"seg_data": "SGN_v2", "subtype": ["Prph"], "output_seg": "SGN_v2b", "intensity": "ratio"},
    "M_LR_000214_L": {"seg_data": "PV_SGN_v2", "subtype": ["Calb1"], "intensity": "ratio"},
    "M_LR_000260_L": {"seg_data": "SGN_v2", "subtype": ["Prph", "Tuj1"], "intensity": "ratio"},
    "M_LR_N110_L": {"seg_data": "SGN_v2", "subtype": ["Calb1", "Ntng1"], "intensity": "ratio"},
    "M_LR_N110_R": {"seg_data": "SGN_v2", "subtype": ["Calb1", "Ntng1"], "intensity": "ratio"},
    "M_LR_N152_L": {"seg_data": "SGN_v2", "subtype": ["CR", "Ntng1"], "intensity": "ratio"},
    "M_AMD_N180_L": {"seg_data": "SGN_merged", "subtype": ["CR", "Lypd1", "Ntng1"], "intensity": "absolute"},
    "M_AMD_N180_R": {"seg_data": "SGN_merged", "subtype": ["CR", "Ntng1"], "intensity": "absolute"},
}

CUSTOM_THRESHOLDS = {
    "M_LR_000099_L": {"Lypd1": 0.65},
    "M_LR_000184_L": {"Prph": 1},
    "M_LR_000260_L": {"Prph": 0.7},
}

def plot_intensity_thresholds(input_dir, output_dir, cochlea, plot=False):
    """Plot histograms for positive and negative populations of subtype markers based on thresholding.
    """
    om_paths = [entry.path for entry in os.scandir(input_dir) if "_om.json" in entry.name]
    cochlea_str = "-".join(cochlea.split("_"))
    om_paths = [p for p in om_paths if cochlea_str in p]
    stains = [os.path.basename(p).split(f"{cochlea_str}_")[1].split("_om")[0] for p in om_paths]

    data_name = COCHLEAE[cochlea]["seg_data"]
    print(f"Evaluating {cochlea} with stains {" and ".join(stains)}.")
    for stain, json_file in zip(stains, om_paths):
        save_path = os.path.join(output_dir, f"{cochlea}_{stain}_thresholds.png")

        with open(json_file, 'r') as myfile:
            data = myfile.read()
        param_dicts = json.loads(data)
        center_strs = param_dicts["center_strings"]
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
        if CUSTOM_THRESHOLDS.get(cochlea, {}).get(stain) is not None:
            rows = 2
        else:
            rows = 1

        columns = number_plots
        fig, axes = plt.subplots(rows, columns, figsize=(columns*4, rows*4), sharex=True)
        ax = axes.flatten()
        table_path_s3, fs = get_s3_path(table_measurement_path)
        with fs.open(table_path_s3, "r") as f:
            table_measurement = pd.read_csv(f, sep="\t")

        for num, center_str in enumerate(center_strs):
            seg_ids = param_dicts[center_str]["seg_ids"]
            threshold = float(param_dicts[center_str]["median_intensity"][0])

            subset = table_measurement[table_measurement["label_id"].isin(seg_ids)]
            subset_neg = subset[(subset[column] < threshold)]
            subset_pos = subset[(subset[column] > threshold)]
            neg_values = list(subset_neg[column])
            pos_values = list(subset_pos[column])

            bins = np.linspace(min(neg_values + pos_values),
                               max(neg_values + pos_values), 30)

            ax[num].hist(pos_values, bins=bins, alpha=0.6, label='Positive', color='tab:blue')
            ax[num].hist(neg_values, bins=bins, alpha=0.6, label='Negative', color='tab:orange')
            ax[num].set_ylabel('Count')
            ax[num].set_xlabel('Intensity')
            ax[num].legend()
            ax[num].set_title(center_str)

        if rows  == 2:
            for num, center_str in enumerate(center_strs):
                seg_ids = param_dicts[center_str]["seg_ids"]
                threshold = CUSTOM_THRESHOLDS[cochlea][stain]

                subset = table_measurement[table_measurement["label_id"].isin(seg_ids)]
                subset_neg = subset[(subset[column] < threshold)]
                subset_pos = subset[(subset[column] > threshold)]
                neg_values = list(subset_neg[column])
                pos_values = list(subset_pos[column])

                bins = np.linspace(min(neg_values + pos_values),
                                   max(neg_values + pos_values), 30)

                ax[columns + num].hist(pos_values, bins=bins, alpha=0.6, label='Positive (custom)', color='tab:blue')
                ax[columns + num].hist(neg_values, bins=bins, alpha=0.6, label='Negative (custom)', color='tab:orange')
                ax[columns + num].set_ylabel('Count')
                ax[columns + num].set_xlabel('Intensity')
                ax[columns + num].legend()
                ax[columns + num].set_title(center_str)

        fig.suptitle(f"{cochlea} - {stain}", fontsize=30)

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
    parser.add_argument("-i", "--input_dir", type=str,
                        default="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/Subtype_marker",
                        help="Directory containing object measures of the cochleae.") #noqa

    args = parser.parse_args()
    for cochlea in args.cochlea:
        plot_intensity_thresholds(args.input_dir, args.figure_dir, cochlea)


if __name__ == "__main__":
    main()
