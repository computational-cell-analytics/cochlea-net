import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from util import get_flatline_handle
from util import prism_cleanup_axes, export_legend

png_dpi = 300
FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"


def plot_legend_supp_fig02(save_path):
    """Plot common legend for figure 2c.

    Args:
        save_path: save path to save legend.
    """
    # Colors
    color = [COLOR_P, COLOR_R, COLOR_F, COLOR_T]
    label = ["Precision", "Recall", "F1-score", "Processing time"]

    handles = [get_flatline_handle(c) for c in color]
    legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def supp_fig_02(
    save_path: str,
    plot: bool = False,
    segm: str = "SGN",
    mode: str = "precision",
):
    """Plot panels for Supplementary Figure 2.

    Args:
        save_path: Path for saving figure.
        plot:
        segm: Segmentation type. Either "SGN" or "IHC".
        mode: Mode for plotting. Either "precision" or "runtime".
    """
    # SGN
    value_dict = {
        "SGN": {
            "stardist": {
                "label": "Stardist",
                "precision": 0.706,
                "recall": 0.630,
                "f1-score": 0.628,
                "marker": "*",
                "runtime": 536.5,
                "runtime_std": 148.4
            },
            "micro_sam": {
                "label": "µSAM",
                "precision": 0.140,
                "recall": 0.782,
                "f1-score": 0.228,
                "marker": "D",
                "runtime": 407.5,
                "runtime_std": 107.5
            },
            "cellpose_3": {
                "label": "Cellpose 3",
                "precision": 0.117,
                "recall": 0.607,
                "f1-score": 0.186,
                "marker": "v",
                "runtime": 167.9116359,
                "runtime_std": 40.2,
            },
            "cellpose_sam": {
                "label": "Cellpose-SAM",
                "precision": 0.250,
                "recall": 0.003,
                "f1-score": 0.005,
                "marker": "^",
                "runtime": 2232.007748,
                "runtime_std": None,
            },
            "spiner2D": {
                "label": "Spiner",
                "precision": 0.373,
                "recall": 0.340,
                "f1-score": 0.326,
                "marker": "o",
                "runtime": None,
                "runtime_std": None,
            },
            "distance_unet": {
                "label": "CochleaNet",
                "precision": 0.886,
                "recall": 0.804,
                "f1-score": 0.837,
                "marker": "s",
                "runtime": 168.8,
                "runtime_std": 21.8
            },
        },
        "IHC": {
            "micro_sam": {
                "label": "µSAM",
                "precision": 0.053,
                "recall": 0.684,
                "f1-score": 0.094,
                "marker": "D",
                "runtime": 445.6,
                "runtime_std": 106.6
            },
            "cellpose_3": {
                "label": "Cellpose 3",
                "precision": 0.375,
                "recall": 0.554,
                "f1-score": 0.329,
                "marker": "v",
                "runtime": 162.3493934,
                "runtime_std": 30.1,
            },
            "cellpose_sam": {
                "label": "Cellpose-SAM",
                "precision": 0.636,
                "recall": 0.025,
                "f1-score": 0.047,
                "marker": "^",
                "runtime": 2137.944779,
                "runtime_std": None
            },
            "distance_unet": {
                "label": "CochleaNet",
                "precision": 0.693,
                "recall": 	0.567,
                "f1-score": 0.618,
                "marker": "s",
                "runtime": 69.01,
                "runtime_std": None
            },
        }
    }

    # Convert setting labels to numerical x positions
    offset = 0.08  # horizontal shift for scatter separation

    # Plot
    tick_rotation = 0

    main_label_size = 20
    main_tick_size = 16
    marker_size = 200

    labels = [value_dict[segm][key]["label"] for key in value_dict[segm].keys()]

    if mode == "precision":
        fig, ax = plt.subplots(figsize=(10, 5))
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for num, key in enumerate(list(value_dict[segm].keys())):
            precision = [value_dict[segm][key]["precision"]]
            recall = [value_dict[segm][key]["recall"]]
            f1score = [value_dict[segm][key]["f1-score"]]
            marker = value_dict[segm][key]["marker"]
            x_pos = num + 1

            plt.scatter([x_pos - offset], precision, label="Precision manual",
                        color=COLOR_P, marker=marker, s=marker_size)
            plt.scatter([x_pos],         recall, label="Recall manual",
                        color=COLOR_R, marker=marker, s=marker_size)
            plt.scatter([x_pos + offset], f1score, label="F1-score manual",
                        color=COLOR_F, marker=marker, s=marker_size)

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=main_tick_size, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        plt.ylim(-0.1, 1)
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif mode == "runtime":
        fig, ax = plt.subplots(figsize=(8.5, 5))
        if "Spiner" in labels:
            labels.remove("Spiner")

        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        x_pos = 1
        for num, key in enumerate(list(value_dict[segm].keys())):
            runtime = [value_dict[segm][key]["runtime"]]
            if runtime[0] is None:
                continue
            marker = value_dict[segm][key]["marker"]
            plt.scatter([x_pos], runtime, label="Runtime", color=COLOR_T, marker=marker, s=marker_size)
            x_pos = x_pos + 1

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=16, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Processing time [s]", fontsize=main_label_size)
        plt.ylim(10, 2600)
        plt.yscale('log')
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    else:
        raise ValueError("Unsupported mode for plotting.")

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Supplementary Fig 2 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.",
                        default="./panels/supp_fig2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Supplementary Figure 2: Comparing other methods in terms of segmentation accuracy and runtime
    plot_legend_supp_fig02(save_path=os.path.join(args.figure_dir, f"supp_fig_02_legend_colors.{FILE_EXTENSION}"))
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02a_sgn_accuracy.{FILE_EXTENSION}"), segm="SGN")
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02b_ihc_accuracy.{FILE_EXTENSION}"), segm="IHC")
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02a_sgn_time.{FILE_EXTENSION}"),
                segm="SGN", mode="runtime")
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02b_ihc_time.{FILE_EXTENSION}"),
                segm="IHC", mode="runtime")


if __name__ == "__main__":
    main()
