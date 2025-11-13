import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from util import literature_reference_values, get_marker_handle, get_flatline_handle, SYNAPSE_DIR_ROOT
from util import prism_style, prism_cleanup_axes, export_legend, custom_formatter_2

png_dpi = 300
FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"

COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"


def plot_legend_suppfig02(save_path):
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


def supp_fig_02(save_path, plot=False, segm="SGN", mode="precision"):
    # SGN
    value_dict = {
        "SGN": {
            "stardist": {
                "label": "Stardist",
                "precision": 0.706,
                "recall": 0.630,
                "f1-score": 0.628,
                "marker": "o",
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
                "marker": "*",
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
    fig, ax = plt.subplots(figsize=(8, 5))

    main_label_size = 20
    main_tick_size = 16

    labels = [value_dict[segm][key]["label"] for key in value_dict[segm].keys()]

    if mode == "precision":
        # Convert setting labels to numerical x positions
        offset = 0.08  # horizontal shift for scatter separation
        for num, key in enumerate(list(value_dict[segm].keys())):
            precision = [value_dict[segm][key]["precision"]]
            recall = [value_dict[segm][key]["recall"]]
            f1score = [value_dict[segm][key]["f1-score"]]
            marker = value_dict[segm][key]["marker"]
            x_pos = num + 1

            plt.scatter([x_pos - offset], precision, label="Precision manual", color=COLOR_P, marker=marker, s=80)
            plt.scatter([x_pos],         recall, label="Recall manual", color=COLOR_R, marker=marker, s=80)
            plt.scatter([x_pos + offset], f1score, label="F1-score manual", color=COLOR_F, marker=marker, s=80)

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=main_tick_size)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        plt.ylim(-0.1, 1)
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif mode == "runtime":
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
            plt.scatter([x_pos], runtime, label="Runtime", color=COLOR_T, marker=marker, s=80)
            x_pos = x_pos + 1

        # Labels and formatting
        x_pos = np.arange(1, len(labels)+1)
        plt.xticks(x_pos, labels, fontsize=16)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Processing time [s]", fontsize=main_label_size)
        plt.ylim(10, 2400)
        plt.yscale('log')
        # plt.legend(loc="lower right", fontsize=legendsize)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

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


def plot_legend_fig02c(save_path, plot_mode="shapes"):
    """Plot common legend for figure 2c.

    Args:.
        save_path: save path to save legend.
        plot_mode: Plot either 'shapes' or 'colors' of data points.
    """
    if plot_mode == "shapes":
        # Shapes
        color = ["black", "black"]
        marker = ["o", "s"]
        label = ["Manual", "Automatic"]

        handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]
        legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
        export_legend(legend, save_path)
        legend.remove()
        plt.close()

    elif plot_mode == "colors":
        # Colors
        color = [COLOR_P, COLOR_R, COLOR_F]
        label = ["Precision", "Recall", "F1-score"]

        handles = [get_flatline_handle(c) for c in color]
        legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
        export_legend(legend, save_path)
        legend.remove()
        plt.close()

    else:
        raise ValueError("Choose either 'shapes' or 'colors' as plot_mode.")


def fig_02c(save_path, plot=False, all_versions=False):
    """Scatter plot showing the precision, recall, and F1-score of SGN (distance U-Net, manual),
    IHC (distance U-Net, manual), and synapse detection (U-Net).
    """
    prism_style()
    # precision, recall, f1-score
    sgn_unet = [0.887, 0.88, 0.884]
    sgn_annotator = [0.95, 0.849, 0.9]

    ihc_v4c = [0.905, 0.831, 0.866]

    ihc_annotator = [0.958, 0.956, 0.957]
    syn_unet = [0.931, 0.905, 0.918]

    setting = ["SGN", "IHC", "Synapse"]

    # This is the version with IHC v4c segmentation:
    # 4th version of the network with optimized segmentation params and split of falsely merged IHCs
    manual = [sgn_annotator, ihc_annotator]
    automatic = [sgn_unet, ihc_v4c, syn_unet]

    precision_manual = [i[0] for i in manual]
    recall_manual = [i[1] for i in manual]
    f1score_manual = [i[2] for i in manual]

    precision_automatic = [i[0] for i in automatic]
    recall_automatic = [i[1] for i in automatic]
    f1score_automatic = [i[2] for i in automatic]

    # Convert setting labels to numerical x positions
    x_manual = np.array([0.8, 1.8])
    x_automatic = np.array([1.2, 2.2, 3])
    offset = 0.08  # horizontal shift for scatter separation

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))

    main_label_size = 20
    main_tick_size = 16

    plt.scatter(x_manual - offset, precision_manual, label="Precision manual", color=COLOR_P, marker="o", s=80)
    plt.scatter(x_manual,         recall_manual, label="Recall manual", color=COLOR_R, marker="o", s=80)
    plt.scatter(x_manual + offset, f1score_manual, label="F1-score manual", color=COLOR_F, marker="o", s=80)

    plt.scatter(x_automatic - offset, precision_automatic, label="Precision automatic", color=COLOR_P, marker="s", s=80)
    plt.scatter(x_automatic,         recall_automatic, label="Recall automatic", color=COLOR_R, marker="s", s=80)
    plt.scatter(x_automatic + offset, f1score_automatic, label="F1-score automatic", color=COLOR_F, marker="s", s=80)

    # Labels and formatting
    plt.xticks([1, 2, 3], setting, fontsize=main_label_size)
    plt.yticks(fontsize=main_tick_size)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_2))
    plt.ylabel("Value", fontsize=main_label_size)
    plt.ylim(0.76, 1)
    # plt.legend(loc="lower right", fontsize=legendsize)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

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


# Load the synapse counts for all IHCs from the relevant tables.
def _load_ribbon_synapse_counts():
    ihc_version = "ihc_counts_v4c"
    synapse_dir = os.path.join(SYNAPSE_DIR_ROOT, ihc_version)
    tables = [entry.path for entry in os.scandir(synapse_dir) if "ihc_count_M_LR" in entry.name]
    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        syn_counts.extend(x["synapse_count"].values.tolist())
    return syn_counts


def fig_02d(save_path, plot=False, plot_average_ribbon_synapses=False):
    """Box plot showing the counts for SGN and IHC per (mouse) cochlea in comparison to literature values.
    """
    prism_style()
    main_tick_size = 16
    main_label_size = 20

    rows = 1
    columns = 3 if plot_average_ribbon_synapses else 2

    sgn_values = [11153, 11398, 10333, 11820]
    ihc_values = [712, 710, 721, 675]

    fig, axes = plt.subplots(rows, columns, figsize=(10, 4.5))
    ax = axes.flatten()
    box_plot = ax[0].boxplot(sgn_values, patch_artist=True, zorder=1)
    for median in box_plot['medians']:
        median.set_color(COLOR_MEASUREMENT)
    for boxcolor in box_plot['boxes']:
        boxcolor.set_facecolor("white")

    box_plot = ax[1].boxplot(ihc_values, patch_artist=True, zorder=1)
    for median in box_plot['medians']:
        median.set_color(COLOR_MEASUREMENT)
    for boxcolor in box_plot['boxes']:
        boxcolor.set_facecolor("white")

    # Labels and formatting
    ax[0].set_xticklabels(["SGN"], fontsize=main_label_size)

    ylim0 = 8500
    ylim1 = 12500
    y_ticks = [i for i in range(9000, 12000 + 1, 1000)]

    ax[0].set_ylabel("Count per cochlea", fontsize=main_label_size)
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[0].set_ylim(ylim0, ylim1)
    ax[0].yaxis.set_ticks_position("left")

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    ax[0].set_xlim(xmin, xmax)
    lower_y, upper_y = literature_reference_values("SGN")
    ax[0].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[0].text(1., lower_y + (upper_y - lower_y) * 0.2, "literature",
               color=COLOR_LITERATURE, fontsize=main_label_size, ha="center")
    ax[0].fill_between([xmin, xmax], lower_y, upper_y, color="C0", alpha=0.05, interpolate=True)

    ylim0 = 600
    ylim1 = 800
    y_ticks = [i for i in range(600, 800 + 1, 100)]

    ax[1].set_xticklabels(["IHC"], fontsize=main_label_size)
    ax[1].set_yticks(y_ticks)
    ax[1].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[1].set_ylim(ylim0, ylim1)
    if not plot_average_ribbon_synapses:
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_ticks_position("right")

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    lower_y, upper_y = literature_reference_values("IHC")
    ax[1].set_xlim(xmin, xmax)
    ax[1].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[1].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

    if plot_average_ribbon_synapses:
        ribbon_synapse_counts = _load_ribbon_synapse_counts()
        ylim0 = -1
        ylim1 = 41
        y_ticks = [0, 10, 20, 30, 40, 50]

        box_plot = ax[2].boxplot(ribbon_synapse_counts, patch_artist=True, zorder=1)
        for median in box_plot['medians']:
            median.set_color(COLOR_MEASUREMENT)
        for boxcolor in box_plot['boxes']:
            boxcolor.set_facecolor("white")

        ax[2].set_xticklabels(["Synapses per IHC"], fontsize=main_label_size)
        ax[2].set_yticks(y_ticks)
        ax[2].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
        ax[2].set_ylim(ylim0, ylim1)

        # set range of literature values
        xmin = 0.5
        xmax = 1.5
        lower_y, upper_y = literature_reference_values("synapse")
        ax[2].set_xlim(xmin, xmax)
        ax[2].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
        ax[2].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

    prism_cleanup_axes(axes)
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
    parser = argparse.ArgumentParser(description="Generate plots for Fig 2 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig2")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: Evaluation of the segmentation results:
    fig_02c(save_path=os.path.join(args.figure_dir, f"fig_02c.{FILE_EXTENSION}"), plot=args.plot, all_versions=False)
    plot_legend_fig02c(os.path.join(args.figure_dir, f"fig_02c_legend_shapes.{FILE_EXTENSION}"), plot_mode="shapes")
    plot_legend_fig02c(os.path.join(args.figure_dir, f"fig_02c_legend_colors.{FILE_EXTENSION}"), plot_mode="colors")

    # Panel D: The number of SGNs, IHCs and average number of ribbon synapses per IHC
    fig_02d(save_path=os.path.join(args.figure_dir, f"fig_02d.{FILE_EXTENSION}"),
            plot=args.plot, plot_average_ribbon_synapses=True)

    # Supplementary Figure 2: Comparing other methods in terms of segmentation accuracy and runtime
    plot_legend_suppfig02(save_path=os.path.join(args.figure_dir, f"figsupp_02_legend_colors.{FILE_EXTENSION}"))
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"figsupp_02_sgn.{FILE_EXTENSION}"), segm="SGN")
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"figsupp_02_ihc.{FILE_EXTENSION}"), segm="IHC")
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"figsupp_02_sgn_time.{FILE_EXTENSION}"),
                segm="SGN", mode="runtime")
    supp_fig_02(save_path=os.path.join(args.figure_dir, f"figsupp_02_ihc_time.{FILE_EXTENSION}"),
                segm="IHC", mode="runtime")


if __name__ == "__main__":
    main()
