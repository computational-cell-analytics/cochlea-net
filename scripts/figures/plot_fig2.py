import argparse
import json
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import literature_reference_values, get_marker_handle, get_flatline_handle, SYNAPSE_DIR_ROOT, VALUE_DICT
from util import iteration_statistics, prism_style, prism_cleanup_axes, export_legend, custom_formatter

png_dpi = 300
FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"

COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"

# Per structure of panel 2c: the x-axis label, the annotator and network accuracy files, and the
# network iteration entries. Key order determines the left-to-right order on the x-axis. The first
# iteration entry is the base version (variant 0); the others are repeat trainings of the same
# network on the same training and validation data.
PANEL_02C_DICT = {
    "SGN_v2": {
        "label": "SGN",
        "accuracy_file": "SGN_3D.json",
        "consensus_file": "consensus_SGN.json",
        "iterations": ["v2", "v2-1", "v2-2", "v2-3", "v2-4"],
    },
    "IHC_v11": {
        "label": "IHC",
        "accuracy_file": "IHC_3D.json",
        "consensus_file": "consensus_IHC.json",
        "iterations": ["v11", "v11-1", "v11-2", "v11-3", "v11-4"],
    },
    "synapses_v3": {
        "label": "Synapse",
        "accuracy_file": "synapses.json",
        "consensus_file": "consensus_synapses.json",
        "iterations": ["v3", "v3-1", "v3-2", "v3-3", "v3-4"],
    },
}

METRICS = ("precision", "recall", "f1-score")


def plot_legend_fig02c(
    save_path: str,
    plot_mode: str = "shapes",
):
    """Plot common legend for Figure 2c.

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


def _read_metrics(data_dir: str, file_name: str) -> dict:
    """Read one accuracy JSON file.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.

    Returns:
        The accuracy entries of the file, keyed by network version or annotator scenario.
    """
    with open(os.path.join(data_dir, file_name), "r") as f:
        return json.load(f)


def _select_scores(metrics: dict, key: str, source: str) -> list:
    """Select the precision, recall, and F1-score of one entry of an accuracy file.

    Args:
        metrics: Accuracy entries of one accuracy JSON file.
        key: Entry to select, e.g. a network version or an annotator scenario.
        source: Name of the accuracy file, for the error message.

    Returns:
        The precision, recall, and F1-score of the entry.
    """
    if key not in metrics:
        raise KeyError(f"{source} has no entry '{key}'. Available entries: {sorted(metrics)}.")
    return [metrics[key][metric] for metric in METRICS]


def _read_scores(data_dir: str, file_name: str, key: str) -> list:
    """Read the precision, recall, and F1-score of one entry from an accuracy JSON file.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.
        key: Top-level entry of the file, e.g. a network version or an annotator scenario.

    Returns:
        The precision, recall, and F1-score of the entry.
    """
    return _select_scores(_read_metrics(data_dir, file_name), key, os.path.join(data_dir, file_name))


def _read_network_scores(data_dir: str, entry: dict, show_variation: bool) -> tuple:
    """Read the network scores of one structure of panel 2c.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        entry: Entry of PANEL_02C_DICT for the structure.
        show_variation: Average over the network iterations instead of reading the base version.

    Returns:
        The precision, recall, and F1-score, and their standard deviation over the network
        iterations. The standard deviation is None if fewer than two iterations are available.
    """
    file_name = entry["accuracy_file"]
    iterations = entry["iterations"]
    metrics = _read_metrics(data_dir, file_name)

    if not show_variation:
        return _select_scores(metrics, iterations[0], os.path.join(data_dir, file_name)), None

    stats, present = iteration_statistics(metrics, iterations, metric_names=METRICS)
    # A metric is absent from stats if every available iteration stores None for it.
    if len(present) < 2 or any(metric not in stats for metric in METRICS):
        print(f"{file_name}: only {present} of the iterations {iterations} are available. "
              f"Plotting '{iterations[0]}' without variation.")
        return _select_scores(metrics, iterations[0], os.path.join(data_dir, file_name)), None

    missing = [key for key in iterations if key not in present]
    if missing:
        print(f"Warning: {file_name}: {', '.join(missing)} not found. "
              f"Averaging over {len(present)} of {len(iterations)} iterations.")
    for metric in METRICS:
        mean, std = stats[metric]
        print(f"{file_name} {metric}: {mean:.3f} +- {std:.3f} (n={len(present)})")

    return [stats[metric][0] for metric in METRICS], [stats[metric][1] for metric in METRICS]


def fig_02c(
    save_path: str,
    data_dir: str,
    plot: bool = False,
    annotator_keyword: str = "all",
    show_variation: bool = False,
    ylim: Optional[List[float]] = None,
):
    """Scatter plot showing the precision, recall, and F1-score of SGN (distance U-Net, manual),
    IHC (distance U-Net, manual), and synapse detection (U-Net).

    Args:
        save_path: Path for saving the figure.
        data_dir: Directory containing the accuracy JSON files.
        plot: Whether to display the plot interactively.
        annotator_keyword: Entry of the annotator accuracy files. Either 'all' or 'pairwise'.
        show_variation: Plot the automatic value as the mean over the network iterations of
            PANEL_02C_DICT, with the standard deviation as error bar. A structure with fewer than
            two iterations in its accuracy file keeps the value of its base version.
        ylim: Lower and upper y-axis limit.
    """
    prism_style()

    setting = [entry["label"] for entry in PANEL_02C_DICT.values()]

    manual = [
        _read_scores(data_dir, entry["consensus_file"], annotator_keyword)
        for entry in PANEL_02C_DICT.values()
    ]
    network_scores = [
        _read_network_scores(data_dir, entry, show_variation) for entry in PANEL_02C_DICT.values()
    ]
    automatic = [scores for scores, _ in network_scores]
    automatic_std = [std for _, std in network_scores]

    precision_manual = [i[0] for i in manual]
    recall_manual = [i[1] for i in manual]
    f1score_manual = [i[2] for i in manual]

    precision_automatic = [i[0] for i in automatic]
    recall_automatic = [i[1] for i in automatic]
    f1score_automatic = [i[2] for i in automatic]

    # Convert setting labels to numerical x positions
    x_manual = np.array([0.8, 1.8, 2.8])
    x_automatic = np.array([1.2, 2.2, 3.2])
    offset = 0.08  # horizontal shift for scatter separation

    # Plot
    fig, ax = plt.subplots(figsize=(8, 4.5))

    main_label_size = 20
    main_tick_size = 16
    capsize = 4

    for x_pos, scores, stds in zip(x_automatic, automatic, automatic_std):
        if stds is None:
            continue
        for score, std, shift in zip(scores, stds, (-offset, 0, offset)):
            plt.errorbar([x_pos + shift], [score], yerr=[std], fmt="none", color="black",
                         capsize=capsize, zorder=1)

    plt.scatter(x_manual - offset, precision_manual, label="Precision manual", color=COLOR_P, marker="o", s=80)
    plt.scatter(x_manual, recall_manual, label="Recall manual", color=COLOR_R, marker="o", s=80)
    plt.scatter(x_manual + offset, f1score_manual, label="F1-score manual", color=COLOR_F, marker="o", s=80)

    plt.scatter(x_automatic - offset, precision_automatic, label="Precision automatic", color=COLOR_P, marker="s", s=80)
    plt.scatter(x_automatic, recall_automatic, label="Recall automatic", color=COLOR_R, marker="s", s=80)
    plt.scatter(x_automatic + offset, f1score_automatic, label="F1-score automatic", color=COLOR_F, marker="s", s=80)

    # Labels and formatting
    plt.xticks([1, 2, 3], setting, fontsize=main_label_size)
    plt.yticks(fontsize=main_tick_size)
    ax.yaxis.set_major_formatter(custom_formatter(2))
    plt.ylabel("Value", fontsize=main_label_size)
    if ylim is None:
        plt.ylim(0.69, 1)
    else:
        plt.ylim(ylim[0], ylim[1])
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
def _load_ribbon_synapse_counts(
    synapse_dir: str = None,
    ihc_version: str = "v11",
    cochleae: list[str] = [
        "M_LR_000226_L",
        "M_LR_000226_R",
        "M_LR_000227_L",
        "M_LR_000227_R",
    ],
) -> list:
    if synapse_dir is None:
        measure_synapse_dir = f"ihc_counts_{ihc_version}"
        synapse_dir = os.path.join(SYNAPSE_DIR_ROOT, measure_synapse_dir)
    tables = [entry.path for entry in os.scandir(synapse_dir) if any(c in entry.name for c in cochleae)]
    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        syn_counts.extend(x["synapse_count"].values.tolist())
    return syn_counts


def fig_02d(
    save_path: str,
    plot: bool = False,
    plot_average_ribbon_synapses: bool = False,
    ihc_version: str = "IHC_v11",
    sgn_version: str = "SGN_v2",
):
    """Box plot showing the counts for SGN and IHC per (mouse) cochlea in comparison to literature values.
    """
    prism_style()
    main_tick_size = 16
    main_label_size = 20

    rows = 1
    columns = 3 if plot_average_ribbon_synapses else 2

    cochleae = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]
    sgn_values = []
    ihc_values = []
    for c in cochleae:
        if c in list(VALUE_DICT.keys()):
            ihc_values.append(VALUE_DICT[c]["IHC"][ihc_version]["count"])
            sgn_values.append(VALUE_DICT[c]["SGN"][sgn_version]["count"])
        else:
            raise ValueError(f"Cochlea {c} is not in value dictionary.")

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
    parser = argparse.ArgumentParser(description="Generate plots for Figure 2 of the CochleaNet paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig2")
    parser.add_argument("--plot", action="store_true")
    _default_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reproducibility", "model_accuracy",
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default=_default_data_dir,
        help="Directory containing the model accuracy files (SGN_3D.json, IHC_3D.json, "
             "synapses.json) and the annotator accuracy files (consensus_SGN.json, consensus_IHC.json, "
             f"consensus_synapses.json). Defaults to {_default_data_dir}.",
    )
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: Evaluation of the segmentation results:
    fig_02c(save_path=os.path.join(args.figure_dir, f"fig_02c.{FILE_EXTENSION}"),
            data_dir=args.data_dir, plot=args.plot)

    annotator_keyword = "pairwise"
    fig_02c(save_path=os.path.join(args.figure_dir, f"fig_02c_{annotator_keyword}.{FILE_EXTENSION}"),
            data_dir=args.data_dir, plot=args.plot,
            annotator_keyword=annotator_keyword,)

    # The same panels, with the automatic value averaged over the network training iterations.
    fig_02c(save_path=os.path.join(args.figure_dir, f"fig_02c_variation.{FILE_EXTENSION}"),
            data_dir=args.data_dir, plot=args.plot, show_variation=True)

    fig_02c(save_path=os.path.join(args.figure_dir, f"fig_02c_{annotator_keyword}_variation.{FILE_EXTENSION}"),
            data_dir=args.data_dir, plot=args.plot,
            annotator_keyword=annotator_keyword, show_variation=True)

    plot_legend_fig02c(os.path.join(args.figure_dir, f"fig_02c_legend_shapes.{FILE_EXTENSION}"), plot_mode="shapes")
    plot_legend_fig02c(os.path.join(args.figure_dir, f"fig_02c_legend_colors.{FILE_EXTENSION}"), plot_mode="colors")

    # Panel D: The number of SGNs, IHCs and average number of ribbon synapses per IHC
    fig_02d(save_path=os.path.join(args.figure_dir, f"fig_02d.{FILE_EXTENSION}"),
            plot=args.plot, plot_average_ribbon_synapses=True)


if __name__ == "__main__":
    main()
