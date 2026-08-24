import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import literature_reference_values, get_marker_handle, get_flatline_handle, SYNAPSE_DIR_ROOT, VALUE_DICT
from util import prism_style, prism_cleanup_axes, export_legend, custom_formatter

png_dpi = 300
FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"

COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"


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


# Scatter marker area. Kept small enough that a marker does not hide the error bar drawn behind
# it: the SGN and IHC seed standard deviations are ~0.005, only a few points tall on these axes.
MARKER_SIZE = 28

# Training-seed replicates behind the error bars of the automatic results. Each entry is
# (accuracy file, reference key of the released model, replicate keys). The replicates share the
# training data and the split with the reference and differ only in the training seed, so their
# spread is the run-to-run variability of the released model. All entries are evaluated at the
# production settings: SGN with component filtering, IHC with component filtering and without the
# zero-synapse filter, synapses with the assignment to the IHCs.
REPLICATE_KEYS = {
    "SGN": ("SGN_3D.json", "v2", ["v2-1", "v2-2", "v2-3", "v2-4"]),
    "IHC": ("IHC_3D.json", "v11", ["v11-1", "v11-2", "v11-3", "v11-4"]),
    "Synapse": ("synapses.json", "v3", ["v3-1", "v3-2", "v3-3", "v3-4"]),
}

# Accepted values of the 'error_bars' argument of fig_02c.
ERROR_BAR_MODES = ("none", "production", "mean")


def _read_replicate_stats(data_dir: str, file_name: str, keys: list) -> tuple:
    """Read the mean and standard deviation of precision, recall and F1 over several entries.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.
        keys: Top-level entries to aggregate over, e.g. the training-seed replicates.

    Returns:
        The per-metric means and the per-metric standard deviations, each in the order
        precision, recall, F1-score.
    """
    scores = np.array([_read_scores(data_dir, file_name, key) for key in keys])
    # np.std's default normalization, to match the seed panel of plot_supp_fig2.py.
    return scores.mean(axis=0), scores.std(axis=0)


def _read_scores(data_dir: str, file_name: str, key: str) -> list:
    """Read the precision, recall, and F1-score of one entry from an accuracy JSON file.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.
        key: Top-level entry of the file, e.g. a network version or an annotator scenario.

    Returns:
        The precision, recall, and F1-score of the entry.
    """
    json_file = os.path.join(data_dir, file_name)
    with open(json_file, "r") as f:
        param_dicts = json.load(f)
    if key not in param_dicts:
        raise KeyError(f"{json_file} has no entry '{key}'. Available entries: {sorted(param_dicts)}.")

    scores = param_dicts[key]
    return [scores["precision"], scores["recall"], scores["f1-score"]]


def fig_02c(
    save_path: str,
    data_dir: str,
    plot: bool = False,
    annotator_keyword: str = "all",
    error_bars: str = "none",
):
    """Scatter plot showing the precision, recall, and F1-score of SGN (distance U-Net, manual),
    IHC (distance U-Net, manual), and synapse detection (U-Net).

    Args:
        save_path: Path for saving the figure.
        data_dir: Directory containing the accuracy JSON files.
        plot: Whether to display the plot interactively.
        annotator_keyword: Entry to read from the annotator accuracy files.
        error_bars: How to show the training-seed variability of the automatic results.
            'none' plots the released model without an error bar, 'production' keeps the released
            model as the marker and adds the replicate standard deviation as the error bar, and
            'mean' plots the mean over the replicates instead. The manual results have no
            replicates and never get an error bar.
    """
    if error_bars not in ERROR_BAR_MODES:
        raise ValueError(f"Invalid error_bars '{error_bars}'. Expected one of {ERROR_BAR_MODES}.")
    prism_style()

    # SGN
    sgn_unet = _read_scores(data_dir, "SGN_3D.json", "v2")
    sgn_annotator = _read_scores(data_dir, "consensus_SGN.json", annotator_keyword)

    # IHC
    ihc_unet = _read_scores(data_dir, "IHC_3D.json", "v11")
    ihc_annotator = _read_scores(data_dir, "consensus_IHC.json", annotator_keyword)

    # synapses
    syn_unet = _read_scores(data_dir, "synapses.json", "v3")
    syn_annotator = _read_scores(data_dir, "consensus_synapses.json", annotator_keyword)

    setting = ["SGN", "IHC", "Synapse"]

    # This is the version with IHC v4c segmentation:
    # 4th version of the network with optimized segmentation params and split of falsely merged IHCs
    manual = [sgn_annotator, ihc_annotator, syn_annotator]
    automatic = [sgn_unet, ihc_unet, syn_unet]

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

    plt.scatter(x_manual - offset, precision_manual, label="Precision manual", color=COLOR_P,
                marker="o", s=MARKER_SIZE)
    plt.scatter(x_manual, recall_manual, label="Recall manual", color=COLOR_R,
                marker="o", s=MARKER_SIZE)
    plt.scatter(x_manual + offset, f1score_manual, label="F1-score manual", color=COLOR_F,
                marker="o", s=MARKER_SIZE)

    # The automatic values, optionally as mean +- standard deviation over the training-seed
    # replicates. 'production' keeps the released model on the marker and only takes the spread
    # from the replicates, so the plotted values stay the ones reported elsewhere.
    automatic_std = None
    if error_bars != "none":
        means, stds = [], []
        for structure in setting:
            file_name, _, replicate_keys = REPLICATE_KEYS[structure]
            mean, std = _read_replicate_stats(data_dir, file_name, replicate_keys)
            means.append(mean)
            stds.append(std)
        automatic_std = np.array(stds)
        if error_bars == "mean":
            means = np.array(means)
            precision_automatic = means[:, 0].tolist()
            recall_automatic = means[:, 1].tolist()
            f1score_automatic = means[:, 2].tolist()

    for values, color, shift, label, column in (
        (precision_automatic, COLOR_P, -offset, "Precision automatic", 0),
        (recall_automatic, COLOR_R, 0.0, "Recall automatic", 1),
        (f1score_automatic, COLOR_F, offset, "F1-score automatic", 2),
    ):
        if automatic_std is not None:
            plt.errorbar(x_automatic + shift, values, yerr=automatic_std[:, column],
                         fmt="none", color="black", capsize=4, zorder=1)
        plt.scatter(x_automatic + shift, values, label=label, color=color, marker="s",
                    s=MARKER_SIZE, zorder=2)

    # Labels and formatting
    plt.xticks([1, 2, 3], setting, fontsize=main_label_size)
    plt.yticks(fontsize=main_tick_size)
    ax.yaxis.set_major_formatter(custom_formatter(2))
    plt.ylabel("Value", fontsize=main_label_size)
    plt.ylim(0.69, 1)
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
    parser.add_argument("--pairwise", action="store_true",
                        help="Additionally plot panel C against the pairwise annotator agreement "
                             "instead of the consensus. Only the manual points differ.")
    _default_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reproducibility", "model_accuracy",
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default=_default_data_dir,
        help="Directory containing the model accuracy files (SGN_3D.json, IHC.json, IHC_3D.json, "
             "synapses.json) and the annotator accuracy files (consensus_SGN.json, consensus_IHC.json, "
             f"consensus_synapses.json). Defaults to {_default_data_dir}.",
    )
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: Evaluation of the segmentation results. Three variants of the automatic results:
    # without an error bar, and with the training-seed standard deviation around either the
    # released model or the mean over the replicates.
    variants = (("none", ""), ("production", "_sd_production"), ("mean", "_sd_mean"))
    for error_bars, suffix in variants:
        fig_02c(save_path=os.path.join(args.figure_dir, f"fig_02c{suffix}.{FILE_EXTENSION}"),
                data_dir=args.data_dir, plot=args.plot, error_bars=error_bars)

    # The same panel against the pairwise annotator agreement instead of the consensus. Only the
    # manual points change, so this is a reference plot and is off by default.
    if args.pairwise:
        annotator_keyword = "pairwise"
        for error_bars, suffix in variants:
            fig_02c(save_path=os.path.join(
                        args.figure_dir, f"fig_02c_{annotator_keyword}{suffix}.{FILE_EXTENSION}"),
                    data_dir=args.data_dir, plot=args.plot,
                    annotator_keyword=annotator_keyword, error_bars=error_bars)

    plot_legend_fig02c(os.path.join(args.figure_dir, f"fig_02c_legend_shapes.{FILE_EXTENSION}"), plot_mode="shapes")
    plot_legend_fig02c(os.path.join(args.figure_dir, f"fig_02c_legend_colors.{FILE_EXTENSION}"), plot_mode="colors")

    # Panel D: The number of SGNs, IHCs and average number of ribbon synapses per IHC
    fig_02d(save_path=os.path.join(args.figure_dir, f"fig_02d.{FILE_EXTENSION}"),
            plot=args.plot, plot_average_ribbon_synapses=True)


if __name__ == "__main__":
    main()
