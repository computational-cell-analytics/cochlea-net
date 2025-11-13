import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from util import literature_reference_values_gerbil, prism_cleanup_axes, prism_style, SYNAPSE_DIR_ROOT

FILE_EXTENSION = "png"
png_dpi = 300

COLOR_LEFT = "#8E00DB"
COLOR_RIGHT = "#DB0063"
MARKER_LEFT = "o"
MARKER_RIGHT = "^"
COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"
COLOR_UNTREATED = "#DB7B00"


# Load the synapse counts for all IHCs from the relevant tables.
def _load_ribbon_synapse_counts():
    ihc_version = "ihc_counts_v6"
    synapse_dir = os.path.join(SYNAPSE_DIR_ROOT, ihc_version)
    tables = [entry.path for entry in os.scandir(synapse_dir) if "ihc_count_G_" in entry.name]
    print(f"Synapse count for tables {tables}.")
    syn_counts = []
    for tab in tables:
        x = pd.read_csv(tab, sep="\t")
        syn_counts.extend(x["synapse_count"].values.tolist())
    return syn_counts


def fig_05c(save_path, plot=False):
    """Box plot showing the counts for SGN and IHC per gerbil cochlea in comparison to literature values.
    """
    main_tick_size = 20
    main_label_size = 20
    prism_style()

    rows = 1
    columns = 3

    fig, ax = plt.subplots(rows, columns, figsize=(8.5, 4.5))

    sgn_values = [18541]
    ihc_values = [1180]

    ax[0].scatter([1], sgn_values, color=COLOR_MEASUREMENT, marker="x", s=100)
    ax[1].scatter([1], ihc_values, color=COLOR_MEASUREMENT, marker="x", s=100)

    # Labels and formatting
    ax[0].set_xticks([1])
    ax[0].set_xticklabels(["SGN"], fontsize=main_label_size)

    ylim0 = 14000
    ylim1 = 30000
    ytick_gap = 4000
    y_ticks = [i for i in range((((ylim0 - 1) // ytick_gap) + 1) * ytick_gap, ylim1 + 1, ytick_gap)]

    ax[0].set_ylabel('Count per cochlea', fontsize=main_label_size)
    ax[0].set_yticks(y_ticks)
    ax[0].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[0].set_ylim(ylim0, ylim1)

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    ax[0].set_xlim(xmin, xmax)
    lower_y, upper_y = literature_reference_values_gerbil("SGN")
    ax[0].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[0].text(1, upper_y - 2000, "literature", color=COLOR_LITERATURE, fontsize=main_tick_size, ha="center")
    ax[0].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

    ylim0 = 900
    ylim1 = 1400
    ytick_gap = 200
    y_ticks = [i for i in range((((ylim0 - 1) // ytick_gap) + 1) * ytick_gap, ylim1 + 1, ytick_gap)]

    ax[1].set_xticks([1])
    ax[1].set_xticklabels(["IHC"], fontsize=main_label_size)

    ax[1].set_yticks(y_ticks)
    ax[1].set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
    ax[1].set_ylim(ylim0, ylim1)

    # set range of literature values
    xmin = 0.5
    xmax = 1.5
    ax[1].set_xlim(xmin, xmax)
    lower_y, upper_y = literature_reference_values_gerbil("IHC")
    ax[1].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[1].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

    ribbon_synapse_counts = _load_ribbon_synapse_counts()
    ylim0 = -1
    ylim1 = 80
    ytick_gap = 20
    y_ticks = [i for i in range((((ylim0 - 1) // ytick_gap) + 1) * ytick_gap, ylim1 + 1, ytick_gap)]

    box_plot = ax[2].boxplot(ribbon_synapse_counts, patch_artist=True)
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
    lower_y, upper_y = literature_reference_values_gerbil("synapse")
    ax[2].set_xlim(xmin, xmax)
    ax[2].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[2].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

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


def fig_05d(save_path, plot=False):
    """Box plot showing the SGN counts of ChReef treated cochleae compared to healthy ones.
    """
    prism_style()
    values_left = [11351]
    values_right = [21995]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16

    offset = 0.08
    x_left = 1
    x_right = 2

    x_pos_inj = [x_left - len(values_left) // 2 * offset + offset * i for i in range(len(values_left))]
    x_pos_non = [x_right - len(values_right) // 2 * offset + offset * i for i in range(len(values_right))]

    # lines between cochleae of same animal
    for num, (left, right) in enumerate(zip(values_left, values_right)):
        ax.plot(
            [x_pos_inj[num], x_pos_non[num]],
            [left, right],
            linestyle="solid",
            color="grey",
            alpha=0.4,
            zorder=0
        )
    plt.scatter(x_pos_inj, values_left, label="Injected",
                color=COLOR_LEFT, marker=MARKER_LEFT, s=80, zorder=1)
    plt.scatter(x_pos_non, values_right, label="Non-Injected",
                color=COLOR_RIGHT, marker=MARKER_RIGHT, s=80, zorder=1)

    # Labels and formatting
    plt.xticks([x_left, x_right], ["Injected", "Non-\nInjected"], fontsize=sub_label_size)
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment('center')
    ax.tick_params(axis='x', which='major', pad=16)

    plt.ylim(10000, 24000)
    y_ticks = [i for i in range(10000, 24000, 4000)]

    plt.yticks(y_ticks, fontsize=main_tick_size)
    plt.ylabel("SGN count per cochlea", fontsize=main_label_size)
    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    sgn_values = [18541]  # G_EK_000233_L
    sgn_value = np.mean(sgn_values)
    sgn_std = np.std(sgn_values)

    upper_y = sgn_value + 1.96 * sgn_std
    lower_y = sgn_value - 1.96 * sgn_std

    c_untreated = COLOR_UNTREATED

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=[c_untreated for _ in range(2)], zorder=-1)
    plt.text((xmin + xmax) / 2, upper_y + 200, "untreated cochleae\n(95% confidence interval)",
             color=c_untreated, fontsize=11, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color=c_untreated, alpha=0.05, interpolate=True)

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
    parser = argparse.ArgumentParser(description="Generate plots for Fig 5 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig5")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: The number of SGNs, IHCs and average number of ribbon synapses per IHC
    fig_05c(save_path=os.path.join(args.figure_dir, "fig_05c"), plot=args.plot)

    # Panel D: Tonotopic mapping of the intensities.
    fig_05d(save_path=os.path.join(args.figure_dir, f"fig_05d.{FILE_EXTENSION}"), plot=args.plot)


if __name__ == "__main__":
    main()
