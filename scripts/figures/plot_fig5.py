import argparse
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_fig2 import _load_ribbon_synapse_counts
from plot_fig4 import group_lr
from plot_sgn_density_profile import FCHRIMSON_GERBIL, WT_GERBIL, get_sgn_length_data
from util import (
    VALUE_DICT,
    animal_colors,
    cochlea_label,
    cochleae_for,
    literature_reference_values,
    png_dpi,
    prism_cleanup_axes,
    prism_style,
)

FILE_EXTENSION = "png"

MARKER_LEFT = "o"
MARKER_RIGHT = "^"
COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"
COLOR_UNTREATED = "#DB7B00"

# Gerbil cohorts of figure 5. The cochlea lists are the canonical ones of plot_sgn_density_profile,
# the counts are documented in util.VALUE_DICT.
GERBIL_COHORT_DICT = {
    "wild_type": WT_GERBIL,
    "f-Chrimson": FCHRIMSON_GERBIL,
}

FCHRIMSON_COCHLEAE_DICT = cochleae_for(GERBIL_COHORT_DICT["f-Chrimson"], "SGN", "SGN_v2")


def _wt_counts(structure: str, source_name: str) -> list:
    """Documented wild type gerbil counts, read from util.VALUE_DICT."""
    counts = []
    for cochlea in GERBIL_COHORT_DICT["wild_type"]:
        if cochlea not in VALUE_DICT:
            raise ValueError(f"Cochlea {cochlea} is not in value dictionary.")
        counts.append(VALUE_DICT[cochlea][structure][source_name]["count"])
    return counts


def _count_ylim(values, lower, upper, tick_step: int = 4000, round_to: int = 2000):
    """Y-axis bounds and ticks that fit the data points, the reference band and the band label.

    Both the counts and the reference band follow the segmentation version, so a fixed range would
    have to be retuned whenever the tables or util.VALUE_DICT change.

    Args:
        values: Plotted counts. NaN entries, which group_lr uses for a missing cochlea, are ignored.
        lower: Lower bound of the reference band.
        upper: Upper bound of the reference band.
        tick_step: Distance between two y ticks.
        round_to: Granularity the axis bounds are rounded outwards to.

    Returns:
        (ylim0, ylim1, y_ticks). ylim1 keeps extra headroom for the band label above upper.
    """
    finite = [v for v in values if not np.isnan(v)]
    lower_value = min(finite + [lower])
    upper_value = max(finite + [upper])
    span = upper_value - lower_value
    ylim0 = math.floor((lower_value - 0.05 * span) / round_to) * round_to
    ylim1 = math.ceil((upper_value + 0.12 * span) / round_to) * round_to
    y_ticks = list(range(math.ceil(ylim0 / tick_step) * tick_step, ylim1 + 1, tick_step))
    return ylim0, ylim1, y_ticks


def fig_05c(
    save_path: str,
    plot: bool = False,
    ihc_version: str = "IHC_v11",
    sgn_version: str = "SGN_v2",
    synapse_dir: str = None,
):
    """Box plot showing the counts for SGN and IHC per gerbil cochlea in comparison to literature values.

    Args:
        save_path: File path to save the figure.
        plot: Plot figure.
        ihc_version: IHC segmentation source key in util.VALUE_DICT.
        sgn_version: SGN segmentation source key in util.VALUE_DICT.
        synapse_dir: Directory with the per-IHC synapse count tables. Defaults to the version
            directory below util.SYNAPSE_DIR_ROOT.
    """
    main_tick_size = 20
    main_label_size = 20
    prism_style()

    rows = 1
    columns = 3

    fig, ax = plt.subplots(rows, columns, figsize=(8.5, 4.5))

    sgn_values = _wt_counts("SGN", sgn_version)
    ihc_values = _wt_counts("IHC", ihc_version)

    box_plot = ax[0].boxplot(sgn_values, patch_artist=True, zorder=1)
    for median in box_plot["medians"]:
        median.set_color(COLOR_MEASUREMENT)
    for boxcolor in box_plot["boxes"]:
        boxcolor.set_facecolor("white")

    box_plot = ax[1].boxplot(ihc_values, patch_artist=True, zorder=1)
    for median in box_plot["medians"]:
        median.set_color(COLOR_MEASUREMENT)
    for boxcolor in box_plot["boxes"]:
        boxcolor.set_facecolor("white")

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
    lower_y, upper_y = literature_reference_values("SGN", animal="gerbil")
    ax[0].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[0].text(1, upper_y - 2000, "literature", color=COLOR_LITERATURE, fontsize=main_tick_size, ha="center")
    ax[0].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

    ylim0 = 900
    ylim1 = 1200
    ytick_gap = 100
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
    lower_y, upper_y = literature_reference_values("IHC", animal="gerbil")
    ax[1].hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    ax[1].fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

    ribbon_synapse_counts = _load_ribbon_synapse_counts(
        synapse_dir=synapse_dir, ihc_version="v11", cochleae=GERBIL_COHORT_DICT["wild_type"],
    )
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
    lower_y, upper_y = literature_reference_values("synapse", animal="gerbil")
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


def fig_05d(
    sgn_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    sgn_version: str = "SGN_v2",
):
    """Plot showing the SGN counts of f-Chrimson treated gerbil cochleae compared to healthy ones.

    Args:
        sgn_data: Per-cochlea SGN tables of the f-Chrimson gerbils, as returned by
            get_sgn_length_data. One row is one SGN, so the row count is the SGN count.
        save_path: File path to save the figure.
        plot: Plot figure.
        use_alias: Use alias.
        sgn_version: SGN segmentation source key of the wild type reference counts in
            util.VALUE_DICT.
    """
    prism_style()

    colors_by_animal = animal_colors(FCHRIMSON_COCHLEAE_DICT, use_alias)
    alias = [cochlea_label(name, FCHRIMSON_COCHLEAE_DICT[name], use_alias) for name in sgn_data]
    values = [len(table) for table in sgn_data.values()]
    alias, values_left, values_right = group_lr(alias, values)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    fontsize_untreated = 16

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
    for num, animal in enumerate(alias):
        ax.scatter(x_pos_inj[num], values_left[num], label=animal,
                   color=colors_by_animal[animal], marker=MARKER_LEFT, s=80, zorder=1)
        ax.scatter(x_pos_non[num], values_right[num],
                   color=colors_by_animal[animal], marker=MARKER_RIGHT, s=80, zorder=1)

    # Labels and formatting
    plt.xticks([x_left, x_right], ["Injected", "Non-\nInjected"], fontsize=sub_label_size)
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment('center')
    ax.tick_params(axis='x', which='major', pad=16)

    reference_values = _wt_counts("SGN", sgn_version)
    sgn_value = np.mean(reference_values)
    sgn_std = np.std(reference_values)

    upper_y = sgn_value + 1.96 * sgn_std
    lower_y = sgn_value - 1.96 * sgn_std

    ylim0, ylim1, y_ticks = _count_ylim(values_left + values_right, lower_y, upper_y)
    plt.ylim(ylim0, ylim1)
    plt.yticks(y_ticks, fontsize=main_tick_size)
    plt.ylabel("SGN count per cochlea", fontsize=main_label_size)
    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    c_untreated = COLOR_UNTREATED
    text_offset = (ylim1 - ylim0) / 40

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=[c_untreated for _ in range(2)], zorder=-1)
    plt.text((xmin + xmax) / 2, upper_y + text_offset, "untreated cochleae\n95% CI",
             color=c_untreated, fontsize=fontsize_untreated, ha="center")
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
    parser.add_argument("--no_alias", action="store_true")
    parser.add_argument(
        "--refresh_cache", action="store_true",
        help="Ignore the cached pickle and fetch the SGN tables from S3 again.",
    )
    parser.add_argument(
        "--synapse_dir", type=str, default=None,
        help="Directory with the per-IHC synapse count tables, e.g. ./ihc_counts_v11.",
    )
    args = parser.parse_args()

    use_alias = not args.no_alias
    os.makedirs(args.figure_dir, exist_ok=True)

    # Panel C: The number of SGNs, IHCs and average number of ribbon synapses per IHC
    fig_05c(save_path=os.path.join(args.figure_dir, f"fig_05c.{FILE_EXTENSION}"), plot=args.plot,
            synapse_dir=args.synapse_dir)

    # Panel D: The SGN count per f-Chrimson treated cochlea.
    sgn_data = get_sgn_length_data(GERBIL_COHORT_DICT["f-Chrimson"], force_download=args.refresh_cache)
    fig_05d(sgn_data, save_path=os.path.join(args.figure_dir, f"fig_05d.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias)


if __name__ == "__main__":
    main()
