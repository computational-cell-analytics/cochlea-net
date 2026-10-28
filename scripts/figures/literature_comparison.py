"""Box plots of measured values against the reference interval from the literature."""
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt

from util import prism_cleanup_axes, prism_style, save_figure

COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"


def plot_counts_vs_literature(
    save_path: str,
    axes_specs: Sequence[dict],
    figsize: Tuple[float, float],
    tick_size: int,
    label_size: int = 20,
    ylabel: str = "Count per cochlea",
    plot: bool = False,
):
    """Plot one box plot per measured quantity, each with the literature interval as a band.

    Args:
        save_path: File path to save the figure.
        axes_specs: One dict per axis, from left to right, with the keys
            "values": the measured values,
            "label": the x tick label,
            "literature": the (lower, upper) bounds of the literature interval,
            "ylim": the (lower, upper) y-axis limits,
            "yticks": the y tick positions,
            and the optional keys
            "literature_label": write "literature" into the band (default False),
            "fill_color": color of the band (default COLOR_LITERATURE),
            "box_zorder": zorder of the box plot (default 1, None for the matplotlib default).
        figsize: Figure size.
        tick_size: Font size of the y tick labels.
        label_size: Font size of the x tick labels, the y label and the band label.
        ylabel: Y-axis label of the first axis.
        plot: Plot figure.
    """
    prism_style()
    fig, axes = plt.subplots(1, len(axes_specs), figsize=figsize)

    xmin, xmax = 0.5, 1.5
    for num, (ax, spec) in enumerate(zip(axes, axes_specs)):
        box_plot = ax.boxplot(spec["values"], patch_artist=True, zorder=spec.get("box_zorder", 1))
        for median in box_plot["medians"]:
            median.set_color(COLOR_MEASUREMENT)
        for boxcolor in box_plot["boxes"]:
            boxcolor.set_facecolor("white")

        ax.set_xticks([1])
        ax.set_xticklabels([spec["label"]], fontsize=label_size)
        if num == 0:
            ax.set_ylabel(ylabel, fontsize=label_size)
        y_ticks: List[float] = spec["yticks"]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=tick_size)
        ax.set_ylim(*spec["ylim"])

        ax.set_xlim(xmin, xmax)
        lower_y, upper_y = spec["literature"]
        ax.hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
        if spec.get("literature_label", False):
            ax.text(1, lower_y + (upper_y - lower_y) * 0.2, "literature",
                    color=COLOR_LITERATURE, fontsize=label_size, ha="center")
        ax.fill_between([xmin, xmax], lower_y, upper_y, color=spec.get("fill_color", COLOR_LITERATURE),
                        alpha=0.05, interpolate=True)

    plt.tight_layout()
    prism_cleanup_axes(axes)
    save_figure(save_path, plot)
