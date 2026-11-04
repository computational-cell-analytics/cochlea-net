"""Box plots of measured values against the reference interval from the literature."""
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

from util import prism_cleanup_axes, prism_style, save_figure

COLOR_MEASUREMENT = "#9C7427"
COLOR_LITERATURE = "#27339C"

# A box plot of fewer values only shows interpolated quartiles, so these are drawn as points.
MIN_BOXPLOT_N = 5
POINT_OFFSET = 0.06
N_LABEL_SIZE = 10


def _draw_values(ax, spec: dict, force_boxplot: bool, label_size: int):
    """Draw the values of one spec as a box plot or as points, and the literature band."""
    values = spec["values"]
    if len(values) < MIN_BOXPLOT_N and not force_boxplot:
        # Neighbors in y alternate between the two sides, so that close values do not overlap.
        offsets = [POINT_OFFSET * (-1) ** (rank + 1) for rank in range(len(values))]
        ax.scatter([1 + offset for offset in offsets], sorted(values), color="black", s=60, zorder=2)
        ax.hlines(np.mean(values), 0.8, 1.2, color=COLOR_MEASUREMENT, linewidth=2, zorder=1)
    else:
        box_plot = ax.boxplot(values, patch_artist=True, zorder=spec.get("box_zorder", 1))
        for median in box_plot["medians"]:
            median.set_color(COLOR_MEASUREMENT)
        for boxcolor in box_plot["boxes"]:
            boxcolor.set_facecolor("white")

    xmin, xmax = 0.5, 1.5
    ax.set_xlim(xmin, xmax)
    lower_y, upper_y = spec["literature"]
    ax.hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
    if spec.get("literature_label", False):
        ax.text(1, lower_y + (upper_y - lower_y) * 0.2, "literature",
                color=COLOR_LITERATURE, fontsize=label_size, ha="center", clip_on=True)
    ax.fill_between([xmin, xmax], lower_y, upper_y, color=spec.get("fill_color", COLOR_LITERATURE),
                    alpha=0.05, interpolate=True)


def _mark_break(top, bottom):
    """Draw the diagonal marks of a broken y-axis on the left spine."""
    style = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=12, linestyle="none", color="black",
                 mec="black", mew=1.2, clip_on=False)
    top.plot([0], [0], transform=top.transAxes, **style)
    bottom.plot([0], [1], transform=bottom.transAxes, **style)


def plot_counts_vs_literature(
    save_path: str,
    axes_specs: Sequence[dict],
    figsize: Tuple[float, float],
    tick_size: int,
    label_size: int = 20,
    ylabel: str = "Count per cochlea",
    plot: bool = False,
    force_boxplot: bool = False,
    ybreak: bool = True,
):
    """Plot one box plot per measured quantity, each with the literature interval as a band.

    A quantity with fewer than MIN_BOXPLOT_N values is drawn as points with a mean line instead.

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
            "box_zorder": zorder of the box plot (default 1, None for the matplotlib default),
            "n_label": sample size text above the upper right corner (default "n = <number of values>"),
            "ybreak": the (lower, upper) y range that a broken y-axis leaves out. Only one spec
            can have it.
        figsize: Figure size.
        tick_size: Font size of the y tick labels.
        label_size: Font size of the x tick labels, the y label and the band label.
        ylabel: Y-axis label of the first axis.
        plot: Plot figure.
        force_boxplot: Draw a box plot also for fewer than MIN_BOXPLOT_N values.
        ybreak: Break the y-axis of the spec with a "ybreak" key. Without it, that axis is continuous.
    """
    prism_style()
    fig = plt.figure(figsize=figsize)
    gaps = [spec.get("ybreak") if ybreak else None for spec in axes_specs]
    # The two parts of a broken axis get the heights of their y ranges, so both have the same scale.
    height_ratios = [1]
    for spec, gap in zip(axes_specs, gaps):
        if gap:
            height_ratios = [spec["ylim"][1] - gap[1], gap[0] - spec["ylim"][0]]
    grid = fig.add_gridspec(len(height_ratios), len(axes_specs), height_ratios=height_ratios)

    all_axes = []
    for num, (spec, gap) in enumerate(zip(axes_specs, gaps)):
        if gap:
            top, ax = fig.add_subplot(grid[0, num]), fig.add_subplot(grid[1, num])
            parts = [(top, (gap[1], spec["ylim"][1])), (ax, (spec["ylim"][0], gap[0]))]
            top.tick_params(axis="x", bottom=False, labelbottom=False)
            top.spines["bottom"].set_visible(False)
            _mark_break(top, ax)
        else:
            top = ax = fig.add_subplot(grid[:, num])
            parts = [(ax, spec["ylim"])]

        for part, (lower, upper) in parts:
            _draw_values(part, spec, force_boxplot, label_size)
            y_ticks: List[float] = [tick for tick in spec["yticks"] if lower <= tick <= upper]
            part.set_yticks(y_ticks)
            part.set_yticklabels(y_ticks, rotation=0, fontsize=tick_size)
            part.set_ylim(lower, upper)
            all_axes.append(part)

        ax.set_xticks([1])
        ax.set_xticklabels([spec["label"]], fontsize=label_size)
        # Above the axis, because outliers at the top of the axis would collide with the label.
        top.annotate(spec.get("n_label", f"n = {len(spec['values'])}"), xy=(1, 1), xycoords="axes fraction",
                     xytext=(0, 4), textcoords="offset points", ha="right", va="bottom", fontsize=N_LABEL_SIZE)
        if num == 0:
            ax.set_ylabel(ylabel, fontsize=label_size)
        hidden = sum(not any(lower <= value <= upper for _, (lower, upper) in parts) for value in spec["values"])
        if hidden:
            shown = " and ".join(str(y_range) for _, y_range in parts)
            print(f"Warning: {save_path}, {spec['label']}: {hidden} values lie outside the y range {shown}.")

    plt.tight_layout()
    prism_cleanup_axes(all_axes)
    save_figure(save_path, plot)
