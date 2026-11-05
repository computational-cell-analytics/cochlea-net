"""Gene therapy plots: injected against non-injected cochleae of one animal.

The left cochlea of an animal is injected, the right one is the non-injected control.
"""
import math
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import LAVISION_COCHLEAE_DICT
from util import (
    COCHLEA_DICT, COLOR_LEFT, COLOR_RIGHT, COLOR_UNTREATED, animal_colors, band_table, band_trendline,
    cochlea_colors, cochlea_label, cochleae_for, cohort_cochleae, cohort_postnatal, custom_formatter,
    frequency_mapping, get_marker_handle, get_trendline_handle, group_lr, paired_x_positions,
    prism_cleanup_axes, prism_style, save_figure, save_legend,
)

CHREEF_COCHLEAE_DICT = cochleae_for(cohort_cochleae("chreef_mouse"), "SGN", "SGN_v2")
OTOF_COCHLEAE_DICT = cochleae_for(cohort_cochleae("otof_mouse"), "IHC", "IHC_v11")
FCHRIMSON_COCHLEAE_DICT = cochleae_for(cohort_cochleae("fchrimson_gerbil"), "SGN", "SGN_v2")

# G_EK_000049 received the injection postnatally, the other three animals as adults. The two
# groups are not comparable, so a panel can plot them apart and average only the adults.
POSTNATAL_COCHLEAE = cohort_postnatal("fchrimson_gerbil")
ADULT_COCHLEAE = [name for name in cohort_cochleae("fchrimson_gerbil") if name not in POSTNATAL_COCHLEAE]

MARKER_LEFT = "o"
MARKER_RIGHT = "^"

# Fraction range along Rosenthal's canal for each cochlear region. A density entry is assigned to
# a region by its reference_fraction, not by its key in the density JSON, so that a region can
# hold several measurements. The bounds are inclusive and the first matching region wins, hence
# 0.3 belongs to "apex" and 0.7 to "mid".
POSITION_DICT = {
    "apex": {"min_fraction": 0.05, "max_fraction": 0.3},
    "mid": {"min_fraction": 0.3, "max_fraction": 0.7},
    "base": {"min_fraction": 0.7, "max_fraction": 0.95},
}
POSITIONS = list(POSITION_DICT)
POSITION_LABELS = {"apex": "Apex", "mid": "Mid", "base": "Base"}


def _expression_efficiency(marker_labels) -> float:
    """Fraction of marker-positive (1) among the positive and negative (2) instances."""
    n_pos = (marker_labels == 1).sum()
    n_neg = (marker_labels == 2).sum()
    return float(n_pos) / (n_pos + n_neg)


def plot_pair_legend(
    data: dict,
    save_path: str,
    cochleae_dict: Optional[Dict] = None,
    use_alias: bool = True,
    alignment: str = "horizontal",
):
    """Plot the legend with the injected and the non-injected marker of every animal.

    Args:
        data: Data of the cochleae to show, mapping cochlea name to measurements.
        save_path: File path to save legend.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
            Defaults to CHREEF_COCHLEAE_DICT.
        use_alias: Use alias.
        alignment: Alignment of legend, "horizontal" or "vertical".
    """
    cochleae_dict = cochleae_dict if cochleae_dict is not None else CHREEF_COCHLEAE_DICT
    colors_by_animal = animal_colors(cochleae_dict, use_alias)

    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in data.keys()]
    alias, _, _ = group_lr(alias, [0] * len(alias))

    colors, labels, markers = [], [], []
    ncol = len(alias)
    for a in alias:
        colors += [colors_by_animal[a], colors_by_animal[a]]
        labels += [f"{a}L", f"{a}R"]
        markers += [MARKER_LEFT, MARKER_RIGHT]
    if alignment == "vertical":
        colors = colors[::2] + colors[1::2]
        labels = labels[::2] + labels[1::2]
        markers = markers[::2] + markers[1::2]
        ncol = 2

    handles = [get_marker_handle(c, m) for (c, m) in zip(colors, markers)]
    save_legend(handles, labels, save_path, ncol=ncol)


def plot_side_trendline_legend(save_path: str):
    """Plot the legend of the injected (dashed) and the non-injected (dotted) trendline.

    Args:
        save_path: Path for output.
    """
    lw = 3
    handles = [get_trendline_handle(style, lw) for style in ["dashed", "dotted"]]
    save_legend(handles, ["Injected", "Non-Injected"], save_path, ncol=1, handlelength=lw * 1.5)


def plot_adult_mean_legend(save_path: str):
    """Plot the legend of the adult injected mean of plot_efficiency_per_band.

    Args:
        save_path: File path for the legend.
    """
    handle = get_trendline_handle("dashed", 3)
    handle.set_color(COLOR_LEFT)
    save_legend([handle], ["Mean, adult injected"], save_path, handlelength=4.5)


def plot_side_legend(
    save_path: str,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
):
    """Plot the legend of a panel that colors the points by side, e.g. a single cochlea pair.

    Args:
        save_path: File path to save legend.
        labels: One label per entry. Defaults to the aliases of the postnatal gerbil cochleae.
        colors: One color per entry. Defaults to the injected and non-injected side colors.
        markers: One marker per entry. Defaults to the injected and non-injected markers.
    """
    colors = [COLOR_LEFT, COLOR_RIGHT] if colors is None else colors
    if labels is None:
        labels = [COCHLEA_DICT[name]["alias"] for name in POSTNATAL_COCHLEAE]
    markers = [MARKER_LEFT, MARKER_RIGHT] if markers is None else markers

    handles = [get_marker_handle(c, m) for (c, m) in zip(colors, markers)]
    save_legend(handles, labels, save_path, ncol=len(labels))


def _count_ylim(values, lower, upper, tick_step: int = 4000, round_to: int = 2000):
    """Y-axis bounds and ticks that fit the data points, the reference band and the band label.

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


def _efficiency_ylim(values, margin: float = 0.05):
    """Data-driven y-axis bounds for an expression-efficiency panel.

    Gerbil expression efficiency spans a much wider range than mouse (some non-injected
    cochleae are entirely marker-negative, i.e. efficiency 0), so the axis is derived from
    the actual data rather than a fixed range tuned for one animal.

    Returns:
        (ylim_lo, ylim_hi, tick_lo, tick_hi) with tick_lo/tick_hi rounded to the nearest 0.1
        for use with np.arange(tick_lo, tick_hi + eps, 0.1).
    """
    finite = [v for v in values if not np.isnan(v)]
    data_min, data_max = min(finite), max(finite)
    ylim_lo = max(0.0, data_min - margin)
    ylim_hi = min(1.0, data_max + margin)
    tick_lo = math.floor(ylim_lo * 10) / 10
    tick_hi = math.ceil(ylim_hi * 10) / 10
    return ylim_lo, ylim_hi, tick_lo, tick_hi


def _draw_pairs(ax, alias, values_left, values_right, colors_by_animal, x_left=1, x_right=2, offset=0.08):
    """Draw the injected and non-injected point of every animal, joined by a grey line."""
    x_pos_inj = paired_x_positions(len(values_left), x_left, offset)
    x_pos_non = paired_x_positions(len(values_right), x_right, offset)
    for left, right, xi, xn in zip(values_left, values_right, x_pos_inj, x_pos_non):
        ax.plot([xi, xn], [left, right], linestyle="solid", color="grey", alpha=0.4, zorder=0)
    for num, animal in enumerate(alias):
        ax.scatter(x_pos_inj[num], values_left[num], label=animal,
                   color=colors_by_animal[animal], marker=MARKER_LEFT, s=80, zorder=1)
        ax.scatter(x_pos_non[num], values_right[num],
                   color=colors_by_animal[animal], marker=MARKER_RIGHT, s=80, zorder=1)


def _pair_xticks(ax, xtick_labels, sub_label_size, x_left=1, x_right=2):
    plt.xticks([x_left, x_right], list(xtick_labels), fontsize=sub_label_size)
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment("center")
    ax.tick_params(axis="x", which="major", pad=16)


def plot_count_vs_untreated(
    data: dict,
    save_path: str,
    cochleae_dict: Dict,
    reference_values: Sequence[float],
    plot: bool = False,
    use_alias: bool = True,
    count_label: str = "SGN count per cochlea",
    xtick_labels: Tuple[str, str] = ("Injected", "Non-\nInjected"),
    ylim: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[List[float]] = None,
):
    """Plot the instance count per cochlea against the 95% CI band of untreated cochleae.

    Args:
        data: Table per cochlea with one row per instance, so the row count is the instance count.
        save_path: File path to save the figure.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
        reference_values: Counts of the untreated cochleae. The band is mean +/- 1.96 * std.
        plot: Plot figure.
        use_alias: Use alias.
        count_label: Y-axis label.
        xtick_labels: Labels for the left (injected) and right (non-injected) x positions.
        ylim: Lower and upper y-axis limit. Defaults to bounds that fit the points and the band.
        y_ticks: Y-axis tick positions. Ignored when ylim is None.
    """
    prism_style()
    colors_by_animal = animal_colors(cochleae_dict, use_alias)
    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in data.keys()]
    alias, values_left, values_right = group_lr(alias, [len(table) for table in data.values()])

    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    fontsize_untreated = 16

    _draw_pairs(ax, alias, values_left, values_right, colors_by_animal)
    _pair_xticks(ax, xtick_labels, sub_label_size)

    mean, std = np.mean(reference_values), np.std(reference_values)
    upper_y = mean + 1.96 * std
    lower_y = mean - 1.96 * std
    if ylim is None:
        ylim0, ylim1, y_ticks = _count_ylim(values_left + values_right, lower_y, upper_y)
        ylim = (ylim0, ylim1)

    plt.yticks(y_ticks, fontsize=main_tick_size)
    plt.ylabel(count_label, fontsize=main_label_size)
    plt.ylim(*ylim)

    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=[COLOR_UNTREATED for _ in range(2)], zorder=-1)
    text_offset = (ylim[1] - ylim[0]) / 40
    plt.text((xmin + xmax) / 2, upper_y + text_offset, "untreated cochleae\n95% CI",
             color=COLOR_UNTREATED, fontsize=fontsize_untreated, ha="center")
    plt.fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_UNTREATED, alpha=0.05, interpolate=True)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_expression_efficiency(
    data: dict,
    save_path: str,
    cochleae_dict: Dict,
    plot: bool = False,
    use_alias: bool = True,
    xtick_labels: Tuple[str, str] = ("Injected", "Non-\nInjected"),
    ylim: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[List[float]] = None,
):
    """Plot the expression efficiency per cochlea, injected against non-injected.

    The marker_labels column holds 0 for unlabeled, 1 for positive and 2 for negative instances.

    Args:
        data: Table per cochlea with a marker_labels column.
        save_path: File path to save the figure.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
        plot: Plot figure.
        use_alias: Use alias.
        xtick_labels: Labels for the left (injected) and right (non-injected) x positions.
        ylim: Lower and upper y-axis limit. Defaults to bounds that fit the points, with ticks
            every 0.1.
        y_ticks: Y-axis tick positions. Only used together with ylim.
    """
    prism_style()
    colors_by_animal = animal_colors(cochleae_dict, use_alias)
    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in data.keys()]
    values = [_expression_efficiency(vals["marker_labels"].values) for vals in data.values()]
    alias, values_left, values_right = group_lr(alias, values)

    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16

    x_pos_inj = paired_x_positions(len(values_left), 1, 0.08)
    x_pos_non = paired_x_positions(len(values_right), 2, 0.08)
    for num, animal in enumerate(alias):
        ax.scatter(x_pos_inj[num], values_left[num], label=animal,
                   color=colors_by_animal[animal], marker=MARKER_LEFT, s=80, zorder=1)
        ax.scatter(x_pos_non[num], values_right[num],
                   color=colors_by_animal[animal], marker=MARKER_RIGHT, s=80, zorder=1)
    for left, right, xi, xn in zip(values_left, values_right, x_pos_inj, x_pos_non):
        ax.plot([xi, xn], [left, right], linestyle="solid", color="grey", alpha=0.4, zorder=0)

    if ylim is None:
        ylim_lo, ylim_hi, tick_lo, tick_hi = _efficiency_ylim(values_left + values_right)
        plt.ylim(ylim_lo, ylim_hi)
        plt.yticks(np.arange(tick_lo, tick_hi + 1e-9, 0.1), fontsize=main_tick_size)
    else:
        plt.ylim(*ylim)
        if y_ticks is not None:
            plt.yticks(y_ticks, fontsize=main_tick_size)

    _pair_xticks(ax, xtick_labels, sub_label_size)
    plt.ylabel("Expression efficiency", fontsize=main_label_size)
    ax.yaxis.set_major_formatter(custom_formatter(1))
    plt.xlim(0.5, 2.5)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def _band_x_positions(result: pd.DataFrame, offset_map: dict, offset: float) -> pd.Series:
    """Assign the x position of every point, jittered inside the column of its side.

    The positions are derived from every cochlea in result, not only from the plotted ones,
    so a trendline of cochleae that are left out still spans the column they would occupy.

    Args:
        result: Table as returned by util.band_table.
        offset_map: X offset of the injected and the non-injected column.
        offset: X distance between two cochleae inside one column.

    Returns:
        The x position per row of result.
    """
    aliases = sorted(result["cochlea"].unique())
    n_animals = len({alias[:-1] for alias in aliases})

    x_positions = pd.Series(index=result.index, dtype=float)
    for num, alias in enumerate(aliases):
        rows = result["cochlea"] == alias
        x_positions[rows] = result.loc[rows, "x_pos"] + offset_map[alias[-1]] - n_animals / 2 * offset + offset * num
    return x_positions


def _trend_dict(result: pd.DataFrame, aliases: List[str]) -> dict:
    """Collect the per-cochlea curves that a trendline averages over."""
    trend_dict = {}
    for alias in sorted(aliases):
        grp = result[result["cochlea"] == alias].sort_values("x")
        trend_dict[alias] = {
            "x_sorted": grp["x"].to_numpy(),
            "y_sorted": grp["value"].to_numpy(),
            "side": alias[-1],
        }
    return trend_dict


def _draw_band_trendline(
    ax,
    trend_dict: dict,
    side: str,
    color: str,
    x_bounds: tuple,
    linestyle: str = "dashed",
    alpha: float = 0.6,
    zorder: int = 2,
    show_std: bool = False,
    linewidth: int = 3,
) -> List[float]:
    """Draw the mean of one side as a step line over the octave bands.

    Args:
        ax: Axes to draw on.
        trend_dict: Per-cochlea curves, as returned by _trend_dict.
        side: 'L' for the injected or 'R' for the non-injected cochleae.
        color: Line color.
        x_bounds: X values the line is extended to, so that it spans the full axis.
        linestyle: Line style.
        alpha: Line alpha.
        zorder: Draw order.
        show_std: Draw the standard deviation as two bounds and a filled band.
        linewidth: Line width.

    Returns:
        The drawn y values, to include in the y-axis limits.
    """
    x_sorted, y_center, y_upper, y_lower = band_trendline(trend_dict, side)
    min_x, max_x = x_bounds
    x_sorted.insert(0, min_x)
    x_sorted.append(max_x)
    y_center.insert(0, y_center[0])
    y_center.append(y_center[-1])

    ax.plot(x_sorted, y_center, linestyle=linestyle, color=color, alpha=alpha,
            linewidth=linewidth, zorder=zorder)

    drawn = list(y_center)
    if show_std:
        for y_bound in (y_lower, y_upper):
            y_bound.insert(0, y_bound[0])
            y_bound.append(y_bound[-1])
            ax.plot(x_sorted, y_bound, linestyle="solid", color=color, alpha=0.08, zorder=0)
        ax.fill_between(x_sorted, y_lower, y_upper, color=color, alpha=0.05, interpolate=True)
        drawn += y_lower + y_upper
    return drawn


def plot_efficiency_per_band(
    data: dict,
    save_path: str,
    cochleae_dict: Dict,
    animal: str = "mouse",
    plot: bool = False,
    use_alias: bool = True,
    trendlines: bool = False,
    trendline_std: bool = False,
    trendline_colors: Tuple[str, str] = (COLOR_LEFT, COLOR_RIGHT),
    cochleae: Optional[List[str]] = None,
    color_by_side: bool = False,
    reference_cochleae: Optional[List[str]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[List[float]] = None,
):
    """Plot the expression efficiency per octave band, injected (I) next to non-injected (N).

    Args:
        data: Table per cochlea with the columns frequency[kHz] and marker_labels. Every cochlea
            is read, so that a reference trendline stays available when only a subset is plotted.
        save_path: File path to save the figure.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
        animal: "mouse" or "gerbil", selects the octave bands.
        plot: Plot figure.
        use_alias: Use alias.
        trendlines: Draw the injected (dashed) and the non-injected (dotted) mean of the plotted
            cochleae.
        trendline_std: Add the standard deviation to every drawn trendline.
        trendline_colors: Colors of the injected and the non-injected trendline.
        cochleae: Cochleae to plot. Defaults to every cochlea in data.
        color_by_side: Color the points by side instead of per animal. Use it when the plot holds
            a single pair, where the animal color carries no information.
        reference_cochleae: Injected cochleae whose mean is drawn as a dashed reference line, also
            when they are not plotted.
        ylim: Lower and upper y-axis limit. Defaults to bounds that fit the points and the
            trendlines, with ticks every 0.1.
        y_ticks: Y-axis tick positions. Only used together with ylim.
    """
    prism_style()

    cochleae = list(data) if cochleae is None else cochleae

    binned = {
        cochlea_label(name, cochleae_dict[name], use_alias): frequency_mapping(
            values["frequency[kHz]"].values, values["marker_labels"].values, animal=animal,
            transduction_efficiency=True)
        for name, values in data.items()
    }
    result, bin_labels = band_table(binned)

    offset_map = {"L": -0.2, "R": 0.2}
    offset = 0.018
    result["x"] = _band_x_positions(result, offset_map, offset)

    plotted = sorted(cochlea_label(name, cochleae_dict[name], use_alias) for name in cochleae)

    fig, ax = plt.subplots(figsize=(8, 5))

    sub_tick_label_size = 12
    tick_label_size = 14
    yaxis_tick_size = 16
    label_size = 20
    band_label_offset_y = 0.08

    if color_by_side:
        color_map = {alias: COLOR_LEFT if alias.endswith("L") else COLOR_RIGHT for alias in plotted}
    else:
        colors_by_animal = animal_colors(cochleae_dict, use_alias)
        color_map = {alias: colors_by_animal[alias[:-1]] for alias in plotted}

    for alias in plotted:
        grp = result[result["cochlea"] == alias]
        ax.scatter(
            grp["x"],
            grp["value"],
            s=60,
            alpha=0.8,
            marker=MARKER_LEFT if alias.endswith("L") else MARKER_RIGHT,
            color=color_map[alias],
            zorder=1,
        )

    # The y limits have to cover every drawn trendline as well. A reference line averaged over
    # cochleae that are not plotted can otherwise fall outside the axis.
    axis_values = result[result["cochlea"].isin(plotted)]["value"].tolist()

    xlim_left, xlim_right = ax.get_xlim()
    x_bounds = (result["x"].min(), result["x"].max())

    if trendlines:
        trend_dict = _trend_dict(result, plotted)
        axis_values += _draw_band_trendline(
            ax, trend_dict, "L", trendline_colors[0], x_bounds,
            linestyle="dashed", alpha=0.6, zorder=2, show_std=trendline_std)
        axis_values += _draw_band_trendline(
            ax, trend_dict, "R", trendline_colors[1], x_bounds,
            linestyle="dotted", alpha=0.7, zorder=0, show_std=trendline_std)

    if reference_cochleae:
        reference = [cochlea_label(name, cochleae_dict[name], use_alias) for name in reference_cochleae]
        axis_values += _draw_band_trendline(
            ax, _trend_dict(result, reference), "L", COLOR_LEFT, x_bounds,
            linestyle="dashed", alpha=0.6, zorder=2, show_std=trendline_std)

    if ylim is None:
        ylim_min, ylim_max, ytick_min, ytick_max = _efficiency_ylim(axis_values)
        ylim = (ylim_min, ylim_max)
        y_ticks = np.arange(ytick_min, ytick_max + 1e-9, 0.1)

    plt.xlim(xlim_left, xlim_right)
    main_ticks = range(len(bin_labels))
    ax.yaxis.set_major_formatter(custom_formatter(1))
    ax.set_ylim(*ylim)
    plt.yticks(y_ticks, fontsize=yaxis_tick_size)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    ax.set_xticks([pos + offset_map["L"] for pos in main_ticks] + [pos + offset_map["R"] for pos in main_ticks])
    ax.set_xticklabels(["I"] * len(main_ticks) + ["N"] * len(main_ticks), fontsize=sub_tick_label_size)

    for i, label in enumerate(bin_labels):
        ax.text(i, ax.get_ylim()[0] - band_label_offset_y * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label, ha="center", va="top", fontsize=tick_label_size, fontweight="bold")

    ax.set_xlabel("Octave band [kHz]", fontsize=label_size)
    ax.xaxis.set_label_coords(.5, -.16)
    ax.set_ylabel("Expression efficiency", fontsize=label_size)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def _region_of_fraction(fraction: float) -> Optional[str]:
    """Region of POSITION_DICT whose fraction range contains `fraction`, or None."""
    for region, limits in POSITION_DICT.items():
        if limits["min_fraction"] <= fraction <= limits["max_fraction"]:
            return region
    return None


def group_density_by_region(cochlea_density: dict) -> Dict[str, List[dict]]:
    """Group the position entries of one cochlea by cochlear region.

    Entries are assigned by their reference_fraction, so both the preset density file
    (keys apex/mid/base) and the extended one (keys of fraction strings) work the same way.

    Args:
        cochlea_density: Parsed density JSON of a single cochlea.

    Returns:
        Mapping of region name to its entries, sorted by reference_fraction.
    """
    regions = {region: [] for region in POSITION_DICT}
    unassigned = []
    for key, entry in cochlea_density.items():
        fraction = entry.get("reference_fraction")
        if not isinstance(fraction, (int, float)):
            unassigned.append(key)
            continue
        region = _region_of_fraction(float(fraction))
        if region is None:
            unassigned.append(f"{fraction}")
        else:
            regions[region].append(entry)

    if unassigned:
        print(f"Skipping density positions outside of any region: {', '.join(unassigned)}")

    return {region: sorted(entries, key=lambda e: e["reference_fraction"]) for region, entries in regions.items()}


def _density_value(entry: dict) -> float:
    """Extract SGN density in cells/mm^2 from a SGN_density_2d.json position entry."""
    density = entry.get("density")
    if density is None or (isinstance(density, float) and np.isnan(density)):
        return np.nan
    return density * 1e6  # stored as cells/um^2 -> cells/mm^2


def untreated_reference_values(density_data: dict) -> Dict[str, List[float]]:
    """Get the mean SGN density per cochlear region for every untreated cochlea, in cells/mm^2.

    Averaging the positions of a region per cochlea first gives one value per animal, so that the
    reference band reflects the spread between animals and not the spread between the positions
    inside one cochlea.

    Args:
        density_data: Parsed SGN density data of the untreated gerbil cochleae.

    Returns:
        Mapping of region name to the per-cochlea mean densities of that region.
    """
    regions = {name: group_density_by_region(entries) for name, entries in density_data.items()}

    reference_values = {}
    for position in POSITIONS:
        means = []
        for name in density_data:
            values = np.asarray([_density_value(entry) for entry in regions[name][position]], dtype=float)
            values = values[np.isfinite(values)]
            if values.size:
                means.append(float(np.mean(values)))
        reference_values[position] = means
    return reference_values


def plot_density_by_region(
    density_data: dict,
    save_path: str,
    cochleae_dict: Optional[Dict] = None,
    plot: bool = False,
    use_alias: bool = True,
    show_std: bool = False,
    reference_values: Optional[Dict[str, List[float]]] = None,
):
    """Plot the SGN density at apex/mid/base, injected against non-injected.

    Each data point is the average of all density positions that fall into the region, following
    the fraction ranges of POSITION_DICT.

    Args:
        density_data: Parsed SGN density data per cochlea.
        save_path: File path to save the figure.
        cochleae_dict: Mapping of cochlea name to its metadata. Defaults to FCHRIMSON_COCHLEAE_DICT.
        plot: Plot figure.
        use_alias: Use alias.
        show_std: Draw the standard deviation of the density positions of a region as error bars.
            Only regions with more than one position get an error bar.
        reference_values: Optional per-position list of untreated SGN density values
            [cells/mm^2], as returned by untreated_reference_values. A 95% CI band
            (mean +/- 1.96 * std) is drawn for any position present with a non-empty list;
            positions absent from the dict, or mapped to an empty list, are drawn without a band.
    """
    prism_style()

    cochleae_dict = FCHRIMSON_COCHLEAE_DICT if cochleae_dict is None else cochleae_dict
    colors_by_animal = animal_colors(cochleae_dict, use_alias)
    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in density_data.keys()]

    regions = {name: group_density_by_region(entries) for name, entries in density_data.items()}

    fig, ax = plt.subplots(figsize=(10, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    fontsize_untreated = 16

    offset = 0.08
    group_spacing = 2.5
    col_width = 1.0

    group_x_centers = []
    bands = []
    for g, position in enumerate(POSITIONS):
        x_left = g * group_spacing + 1
        x_right = x_left + col_width
        group_x_centers.append((position, x_left, x_right))

        means, stds = [], []
        for name in density_data:
            vals = np.asarray([_density_value(entry) for entry in regions[name][position]], dtype=float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.mean(vals)) if vals.size else np.nan)
            stds.append(float(np.std(vals)) if vals.size > 1 else np.nan)

        animals, values_left, values_right = group_lr(alias, means)
        _, stds_left, stds_right = group_lr(alias, stds)

        x_pos_inj = paired_x_positions(len(values_left), x_left, offset)
        x_pos_non = paired_x_positions(len(values_right), x_right, offset)

        for left, right, xi, xn in zip(values_left, values_right, x_pos_inj, x_pos_non):
            ax.plot([xi, xn], [left, right], linestyle="solid", color="grey", alpha=0.4, zorder=0)

        for num, animal in enumerate(animals):
            ax.scatter(x_pos_inj[num], values_left[num], color=colors_by_animal[animal],
                       marker=MARKER_LEFT, s=80, zorder=1)
            ax.scatter(x_pos_non[num], values_right[num], color=colors_by_animal[animal],
                       marker=MARKER_RIGHT, s=80, zorder=1)

            if show_std:
                # The columns are only `offset` apart, so the error bar takes the color of its
                # own data point to stay attributable.
                for x_val, y_val, err in ((x_pos_inj[num], values_left[num], stds_left[num]),
                                          (x_pos_non[num], values_right[num], stds_right[num])):
                    if np.isfinite(err):
                        ax.errorbar([x_val], [y_val], yerr=[err], fmt="none",
                                    color=colors_by_animal[animal], zorder=1)

        ref = (reference_values or {}).get(position)
        if ref:
            ref_arr = np.asarray(ref, dtype=float)
            mean, std = ref_arr.mean(), ref_arr.std()
            lower, upper = mean - 1.96 * std, mean + 1.96 * std
            xmin_ref, xmax_ref = x_left - 0.5, x_right + 0.5
            # The band is drawn here so that the y-axis autoscaling includes it. Its label
            # follows below, once the final y limits are known.
            ax.hlines([lower, upper], xmin_ref, xmax_ref, colors=[COLOR_UNTREATED] * 2, zorder=-1)
            ax.fill_between([xmin_ref, xmax_ref], lower, upper,
                            color=COLOR_UNTREATED, alpha=0.05, interpolate=True)
            tops = np.asarray(means, dtype=float)
            if show_std:
                errors = np.asarray(stds, dtype=float)
                tops = tops + np.where(np.isfinite(errors), errors, 0.0)
            region_top = float(np.nanmax(tops)) if np.isfinite(tops).any() else -np.inf
            bands.append((xmin_ref, xmax_ref, upper, region_top))

    xticks = [x for _, x_left, x_right in group_x_centers for x in (x_left, x_right)]
    xticklabels = ["Injected", "Non-\nInjected"] * len(POSITIONS)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, fontsize=sub_label_size)
    for label in ax.get_xticklabels():
        label.set_verticalalignment("center")
    ax.tick_params(axis="x", which="major", pad=16)

    xmin = group_x_centers[0][1] - 0.5
    xmax = group_x_centers[-1][2] + 0.5
    ax.set_xlim(xmin, xmax)

    ax.tick_params(axis="y", labelsize=main_tick_size)
    ax.set_ylabel("SGN density [cells/mm²]", fontsize=main_label_size)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin
    ax.set_ylim(ymin - 0.15 * yrange, ymax + (0.1 * yrange if bands else 0.0))
    for position, x_left, x_right in group_x_centers:
        ax.text((x_left + x_right) / 2, ymin - 0.05 * yrange, POSITION_LABELS[position],
                ha="center", va="top", fontsize=main_label_size, fontweight="bold")

    # One label for all bands, because every band shows the same quantity and repeating the label
    # per region would only add clutter. It goes above the highest band that has no data point
    # over it, so that it never sits on a marker. If every region has a point above its band, the
    # label is lifted over the data of the highest band instead.
    if bands:
        clear = [band for band in bands if band[3] <= band[2]]
        xmin_ref, xmax_ref, upper, region_top = max(clear or bands, key=lambda band: band[2])
        ylim0, ylim1 = ax.get_ylim()
        ax.text((xmin_ref + xmax_ref) / 2, max(upper, region_top) + (ylim1 - ylim0) / 40,
                "untreated cochleae\n95% CI",
                color=COLOR_UNTREATED, fontsize=fontsize_untreated, ha="center")

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_lavision_efficiency_per_band(
    otof_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    trendline_mode: Optional[str] = None,
    mapping: str = "mueller",
    cochleae_dict: Optional[Dict] = None,
):
    """Plot the expression efficiency per octave band of the LaVision OTOF cochleae.

    Args:
        otof_data: Table per cochlea with the frequency columns and expression_classification.
        save_path: Output path for figure.
        plot: Plot figure.
        use_alias: Use alias for animal name.
        trendline_mode: Mode for trendline- either "filled", "mean", or None.
        mapping: Function for tonotopic mapping, either "default" or "mueller".
        cochleae_dict: Mapping of cochlea name to its metadata. Defaults to data.LAVISION_COCHLEAE_DICT.
    """
    prism_style()
    label_size = 20
    tick_label_size = 14

    cochleae_dict = LAVISION_COCHLEAE_DICT if cochleae_dict is None else cochleae_dict
    binned = {}
    expression_eff_dic = {}
    color_dict = cochlea_colors(cochleae_dict, otof_data, use_alias)
    for name, values in otof_data.items():
        alias = cochlea_label(name, cochleae_dict[name], use_alias)
        if mapping == "default":
            freq = values["frequency[kHz]"].values
            bin_edges, bin_labels = None, None
        elif mapping == "mueller":
            freq = values["frequency-mueller[kHz]"].values
            # We need custom bin edges and bin labels in this case.
            bin_edges = [0, 8, 12, 16, 24, np.inf]
            bin_labels = ["4-8", "8-12", "12-16", "16-24", "24-32"]
            assert len(bin_edges) == len(bin_labels) + 1
        else:
            raise ValueError("Choose either 'default' or 'mueller' for tonotopic mapping.")
        marker_labels = values["expression_classification"].values
        marker_pos = len([1 for i in marker_labels if i == 1])
        marker_neg = len([1 for i in marker_labels if i == 2])
        expression_eff = marker_pos / (marker_pos + marker_neg)
        print(f"Cochlea {name}, average expression efficiency {expression_eff}")
        print(f"Cochlea {name}, number of IHCs: {len(freq)}")
        binned[alias] = frequency_mapping(
            freq, marker_labels, animal="mouse", transduction_efficiency=True,
            bin_edges=bin_edges, bin_labels=bin_labels
        )
        expression_eff_dic[alias] = expression_eff

    result, bin_labels = band_table(binned)

    fig, ax = plt.subplots(figsize=(8, 4))

    offset = 0.08
    trend_dict = {}
    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        x_positions = [x - len(grp["x_pos"]) // 2 * offset + offset * num for x in grp["x_pos"]]
        ax.scatter(x_positions, grp["value"], marker="o", label=name, s=80, alpha=1, color=color_dict[name])

        if trendline_mode == "filled":
            sorted_idx = np.argsort(x_positions)
            trend_dict[name] = {"x_sorted": np.array(x_positions)[sorted_idx],
                                "y_sorted": np.array(grp["value"])[sorted_idx]}

    if trendline_mode == "filled":
        x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = band_trendline(trend_dict)
        ax.plot(x_sorted, y_sorted, linestyle="dotted", color="gray", alpha=0.6, linewidth=3, zorder=2)
        ax.plot(x_sorted, y_sorted_upper, linestyle="solid", color="gray", alpha=0.08, zorder=0)
        ax.plot(x_sorted, y_sorted_lower, linestyle="solid", color="gray", alpha=0.08, zorder=0)
        plt.fill_between(x_sorted, y_sorted_lower, y_sorted_upper, color="gray", alpha=0.05, interpolate=True)

    elif trendline_mode == "mean":
        xlim_left, xlim_right = ax.get_xlim()
        y_offset = [0.01, -0.04]
        x_offset = 0.5
        plt.xlim(xlim_left, xlim_right)
        for num, key in enumerate(color_dict.keys()):
            color = color_dict[key]
            expression_eff = expression_eff_dic[key]
            ax.text(xlim_left + x_offset, expression_eff + y_offset[num], "mean",
                    color=color, fontsize=tick_label_size, ha="center")
            ax.plot([xlim_left, xlim_right], [expression_eff, expression_eff],
                    linestyle="dashed", color=color, alpha=0.7, zorder=0)

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.set_xlabel("Octave band [kHz]", fontsize=label_size)

    ax.set_ylabel("Expression efficiency")
    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)
