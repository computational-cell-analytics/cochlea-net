import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from flamingo_tools.analysis.density_utils import report_density_overlap
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from plot_fig4 import group_lr, plot_legend_fig04_trendline
from util import (
    COCHLEA_DICT,
    COLOR_LEFT,
    COLOR_RIGHT,
    animal_colors,
    cochlea_label,
    custom_formatter_1,
    export_legend,
    frequency_mapping,
    get_marker_handle,
    png_dpi,
    prism_cleanup_axes,
    prism_style,
)

SOURCE_NAME = "SGN_v2"
TABLE_FILENAME = "GFP_SGN-v2.tsv"
DENSITY_FILENAME = "SGN_density_2d.json"
DENSITY_FILENAME_EXTENDED = "SGN_density_2d_extended.json"

SGN_CACHE_PATH = "./gerbil_chreef_sgn_data.pkl"
DENSITY_CACHE_PATH = "./gerbil_chreef_density_data.pkl"

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

# The gerbil cochleae for the ChReef analysis. The metadata lives in util.COCHLEA_DICT.
COCHLEAE = [
    "G_EK_000049_L",
    "G_EK_000049_R",
    "G_EK_000071_L",
    "G_EK_000071_R",
    "G_EK_000074_L",
    "G_EK_000074_R",
    "G_EK_000076_L",
    "G_EK_000076_R",
]

COCHLEAE_DICT = {name: COCHLEA_DICT[name] for name in COCHLEAE}

FILE_EXTENSION = "png"

COLOR_REFERENCE = "#DB7B00"
MARKER_LEFT = "o"
MARKER_RIGHT = "^"


def _match_column(columns, prefix: str, exclude: Optional[str] = None) -> str:
    """Find the single column starting with `prefix`, tolerating encoding-mangled headers."""
    candidates = [c for c in columns if c.lower().startswith(prefix) and c != exclude]
    assert len(candidates) == 1, f"Expected one column starting with {prefix!r}, found {candidates}"
    return candidates[0]


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


def get_gerbil_chreef_data(force_download: bool = False) -> Dict[str, pd.DataFrame]:
    """Create (pickled) dictionary of gerbil cochleae used for the optogenetic therapy figure.

    Args:
        force_download: Ignore the cached pickle and fetch the tables from S3 again.

    Returns:
        Mapping of cochlea name to a table with columns label_id, frequency[kHz], marker_labels.
    """
    if not force_download and os.path.exists(SGN_CACHE_PATH):
        with open(SGN_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    s3 = create_s3_target()
    chreef_data = {}
    for cochlea, meta in COCHLEAE_DICT.items():
        print("Processing cochlea:", cochlea)
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        source = info["sources"][SOURCE_NAME]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]

        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, TABLE_FILENAME), mode="rb")
        table = pd.read_csv(table_content, sep="\t")
        table = table[table.component_labels.isin(meta["component"])]

        freq_col = _match_column(table.columns, "frequency")
        values = table[["label_id", freq_col, "marker_labels"]].rename(columns={freq_col: "frequency[kHz]"})

        marker_labels = values["marker_labels"].to_numpy()
        if not np.isin(marker_labels, [1, 2]).any():
            # Non-injected cochleae are sometimes left fully unannotated (0) instead of being
            # individually marked negative (2). A fully non-injected cochlea is entirely
            # marker-negative, so treat this case as all-negative rather than leaving a 0/0
            # efficiency downstream.
            values = values.assign(marker_labels=2)

        chreef_data[cochlea] = values

    with open(SGN_CACHE_PATH, "wb") as f:
        pickle.dump(chreef_data, f)
    return chreef_data


def get_gerbil_density_data(force_download: bool = False) -> Dict[str, dict]:
    """Create (pickled) dictionary of the SGN density data for the gerbil cochleae.

    The extended density file is preferred, because it holds several positions per cochlear
    region. Cochleae without it fall back to the preset file with one position per region.

    Args:
        force_download: Ignore the cached pickle and fetch the density files from S3 again.

    Returns:
        Mapping of cochlea name to its parsed density dict, keyed by position.
    """
    if not force_download and os.path.exists(DENSITY_CACHE_PATH):
        with open(DENSITY_CACHE_PATH, "rb") as f:
            return pickle.load(f)

    s3 = create_s3_target()
    density_data = {}
    for cochlea in COCHLEAE_DICT:
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        source = info["sources"][SOURCE_NAME]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]

        table_dir = os.path.join(BUCKET_NAME, cochlea, rel_path)
        density_path = os.path.join(table_dir, DENSITY_FILENAME_EXTENDED)
        if not s3.exists(density_path):
            density_path = os.path.join(table_dir, DENSITY_FILENAME)
        print(f"Processing cochlea: {cochlea}, density file: {os.path.basename(density_path)}")

        density_content = s3.open(density_path, mode="r", encoding="utf-8")
        density_data[cochlea] = json.loads(density_content.read())

    with open(DENSITY_CACHE_PATH, "wb") as f:
        pickle.dump(density_data, f)
    return density_data


def plot_legend_gerbil(
    chreef_data: dict,
    save_path: str,
    use_alias: bool = True,
    alignment: str = "horizontal",
):
    """Plot common legend for the gerbil ChReef figure panels.

    Args:
        chreef_data: Data of ChReef gerbil cochleae.
        save_path: File path to save legend.
        use_alias: Use alias.
        alignment: Alignment of legend.
    """
    colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)

    alias = [cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in chreef_data.keys()]
    alias, _, _ = group_lr(alias, [0] * len(alias))

    colors = []
    labels = []
    markers = []
    for a in alias:
        colors.append(colors_by_animal[a])
        colors.append(colors_by_animal[a])
        labels.append(f"{a}L")
        labels.append(f"{a}R")
        markers.append(MARKER_LEFT)
        markers.append(MARKER_RIGHT)

    if alignment == "vertical":
        colors = colors[::2] + colors[1::2]
        labels = labels[::2] + labels[1::2]
        markers = markers[::2] + markers[1::2]
        ncol = 2
    else:
        ncol = len(alias)

    handles = [get_marker_handle(c, m) for (c, m) in zip(colors, markers)]
    legend = plt.legend(handles, labels, loc=3, ncol=ncol, framealpha=1, frameon=False)

    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def _density_value(entry: dict) -> float:
    """Extract SGN density in cells/mm^2 from a SGN_density_2d.json position entry."""
    density = entry.get("density")
    if density is None or (isinstance(density, float) and np.isnan(density)):
        return np.nan
    return density * 1e6  # stored as cells/um^2 -> cells/mm^2


def fig_c_gerbil(
    density_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    show_std: bool = False,
    reference_values: Optional[Dict[str, List[float]]] = None,
):
    """Plot showing gerbil SGN density at apex/mid/base, Injected vs Non-Injected.

    Each data point is the average of all density positions that fall into the region, following
    the fraction ranges of POSITION_DICT.

    Args:
        density_data: Parsed SGN density data of ChReef gerbil cochleae.
        save_path: File path to save the figure.
        plot: Plot figure.
        use_alias: Use alias.
        show_std: Draw the standard deviation of the density positions of a region as error bars.
            Only regions with more than one position get an error bar.
        reference_values: Optional per-position list of healthy/untreated SGN density values
            [cells/mm^2]. A 95% CI band (mean +/- 1.96 * std) is drawn for any position present
            with a non-empty list; positions absent from the dict, or mapped to an empty list,
            are drawn without a band.
    """
    prism_style()

    colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)
    alias = [cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in density_data.keys()]

    regions = {name: group_density_by_region(entries) for name, entries in density_data.items()}

    fig, ax = plt.subplots(figsize=(10, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    fontsize_reference = 14

    offset = 0.08
    group_spacing = 2.5
    col_width = 1.0

    group_x_centers = []
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

        x_pos_inj = [x_left - len(values_left) // 2 * offset + offset * i for i in range(len(values_left))]
        x_pos_non = [x_right - len(values_right) // 2 * offset + offset * i for i in range(len(values_right))]

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
            ax.hlines([lower, upper], xmin_ref, xmax_ref, colors=[COLOR_REFERENCE, COLOR_REFERENCE], zorder=-1)
            ax.fill_between([xmin_ref, xmax_ref], lower, upper,
                            color=COLOR_REFERENCE, alpha=0.05, interpolate=True)
            ax.text((xmin_ref + xmax_ref) / 2, upper + (upper - lower) * 0.05, "reference\n95% CI",
                    color=COLOR_REFERENCE, fontsize=fontsize_reference, ha="center")

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
    ax.set_ylim(ymin - 0.15 * yrange, ymax)
    for position, x_left, x_right in group_x_centers:
        ax.text((x_left + x_right) / 2, ymin - 0.05 * yrange, POSITION_LABELS[position],
                ha="center", va="top", fontsize=main_label_size, fontweight="bold")

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


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


def fig_d_gerbil(
    chreef_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
):
    """Expression efficiency per gerbil cochlea, Injected vs Non-Injected.

    Args:
        chreef_data: Data of ChReef gerbil cochleae.
        save_path: File path to save the figure.
        plot: Plot figure.
        use_alias: Use alias.
    """
    prism_style()
    colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)
    alias = [cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in chreef_data.keys()]

    values = []
    for vals in chreef_data.values():
        marker_labels = vals["marker_labels"].values
        n_pos = (marker_labels == 1).sum()
        n_neg = (marker_labels == 2).sum()
        values.append(float(n_pos) / (n_pos + n_neg))

    alias, values_left, values_right = group_lr(alias, values)

    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16

    x_left = 1
    x_right = 2
    offset = 0.08

    x_pos_inj = [x_left - len(values_left) // 2 * offset + offset * i for i in range(len(values_left))]
    x_pos_non = [x_right - len(values_right) // 2 * offset + offset * i for i in range(len(values_right))]

    for num, animal in enumerate(alias):
        ax.scatter(x_pos_inj[num], values_left[num], label=animal,
                   color=colors_by_animal[animal], marker=MARKER_LEFT, s=80, zorder=1)
        ax.scatter(x_pos_non[num], values_right[num],
                   color=colors_by_animal[animal], marker=MARKER_RIGHT, s=80, zorder=1)

    for left, right, xi, xn in zip(values_left, values_right, x_pos_inj, x_pos_non):
        ax.plot([xi, xn], [left, right], linestyle="solid", color="grey", alpha=0.4, zorder=0)

    ylim_lo, ylim_hi, tick_lo, tick_hi = _efficiency_ylim(values_left + values_right)
    plt.ylim(ylim_lo, ylim_hi)
    plt.yticks(np.arange(tick_lo, tick_hi + 1e-9, 0.1), fontsize=main_tick_size)

    plt.xticks([x_left, x_right], ["Injected", "Non-\nInjected"], fontsize=sub_label_size)
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment("center")
    ax.tick_params(axis="x", which="major", pad=16)
    plt.ylabel("Expression efficiency", fontsize=main_label_size)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_1))

    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def _get_trendline_dict(trend_dict, side):
    x_sorted = [trend_dict[k]["x_sorted"] for k in trend_dict.keys() if trend_dict[k]["side"] == side]
    x_dict = {}
    for num in range(len(x_sorted[0])):
        x_dict[num] = {"pos": num, "values": []}

    for s in x_sorted:
        for num, pos in enumerate(s):
            x_dict[num]["values"].append(pos)

    y_sorted_all = [trend_dict[k]["y_sorted"] for k in trend_dict.keys() if trend_dict[k]["side"] == side]
    y_dict = {}
    for num in range(len(x_sorted[0])):
        y_dict[num] = {"pos": num, "values": []}

    for num in range(len(x_sorted[0])):
        y_dict[num]["mean"] = np.mean([y[num] for y in y_sorted_all])
        y_dict[num]["stdv"] = np.std([y[num] for y in y_sorted_all])
    return x_dict, y_dict


def _get_trendline_params(trend_dict, side):
    x_dict, y_dict = _get_trendline_dict(trend_dict, side)

    x_values = []
    for key in x_dict.keys():
        x_values.append(min(x_dict[key]["values"]))
        x_values.append(max(x_dict[key]["values"]))

    y_values_center = []
    y_values_upper = []
    y_values_lower = []
    for key in y_dict.keys():
        y_values_center.append(y_dict[key]["mean"])
        y_values_center.append(y_dict[key]["mean"])

        y_values_upper.append(y_dict[key]["mean"] + y_dict[key]["stdv"])
        y_values_upper.append(y_dict[key]["mean"] + y_dict[key]["stdv"])

        y_values_lower.append(y_dict[key]["mean"] - y_dict[key]["stdv"])
        y_values_lower.append(y_dict[key]["mean"] - y_dict[key]["stdv"])

    return x_values, y_values_center, y_values_upper, y_values_lower


def fig_e_gerbil(
    chreef_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    trendlines: bool = False,
    trendline_std: bool = False,
):
    """Expression efficiency per octave band for gerbil cochleae.

    Args:
        chreef_data: Data of ChReef gerbil cochleae.
        save_path: File path to save the figure.
        plot: Plot figure.
        use_alias: Use alias.
        trendlines: Use trendline of averages.
        trendline_std: Use standard deviation for upper and lower trendlines.
    """
    prism_style()

    result = {"cochlea": [], "octave_band": [], "value": []}
    for name, values in chreef_data.items():
        alias = cochlea_label(name, COCHLEAE_DICT[name], use_alias)

        freq = values["frequency[kHz]"].values
        marker_labels = values["marker_labels"].values
        octave_binned = frequency_mapping(freq, marker_labels, animal="gerbil", transduction_efficiency=True)

        result["cochlea"].extend([alias] * len(octave_binned))
        result["octave_band"].extend(octave_binned.axes[0].values.tolist())
        result["value"].extend(octave_binned.values.tolist())

    result = pd.DataFrame(result)
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    result["x_pos"] = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(8, 5))

    sub_tick_label_size = 12
    tick_label_size = 14
    yaxis_tick_size = 16
    label_size = 20
    band_label_offset_y = 0.08
    ylim_min, ylim_max, ytick_min, ytick_max = _efficiency_ylim(result["value"].tolist())

    offset_map = {"L": -0.2, "R": 0.2}
    cochleas = sorted({name_lr[:-1] for name_lr in result["cochlea"].unique()})

    color_map = {}
    for name_lr in sorted(result["cochlea"].unique()):
        color_map[name_lr] = COLOR_LEFT if name_lr[-1] == "L" else COLOR_RIGHT

    legend_added = set()
    offset = 0.018
    trend_dict = {}

    for num, (name_lr, grp) in enumerate(result.groupby("cochlea")):
        name, side = name_lr[:-1], name_lr[-1]
        color = color_map[name_lr]

        x_positions = grp["x_pos"] + offset_map[side] - len(cochleas) / 2 * offset + offset * num
        ax.scatter(
            x_positions,
            grp["value"],
            label=name if name not in legend_added else None,
            s=60,
            alpha=0.8,
            marker=MARKER_LEFT if side == "L" else MARKER_RIGHT,
            color=color,
            zorder=1,
        )
        legend_added.add(name)

        if trendlines:
            sorted_idx = np.argsort(x_positions)
            x_sorted = np.array(x_positions)[sorted_idx]
            y_sorted = np.array(grp["value"])[sorted_idx]
            trend_dict[name_lr] = {"x_sorted": x_sorted, "y_sorted": y_sorted, "side": side}

    xlim_left, xlim_right = ax.get_xlim()

    if trendlines:
        trendline_width = 3

        x_sorted_r, _, _, _ = _get_trendline_params(trend_dict, "R")
        x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = _get_trendline_params(trend_dict, "L")
        min_x = min(min(x_sorted_r), min(x_sorted))
        max_x = max(max(x_sorted_r), max(x_sorted))
        x_sorted.insert(0, min_x)
        x_sorted.append(max_x)
        y_sorted.insert(0, y_sorted[0])
        y_sorted.append(y_sorted[-1])

        ax.plot(x_sorted, y_sorted, linestyle="dashed", color=COLOR_LEFT, alpha=0.6,
                linewidth=trendline_width, zorder=2)

        if trendline_std:
            y_sorted_lower.insert(0, y_sorted_lower[0])
            y_sorted_lower.append(y_sorted_lower[-1])
            y_sorted_upper.insert(0, y_sorted_upper[0])
            y_sorted_upper.append(y_sorted_upper[-1])
            ax.plot(x_sorted, y_sorted_upper, linestyle="solid", color=COLOR_LEFT, alpha=0.08, zorder=0)
            ax.plot(x_sorted, y_sorted_lower, linestyle="solid", color=COLOR_LEFT, alpha=0.08, zorder=0)
            ax.fill_between(x_sorted, y_sorted_lower, y_sorted_upper,
                            color=COLOR_LEFT, alpha=0.05, interpolate=True)

        x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = _get_trendline_params(trend_dict, "R")
        x_sorted.insert(0, min_x)
        x_sorted.append(max_x)
        y_sorted.insert(0, y_sorted[0])
        y_sorted.append(y_sorted[-1])

        ax.plot(x_sorted, y_sorted, linestyle="dotted", color=COLOR_RIGHT, alpha=0.7,
                linewidth=trendline_width, zorder=0)

        if trendline_std:
            y_sorted_lower.insert(0, y_sorted_lower[0])
            y_sorted_lower.append(y_sorted_lower[-1])
            y_sorted_upper.insert(0, y_sorted_upper[0])
            y_sorted_upper.append(y_sorted_upper[-1])
            ax.plot(x_sorted, y_sorted_upper, linestyle="solid", color=COLOR_RIGHT, alpha=0.08, zorder=0)
            ax.plot(x_sorted, y_sorted_lower, linestyle="solid", color=COLOR_RIGHT, alpha=0.08, zorder=0)
            ax.fill_between(x_sorted, y_sorted_lower, y_sorted_upper,
                            color=COLOR_RIGHT, alpha=0.05, interpolate=True)

    plt.xlim(xlim_left, xlim_right)
    main_ticks = range(len(bin_labels))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_1))
    ax.set_ylim(ylim_min, ylim_max)
    plt.yticks(np.arange(ytick_min, ytick_max + 1e-9, 0.1), fontsize=yaxis_tick_size)
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

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for the gerbil ChReef figure of the cochlea paper.")
    parser.add_argument(
        "--figure_dir", "-f", type=str, help="Output directory for plots.",
        default="./panels/fig_gerbil_chreef",
    )
    parser.add_argument("--no_alias", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--refresh_cache", action="store_true",
        help="Ignore the cached pickles and fetch the tables and density files from S3 again.",
    )
    parser.add_argument(
        "--no_overlap_report", action="store_true",
        help="Do not print the position overlap of the density data.",
    )
    args = parser.parse_args()

    use_alias = not args.no_alias
    os.makedirs(args.figure_dir, exist_ok=True)

    chreef_data = get_gerbil_chreef_data(force_download=args.refresh_cache)
    density_data = get_gerbil_density_data(force_download=args.refresh_cache)

    if not args.no_overlap_report:
        for cochlea, results in density_data.items():
            report_density_overlap(results, name=cochlea)

    plot_legend_gerbil(chreef_data, save_path=os.path.join(args.figure_dir, f"fig_gerbil_legend.{FILE_EXTENSION}"))
    plot_legend_fig04_trendline(
        save_path=os.path.join(args.figure_dir, f"fig_gerbil_legend_trendline.{FILE_EXTENSION}")
    )

    # C: SGN density at apex/mid/base, Injected vs Non-Injected.
    fig_c_gerbil(density_data,
                 save_path=os.path.join(args.figure_dir, f"fig_c_gerbil_density.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias)
    fig_c_gerbil(density_data,
                 save_path=os.path.join(args.figure_dir, f"fig_c_gerbil_density_std.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias, show_std=True)

    # D: The expression efficiency per cochlea.
    fig_d_gerbil(chreef_data,
                 save_path=os.path.join(args.figure_dir, f"fig_d_gerbil_transduction.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias)

    # E: The expression efficiency per octave band.
    fig_e_gerbil(chreef_data,
                 save_path=os.path.join(args.figure_dir, f"fig_e_gerbil_transduction.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias, trendlines=True)
    fig_e_gerbil(chreef_data,
                 save_path=os.path.join(args.figure_dir, f"fig_e_gerbil_transduction_std.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias, trendlines=True, trendline_std=True)


if __name__ == "__main__":
    main()
