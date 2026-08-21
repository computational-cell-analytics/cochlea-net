import argparse
import json
import math
import os
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
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
    cochleae_for,
    cohort_cochleae,
    cohort_postnatal,
    COLOR_UNTREATED,
    custom_formatter,
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

SGN_CACHE_PATH = "./gerbil_fchrimson_sgn_data.pkl"
DENSITY_CACHE_PATH = "./gerbil_fchrimson_density_data.pkl"

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

# The gerbil cochleae for the f-Chrimson analysis. The metadata lives in util.COCHLEA_DICT.
COCHLEAE = cohort_cochleae("fchrimson_gerbil")

COCHLEAE_DICT = cochleae_for(COCHLEAE, "SGN", SOURCE_NAME)

# G_EK_000049 received the injection postnatally, the other three animals as adults. The two
# groups are not comparable, so panel E can plot them apart and average only the adults.
POSTNATAL_COCHLEAE = cohort_postnatal("fchrimson_gerbil")
ADULT_COCHLEAE = [name for name in COCHLEAE if name not in POSTNATAL_COCHLEAE]

# Untreated gerbil cochleae, plotted as the SGN density reference of panel C.
WT_COCHLEAE = cohort_cochleae("wt_gerbil")

FILE_EXTENSION = "png"

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


def get_gerbil_fchrimson_data(force_download: bool = False) -> Dict[str, pd.DataFrame]:
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
    fchrimson_data = {}
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

        fchrimson_data[cochlea] = values

    with open(SGN_CACHE_PATH, "wb") as f:
        pickle.dump(fchrimson_data, f)
    return fchrimson_data


def get_gerbil_density_data(force_download: bool = False) -> Dict[str, dict]:
    """Create (pickled) dictionary of the SGN density data for the gerbil cochleae.

    Both the f-Chrimson and the untreated cochleae are loaded, so that panel C and its reference
    band come from one cache. The extended density file is preferred, because it holds several
    positions per cochlear region. Cochleae without it fall back to the preset file with one
    position per region.

    Args:
        force_download: Ignore the cached pickle and fetch the density files from S3 again.

    Returns:
        Mapping of cochlea name to its parsed density dict, keyed by position.
    """
    cochleae = list(COCHLEAE) + list(WT_COCHLEAE)
    if not force_download and os.path.exists(DENSITY_CACHE_PATH):
        with open(DENSITY_CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        # A cache written for a smaller cohort would raise a KeyError downstream instead.
        if all(cochlea in cached for cochlea in cochleae):
            return cached

    s3 = create_s3_target()
    density_data = {}
    for cochlea in cochleae:
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
    fchrimson_data: dict,
    save_path: str,
    use_alias: bool = True,
    alignment: str = "horizontal",
):
    """Plot common legend for the gerbil f-Chrimson figure panels.

    Args:
        fchrimson_data: Data of f-Chrimson gerbil cochleae.
        save_path: File path to save legend.
        use_alias: Use alias.
        alignment: Alignment of legend.
    """
    colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)

    alias = [cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in fchrimson_data.keys()]
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


def plot_legend_fig05e_gerbil(
    save_path: str,
    color: Optional[List[str]] = None,
    label: Optional[List[str]] = None,
    marker: Optional[List[str]] = None,
):
    """Plot the legend of a figure 5e panel that shows a single cochlea pair.

    Args:
        save_path: File path to save legend.
        color: One color per entry. Defaults to the injected and non-injected side colors.
        label: One label per entry. Defaults to the aliases of the postnatal cochleae.
        marker: One marker per entry. Defaults to the injected and non-injected markers.
    """
    color = [COLOR_LEFT, COLOR_RIGHT] if color is None else color
    if label is None:
        label = [COCHLEA_DICT[name]["alias"] for name in POSTNATAL_COCHLEAE]
    marker = [MARKER_LEFT, MARKER_RIGHT] if marker is None else marker

    handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]
    legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


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
        density_data: Parsed SGN density data of f-Chrimson gerbil cochleae.
        save_path: File path to save the figure.
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

    colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)
    alias = [cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in density_data.keys()]

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
    fchrimson_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
):
    """Expression efficiency per gerbil cochlea, Injected vs Non-Injected.

    Args:
        fchrimson_data: Data of f-Chrimson gerbil cochleae.
        save_path: File path to save the figure.
        plot: Plot figure.
        use_alias: Use alias.
    """
    prism_style()
    colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)
    alias = [cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in fchrimson_data.keys()]

    values = []
    for vals in fchrimson_data.values():
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
    ax.yaxis.set_major_formatter(custom_formatter(1))

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


def _efficiency_by_band(fchrimson_data: dict, use_alias: bool = True) -> pd.DataFrame:
    """Expression efficiency per octave band, one row per cochlea and band.

    Args:
        fchrimson_data: Data of f-Chrimson gerbil cochleae.
        use_alias: Use alias.

    Returns:
        Table with the columns cochlea, octave_band, value and x_pos, where x_pos is the index
        of the octave band.
    """
    result = {"cochlea": [], "octave_band": [], "value": []}
    for name, values in fchrimson_data.items():
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
    return result


def _band_x_positions(result: pd.DataFrame, offset_map: dict, offset: float) -> pd.Series:
    """Assign the x position of every point, jittered inside the column of its side.

    The positions are derived from every cochlea of the cohort, not only from the plotted ones,
    so a trendline of cochleae that are left out still spans the column they would occupy.

    Args:
        result: Table as returned by _efficiency_by_band.
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
        column = offset_map[alias[-1]] - n_animals / 2 * offset + offset * num
        x_positions[rows] = result.loc[rows, "x_pos"] + column
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
    x_sorted, y_center, y_upper, y_lower = _get_trendline_params(trend_dict, side)
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


def fig_05e(
    fchrimson_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    trendlines: bool = False,
    trendline_std: bool = False,
    cochleae: Optional[List[str]] = None,
    color_by_side: bool = False,
    adult_trendline: bool = False,
):
    """Expression efficiency per octave band for gerbil cochleae.

    Args:
        fchrimson_data: Data of f-Chrimson gerbil cochleae. Every cochlea is read, so that the
            adult trendline stays available when only a subset is plotted.
        save_path: File path to save the figure.
        plot: Plot figure.
        use_alias: Use alias.
        trendlines: Draw the injected and the non-injected mean of the plotted cochleae.
        trendline_std: Add the standard deviation to every drawn trendline.
        cochleae: Cochleae to plot. Defaults to every cochlea in fchrimson_data.
        color_by_side: Color the points by side instead of per animal. Use it when the plot holds
            a single pair, where the animal color carries no information.
        adult_trendline: Draw the mean of the adult injected cochleae as a dashed reference. The
            postnatal animal is left out of that mean, because it is not comparable to them.
    """
    prism_style()

    cochleae = list(fchrimson_data) if cochleae is None else cochleae

    result = _efficiency_by_band(fchrimson_data, use_alias)
    bin_labels = pd.unique(result["octave_band"])

    offset_map = {"L": -0.2, "R": 0.2}
    offset = 0.018
    result["x"] = _band_x_positions(result, offset_map, offset)

    plotted = sorted(cochlea_label(name, COCHLEAE_DICT[name], use_alias) for name in cochleae)
    adult_injected = [cochlea_label(name, COCHLEAE_DICT[name], use_alias)
                      for name in ADULT_COCHLEAE if name.endswith("_L")]

    fig, ax = plt.subplots(figsize=(8, 5))

    sub_tick_label_size = 12
    tick_label_size = 14
    yaxis_tick_size = 16
    label_size = 20
    band_label_offset_y = 0.08

    if color_by_side:
        color_map = {alias: COLOR_LEFT if alias.endswith("L") else COLOR_RIGHT for alias in plotted}
    else:
        colors_by_animal = animal_colors(COCHLEAE_DICT, use_alias)
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
            ax, trend_dict, "L", COLOR_LEFT, x_bounds,
            linestyle="dashed", alpha=0.6, zorder=2, show_std=trendline_std)
        axis_values += _draw_band_trendline(
            ax, trend_dict, "R", COLOR_RIGHT, x_bounds,
            linestyle="dotted", alpha=0.7, zorder=0, show_std=trendline_std)

    if adult_trendline:
        axis_values += _draw_band_trendline(
            ax, _trend_dict(result, adult_injected), "L", COLOR_LEFT, x_bounds,
            linestyle="dashed", alpha=0.6, zorder=2, show_std=trendline_std)

    ylim_min, ylim_max, ytick_min, ytick_max = _efficiency_ylim(axis_values)

    plt.xlim(xlim_left, xlim_right)
    main_ticks = range(len(bin_labels))
    ax.yaxis.set_major_formatter(custom_formatter(1))
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
    parser = argparse.ArgumentParser(
        description="Generate plots for the gerbil f-Chrimson figure of the cochlea paper."
    )
    parser.add_argument(
        "--figure_dir", "-f", type=str, help="Output directory for plots.",
        default="./panels/fig_gerbil_fchrimson",
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

    fchrimson_data = get_gerbil_fchrimson_data(force_download=args.refresh_cache)
    density_data = get_gerbil_density_data(force_download=args.refresh_cache)
    fchrimson_density = {name: density_data[name] for name in COCHLEAE}
    reference_values = untreated_reference_values({name: density_data[name] for name in WT_COCHLEAE})

    if not args.no_overlap_report:
        for cochlea, results in density_data.items():
            report_density_overlap(results, name=cochlea)

    plot_legend_gerbil(fchrimson_data, save_path=os.path.join(args.figure_dir, f"fig_gerbil_legend.{FILE_EXTENSION}"))
    plot_legend_fig04_trendline(
        save_path=os.path.join(args.figure_dir, f"fig_gerbil_legend_trendline.{FILE_EXTENSION}")
    )

    # C: SGN density at apex/mid/base, Injected vs Non-Injected, against the untreated band.
    fig_c_gerbil(fchrimson_density,
                 save_path=os.path.join(args.figure_dir, f"fig_c_gerbil_density.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias, reference_values=reference_values)
    fig_c_gerbil(fchrimson_density,
                 save_path=os.path.join(args.figure_dir, f"fig_c_gerbil_density_std.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias, show_std=True, reference_values=reference_values)

    # D: The expression efficiency per cochlea.
    fig_d_gerbil(fchrimson_data,
                 save_path=os.path.join(args.figure_dir, f"fig_d_gerbil_transduction.{FILE_EXTENSION}"),
                 plot=args.plot, use_alias=use_alias)

    # E: The expression efficiency per octave band, every cochlea against the adult injected mean.
    fig_05e(fchrimson_data,
            save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_transduction_all.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, adult_trendline=True)
    fig_05e(fchrimson_data,
            save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_transduction_std.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, adult_trendline=True, trendline_std=True)

    # The postnatal pair on its own, against the same adult injected mean.
    fig_05e(fchrimson_data,
            save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_transduction_postnatal.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, cochleae=POSTNATAL_COCHLEAE,
            color_by_side=True, adult_trendline=True)
    plot_legend_fig05e_gerbil(
        save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_legend_postnatal.{FILE_EXTENSION}"),
    )


if __name__ == "__main__":
    main()
