import argparse
import json
import os
import pickle
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import frequency_mapping, prism_style, prism_cleanup_axes
from util import export_legend, custom_formatter, get_marker_handle, get_trendline_handle
from util import animal_colors, cochlea_label
from util import COLOR_LEFT, COLOR_RIGHT, COLOR_UNTREATED, VALUE_DICT, cochleae_for, cohort_cochleae

# from statsmodels.nonparametric.smoothers_lowess import lowess

INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements2"  # noqa

# The ChReef cohort, plus the two G_EK_000049 gerbils that are plotted alongside it.
# The metadata lives in util.COCHLEA_DICT.
COCHLEAE = cohort_cochleae("chreef_mouse") + ["G_EK_000049_L", "G_EK_000049_R"]

COCHLEAE_DICT = cochleae_for(COCHLEAE, "SGN", "SGN_v2")

REFERENCE_COCHLEAE = cohort_cochleae("idisco")

# The cochleae for the OTOF gene therapy. The metadata lives in util.COCHLEA_DICT.
# The left cochlea is treated, the right one is the untreated control.
OTOF_COCHLEAE = cohort_cochleae("otof_mouse")

OTOF_COCHLEAE_DICT = cochleae_for(OTOF_COCHLEAE, "IHC", "IHC_v11")

FILE_EXTENSION = "png"
png_dpi = 300

MARKER_LEFT = "o"
MARKER_RIGHT = "^"


def get_chreef_data(
    animal: str = "mouse",
):
    """Create (pickled) dictionary for mouse or gerbil cochleae used for optogenetic therapy.
    """
    s3 = create_s3_target()
    source_name = "SGN_v2"

    if animal == "mouse":
        cache_path = "./chreef_data.pkl"
        cochleae = [key for key in COCHLEAE_DICT.keys() if "M_" in key]
    else:
        cache_path = "./chreef_data_gerbil.pkl"
        cochleae = [key for key in COCHLEAE_DICT.keys() if "G_" in key]

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chreef_data = {}
    for cochlea in cochleae:
        print("Processsing cochlea:", cochlea)
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the seg table and filter the compartments.
        source = sources[source_name]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # May need to be adjusted for some cochleae.
        component_labels = COCHLEAE_DICT[cochlea]["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        # The relevant values for analysis.
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "marker_labels"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skippped.")
            continue

        fname = f"{cochlea.replace('_', '-')}_GFP_SGN-v2_object-measures.tsv"
        intensity_file = os.path.join(INTENSITY_ROOT, fname)
        assert os.path.exists(intensity_file), intensity_file
        intensity_table = pd.read_csv(intensity_file, sep="\t")
        values = values.merge(intensity_table, on="label_id")

        chreef_data[cochlea] = values

    with open(cache_path, "wb") as f:
        pickle.dump(chreef_data, f)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_reference_counts(
    structure: str,
    source_name: str,
    cochleae: List[str] = REFERENCE_COCHLEAE,
):
    """Get reference counts for healthy cochleae from util.VALUE_DICT.

    Args:
        structure: Structure key in VALUE_DICT, e.g. "SGN" or "IHC".
        source_name: Segmentation source key, e.g. "SGN_v2" or "IHC_v11".
        cochleae: Reference cochleae to read counts for.

    Returns:
        One count per cochlea, in the given order.
    """
    return [VALUE_DICT[cochlea][structure][source_name]["count"] for cochlea in cochleae]


def get_otof_data(
    cache_path: str = "./otof_data.pkl",
):
    """Create (pickled) dictionary of IHC_v11 measurements for OTOF gene-therapy cochleae.
    """
    s3 = create_s3_target()
    source_name = "IHC_v11"

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    otof_data = {}
    for cochlea, meta in OTOF_COCHLEAE_DICT.items():
        print("Processsing cochlea:", cochlea)
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        source = sources[source_name]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        component_labels = meta["component"]
        print(cochlea, component_labels)
        table = table[table.component_labels.isin(component_labels)]
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "marker_labels"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skipped.")
            continue

        otof_data[cochlea] = values

    with open(cache_path, "wb") as f:
        pickle.dump(otof_data, f)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def group_lr(
    names_lr: List[str],
    values: List[float],
):
    """Group values of left and right cochleae.

    Args:
        names_lr: List of cochleae names or aliases with "L" or "R" appendix.
        values: List of values.

    Returns:
        Sorted animal names or aliases.
        Values of left cochleae.
        Values of right cochleae.
    """
    assert len(names_lr) == len(values)
    names = []
    values_left, values_right = {}, {}
    for name_lr, val in zip(names_lr, values):
        name, side = name_lr[:-1], name_lr[-1]
        names.append(name)
        if side == "R":
            values_right[name] = val
        elif side == "L":
            values_left[name] = val
        else:
            raise RuntimeError
    names = sorted(list(set(names)))

    values_left = [values_left.get(name, np.nan) for name in names]
    values_right = [values_right.get(name, np.nan) for name in names]

    return names, values_left, values_right


def plot_legend_fig04(
    data: dict,
    save_path: str,
    use_alias: bool = True,
    alignment: str = "horizontal",
    cochleae_dict: Optional[Dict] = None,
):
    """Plot common legend for Figures 4c, 4d, and 4e.

    Args:
        data: Data of the cochleae to show, mapping cochlea name to measurements.
        save_path: File path to save legend.
        use_alias: Use alias.
        alignment: Alignment of legend.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
            Defaults to the module-level COCHLEAE_DICT (ChReef cochleae).
    """
    cochleae_dict = cochleae_dict if cochleae_dict is not None else COCHLEAE_DICT
    colors_by_animal = animal_colors(cochleae_dict, use_alias)

    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in data.keys()]
    alias, _, _ = group_lr(alias, [0] * len(alias))

    colors = []
    labels = []
    markers = []
    ncol = len(alias)
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

    handles = [get_marker_handle(c, m) for (c, m) in zip(colors, markers)]
    legend = plt.legend(handles, labels, loc=3, ncol=ncol, framealpha=1, frameon=False)

    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def plot_legend_fig04_trendline(
    save_path: str,
):
    """Plot legend for Figure 4 for trendlines used in Figure 4e.

    Args:
        save_path: Path for output.
    """
    labels = ["Injected", "Non-Injected"]
    linestyles = ["dashed", "dotted"]
    lw = 3
    linewidth = [lw for _ in labels]
    handlelength = lw * 1.5

    handles = [get_trendline_handle(style, width) for (style, width) in zip(linestyles, linewidth)]
    legend = plt.legend(handles, labels, loc=3, ncol=1, framealpha=1, handlelength=handlelength, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def plot_legend_fig05e_gerbil(
    save_path: str,
):
    """Plot common legend for figure 5e gerbil.

    Args:
        save_path: Path for output.
    """
    # Shapes
    color = [COLOR_LEFT, COLOR_RIGHT]
    marker = ["o", "^"]
    label = ["G1L", "G1R"]

    handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]
    legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def fig_04c(
    chreef_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    cochleae_dict: Optional[Dict] = None,
    count_label: str = "SGN count per cochlea",
    xtick_labels: Tuple[str, str] = ("Injected", "Non-\nInjected"),
    ylim: Tuple[float, float] = (5000, 14000),
    y_ticks: Optional[List[float]] = None,
    reference_values: Optional[List[float]] = None,
):
    """Box plot showing the SGN counts of ChReef treated cochleae compared to healthy ones.

    Args:
        chreef_data: Data of ChReef cochleae.
        save_path: File path to save legend.
        plot: Plot figure.
        use_alias: Use alias.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
            Defaults to the module-level COCHLEAE_DICT (ChReef cochleae).
        count_label: Y-axis label.
        xtick_labels: Labels for the left (treated) and right (untreated) x positions.
        ylim: Lower and upper y-axis limit.
        y_ticks: Y-axis tick positions. Defaults to the ChReef SGN-count ticks.
        reference_values: Reference counts for the "untreated cochleae 95% CI" band. Defaults to
            the SGN_v2 counts of the healthy iDISCO cochleae in util.VALUE_DICT.
    """
    prism_style()
    cochleae_dict = cochleae_dict if cochleae_dict is not None else COCHLEAE_DICT
    colors_by_animal = animal_colors(cochleae_dict, use_alias)

    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in chreef_data.keys()]

    sgns = [len(vals) for vals in chreef_data.values()]

    alias, values_left, values_right = group_lr(alias, sgns)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16
    fontsize_untreated = 16

    offset = 0.08
    x_left = 1
    x_right = 2
    y_ticks = y_ticks if y_ticks is not None else list(range(6000, 13000, 2000))

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

    for num, a in enumerate(alias):
        plt.scatter(x_pos_inj[num], values_left[num], label=a,
                    color=colors_by_animal[a], marker=MARKER_LEFT, s=80, zorder=1)
        plt.scatter(x_pos_non[num], values_right[num],
                    color=colors_by_animal[a], marker=MARKER_RIGHT, s=80, zorder=1)

    # Labels and formatting
    plt.xticks([x_left, x_right], list(xtick_labels), fontsize=sub_label_size)
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment('center')
    ax.tick_params(axis='x', which='major', pad=16)
    plt.yticks(y_ticks, fontsize=main_tick_size)
    plt.ylabel(count_label, fontsize=main_label_size)
    plt.ylim(*ylim)

    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    reference_values = reference_values if reference_values is not None else get_reference_counts("SGN", "SGN_v2")
    sgn_value = np.mean(reference_values)
    sgn_std = np.std(reference_values)

    upper_y = sgn_value + 1.96 * sgn_std
    lower_y = sgn_value - 1.96 * sgn_std

    c_untreated = COLOR_UNTREATED

    plt.hlines([lower_y, upper_y], xmin, xmax, colors=[c_untreated for _ in range(2)], zorder=-1)
    text_offset = (ylim[1] - ylim[0]) / 40
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


def fig_04d(
    chreef_data: dict,
    save_path: str,
    plot: bool = False,
    intensity: bool = False,
    gerbil: bool = False,
    use_alias: bool = True,
    cochleae_dict: Optional[Dict] = None,
    xtick_labels: Tuple[str, str] = ("Injected", "Non-\nInjected"),
    ylim: Optional[Tuple[float, float]] = None,
    y_ticks: Optional[List[float]] = None,
):
    """Expression efficiency per cochlea.

    Args:
        chreef_data: Data of ChReef cochleae.
        save_path: File path to save legend.
        plot: Plot figure.
        intensity: Use intensity instead of expression efficiency.
        gerbil: Use gerbil data instead of mouse data.
        use_alias: Use alias.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
            Defaults to the module-level COCHLEAE_DICT (ChReef cochleae).
        xtick_labels: Labels for the left (treated) and right (untreated) x positions.
        ylim: Lower and upper y-axis limit. Defaults to the gerbil/mouse ChReef bounds.
        y_ticks: Y-axis tick positions. Defaults to the gerbil/mouse ChReef ticks.
    """
    prism_style()
    cochleae_dict = cochleae_dict if cochleae_dict is not None else COCHLEAE_DICT
    colors_by_animal = animal_colors(cochleae_dict, use_alias)

    alias = [cochlea_label(name, cochleae_dict[name], use_alias) for name in chreef_data.keys()]

    values = []
    for vals in chreef_data.values():
        if intensity:
            intensities = vals["median"].values
            values.append(intensities.mean())
        else:
            # marker labels
            # 0: unlabeled - no median intensity in object_measures table
            # 1: positive
            # 2: negative
            marker_labels = vals["marker_labels"].values
            n_pos = (marker_labels == 1).sum()
            n_neg = (marker_labels == 2).sum()
            eff = float(n_pos) / (n_pos + n_neg)
            values.append(eff)

    alias, values_left, values_right = group_lr(alias, values)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16

    label = "Intensity" if intensity else "Expression efficiency"
    x_left = 1
    x_right = 2
    offset = 0.08

    x_pos_inj = [x_left - len(values_left) // 2 * offset + offset * i for i in range(len(values_left))]
    x_pos_non = [x_right - len(values_right) // 2 * offset + offset * i for i in range(len(values_right))]

    for num, a in enumerate(alias):
        plt.scatter(x_pos_inj[num], values_left[num], label=a,
                    color=colors_by_animal[a], marker=MARKER_LEFT, s=80, zorder=1)
        plt.scatter(x_pos_non[num], values_right[num],
                    color=colors_by_animal[a], marker=MARKER_RIGHT, s=80, zorder=1)

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

    if not intensity:
        if ylim is not None:
            plt.ylim(*ylim)
            if y_ticks is not None:
                plt.yticks(y_ticks, fontsize=main_tick_size)
        elif gerbil:
            plt.ylim(0.25, 0.65)
            plt.yticks(np.arange(0.3, 0.7, 0.1), fontsize=main_tick_size)
        else:
            plt.ylim(0.65, 1.05)
            plt.yticks(np.arange(0.7, 1, 0.1), fontsize=main_tick_size)

    # Labels and formatting
    plt.xticks([x_left, x_right], list(xtick_labels), fontsize=sub_label_size)
    for la in plt.gca().get_xticklabels():
        la.set_verticalalignment('center')
    ax.tick_params(axis='x', which='major', pad=16)
    plt.ylabel(label, fontsize=main_label_size)
    ax.yaxis.set_major_formatter(custom_formatter(1))

    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    # plt.legend(loc="upper right", fontsize=legendsize)

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


def fig_04e(
    chreef_data: dict,
    save_path: str,
    plot: bool = False,
    intensity: bool = False,
    gerbil: bool = False,
    use_alias: bool = True,
    trendlines: bool = False,
    trendline_std: bool = False,
    cochleae_dict: Optional[Dict] = None,
):
    """Expression efficiency per octave band for cochleae.

    Args:
        chreef_data: Data of ChReef cochleae.
        save_path: File path to save legend.
        plot: Plot figure.
        intensity: Use intensity instead of expression efficiency.
        gerbil: Use gerbil data instead of mouse data.
        use_alias: Use alias.
        trendlines: Use trendline of averages.
        trendline_std: Use standard deviation for upper and lower trendlines.
        cochleae_dict: Mapping of cochlea name to its metadata, as returned by util.cochleae_for.
            Defaults to the module-level COCHLEAE_DICT (ChReef cochleae).
    """
    prism_style()
    cochleae_dict = cochleae_dict if cochleae_dict is not None else COCHLEAE_DICT

    if gerbil:
        animal = "gerbil"
    else:
        animal = "mouse"

    result = {"cochlea": [], "octave_band": [], "value": []}
    aliases = []
    for name, values in chreef_data.items():
        alias = cochlea_label(name, cochleae_dict[name], use_alias)

        freq = values["frequency[kHz]"].values
        if intensity:
            intensity_values = values["median"].values
            octave_binned = frequency_mapping(freq, intensity_values, animal=animal)
        else:
            marker_labels = values["marker_labels"].values
            octave_binned = frequency_mapping(freq, marker_labels, animal=animal, transduction_efficiency=True)

        result["cochlea"].extend([alias] * len(octave_binned))
        result["octave_band"].extend(octave_binned.axes[0].values.tolist())
        result["value"].extend(octave_binned.values.tolist())
        aliases.append(alias)

    if gerbil:
        values = []
        for vals in chreef_data.values():
            if intensity:
                intensities = vals["median"].values
                values.append(intensities.mean())
            else:
                # marker labels
                # 0: unlabeled - no median intensity in object_measures table
                # 1: positive
                # 2: negative
                marker_labels = vals["marker_labels"].values
                n_pos = (marker_labels == 1).sum()
                n_neg = (marker_labels == 2).sum()
                eff = float(n_pos) / (n_pos + n_neg)
                values.append(eff)
        alias, values_left, values_right = group_lr(aliases, values)
        print(f"Average expression efficiency left: {round(values_left[0], 4)}")
        print(f"Average expression efficiency right: {round(values_right[0], 4)}")

    result = pd.DataFrame(result)
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    result["x_pos"] = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(8, 5))

    sub_tick_label_size = 12
    tick_label_size = 14
    yaxis_tick_size = 16
    label_size = 20

    if intensity:
        band_label_offset_y = 0.09
    else:
        band_label_offset_y = 0.08
        if gerbil:
            ymin = 0.1
            ymax = 0.81
            ax.set_ylim(0.05, 0.95)
        else:
            ymin = 0.5
            ymax = 1.01
            ax.set_ylim(0.45, 1.05)

    # Offsets within each octave band
    offset_map = {"L": -0.2, "R": 0.2}

    # Assign a color to each cochlea (ignoring side)
    cochleas = sorted({name_lr[:-1] for name_lr in result["cochlea"].unique()})

    if gerbil:
        color_map = {name_lr: COLOR_LEFT if name_lr.endswith("L") else COLOR_RIGHT
                     for name_lr in result["cochlea"].unique()}
    else:
        colors_by_animal = animal_colors(cochleae_dict, use_alias)
        color_map = {name_lr: colors_by_animal[name_lr[:-1]] for name_lr in result["cochlea"].unique()}

    if len(cochleas) == 1:
        color_map = {"L": COLOR_LEFT, "R": COLOR_RIGHT}

    # Track which cochlea names we have already added to the legend
    legend_added = set()

    offset = 0.018
    trend_dict = {}

    for num, (name_lr, grp) in enumerate(result.groupby("cochlea")):
        name, side = name_lr[:-1], name_lr[-1]
        if len(cochleas) == 1:
            label_name = name_lr
            color = color_map[side]
        else:
            label_name = name
            color = color_map[name_lr]

        x_positions = grp["x_pos"] + offset_map[side] - len(cochleas) / 2 * offset + offset * num
        ax.scatter(
            x_positions,
            grp["value"],
            label=label_name if label_name not in legend_added else None,
            s=60,
            alpha=0.8,
            marker=MARKER_LEFT if side == "L" else MARKER_RIGHT,
            color=color,
            zorder=1
        )

        if name not in legend_added:
            legend_added.add(name)

        if trendlines:
            sorted_idx = np.argsort(x_positions)
            x_sorted = np.array(x_positions)[sorted_idx]
            y_sorted = np.array(grp["value"])[sorted_idx]
            trend_dict[name_lr] = {"x_sorted": x_sorted,
                                   "y_sorted": y_sorted,
                                   "side": side,
                                   }

    xlim_left, xlim_right = ax.get_xlim()
    if trendlines:
        trendline_width = 3
        if not gerbil:
            x_sorted_r, _, _, _ = _get_trendline_params(trend_dict, "R")
            x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = _get_trendline_params(trend_dict, "L")
            min_x = min([min(x_sorted_r), min(x_sorted)])
            max_x = max([max(x_sorted_r), max(x_sorted)])
            x_sorted.insert(0, min_x)
            x_sorted.append(max_x)
            y_sorted.insert(0, y_sorted[0])
            y_sorted.append(y_sorted[-1])

            if gerbil:
                color_trend_l = COLOR_LEFT
                color_trend_r = COLOR_RIGHT
            else:
                color_trend_l = "gray"
                color_trend_r = "gray"

            # central line
            trend_l, = ax.plot(
                x_sorted,
                y_sorted,
                linestyle="dashed",
                color=color_trend_l,
                alpha=0.6,
                linewidth=trendline_width,
                zorder=2,
            )

            if trendline_std:
                y_sorted_lower.insert(0, y_sorted_lower[0])
                y_sorted_lower.append(y_sorted_lower[-1])
                y_sorted_upper.insert(0, y_sorted_upper[0])
                y_sorted_upper.append(y_sorted_upper[-1])
                # upper and lower standard deviation
                trend_l_upper, = ax.plot(
                    x_sorted,
                    y_sorted_upper,
                    linestyle="solid",
                    color=color_trend_l,
                    alpha=0.08,
                    zorder=0
                )
                trend_l_lower, = ax.plot(
                    x_sorted,
                    y_sorted_lower,
                    linestyle="solid",
                    color=color_trend_l,
                    alpha=0.08,
                    zorder=0
                )
                plt.fill_between(x_sorted, y_sorted_lower, y_sorted_upper,
                                 color=COLOR_LEFT, alpha=0.05, interpolate=True)

            # Trendline Non-Injected (Right)
            x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = _get_trendline_params(trend_dict, "R")
            x_sorted.insert(0, min_x)
            x_sorted.append(max_x)
            y_sorted.insert(0, y_sorted[0])
            y_sorted.append(y_sorted[-1])
            # central line
            trend_r, = ax.plot(
                x_sorted,
                y_sorted,
                linestyle="dotted",
                color=color_trend_r,
                alpha=0.7,
                linewidth=trendline_width,
                zorder=0
            )

            if trendline_std:
                y_sorted_lower.insert(0, y_sorted_lower[0])
                y_sorted_lower.append(y_sorted_lower[-1])
                y_sorted_upper.insert(0, y_sorted_upper[0])
                y_sorted_upper.append(y_sorted_upper[-1])
                # upper and lower standard deviation
                trend_r_upper, = ax.plot(
                    x_sorted,
                    y_sorted_upper,
                    linestyle="solid",
                    color=color_trend_r,
                    alpha=0.08,
                    zorder=0
                )
                trend_r_lower, = ax.plot(
                    x_sorted,
                    y_sorted_lower,
                    linestyle="solid",
                    color=color_trend_r,
                    alpha=0.08,
                    zorder=0
                )
                plt.fill_between(x_sorted, y_sorted_lower, y_sorted_upper,
                                 color=COLOR_RIGHT, alpha=0.05, interpolate=True)

        else:
            x_sorted = [trend_dict[k]["x_sorted"] for k in trend_dict.keys() if trend_dict[k]["side"] == "L"][0]
            y_left = [values_left[0], values_left[0]]
            y_right = [values_right[0], values_right[0]]
            if gerbil:
                color_trend_l = COLOR_LEFT
                color_trend_r = COLOR_RIGHT
            else:
                color_trend_l = "gray"
                color_trend_r = "gray"

            trend_l, = ax.plot(
                [xlim_left, xlim_right],
                y_left,
                linestyle="dotted",
                color=color_trend_l,
                alpha=0.7,
                zorder=0
            )
            x_offset = 0.5
            y_offset = 0.01
            ax.text(xlim_left + x_offset, y_left[0] + y_offset, "mean",
                    color=color_trend_l, fontsize=tick_label_size, ha="center")
            ax.text(xlim_left + x_offset, y_right[0] + y_offset, "mean",
                    color=color_trend_r, fontsize=tick_label_size, ha="center")
            x_sorted = [trend_dict[k]["x_sorted"] for k in trend_dict.keys() if trend_dict[k]["side"] == "R"][0]
            trend_r, = ax.plot(
                [xlim_left, xlim_right],
                y_right,
                linestyle="dashed",
                color=color_trend_r,
                alpha=0.7,
                zorder=0
            )

    plt.xlim(xlim_left, xlim_right)
    # Create combined tick positions & labels
    main_ticks = range(len(bin_labels))
    ax.yaxis.set_major_formatter(custom_formatter(1))
    plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=yaxis_tick_size)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    # add a final tick for label '>64k'
    ax.set_xticks([pos + offset_map["L"] for pos in main_ticks] +
                  [pos + offset_map["R"] for pos in main_ticks])
    ax.set_xticklabels(["I"] * len(main_ticks) + ["N"] * len(main_ticks), fontsize=sub_tick_label_size)

    # Add main octave band labels above sublabels
    for i, label in enumerate(bin_labels):
        ax.text(i, ax.get_ylim()[0] - band_label_offset_y * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                label, ha='center', va='top', fontsize=tick_label_size, fontweight='bold')

    ax.set_xlabel("Octave band [kHz]", fontsize=label_size)
    ax.xaxis.set_label_coords(.5, -.16)

    if intensity:
        ax.set_ylabel("Marker Intensity", fontsize=label_size)
        ax.set_title("Intensity per octave band (Left/Right)")
    else:
        ax.set_ylabel("Expression efficiency", fontsize=label_size)

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
    parser = argparse.ArgumentParser(description="Generate plots for Fig 4 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig4")
    parser.add_argument("--no_alias", action="store_true")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    use_alias = not args.no_alias
    os.makedirs(args.figure_dir, exist_ok=True)

    # Get the chreef data as a dictionary of cochlea name to measurements.
    chreef_data = get_chreef_data()
    # M_LR_00143_L is a complete outlier
    chreef_data.pop("M_LR_000143_L")
    # remove other cochlea to have only pairs remaining
    chreef_data.pop("M_LR_000143_R")

    # Create the panels:
    plot_legend_fig04(chreef_data, save_path=os.path.join(args.figure_dir, f"fig_04_legend.{FILE_EXTENSION}"))

    plot_legend_fig04_trendline(save_path=os.path.join(args.figure_dir, f"fig_04_legend_trendline.{FILE_EXTENSION}"))

    # C: The SGN count compared to reference values from literature and healthy
    # Maybe remove literature reference from plot?
    fig_04c(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04c.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias)

    # D: The expression efficiency. We also plot GFP intensities.
    fig_04d(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04d_transduction.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias)

    # E: The expression efficiency per octave band.
    # trendlines without standard deviation
    fig_04e(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04e_transduction.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, trendlines=True)
    # trendlines with standard deviation
    fig_04e(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04e_transduction_std.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, trendlines=True, trendline_std=True)

    # OTOF gene therapy: IHC counts and expression efficiency, treated (L) vs. untreated (R).
    otof_data = get_otof_data()
    otof_reference_ihc = get_reference_counts("IHC", "IHC_v11")

    fig_04c(otof_data,
            save_path=os.path.join(args.figure_dir, f"fig_06e_otof_ihc.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, cochleae_dict=OTOF_COCHLEAE_DICT,
            count_label="IHC count per cochlea", xtick_labels=("Injected", "Non-Injected"),
            ylim=(500, 750), y_ticks=list(range(500, 800, 50)),
            reference_values=otof_reference_ihc)

    fig_04d(otof_data,
            save_path=os.path.join(args.figure_dir, f"fig_06e_otof_expression.{FILE_EXTENSION}"),
            plot=args.plot, use_alias=use_alias, cochleae_dict=OTOF_COCHLEAE_DICT,
            xtick_labels=("Injected", "Non-Injected"),
            ylim=(-0.03, 0.4), y_ticks=[0.0, 0.1, 0.2, 0.3])

    plot_legend_fig04(otof_data, cochleae_dict=OTOF_COCHLEAE_DICT,
                      save_path=os.path.join(args.figure_dir, f"fig_06_otof_legend.{FILE_EXTENSION}"))

    # Figures for gerbil (Figure 5)
    chreef_data_gerbil = get_chreef_data(animal="gerbil")
    fig_04e(chreef_data_gerbil,
            save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_transduction.{FILE_EXTENSION}"),
            plot=args.plot, gerbil=True, use_alias=use_alias, trendlines=True)

    plot_legend_fig05e_gerbil(save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_legend.{FILE_EXTENSION}"))


if __name__ == "__main__":
    main()
