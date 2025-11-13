import argparse
import json
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import frequency_mapping, prism_style, prism_cleanup_axes
from util import export_legend, custom_formatter_1, get_marker_handle, get_trendline_handle

# from statsmodels.nonparametric.smoothers_lowess import lowess

INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements2"  # noqa

# The cochlea for the CHReef analysis.
COCHLEAE_DICT = {
    "M_LR_000143_L": {"alias": "M0L", "component": [1]},
    "M_LR_000144_L": {"alias": "M_05L", "component": [1]},
    "M_LR_000145_L": {"alias": "M_06L", "component": [1]},
    "M_LR_000153_L": {"alias": "M_07L", "component": [1, 2, 3]},
    "M_LR_000155_L": {"alias": "M_08L", "component": [1]},
    "M_LR_000189_L": {"alias": "M_09L", "component": [1]},
    "M_LR_000143_R": {"alias": "M0R", "component": [1]},
    "M_LR_000144_R": {"alias": "M_05R", "component": [1]},
    "M_LR_000145_R": {"alias": "M_06R", "component": [1]},
    "M_LR_000153_R": {"alias": "M_07R", "component": [1]},
    "M_LR_000155_R": {"alias": "M_08R", "component": [1]},
    "M_LR_000189_R": {"alias": "M_09R", "component": [1]},
    "G_EK_000049_L": {"alias": "G_1L", "component": [1, 3, 4, 5]},
    "G_EK_000049_R": {"alias": "G_1R", "component": [1, 2]},
}

COLORS_ANIMAL = {
    "M05": "#9C5027",
    "M06": "#279C52",
    "M07": "#67279C",
    "M08": "#27339C",
    "M09": "#9C276F"
}

COLORS_LEFT = {
    "M05R": "#A600FF",
    "M06R": "#8F00DB",
    "M07R": "#7D1DB1",
    "M08R": "#672D86",
    "M09R": "#4C2E5C"
}

COLORS_RIGHT = {
    "M05L": "#FF0063",
    "M06L": "#DB0063",
    "M07L": "#B11D60",
    "M08L": "#862D55",
    "M09L": "#5C2E43"
}

FILE_EXTENSION = "png"
png_dpi = 300

COLOR_LEFT = "#8E00DB"
COLOR_RIGHT = "#DB0063"
COLOR_UNTREATED = "#DB7B00"
MARKER_LEFT = "o"
MARKER_RIGHT = "^"


def get_chreef_data(animal="mouse"):
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


def group_lr(names_lr, values):
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


def plot_legend(chreef_data, save_path, grouping="side_mono", use_alias=True,
                alignment="horizontal"):
    """Plot common legend for figures 4c, 4d, and 4e.

    Args:
        chreef_data: Data of ChReef cochleae.
        save_path: save path to save legend.
        grouping: Grouping for cochleae.
            "side_mono" for division in Injected and Non-Injected.
            "side_multi" for division per cochlea.
            "animal" for division per animal.
        use_alias: Use alias.
    """
    if use_alias:
        alias = [COCHLEAE_DICT[k]["alias"] for k in chreef_data.keys()]
    else:
        alias = [name.replace("_", "").replace("0", "") for name in chreef_data.keys()]

    sgns = [len(vals) for vals in chreef_data.values()]
    alias, values_left, values_right = group_lr(alias, sgns)

    colors = ["crimson", "purple", "gold"]
    if grouping == "side_mono":
        colors = [COLOR_LEFT, COLOR_RIGHT]
        labels = ["Injected", "Non-Injected"]
        markers = [MARKER_LEFT, MARKER_RIGHT]
        ncol = 2

    elif grouping == "side_multi":
        colors = []
        labels = []
        markers = []
        keys_left = list(COLORS_LEFT.keys())
        keys_right = list(COLORS_RIGHT.keys())
        for num in range(len(COLORS_LEFT)):
            colors.append(COLORS_LEFT[keys_left[num]])
            colors.append(COLORS_RIGHT[keys_right[num]])
            labels.append(f"{alias[num]}L")
            labels.append(f"{alias[num]}R")
            markers.append(MARKER_LEFT)
            markers.append(MARKER_RIGHT)
        if alignment == "vertical":
            colors = colors[::2] + colors[1::2]
            labels = labels[::2] + labels[1::2]
            markers = markers[::2] + markers[1::2]
            ncol = 2
        else:
            ncol = 5

    elif grouping == "animal":
        colors = []
        labels = []
        markers = []
        ncol = 5
        keys_animal = list(COLORS_ANIMAL.keys())
        for num in range(len(COLORS_ANIMAL)):
            colors.append(COLORS_ANIMAL[keys_animal[num]])
            colors.append(COLORS_ANIMAL[keys_animal[num]])
            labels.append(f"{alias[num]}L")
            labels.append(f"{alias[num]}R")
            markers.append(MARKER_LEFT)
            markers.append(MARKER_RIGHT)
        if alignment == "vertical":
            colors = colors[::2] + colors[1::2]
            labels = labels[::2] + labels[1::2]
            markers = markers[::2] + markers[1::2]
            ncol = 2
        else:
            ncol = 5

    else:
        raise ValueError("Choose a correct 'grouping' parameter.")

    handles = [get_marker_handle(c, m) for (c, m) in zip(colors, markers)]
    legend = plt.legend(handles, labels, loc=3, ncol=ncol, framealpha=1, frameon=False)

    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def plot_legend_trendline(save_path):
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


def plot_legend_fig05e_gerbil(save_path):
    """Plot common legend for figure 5e gerbil.

    Args:
        chreef_data: Data of ChReef cochleae.
        save_path: save path to save legend.
        grouping: Grouping for cochleae.
            "side_mono" for division in Injected and Non-Injected.
            "side_multi" for division per cochlea.
            "animal" for division per animal.
        use_alias: Use alias.
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


def fig_04c(chreef_data, save_path, plot=False, grouping="side_mono", use_alias=True):
    """Box plot showing the SGN counts of ChReef treated cochleae compared to healthy ones.
    """
    prism_style()

    # TODO have central function for alias for all plots?
    if use_alias:
        alias = [COCHLEAE_DICT[k]["alias"] for k in chreef_data.keys()]
    else:
        alias = [name.replace("_", "").replace("0", "") for name in chreef_data.keys()]

    sgns = [len(vals) for vals in chreef_data.values()]

    alias, values_left, values_right = group_lr(alias, sgns)

    # Plot
    fig, ax = plt.subplots(figsize=(4, 5))

    main_label_size = 20
    sub_label_size = 16
    main_tick_size = 16

    offset = 0.08
    x_left = 1
    x_right = 2
    y_ticks = [i for i in range(6000, 13000, 2000)]

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

    if grouping == "side_mono":
        plt.scatter(x_pos_inj, values_left, label="Injected",
                    color=COLOR_LEFT, marker=MARKER_LEFT, s=80, zorder=1)
        plt.scatter(x_pos_non, values_right, label="Non-Injected",
                    color=COLOR_RIGHT, marker=MARKER_RIGHT, s=80, zorder=1)

    elif grouping == "side_multi":
        for num, key in enumerate(COLORS_LEFT.keys()):
            plt.scatter(x_pos_inj[num], values_left[num], label=f"{alias[num]}L",
                        color=COLORS_LEFT[key], marker=MARKER_LEFT, s=80, zorder=1)
        for num, key in enumerate(COLORS_RIGHT.keys()):
            plt.scatter(x_pos_non[num], values_right[num], label=f"{alias[num]}R",
                        color=COLORS_RIGHT[key], marker=MARKER_RIGHT, s=80, zorder=1)

    elif grouping == "animal":
        for num, key in enumerate(COLORS_ANIMAL.keys()):
            plt.scatter(x_pos_inj[num], values_left[num], label=f"{alias[num]}",
                        color=COLORS_ANIMAL[key], marker=MARKER_LEFT, s=80, zorder=1)
            plt.scatter(x_pos_non[num], values_right[num],
                        color=COLORS_ANIMAL[key], marker=MARKER_RIGHT, s=80, zorder=1)

    else:
        raise ValueError("Choose a correct 'grouping' parameter.")

    # Labels and formatting
    plt.xticks([x_left, x_right], ["Injected", "Non-\nInjected"], fontsize=sub_label_size)
    for label in plt.gca().get_xticklabels():
        label.set_verticalalignment('center')
    ax.tick_params(axis='x', which='major', pad=16)
    plt.yticks(y_ticks, fontsize=main_tick_size)
    plt.ylabel("SGN count per cochlea", fontsize=main_label_size)
    plt.ylim(5000, 14000)

    xmin = 0.5
    xmax = 2.5
    plt.xlim(xmin, xmax)

    sgn_values = [11153, 11398, 10333, 11820]
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


def fig_04d(chreef_data, save_path, plot=False, grouping="animal",
            intensity=False, gerbil=False, use_alias=True):
    """Transduction efficiency per cochlea.
    """
    prism_style()
    if use_alias:
        alias = [COCHLEAE_DICT[k]["alias"] for k in chreef_data.keys()]
    else:
        alias = [name.replace("_", "").replace("0", "") for name in chreef_data.keys()]

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

    if grouping == "side_mono":
        plt.scatter(x_pos_inj, values_left, label="Injected",
                    color=COLOR_LEFT, marker=MARKER_LEFT, s=80, zorder=1)
        plt.scatter(x_pos_non, values_right, label="Non-Injected",
                    color=COLOR_RIGHT, marker=MARKER_RIGHT, s=80, zorder=1)

    elif grouping == "side_multi":
        for num, key in enumerate(COLORS_LEFT.keys()):
            plt.scatter(x_pos_inj[num], values_left[num], label=f"{alias[num]}L",
                        color=COLORS_LEFT[key], marker=MARKER_LEFT, s=80, zorder=1)
        for num, key in enumerate(COLORS_RIGHT.keys()):
            plt.scatter(x_pos_non[num], values_right[num], label=f"{alias[num]}R",
                        color=COLORS_RIGHT[key], marker=MARKER_RIGHT, s=80, zorder=1)

    elif grouping == "animal":
        for num, key in enumerate(COLORS_ANIMAL.keys()):
            plt.scatter(x_pos_inj[num], values_left[num], label=f"{alias[num]}",
                        color=COLORS_ANIMAL[key], marker=MARKER_LEFT, s=80, zorder=1)
            plt.scatter(x_pos_non[num], values_right[num],
                        color=COLORS_ANIMAL[key], marker=MARKER_RIGHT, s=80, zorder=1)

    else:
        raise ValueError("Choose a correct 'grouping' parameter.")

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
        if gerbil:
            plt.ylim(0.25, 0.65)
            plt.yticks(np.arange(0.3, 0.7, 0.1), fontsize=main_tick_size)
        else:
            plt.ylim(0.65, 1.05)
            plt.yticks(np.arange(0.7, 1, 0.1), fontsize=main_tick_size)

    # Labels and formatting
    plt.xticks([x_left, x_right], ["Injected", "Non-\nInjected"], fontsize=sub_label_size)
    for la in plt.gca().get_xticklabels():
        la.set_verticalalignment('center')
    ax.tick_params(axis='x', which='major', pad=16)
    plt.ylabel(label, fontsize=main_label_size)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_1))

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


def fig_04e(chreef_data, save_path, plot, intensity=False, gerbil=False,
            use_alias=True, trendlines=False, grouping="side_mono",
            trendline_std=False):
    prism_style()

    if gerbil:
        animal = "gerbil"
    else:
        animal = "mouse"

    result = {"cochlea": [], "octave_band": [], "value": []}
    aliases = []
    for name, values in chreef_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
            alias = name.replace("_", "").replace("0", "")

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

    if grouping == "side_mono":
        colors_l = [COLOR_LEFT for _ in range(5)]
        colors_r = [COLOR_RIGHT for _ in range(5)]

    elif grouping == "side_multi":
        colors_l = [COLORS_LEFT[key] for key in COLORS_LEFT.keys()]
        colors_r = [COLORS_RIGHT[key] for key in COLORS_RIGHT.keys()]

    elif grouping == "animal":
        colors_l = [COLORS_ANIMAL[key] for key in COLORS_ANIMAL.keys()]
        colors_r = [COLORS_ANIMAL[key] for key in COLORS_ANIMAL.keys()]

    else:
        raise ValueError("Choose a correct 'grouping' parameter.")

    color_map = {}
    count_l = 0
    count_r = 0
    for num, (name_lr, grp) in enumerate(result.groupby("cochlea")):
        name, side = name_lr[:-1], name_lr[-1]
        if side == "L":
            color_map[name_lr] = colors_l[count_l]
            count_l += 1
        else:
            color_map[name_lr] = colors_r[count_r]
            count_r += 1

    if len(cochleas) == 1:
        color_map = {"L": colors_l[0], "R": colors_r[1]}

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

            if grouping == "animal":
                color_trend_l = "gray"
                color_trend_r = "gray"
            else:
                color_trend_l = COLOR_LEFT
                color_trend_r = COLOR_RIGHT

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
            if grouping == "animal":
                color_trend_l = "gray"
                color_trend_r = "gray"
            else:
                color_trend_l = COLOR_LEFT
                color_trend_r = COLOR_RIGHT
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
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_1))
    plt.yticks(np.arange(ymin, ymax, 0.1), fontsize=yaxis_tick_size)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    # add a final tick for label '>64k'
    ax.set_xticks([pos + offset_map["L"] for pos in main_ticks] +
                  [pos + offset_map["R"] for pos in main_ticks])
    ax.set_xticklabels(["I"] * len(main_ticks) + ["N"] * len(main_ticks), fontsize=sub_tick_label_size)

    # Add main octave band labels above sublabels
    for i, label in enumerate(bin_labels):
        ax.text(i, ax.get_ylim()[0] - band_label_offset_y*(ax.get_ylim()[1]-ax.get_ylim()[0]),
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
    grouping = "animal"
    plot_legend(chreef_data, grouping=grouping,
                save_path=os.path.join(args.figure_dir, f"fig_04_legend_{grouping}.{FILE_EXTENSION}"))

    plot_legend_trendline(save_path=os.path.join(args.figure_dir, f"fig_04_legend_trendline.{FILE_EXTENSION}"))

    # C: The SGN count compared to reference values from literature and healthy
    # Maybe remove literature reference from plot?
    fig_04c(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04c.{FILE_EXTENSION}"),
            plot=args.plot, grouping=grouping, use_alias=use_alias)

    # D: The expression efficiency. We also plot GFP intensities.
    fig_04d(chreef_data,
            save_path=os.path.join(args.figure_dir,  f"fig_04d_transduction.{FILE_EXTENSION}"),
            plot=args.plot, grouping=grouping, use_alias=use_alias)

    # E: The expression efficiency per octave band.
    # trendlines without standard deviation
    fig_04e(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04e_transduction.{FILE_EXTENSION}"),
            plot=args.plot, grouping=grouping, use_alias=use_alias, trendlines=True)
    # trendlines with standard deviation
    fig_04e(chreef_data,
            save_path=os.path.join(args.figure_dir, f"fig_04e_transduction_std.{FILE_EXTENSION}"),
            plot=args.plot, grouping=grouping, use_alias=use_alias, trendlines=True, trendline_std=True)

    # Figures for gerbil (Figure 5)
    chreef_data_gerbil = get_chreef_data(animal="gerbil")
    fig_04e(chreef_data_gerbil,
            save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_transduction.{FILE_EXTENSION}"),
            plot=args.plot, gerbil=True, use_alias=use_alias, trendlines=True)

    plot_legend_fig05e_gerbil(save_path=os.path.join(args.figure_dir, f"fig_05e_gerbil_legend.{FILE_EXTENSION}"))


if __name__ == "__main__":
    main()
