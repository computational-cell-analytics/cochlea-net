import argparse
import json
import os
import pickle

import imageio.v3 as imageio
from glob import glob

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import cm, colors

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target, get_s3_path
from util import sliding_runlength_sum, frequency_mapping, SYNAPSE_DIR_ROOT, custom_formatter_1, average_by_fraction
from util import prism_style, prism_cleanup_axes, export_legend, get_marker_handle, get_flatline_handle
from flamingo_tools.segmentation.sgn_subtype_utils import stain_to_type, COCHLEAE, ALIAS

INPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/frequency_mapping/M_LR_000227_R/scale3"
MEYER_DATA = "/user/schilling40/u15000/flamingo-tools/scripts/figures/Meyer_et_al-selected.tsv"
MEYER_COLOR = "black"
MEYER_MARKER = "s"

TYPE_TO_CHANNEL = {
    "Type-Ia": "CR",
    "Type-Ib": "Calb1",
    "Type-Ic": "Lypd1",
    "Type-II": "Prph",
}

FILE_EXTENSION = "png"

png_dpi = 300


# Define the animal specific octave bands.
def _get_mapping(animal):
    if animal == "mouse":
        bin_edges = [0, 2, 4, 8, 16, 32, 64, np.inf]
        bin_labels = [
            "<2", "2–4", "4–8", "8–16", "16–32", "32–64", ">64"
        ]
    elif animal == "gerbil":
        bin_edges = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
        bin_labels = [
            "<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", "16–32", ">32"
        ]
    else:
        raise ValueError
    assert len(bin_edges) == len(bin_labels) + 1
    return bin_edges, bin_labels


def frequency_mapping2(frequencies, values, animal="mouse", transduction_efficiency=False):
    # Get the mapping of frequencies to octave bands for the given species.
    bin_edges, bin_labels = _get_mapping(animal)

    # Construct the data frame with octave bands.
    df = pd.DataFrame({"freq_khz": frequencies, "value": values})
    df["octave_band"] = pd.cut(
        df["freq_khz"], bins=bin_edges, labels=bin_labels, right=False
    )

    if transduction_efficiency:  # We compute the transduction efficiency per band.
        num_pos = df[df["value"] == 1].groupby("octave_band", observed=False).size()
        num_tot = df[df["value"].isin([1, 2])].groupby("octave_band", observed=False).size()
        value_by_band = (num_pos / num_tot).reindex(bin_labels)
    else:  # Otherwise, aggregate the values over the octave band using the mean.
        value_by_band = (
            df.groupby("octave_band", observed=True)["value"]
              .sum()
              .reindex(bin_labels)   # keep octave order even if a bin is empty
        )
    return value_by_band


COLORS = {
    "Type Ia": "#859cc8",
    "Type Ib": "#49B38B",
    "Type Ib/Ic": "#7dae7b",
    "Type Ic": "#7EB349",
    "inconclusive": "#B36E49",

    "bbox_outer": "black",
    "bbox_inner": "#7dae7b",
    "Type I": "#68CC6F",
    "Type II": "#E2E271",
    "default": "#279C47"
}

LEGEND_LABEL = {
    "Type Ia": r"$\mathrm{Type}~\mathrm{I}_{\mathrm{a}}$",
    "Type Ib": r"$\mathrm{Type}~\mathrm{I}_{\mathrm{b}}$",
    "Type Ib/Ic": r"$\mathrm{Type}~\mathrm{I}_{\mathrm{b/c}}$",
    "Type Ic": r"$\mathrm{Type}~\mathrm{I}_{\mathrm{c}}$",

}

# The cochlea for the CHReef analysis.
COCHLEAE_DICT = {
    "M_LR_000226_L": {"alias": "M_01L", "component": [1], "color": "#9C5027"},
    "M_LR_000226_R": {"alias": "M_01R", "component": [1], "color": "#279C52"},
    "M_LR_000227_L": {"alias": "M_02L", "component": [1], "color": "#67279C"},
    "M_LR_000227_R": {"alias": "M_02R", "component": [1], "color": "#27339C"},
}

GROUPINGS = {
    "Type Ia;Type Ib;Type Ic;Type II": ["M_LR_000098_L", "M_LR_N152_L"],  # "M_LR_N98_R"
    # , "M_AMD_N180_L", "M_AMD_N180_R"
    "Type I;Type II": ["M_LR_000184_L", "M_LR_000184_R", "M_LR_000260_L"],
    "Type Ib;Type Ic;inconclusive": ["M_LR_N110_L", "M_LR_N110_R", "M_LR_N152_R"],
    "Type Ib;Type Ic;Type IbIc": ["M_LR_000099_L"],
}


def get_tonotopic_data():
    s3 = create_s3_target()
    source_name = "IHC_v4c"
    ihc_version = source_name.split("_")[1]
    cache_path = "./tonotopic_data.pkl"
    cochleae = [key for key in COCHLEAE_DICT.keys()]

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
        ihc_dir = f"ihc_counts_{ihc_version}"
        synapse_dir = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/{ihc_dir}"
        tab_path = os.path.join(synapse_dir, f"ihc_count_{cochlea}.tsv")
        syn_tab = pd.read_csv(tab_path, sep="\t")
        syn_ids = syn_tab["label_id"].values

        syn_per_ihc = [0 for _ in range(len(table))]
        table.loc[:, "syn_per_IHC"] = syn_per_ihc
        for syn_id in syn_ids:
            table.loc[table["label_id"] == syn_id, "syn_per_IHC"] = syn_tab.at[syn_tab.index[syn_tab["label_id"] == syn_id][0], "synapse_count"]  # noqa

        # The relevant values for analysis.
        try:
            values = table[["label_id", "length[µm]", "length_fraction", "frequency[kHz]", "syn_per_IHC"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skippped.")
            continue

        chreef_data[cochlea] = values

    with open(cache_path, "wb") as f:
        pickle.dump(chreef_data, f)
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def _plot_colormap(vol, title, plot, save_path, cmap="viridis", dark_mode=False):
    # Base style
    matplotlib.rcParams.update({
        "font.size": 24,
        "axes.titlesize": 32,
        "figure.titlesize": 32,
        "xtick.major.size": 5,
        "xtick.major.width": 3,
        "xtick.labelsize": 24,
        "ytick.labelsize": 20,
        "legend.fontsize": 48,
    })

    # Create the colormap figure
    fig, ax = plt.subplots(figsize=(6, 1.3))
    fig.subplots_adjust(bottom=0.5)

    # Compute frequency normalization
    freq_min = np.min(np.nonzero(vol))
    freq_max = vol.max()
    norm = colors.LogNorm(vmin=freq_min, vmax=freq_max, clip=True)
    tick_values = np.array([10, 20, 40, 80])

    cmap = plt.get_cmap(cmap)
    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax,
        orientation="horizontal",
        ticks=tick_values
    )

    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_label("Frequency [kHz]")
    plt.title(title)

    # --- DARK MODE INVERSION ---
    if dark_mode:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        # Invert text, ticks, labels, and spines
        cb.ax.xaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")
        plt.setp(cb.ax.xaxis.get_majorticklabels(), color="white")

        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    plt.tight_layout()

    if plot:
        plt.show()

    # Save figure
    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=300,
                    facecolor=fig.get_facecolor(), transparent=False)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0,
                    facecolor=fig.get_facecolor(), transparent=False)
    plt.close()


def fig_03a(save_path, plot, plot_napari, cmap="viridis", dark_mode=False):
    path_ihc = os.path.join(INPUT_ROOT, "frequencies_IHC_v4c.tif")
    path_sgn = os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif")
    sgn = imageio.imread(path_sgn)
    ihc = imageio.imread(path_ihc)
    _plot_colormap(sgn, title="Tonotopic Mapping", plot=plot, save_path=save_path, cmap=cmap, dark_mode=dark_mode)

    # Show the image in napari for rendering.
    if plot_napari:
        import napari
        from napari.utils import Colormap
        # cmap = plt.get_cmap(cmap)
        mpl_cmap = plt.get_cmap(cmap)

        # Sample it into an array of RGBA values
        colors = mpl_cmap(np.linspace(0, 1, 256))

        # Wrap into napari Colormap
        napari_cmap = Colormap(colors, name=f"{cmap}_custom")

        v = napari.Viewer()
        v.add_image(ihc, colormap=napari_cmap)
        v.add_image(sgn, colormap=napari_cmap)
        napari.run()


def fig_03c_length_fraction(tonotopic_data, save_path, use_alias=True,
                            plot=False, n_bins=10, top_axis=False, trendline=False, trendline_std=False,
                            errorbar=True):
    ihc_version = "ihc_counts_v4c"
    if trendline_std:
        line_alphas = {
            "center": 0.6,
            "upper": 0.08,
            "lower": 0.08,
            "fill": 0.05,
        }
    else:
        line_alphas = {
            "center": 1,
            "upper": 0.,
            "lower": 0.,
            "fill": 0.,
        }
    main_label_size = 24
    tick_size = 16
    prism_style()
    tables = glob(os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_LR*.tsv"))
    assert len(tables) == 4, len(tables)

    result = {"cochlea": [], "runlength": [], "value": []}
    color_dict = {}
    marker_dict = {}
    cochleae_length = []
    for name, values in tonotopic_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
            alias = name.replace("_", "").replace("0", "")

        color_dict[alias] = COCHLEAE_DICT[name]["color"]
        marker_dict[alias] = "o"
        syn_count = values["syn_per_IHC"].values
        length_fraction = values["length_fraction"].values
        run_length = values["length[µm]"].values
        cochleae_length.append(max(run_length))

        # columns: "fraction_midpoint", "mean_syn_per_IHC"
        avg_per_bin = average_by_fraction(length_fraction, syn_count, n_bins=n_bins)
        fraction = avg_per_bin["fraction_midpoint"].values
        syn_per_IHC = avg_per_bin["mean_syn_per_IHC"].values

        result["cochlea"].extend([alias] * len(fraction))
        result["runlength"].extend(fraction)
        result["value"].extend(syn_per_IHC)

    if os.path.isfile(MEYER_DATA):
        table = pd.read_csv(MEYER_DATA, sep="\t")
        table = table.reset_index()  # make sure indexes pair with number of rows
        lf_meyer = []
        syn_meyer = []
        for index, row in table.iterrows():
            if isinstance(row["Ribbons_Cell"], float):
                lf_meyer.append(row["X__from_Apex"] / 100)
                syn_meyer.append(row["Ribbons_Cell"])
        avg_per_bin = average_by_fraction(lf_meyer, syn_meyer, n_bins=n_bins)
        fraction = avg_per_bin["fraction_midpoint"].values
        syn_per_IHC = avg_per_bin["mean_syn_per_IHC"].values

        result["cochlea"].extend(["Meyer"] * len(fraction))
        result["runlength"].extend(fraction)
        result["value"].extend(syn_per_IHC)
        color_dict["Meyer"] = MEYER_COLOR
        marker_dict["Meyer"] = MEYER_MARKER

    avg_length = sum(cochleae_length) / len(cochleae_length)
    print(f"Average total length: {round(avg_length, 2)} µm")
    print(f"Average length per bin: {round(avg_length / n_bins, 2)} µm")

    result = pd.DataFrame(result)
    fig, ax = plt.subplots(figsize=(6.7, 5))

    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        run_length = list(grp["runlength"])
        syn_count = list(grp["value"])
        # print(run_length.shape)
        # print(syn_count.shape)
        run_length = [r for num, r in enumerate(run_length) if not np.isnan(syn_count[num])]
        syn_count = [s for s in syn_count if not np.isnan(s)]
        if name == "Meyer":
            ax.plot(run_length, syn_count, label=name,
                    markeredgecolor=color_dict[name],
                    color=color_dict[name], marker='s', linestyle='solid', linewidth=2)
        else:
            ax.scatter(run_length, syn_count, label=name,
                       color=color_dict[name], marker=marker_dict[name])

    # calculate mean and standard deviation
    trend_dict = {}
    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        if name == "Meyer":
            continue
        run_length = list(grp["runlength"])
        syn_count = list(grp["value"])
        for r, s in zip(run_length, syn_count):
            if r in trend_dict:
                trend_dict[r].append(s)
            else:
                trend_dict[r] = [s]

    x_pos = [k for k in list(trend_dict.keys())]
    center_line = [sum(val) / len(val) for _, val in trend_dict.items()]
    val_std = [np.std(val) for _, val in trend_dict.items()]
    lower_std = [mean - std for (mean, std) in zip(center_line, val_std)]
    upper_std = [mean + std for (mean, std) in zip(center_line, val_std)]

    if errorbar:
        ax.errorbar(x_pos, center_line, val_std, linestyle='dashed', marker='D', color="#D63637", linewidth=1)

    if trendline:
        trend_center, = ax.plot(
            x_pos,
            center_line,
            linestyle="dashed",
            color="gray",
            alpha=line_alphas["center"],
            linewidth=3,
            zorder=2
        )
        trend_upper, = ax.plot(
            x_pos,
            upper_std,
            linestyle="solid",
            color="gray",
            alpha=line_alphas["upper"],
            zorder=0
        )
        trend_lower, = ax.plot(
            x_pos,
            lower_std,
            linestyle="solid",
            color="gray",
            alpha=line_alphas["lower"],
            zorder=0
        )
        plt.fill_between(x_pos, lower_std, upper_std,
                         color="gray", alpha=line_alphas["fill"], interpolate=True)

    if top_axis:
        # Create second x-axis
        ax_top = ax.twiny()

        # Frequencies for ticks (kHz → convert to kHz or Hz depending on preference)
        freq_ticks = np.array([2, 4, 8, 16, 32, 64])  # kHz

        # Given constants
        var_A = 1.46
        var_a = 1.77

        # Inverse mapping length_fraction = log10(f/A) / a
        length_positions = np.log10(freq_ticks / var_A) / var_a

        # Set ticks on top axis
        ax_top.set_xticks(length_positions)
        ax_top.set_xticklabels([f"{f}" for f in freq_ticks], fontsize=tick_size)

        # Label for the new axis
        ax_top.set_xlabel("Frequency [kHz]", fontsize=main_label_size)

        # Ensure both axes align well
        ax_top.set_xlim(ax.get_xlim())

    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_xlabel("Length fraction", fontsize=main_label_size)
    ax.set_ylabel("Synapse per IHC", fontsize=main_label_size)
    # ax.legend(title="cochlea")
    plt.tight_layout()
    # prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_03c_rl(tonotopic_data, save_path, use_alias=True, plot=False):
    ihc_version = "ihc_counts_v4c"
    width = 200
    prism_style()
    tables = glob(os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_LR*.tsv"))
    assert len(tables) == 4, len(tables)

    result = {"cochlea": [], "runlength": [], "value": []}
    color_dict = {}
    for name, values in tonotopic_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
            alias = name.replace("_", "").replace("0", "")

        color_dict[alias] = COCHLEAE_DICT[name]["color"]
        syn_count = values["syn_per_IHC"].values
        run_length = values["length[µm]"].values

        run_length, syn_count_running = sliding_runlength_sum(run_length, syn_count, width=width)

        result["cochlea"].extend([alias] * len(run_length))
        result["runlength"].extend(list(run_length))
        result["value"].extend(list(syn_count_running))

    result = pd.DataFrame(result)
    fig, ax = plt.subplots(figsize=(8, 4))

    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        run_length = grp["runlength"]
        syn_count_running = grp["value"]
        ax.plot(run_length, syn_count_running, label=name, color=color_dict[name])

    ax.set_xlabel("Length [µm]")
    ax.set_ylabel("Synapse Count")
    ax.set_title(f"Ribbon Syn. per IHC: Runnig sum @ {width} µm")
    ax.legend(title="cochlea")
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


def plot_legend_fig03c(save_path, ncol=None):
    color_dict = {}
    for key in COCHLEAE_DICT.keys():
        color_dict[COCHLEAE_DICT[key]["alias"]] = COCHLEAE_DICT[key]["color"]

    marker = ["o" for _ in color_dict]
    label = list(color_dict.keys())
    color = [color_dict[key] for key in color_dict.keys()]
    if ncol is None:
        ncol = 2

    handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]
    legend = plt.legend(handles, label, loc=3, ncol=ncol, framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def plot_legend_suppfig03(save_path, ncol=None):
    """Legend for supplementary Figure showing synapses per IHC compared to values from Meyer
    """
    color_dict = {}
    for key in COCHLEAE_DICT.keys():
        color_dict[COCHLEAE_DICT[key]["alias"]] = COCHLEAE_DICT[key]["color"]

    marker = ["o" for _ in color_dict]
    label = list(color_dict.keys())

    # color_dict["mean"] = "#D63637"
    # marker.append("D")
    # label.append("Mean")

    # add parameters for data from Meyer
    color_dict["Meyer"] = MEYER_COLOR
    marker.append(MEYER_MARKER)
    label.append("Meyer et al.")

    color = [color_dict[key] for key in color_dict.keys()]
    if ncol is None:
        ncol = len(label // 2)

    handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]
    legend = plt.legend(handles, label, loc=3, ncol=ncol, framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def _get_trendline_dict(trend_dict,):
    x_sorted = [trend_dict[k]["x_sorted"] for k in trend_dict.keys()]
    x_dict = {}
    for num in range(len(x_sorted[0])):
        x_dict[num] = {"pos": num, "values": []}

    for s in x_sorted:
        for num, pos in enumerate(s):
            x_dict[num]["values"].append(pos)

    y_sorted_all = [trend_dict[k]["y_sorted"] for k in trend_dict.keys()]
    y_dict = {}
    for num in range(len(x_sorted[0])):
        y_dict[num] = {"pos": num, "values": []}

    for num in range(len(x_sorted[0])):
        y_dict[num]["mean"] = np.mean([y[num] for y in y_sorted_all])
        y_dict[num]["stdv"] = np.std([y[num] for y in y_sorted_all])
    return x_dict, y_dict


def _get_trendline_params(trend_dict):
    x_dict, y_dict = _get_trendline_dict(trend_dict)

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


def fig_03c_octave(tonotopic_data, save_path, plot=False, use_alias=True, trendline=False):
    ihc_version = "ihc_counts_v4c"
    prism_style()
    tables = glob(os.path.join(SYNAPSE_DIR_ROOT, ihc_version, "ihc_count_M_LR*.tsv"))
    assert len(tables) == 4, len(tables)
    main_label_size = 28
    main_tick_size = 20
    xtick_size = 16

    result = {"cochlea": [], "octave_band": [], "value": []}
    color_dict = {}
    for name, values in tonotopic_data.items():
        if use_alias:
            alias = COCHLEAE_DICT[name]["alias"]
        else:
            alias = name.replace("_", "").replace("0", "")

        color_dict[alias] = COCHLEAE_DICT[name]["color"]
        freq = values["frequency[kHz]"].values
        syn_count = values["syn_per_IHC"].values
        octave_binned = frequency_mapping(freq, syn_count, animal="mouse")

        result["cochlea"].extend([alias] * len(octave_binned))
        result["octave_band"].extend(octave_binned.axes[0].values.tolist())
        result["value"].extend(octave_binned.values.tolist())

    result = pd.DataFrame(result)
    bin_labels = pd.unique(result["octave_band"])
    band_to_x = {band: i for i, band in enumerate(bin_labels)}
    result["x_pos"] = result["octave_band"].map(band_to_x)

    fig, ax = plt.subplots(figsize=(7, 5))

    offset = 0.08
    trend_dict = {}
    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        x_sorted = grp["x_pos"]
        x_positions = [x - len(grp["x_pos"]) // 2 * offset + offset * num for x in grp["x_pos"]]
        ax.scatter(x_positions, grp["value"], marker="o", label=name, s=80, alpha=1, color=color_dict[name])

        # y_values.append(list(grp["value"]))

        if trendline:
            sorted_idx = np.argsort(x_positions)
            x_sorted = np.array(x_positions)[sorted_idx]
            y_sorted = np.array(grp["value"])[sorted_idx]
            trend_dict[name] = {"x_sorted": x_sorted,
                                "y_sorted": y_sorted,
                                }

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.tick_params(axis='x', labelsize=xtick_size)
    ax.tick_params(axis='y', labelsize=main_tick_size)
    ax.set_xlabel("Octave band [kHz]", fontsize=main_label_size)
    ax.set_ylabel("Syn. per IHC", fontsize=main_label_size)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    # central line
    if trendline:
        x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = _get_trendline_params(trend_dict)
        trend_center, = ax.plot(
            x_sorted,
            y_sorted,
            linestyle="dotted",
            color="gray",
            alpha=0.6,
            linewidth=3,
            zorder=2
        )
        trend_upper, = ax.plot(
            x_sorted,
            y_sorted_upper,
            linestyle="solid",
            color="gray",
            alpha=0.08,
            zorder=0
        )
        trend_lower, = ax.plot(
            x_sorted,
            y_sorted_lower,
            linestyle="solid",
            color="gray",
            alpha=0.08,
            zorder=0
        )
        plt.fill_between(x_sorted, y_sorted_lower, y_sorted_upper,
                         color="gray", alpha=0.05, interpolate=True)

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


def plot_average_tonotopic_mapping(results, save_path, plot=False, combine_IbIc=False):
    prism_style()
    #
    # Create the average tonotopic mapping for multiple cochleae.
    #
    main_label_size = 28
    main_tick_size = 20
    xtick_size = 16
    summary = {}
    for cochlea, result in results.items():
        classification = result["classification"]
        frequencies = result["frequencies"]
        # get categories
        cats = list(set([c[:c.find(" (")] for c in classification]))
        cats.sort()
        if "Type Ic" in cats and "Type II" in cats:
            # change order of "Type II" and "Type I"
            cats = cats[1:] + cats[:1]

        dic = {}
        for c in cats:
            sub_freq = [frequencies[i] for i in range(len(classification))
                        if classification[i][:classification[i].find(" (")] == c]
            mapping = frequency_mapping2(sub_freq, [1 for _ in range(len(sub_freq))])
            mapping.fillna(0, inplace=True)

            mapping = mapping.astype('float32')
            dic[c] = mapping
            bin_labels = pd.unique(mapping.index)

        for bin in bin_labels:
            total = sum([dic[key][bin] for key in dic.keys()])
            for key in dic.keys():
                dic[key][bin] = float(dic[key][bin] / total)

        summary[cochlea] = dic

    fig, ax = plt.subplots(figsize=(6.7, 5))

    # Collect all data per subtype
    subtype_data = {}

    freq_IbIc = []
    for cochlea, dic in summary.items():
        freq_IbIc.append([])
        for cat, freq_map in dic.items():
            if combine_IbIc and cat in ["Type Ib", "Type Ic", "Type IbIc"]:
                freq_IbIc[-1].append(freq_map)

            else:
                if cat not in subtype_data:
                    subtype_data[cat] = []
                subtype_data[cat].append(freq_map)

    # TODO: generalize function
    if combine_IbIc:
        df_IbIc = []
        for i in range(len(freq_IbIc)):
            df_IbIc.append(freq_IbIc[i][0].add(freq_IbIc[i][1], fill_value=0))
        subtype_data["Type Ib/Ic"] = df_IbIc

    # Compute average and std for each subtype
    for cat, freq_list in subtype_data.items():
        # Align all dataframes on the same index (octave bands)
        df_concat = pd.concat(freq_list, axis=1)
        mean_vals = df_concat.mean(axis=1)
        std_vals = df_concat.std(axis=1)

        bin_labels = mean_vals.index
        x_positions = np.arange(len(bin_labels))
        color = COLORS.get(cat, COLORS["default"])

        ax.scatter(x_positions, mean_vals, label=cat, color=color, s=80)
        ax.fill_between(
            x_positions,
            mean_vals - std_vals,
            mean_vals + std_vals,
            color=color,
            alpha=0.3
        )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(bin_labels)
    ax.tick_params(axis='x', labelsize=xtick_size)
    ax.tick_params(axis='y', labelsize=main_tick_size)
    ax.set_xlabel("Octave band [kHz]", fontsize=main_label_size)
    ax.set_ylabel("Fraction", fontsize=main_label_size)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_1))
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    # Cochleae as title
    # cochleae = [ALIAS[c] for c in list(results.keys())]
    # cochleae_str = " - ".join(cochleae)
    # ax.set_title(f"Cochleae: {cochleae_str}")
    # ax.legend(title="Subtypes")

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if plot:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()


def plot_subtype_fraction(results, save_path, plot=False):
    #
    # Create the average tonotopic mapping for multiple cochleae.
    #
    prism_style()
    main_label_size = 28
    main_tick_size = 20

    summary, types = {}, []
    for num, (cochlea, result) in enumerate(results.items()):
        alias = ALIAS[cochlea]
        classification = result["classification"]
        classification = [cls[:cls.find(" (")] for cls in classification]
        n_tot = len(classification)

        this_types = list(set(classification))
        types.extend(this_types)
        types = list(set(types))
        types.sort()
        if "Type Ia" in types and "Type II" in types:
            # change order of "Type II" and "Type I"
            types = types[1:] + types[:1]
        # account for plotting dataframes from bottom to top
        types.reverse()

        summary[alias] = {}
        for stype in types:
            n_type = len([cls for cls in classification if cls == stype])
            type_ratio = float(n_type) / n_tot
            summary[alias][stype] = type_ratio

    df = pd.DataFrame(summary).fillna(0)  # missing values → 0

    colors = [COLORS[t] for t in types]
    # Plot with reversed order so first entry in `types` is at the top [types[::-1]]
    ax = df.T.plot(
        kind="bar",
        stacked=True,
        figsize=(6.7, 5),
        color=colors,
        legend=False,
    )

    if "Type Ib" in types and "Type Ic" in types:
        for num in range(len(results)):
            # --- Compute the total height and positions of each segment ---
            patches_total = ax.patches  # patches are ordered bottom→top
            patches = patches_total[num::len(results)]
            bar = patches[0].get_x(), patches[0].get_width()  # x position and width

            # find y positions of each patch
            segment_info = []
            for p, subtype in zip(patches, types):
                y0 = p.get_y()
                y1 = y0 + p.get_height()
                segment_info.append((subtype, y0, y1))

            # --- Define which contiguous block to outline ---
            outlined_block = {"Type Ib", "Type Ic"}

            # Get bottom and top of that block
            y_bottom = min(y0 for subtype, y0, y1 in segment_info if subtype in outlined_block)
            y_top = max(y1 for subtype, y0, y1 in segment_info if subtype in outlined_block)

            x, width = bar
            rect = plt.Rectangle(
                (x, y_bottom),
                width,
                y_top - y_bottom,
                linewidth=5,
                edgecolor=COLORS["bbox_outer"],
                facecolor="none",
                zorder=1,
            )
            ax.add_patch(rect)
            rect = plt.Rectangle(
                (x, y_bottom),
                width,
                y_top - y_bottom,
                linewidth=3,
                edgecolor=COLORS["bbox_inner"],
                facecolor="none",
                zorder=2,
            )
            ax.add_patch(rect)

    # Optional: reverse legend order to match your original `types` order
    # handles, labels = ax.get_legend_handles_labels()
    # ax.legend(handles[::-1], labels[::-1], title="Subtype", loc="lower right")

    ax.set_ylabel("Fraction", fontsize=main_label_size)
    # ax.set_xlabel("Cochlea", fontsize=main_label_size)
    # ax.set_title("Subtype Fractions per Cochlea", fontsize=main_label_size)

    plt.yticks(fontsize=main_tick_size)
    plt.xticks(rotation=0, fontsize=main_tick_size)
    plt.tight_layout()
    prism_cleanup_axes(ax)

    if plot:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def fig_03_subtype_fraction(save_path, grouping="Type Ia;Type Ib;Type Ic;Type II", cochleae=None):
    if cochleae is None:
        cochleae = GROUPINGS[grouping]

    results = {}

    for cochlea in cochleae:
        if "output_seg" in list(COCHLEAE[cochlea].keys()):
            seg_name = COCHLEAE[cochlea]["output_seg"]
        else:
            seg_name = COCHLEAE[cochlea]["seg_data"]

        if "component_list" in list(COCHLEAE[cochlea].keys()):
            component_list = COCHLEAE[cochlea]["component_list"]
        else:
            component_list = [1]

        if "label_stains" in list(COCHLEAE[cochlea].keys()):
            stain_channels = COCHLEAE[cochlea]["label_stains"]["subtype_label"]
        else:
            stain_channels = COCHLEAE[cochlea]["subtype_stains"]

        s3_path = f"{cochlea}/tables/{seg_name}/default.tsv"
        table_path, fs = get_s3_path(s3_path)
        with fs.open(table_path, 'r') as f:
            table = pd.read_csv(f, sep="\t")
        table = table[table["component_labels"].isin(component_list)]

        # filter subtype table
        for chan in stain_channels:
            column = f"marker_{chan}"
            table = table.loc[table[column].isin([1, 2])]

        classification = []
        for chan in stain_channels:
            column = f"marker_{chan}"
            subset = table.loc[table[column].isin([1, 2])]
            marker = list(subset[column])
            chan_classification = []
            for m in marker:
                if m == 1:
                    chan_classification.append(f"{chan}+")
                elif m == 2:
                    chan_classification.append(f"{chan}-")
            classification.append(chan_classification)

        # Unify the classification and assign colors
        assert len(classification) in (1, 2)
        if len(classification) == 2:
            cls1, cls2 = classification[0], classification[1]
            assert len(cls1) == len(cls2)
            classification = [f"{c1} / {c2}" for c1, c2 in zip(cls1, cls2)]
        else:
            classification = classification[0]

        classification = [stain_to_type(cls) for cls in classification]
        classification = [f"{stype} ({stain})" for stype, stain in classification]

        # 3.) Plot tonotopic mapping.
        freq = table["frequency[kHz]"].values
        assert len(freq) == len(classification)

        results[cochlea] = {"classification": classification, "frequencies": freq}
    plot_subtype_fraction(results, save_path)


def plot_legend_subtypes(save_path, grouping, ncol=None):
    """Plot common legend for subtype panels in Figure 3.

    Args:
        save_path: save path to save legend.
    """
    subtypes = grouping.split(";")
    labels = [LEGEND_LABEL.get(label, label) for label in subtypes]
    colors = [COLORS[subtype] for subtype in subtypes]
    if ncol is None:
        ncol = len(labels)

    # Colors
    handles = [get_flatline_handle(c) for c in colors]
    legend = plt.legend(handles, labels, loc=3, ncol=ncol, framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def fig_03_subtype_tonotopic(save_path, grouping="Type Ia;Type Ib;Type Ic;Type II", cochleae=None,
                             combine_IbIc=False):
    if cochleae is None:
        cochleae = GROUPINGS[grouping]

    results = {}

    for cochlea in cochleae:
        if "output_seg" in list(COCHLEAE[cochlea].keys()):
            seg_name = COCHLEAE[cochlea]["output_seg"]
        else:
            seg_name = COCHLEAE[cochlea]["seg_data"]

        if "component_list" in list(COCHLEAE[cochlea].keys()):
            component_list = COCHLEAE[cochlea]["component_list"]
        else:
            component_list = [1]

        if "label_stains" in list(COCHLEAE[cochlea].keys()):
            stain_channels = COCHLEAE[cochlea]["label_stains"]["subtype_label"]
        else:
            stain_channels = COCHLEAE[cochlea]["subtype_stains"]

        s3_path = f"{cochlea}/tables/{seg_name}/default.tsv"
        table_path, fs = get_s3_path(s3_path)
        with fs.open(table_path, 'r') as f:
            table = pd.read_csv(f, sep="\t")
        table = table[table["component_labels"].isin(component_list)]

        # filter subtype table
        for chan in stain_channels:
            column = f"marker_{chan}"
            table = table.loc[table[column].isin([1, 2])]

        classification = []
        for chan in stain_channels:
            column = f"marker_{chan}"
            subset = table.loc[table[column].isin([1, 2])]
            marker = list(subset[column])
            chan_classification = []
            for m in marker:
                if m == 1:
                    chan_classification.append(f"{chan}+")
                elif m == 2:
                    chan_classification.append(f"{chan}-")
            classification.append(chan_classification)

        # Unify the classification and assign colors
        assert len(classification) in (1, 2)
        if len(classification) == 2:
            cls1, cls2 = classification[0], classification[1]
            assert len(cls1) == len(cls2)
            classification = [f"{c1} / {c2}" for c1, c2 in zip(cls1, cls2)]
        else:
            classification = classification[0]

        classification = [stain_to_type(cls) for cls in classification]
        classification = [f"{stype} ({stain})" for stype, stain in classification]

        # 3.) Plot tonotopic mapping.
        freq = table["frequency[kHz]"].values
        assert len(freq) == len(classification)

        results[cochlea] = {"classification": classification, "frequencies": freq}
    plot_average_tonotopic_mapping(results, save_path, combine_IbIc=combine_IbIc)


# TODO
def fig_03d_octave(save_path, plot):
    pass


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Fig 3 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.", default="./panels/fig3")
    parser.add_argument("--napari", action="store_true", help="Visualize tonotopic mapping in napari.")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)
    tonotopic_data = get_tonotopic_data()

    # Panel A: Tonotopic mapping of SGNs and IHCs (rendering in napari + heatmap)
    cmap = "plasma"
    fig_03a(save_path=os.path.join(args.figure_dir, f"fig_03a_cmap_{cmap}.{FILE_EXTENSION}"),
            plot=args.plot, plot_napari=args.napari, cmap=cmap, dark_mode=True)

    fig_03c_rl(tonotopic_data=tonotopic_data,
               save_path=os.path.join(args.figure_dir, f"fig_03c_rl.{FILE_EXTENSION}"), plot=args.plot)

    fig_03c_length_fraction(tonotopic_data=tonotopic_data,
                            save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer.{FILE_EXTENSION}"),
                            plot=args.plot, n_bins=25)
#    fig_03c_length_fraction(tonotopic_data=tonotopic_data,
#                            save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_freq.{FILE_EXTENSION}"),
#                            plot=args.plot, n_bins=25, top_axis=True)
#    fig_03c_length_fraction(tonotopic_data=tonotopic_data,
#                            save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_freq_trend.{FILE_EXTENSION}"),
#                            plot=args.plot, n_bins=25, top_axis=True, trendline=True, trendline_std=False)
#    fig_03c_length_fraction(tonotopic_data=tonotopic_data,
#                            save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_freq_std.{FILE_EXTENSION}"),
#                            plot=args.plot, n_bins=25, top_axis=True, trendline=True, trendline_std=True)
    fig_03c_length_fraction(tonotopic_data=tonotopic_data,
                            save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_errorbar.{FILE_EXTENSION}"),
                            plot=args.plot, n_bins=25, top_axis=True, errorbar=True)
    # plot_legend_suppfig03(save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_lgnd1.{FILE_EXTENSION}"), ncol=1)
    # plot_legend_suppfig03(save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_lgnd6.{FILE_EXTENSION}"), ncol=6)
    plot_legend_suppfig03(save_path=os.path.join(args.figure_dir, f"figsupp_03_meyer_lgnd3.{FILE_EXTENSION}"), ncol=3)

    # Panel C: Spatial distribution of synapses across the cochlea (running sum per octave band)
    fig_03c_octave(tonotopic_data=tonotopic_data,
                   save_path=os.path.join(args.figure_dir, f"fig_03c_octave.{FILE_EXTENSION}"),
                   plot=args.plot, trendline=True)
    plot_legend_fig03c(save_path=os.path.join(args.figure_dir, f"fig_03c_legend.{FILE_EXTENSION}"), ncol=1)

    grouping = "Type Ia;Type Ib/Ic;Type II"
    plot_legend_subtypes(save_path=os.path.join(args.figure_dir, f"fig_03_legend_Ia-IbIc-II.{FILE_EXTENSION}"),
                         grouping=grouping)

    grouping = "Type Ia;Type Ib;Type Ic;Type II"
    plot_legend_subtypes(save_path=os.path.join(args.figure_dir, f"fig_03_legend_Ia-Ib-Ic-II.{FILE_EXTENSION}"),
                         grouping=grouping, ncol=1)
    fig_03_subtype_tonotopic(save_path=os.path.join(args.figure_dir, f"fig_03_tonotopic_Ia-IbIc-II.{FILE_EXTENSION}"),
                             grouping=grouping, combine_IbIc=True)
    fig_03_subtype_fraction(save_path=os.path.join(args.figure_dir, f"fig_03_fraction_Ia-Ib-Ic-II.{FILE_EXTENSION}"),
                            grouping=grouping)

    grouping = "Type I;Type II"
    plot_legend_subtypes(save_path=os.path.join(args.figure_dir, f"fig_03_legend_I-II.{FILE_EXTENSION}"),
                         grouping=grouping)
    fig_03_subtype_tonotopic(save_path=os.path.join(args.figure_dir, f"figsupp_03_tonotopic_I-II.{FILE_EXTENSION}"),
                             grouping=grouping)
    fig_03_subtype_fraction(save_path=os.path.join(args.figure_dir, f"fig_03d_fraction_I-II.{FILE_EXTENSION}"),
                            grouping=grouping)

    grouping = "Type Ib/Ic;inconclusive"
    plot_legend_subtypes(save_path=os.path.join(args.figure_dir, f"fig_03_legend_IbIc-inconclusive.{FILE_EXTENSION}"),
                         grouping=grouping, ncol=1)
    grouping = "Type Ib;Type Ic;inconclusive"
    plot_legend_subtypes(save_path=os.path.join(args.figure_dir, f"fig_03_legend_Ib-Ic-inconclusive.{FILE_EXTENSION}"),
                         grouping=grouping, ncol=1)
    fig_03_subtype_tonotopic(save_path=os.path.join(args.figure_dir, f"figsupp_03_tonotopic_IbIc.{FILE_EXTENSION}"),
                             grouping=grouping, combine_IbIc=True)
    fig_03_subtype_fraction(save_path=os.path.join(args.figure_dir, f"figsupp_03_fraction_Ib-Ic.{FILE_EXTENSION}"),
                            grouping=grouping)

    # Panel D: Spatial distribution of SGN sub-types.
    fig_03d_octave(save_path=os.path.join(args.figure_dir, f"fig_03d_octave.{FILE_EXTENSION}"), plot=args.plot)


if __name__ == "__main__":
    main()
