"""Tonotopic mapping: frequency color bar, napari renderings and synapses per IHC along the cochlea."""
import os
from typing import Optional

import imageio.v3 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from matplotlib import cm, colors
from flamingo_tools.s3_utils import BUCKET_NAME, get_s3_path, read_table

from util import (
    MEYER_COLOR, MEYER_MARKER, average_by_fraction, band_table, band_trendline, cochlea_colors, cochlea_label,
    cochleae_for, cohort_cochleae, frequency_mapping, get_flatline_handle, get_marker_handle,
    prism_cleanup_axes, prism_style, save_figure,
)

# Frequency volumes of M_LR_000227_R for the renderings and the color bar.
INPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/frequency_mapping/M_LR_000227_R/scale3"
MEYER_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Meyer_et_al-selected.tsv")

# The iDISCO reference cochleae, for the alias and color of a plotted cochlea.
IDISCO_COCHLEAE_DICT = cochleae_for(cohort_cochleae("idisco"), "IHC", "IHC_v11")


def _napari_colormap(cmap: str):
    from napari.utils import Colormap
    return Colormap(plt.get_cmap(cmap)(np.linspace(0, 1, 256)), name=f"{cmap}_custom")


def show_analysis_overview(
    cmap_tonotopic: str = "plasma",
    cmap_syn_per_ihc: str = "cividis",
    syn_per_IHC_table: Optional[pd.DataFrame] = None,
):
    """Show the tonotopic mapping, the SGN and IHC segmentation and the synapses per IHC in napari.

    The screenshots are the small analysis renderings of the overview figure.

    Args:
        cmap_tonotopic: Color map for tonotopic mapping.
        cmap_syn_per_ihc: Color map for visualizing synapses per IHC.
        syn_per_IHC_table: Segmentation table with a "syn_per_IHC" column. Defaults to the
            IHC_v4c table of M_LR_000227_R on S3.
    """
    import napari

    cochlea = "M_LR_000227_R"
    seg_name = "IHC_v4c"
    sgn = imageio.imread(os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif"))
    ihc = imageio.imread(os.path.join(INPUT_ROOT, "frequencies_IHC_v4c.tif"))

    if syn_per_IHC_table is None:
        syn_per_IHC_table = read_table(f"{cochlea}/tables/{seg_name}/default.tsv", s3=True)

    sgn_mono = sgn.copy()
    sgn_mono[sgn_mono > 0] = 2
    sgn_mono = sgn_mono.astype(np.uint32)

    ihc_mono = ihc.copy()
    ihc_mono[ihc_mono > 0] = 3
    ihc_mono = ihc_mono.astype(np.uint32)

    # Map every IHC label to its synapse count.
    column = "syn_per_IHC"
    subset = syn_per_IHC_table[syn_per_IHC_table["component_labels"] == 1][["label_id", column]].dropna()
    lookup = dict(zip(subset["label_id"].astype(int), subset[column].astype(float)))

    internal_path = os.path.join(cochlea, "images", "ome-zarr", f"{seg_name}.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME)
    ihc_syn = zarr.open(s3_store, mode="r")["s3"][:].astype("uint16")

    max_label = int(ihc_syn.max())
    lut = np.zeros(max_label + 1, dtype=np.uint32)
    for k, v in lookup.items():
        if k <= max_label:
            lut[k] = v
    syn_arr = lut[ihc_syn.astype(np.uint32)].astype(np.uint32)

    napari_cmap = _napari_colormap(cmap_tonotopic)
    v = napari.Viewer()
    v.add_image(ihc, colormap=napari_cmap)
    v.add_image(sgn, colormap=napari_cmap)
    v.add_image(syn_arr, colormap=_napari_colormap(cmap_syn_per_ihc), name="syn_per_IHC")
    v.add_labels(sgn_mono, name="SGN")
    v.add_labels(ihc_mono, name="IHC")
    napari.run()


def show_tonotopic_mapping(cmap: str = "viridis"):
    """Show the SGN and IHC frequency volumes in napari, for the tonotopic mapping rendering.

    Args:
        cmap: Color map.
    """
    import napari

    napari_cmap = _napari_colormap(cmap)
    v = napari.Viewer()
    v.add_image(imageio.imread(os.path.join(INPUT_ROOT, "frequencies_IHC_v4c.tif")), colormap=napari_cmap)
    v.add_image(imageio.imread(os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif")), colormap=napari_cmap)
    napari.run()


def plot_frequency_colorbar(
    save_path: str,
    plot: bool = False,
    cmap: str = "viridis",
    dark_mode: bool = True,
    title: str = "Tonotopic Mapping",
):
    """Plot the logarithmic frequency color bar of the SGN frequency volume.

    The function changes matplotlib.rcParams without a reset, like the original color map script.

    Args:
        save_path: Save path for color map.
        plot: Plot figures.
        cmap: Color map.
        dark_mode: Toggle black background for color map.
        title: Title above the color bar.
    """
    vol = imageio.imread(os.path.join(INPUT_ROOT, "frequencies_SGN_v2.tif"))

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

    fig, ax = plt.subplots(figsize=(6, 1.3))
    fig.subplots_adjust(bottom=0.5)

    freq_min = np.min(np.nonzero(vol))
    freq_max = vol.max()
    norm = colors.LogNorm(vmin=freq_min, vmax=freq_max, clip=True)
    tick_values = np.array([10, 20, 40, 80])

    cb = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap)),
        cax=ax,
        orientation="horizontal",
        ticks=tick_values
    )

    cb.ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    cb.ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    cb.set_label("Frequency [kHz]")
    plt.title(title)

    if dark_mode:
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")

        cb.ax.xaxis.label.set_color("white")
        cb.ax.tick_params(colors="white")
        plt.setp(cb.ax.xaxis.get_majorticklabels(), color="white")

        ax.title.set_color("white")
        for spine in ax.spines.values():
            spine.set_color("white")

    plt.tight_layout()
    save_figure(save_path, plot, facecolor=fig.get_facecolor(), transparent=False)


def plot_synapses_along_cochlea(
    tonotopic_data: dict,
    save_path: str,
    use_alias: bool = True,
    plot: bool = False,
    n_bins: int = 10,
    top_axis: bool = False,
    trendline: bool = False,
    trendline_std: bool = False,
    errorbar: bool = False,
    cohort_dict: dict = None,
    trendline_colors: dict = None,
    show_legend: bool = False,
    length_info: bool = False,
    alpha: Optional[float] = None,
):
    """Plot the synapses per IHC over the length fraction, with the counts of Meyer et al. 2009.

    Args:
        tonotopic_data: Dictionary of tonotopic data.
        save_path: Save path for figure.
        use_alias: Use cochleae aliases.
        plot: Plot figure.
        n_bins: Number of bins to divide the run length into.
        top_axis: Plot top x-axis as frequency range.
        trendline: Visualize trendline as average of each bin.
        trendline_std: Visualize standard deviation of trendline.
        errorbar: Use errorbar for visualizing mean and stdev.
        cohort_dict: Optional per-cochlea metadata dict with keys "alias", "color", "marker".
            Shape: {cochlea_name: {"alias": str, "color": str, "marker": str}}.
            When None, falls back to IDISCO_COCHLEAE_DICT with marker "o".
        trendline_colors: Optional dict mapping cohort name → color string. When provided,
            entries in cohort_dict must also carry a "cohort" key, and one trendline is drawn
            per cohort in its color. When None, a single gray trendline is drawn across all
            non-Meyer cochleae.
        show_legend: Show legend within plot.
        length_info: Print info for length of entire cochlea and bins used for plotting.
        alpha: Alpha of the cochlea points. Defaults to 0.5 with a cohort_dict and 1 without.
    """
    if alpha is None:
        alpha = 0.5 if cohort_dict else 1

    if trendline_std:
        line_alphas = {"center": 0.6, "upper": 0.08, "lower": 0.08, "fill": 0.05}
    else:
        line_alphas = {"center": 1, "upper": 0., "lower": 0., "fill": 0.}
    main_label_size = 24
    tick_size = 16
    prism_style()

    result = {"cochlea": [], "runlength": [], "value": []}
    marker_dict = {}
    alias_to_cohort = {}
    cochleae_length = []
    _meta = cohort_dict if cohort_dict is not None else IDISCO_COCHLEAE_DICT
    color_dict = cochlea_colors(_meta, tonotopic_data, use_alias)
    for name, values in tonotopic_data.items():
        alias = cochlea_label(name, _meta[name], use_alias)
        marker_dict[alias] = _meta[name].get("marker", "o")
        if "cohort" in _meta[name]:
            alias_to_cohort[alias] = _meta[name]["cohort"]
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

    if length_info:
        avg_length = sum(cochleae_length) / len(cochleae_length)
        print(f"Average total length: {round(avg_length, 2)} µm")
        print(f"Average length per bin: {round(avg_length / n_bins, 2)} µm")

    result = pd.DataFrame(result)
    fig, ax = plt.subplots(figsize=(6.7, 6.5 if show_legend else 5))

    for name, grp in result.groupby("cochlea"):
        run_length = list(grp["runlength"])
        syn_count = list(grp["value"])
        run_length = [r for num, r in enumerate(run_length) if not np.isnan(syn_count[num])]
        syn_count = [s for s in syn_count if not np.isnan(s)]
        if name == "Meyer":
            ax.plot(run_length, syn_count, label=name,
                    markeredgecolor=color_dict[name],
                    color=color_dict[name], marker='s', linestyle='solid', linewidth=2)
        else:
            ax.scatter(run_length, syn_count, label=name,
                       color=color_dict[name], marker=marker_dict[name], alpha=alpha)

    # Build trend dict(s): one per cohort when trendline_colors is set, otherwise one combined.
    def _build_trend_dict(aliases):
        td = {}
        for name, grp in result.groupby("cochlea"):
            if name not in aliases:
                continue
            for r, s in zip(grp["runlength"], grp["value"]):
                td.setdefault(r, []).append(s)
        return td

    def _draw_trendline(ax, trend_dict, color):
        x_pos = list(trend_dict.keys())
        center_line = [sum(v) / len(v) for v in trend_dict.values()]
        val_std = [np.std(v) for v in trend_dict.values()]
        lower_std = [m - s for m, s in zip(center_line, val_std)]
        upper_std = [m + s for m, s in zip(center_line, val_std)]

        if errorbar:
            ax.errorbar(x_pos, center_line, val_std,
                        linestyle="dashed", marker="D", color=color, linewidth=1)
        if trendline:
            ax.plot(x_pos, center_line, linestyle="dashed", color=color,
                    alpha=line_alphas["center"], linewidth=3, zorder=2)
            ax.plot(x_pos, upper_std, linestyle="solid", color=color,
                    alpha=line_alphas["upper"], zorder=0)
            ax.plot(x_pos, lower_std, linestyle="solid", color=color,
                    alpha=line_alphas["lower"], zorder=0)
            plt.fill_between(x_pos, lower_std, upper_std,
                             color=color, alpha=line_alphas["fill"], interpolate=True)

    non_meyer = {a for a in color_dict if a != "Meyer"}
    if trendline_colors and alias_to_cohort:
        for cohort, color in trendline_colors.items():
            cohort_aliases = {a for a in non_meyer if alias_to_cohort.get(a) == cohort}
            if cohort_aliases:
                _draw_trendline(ax, _build_trend_dict(cohort_aliases), color)
    elif trendline or errorbar:
        _draw_trendline(ax, _build_trend_dict(non_meyer), "gray")

    if top_axis:
        ax_top = ax.twiny()
        freq_ticks = np.array([2, 4, 8, 16, 32, 64])  # kHz
        # Inverse of the mouse Greenwood mapping: length_fraction = log10(f / A) / a
        var_A = 1.46
        var_a = 1.77
        length_positions = np.log10(freq_ticks / var_A) / var_a
        ax_top.set_xticks(length_positions)
        ax_top.set_xticklabels([f"{f}" for f in freq_ticks], fontsize=tick_size)
        ax_top.set_xlabel("Frequency [kHz]", fontsize=main_label_size)
        ax_top.set_xlim(ax.get_xlim())

    ax.tick_params(axis='x', labelsize=tick_size)
    ax.tick_params(axis='y', labelsize=tick_size)
    ax.set_xlabel("Length fraction", fontsize=main_label_size)
    ax.set_ylabel("Synapse per IHC", fontsize=main_label_size)

    plt.tight_layout()

    if show_legend:
        color = [color_dict[key] for key in color_dict.keys()]
        marker = [marker_dict[key] for key in marker_dict.keys()]
        label = [k for k in marker_dict.keys()]
        for num, lbl in enumerate(label):
            if lbl == "Meyer":
                label[num] = "Meyer et al. 2009"

        handles = [get_marker_handle(c, m) for (c, m) in zip(color, marker)]

        if trendline and trendline_colors:
            for cohort_name, trendline_color in trendline_colors.items():
                handles.insert(-1, get_flatline_handle(trendline_color))
                label.insert(-1, cohort_name)

        ncol = (len(label) + 1) // 2
        ax.legend(handles, label, loc=(-0.15, -0.3),
                  ncol=ncol, framealpha=1, frameon=False)
        fig.subplots_adjust(bottom=0.3)

    save_figure(save_path, plot)


def plot_synapses_per_band(
    tonotopic_data: dict,
    save_path: str,
    plot: bool = False,
    use_alias: bool = True,
    trendline: bool = False,
    cochleae_dict: Optional[dict] = None,
):
    """Plot the synapses per IHC per octave band, one point per cochlea and band.

    Args:
        tonotopic_data: Dictionary of tonotopic data.
        save_path: Save path for figure.
        plot: Plot figure.
        use_alias: Use cochleae aliases.
        trendline: Visualize trendline of averages.
        cochleae_dict: Mapping of cochlea name to its metadata. Defaults to IDISCO_COCHLEAE_DICT.
    """
    prism_style()
    main_label_size = 28
    main_tick_size = 20
    xtick_size = 16

    cochleae_dict = IDISCO_COCHLEAE_DICT if cochleae_dict is None else cochleae_dict
    color_dict = cochlea_colors(cochleae_dict, tonotopic_data, use_alias)
    binned = {
        cochlea_label(name, cochleae_dict[name], use_alias): frequency_mapping(
            values["frequency[kHz]"].values, values["syn_per_IHC"].values, animal="mouse")
        for name, values in tonotopic_data.items()
    }
    result, bin_labels = band_table(binned)

    fig, ax = plt.subplots(figsize=(7, 5))

    offset = 0.08
    trend_dict = {}
    for num, (name, grp) in enumerate(result.groupby("cochlea")):
        x_positions = [x - len(grp["x_pos"]) // 2 * offset + offset * num for x in grp["x_pos"]]
        ax.scatter(x_positions, grp["value"], marker="o", label=name, s=80, alpha=1, color=color_dict[name])

        if trendline:
            sorted_idx = np.argsort(x_positions)
            trend_dict[name] = {"x_sorted": np.array(x_positions)[sorted_idx],
                                "y_sorted": np.array(grp["value"])[sorted_idx]}

    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels)
    ax.tick_params(axis='x', labelsize=xtick_size)
    ax.tick_params(axis='y', labelsize=main_tick_size)
    ax.set_xlabel("Octave band [kHz]", fontsize=main_label_size)
    ax.set_ylabel("Syn. per IHC", fontsize=main_label_size)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    if trendline:
        x_sorted, y_sorted, y_sorted_upper, y_sorted_lower = band_trendline(trend_dict)
        ax.plot(x_sorted, y_sorted, linestyle="dotted", color="gray", alpha=0.6, linewidth=3, zorder=2)
        ax.plot(x_sorted, y_sorted_upper, linestyle="solid", color="gray", alpha=0.08, zorder=0)
        ax.plot(x_sorted, y_sorted_lower, linestyle="solid", color="gray", alpha=0.08, zorder=0)
        plt.fill_between(x_sorted, y_sorted_lower, y_sorted_upper, color="gray", alpha=0.05, interpolate=True)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)
