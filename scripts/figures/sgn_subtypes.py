"""SGN subtype plots: subtype fractions, subtypes per octave band, offsets and marker thresholds."""
import json
import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flamingo_tools.postprocessing.sgn_subtype_utils import ALIAS, COCHLEAE, CUSTOM_THRESHOLDS
from flamingo_tools.s3_utils import read_table
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from util import (
    custom_formatter, export_legend, frequency_mapping, get_flatline_handle, paired_x_positions,
    prism_cleanup_axes, prism_style, save_figure, save_legend,
)

THRESHOLD_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/Subtype_marker"  # noqa

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

COLORS_OFFSET = {
    "M_10L": "#D63637",
    "M_10R": "#237556",
    "M_11L": "#5A36D6",
}

COLORS_THRESHOLD = {
    "CR": {"Positive": "#E26F0B", "Negative": "#4177AA"},
    "Ntng1": {"Positive": "#AA4D41", "Negative": "#41AA53"},
}

MARKER_LEFT = "o"
MARKER_RIGHT = "^"


def _seg_name(cochlea: str) -> str:
    meta = COCHLEAE[cochlea]
    return meta["output_seg"] if "output_seg" in meta else meta["seg_data"]


def plot_subtype_per_band(
    results: dict,
    save_path: str,
    plot: bool = False,
    combine_IbIc: bool = False,
):
    """Plot the mean fraction of every subtype per octave band over several cochleae.

    The band around the mean is the standard deviation over the cochleae.

    Args:
        results: Classification and frequencies per cochlea, as returned by data.get_subtype_data.
        save_path: Path for saving figure.
        plot: Plot figure.
        combine_IbIc: Combine subtypes Ib and Ic for the analysis.
    """
    prism_style()
    main_label_size = 28
    main_tick_size = 20
    xtick_size = 16
    summary = {}
    for cochlea, result in results.items():
        classification = result["classification"]
        frequencies = result["frequencies"]
        cats = list(set([c[:c.find(" (")] for c in classification]))
        cats.sort()
        if "Type Ic" in cats and "Type II" in cats:
            # change order of "Type II" and "Type I"
            cats = cats[1:] + cats[:1]

        dic = {}
        for c in cats:
            sub_freq = [frequencies[i] for i in range(len(classification))
                        if classification[i][:classification[i].find(" (")] == c]
            mapping = frequency_mapping(sub_freq, [1 for _ in range(len(sub_freq))], aggregation="sum")
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
    ax.yaxis.set_major_formatter(custom_formatter(1))
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_subtype_fraction(
    results: dict,
    save_path: str,
    plot: bool = False,
):
    """Create stacked bar plots for multiple cochleae visualizing the fraction of subtypes.

    If the cochleae have the subtypes Ib and Ic, a frame encloses the two of them.

    Args:
        results: Classification and frequencies per cochlea, as returned by data.get_subtype_data.
        save_path: Path for saving figure.
        plot: Plot figure.
    """
    prism_style()
    main_label_size = 28
    main_tick_size = 20

    summary, types = {}, []
    for cochlea, result in results.items():
        alias = ALIAS[cochlea]
        classification = [cls[:cls.find(" (")] for cls in result["classification"]]
        n_tot = len(classification)

        types.extend(list(set(classification)))
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
            summary[alias][stype] = float(n_type) / n_tot

    df = pd.DataFrame(summary).fillna(0)  # missing values → 0

    colors = [COLORS[t] for t in types]
    width_graph = 6.7 * len(results) / 3

    ax = df.T.plot(
        kind="bar",
        stacked=True,
        figsize=(width_graph, 5),
        color=colors,
        legend=False,
    )

    if "Type Ib" in types and "Type Ic" in types:
        for num in range(len(results)):
            # patches are ordered bottom to top
            patches = ax.patches[num::len(results)]
            x, width = patches[0].get_x(), patches[0].get_width()

            segment_info = []
            for p, subtype in zip(patches, types):
                y0 = p.get_y()
                segment_info.append((subtype, y0, y0 + p.get_height()))

            outlined_block = {"Type Ib", "Type Ic"}
            y_bottom = min(y0 for subtype, y0, y1 in segment_info if subtype in outlined_block)
            y_top = max(y1 for subtype, y0, y1 in segment_info if subtype in outlined_block)

            for color, linewidth, zorder in ((COLORS["bbox_outer"], 5, 1), (COLORS["bbox_inner"], 3, 2)):
                ax.add_patch(plt.Rectangle(
                    (x, y_bottom), width, y_top - y_bottom,
                    linewidth=linewidth, edgecolor=color, facecolor="none", zorder=zorder,
                ))

    ax.set_ylabel("Fraction", fontsize=main_label_size)

    plt.yticks(fontsize=main_tick_size)
    plt.xticks(rotation=0, fontsize=main_tick_size)
    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_subtype_legend(
    save_path: str,
    grouping: str,
    ncol: Optional[int] = None,
    outlined_subtypes: Optional[List[str]] = None,
):
    """Plot the legend of the subtype panels.

    Args:
        save_path: File path to output.
        grouping: String for identification of subtypes. Subtypes are separated by ";"
        ncol: Number of columns for legend.
        outlined_subtypes: Subtypes to enclose in one shared frame.
    """
    subtypes = grouping.split(";")
    labels = [LEGEND_LABEL.get(label, label) for label in subtypes]
    colors = [COLORS[subtype] for subtype in subtypes]
    if ncol is None:
        ncol = len(labels)

    handles = [get_flatline_handle(c) for c in colors]
    legend = plt.legend(handles, labels, loc=3, ncol=ncol, framealpha=1, frameon=False)
    if outlined_subtypes:
        fig = legend.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        legend_handles = legend.get_lines()
        legend_texts = legend.get_texts()
        indices = [subtypes.index(subtype) for subtype in outlined_subtypes]
        extents = [
            artist.get_window_extent(renderer)
            for index in indices
            for artist in (legend_handles[index], legend_texts[index])
        ]
        bbox = matplotlib.transforms.Bbox.union(extents)
        padding = 3
        lower_left = fig.transFigure.inverted().transform((bbox.x0 - padding, bbox.y0 - padding))
        upper_right = fig.transFigure.inverted().transform((bbox.x1 + padding, bbox.y1 + padding))
        width, height = upper_right - lower_left
        for color, linewidth in ((COLORS["bbox_outer"], 2), (COLORS["bbox_inner"], 1)):
            outline = plt.Rectangle(
                lower_left, width, height, transform=fig.transFigure,
                linewidth=linewidth, edgecolor=color, facecolor="none", clip_on=False,
            )
            fig.add_artist(outline)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def _prph_cochleae() -> List[str]:
    """The cochleae with Prph as the only subtype stain, which separate Type I from Type II."""
    return [c for c in COCHLEAE.keys() if
            len(COCHLEAE[c]["subtype_stains"]) == 1 and
            "Prph" in COCHLEAE[c]["subtype_stains"]]


def plot_offset_legend(
    save_path: str,
    ncol: Optional[int] = None,
    cochleae: Optional[List[str]] = None,
):
    """Plot the legend of plot_type_offset.

    Args:
        save_path: Path to save legend.
        ncol: Number of columns for legend.
        cochleae: Cochleae of the legend. Defaults to the cochleae with Prph as only subtype stain.
    """
    labels = [ALIAS[k] for k in (_prph_cochleae() if cochleae is None else cochleae)]
    colors = [COLORS_OFFSET[label] for label in labels]
    handles = [get_flatline_handle(c) for c in colors]
    save_legend(handles, labels, save_path, ncol=len(labels) if ncol is None else ncol)


def plot_type_offset(
    save_path: str,
    plot: bool = False,
    plot_type: str = "mean",
    seed: int = 0,
    cochleae: Optional[List[str]] = None,
):
    """Plot the offset of Type I and Type II SGNs from the central path of Rosenthal's canal.

    Args:
        save_path: Path for saving figure.
        plot: Plot figure.
        plot_type: Plot mode. Either "mean" for mean offset or "points" for point cloud representation.
        seed: Seed of the horizontal jitter of the "points" mode.
        cochleae: Cochleae to plot, each with one color of COLORS_OFFSET. Defaults to the cochleae
            with Prph as only subtype stain.
    """
    cochleae = _prph_cochleae() if cochleae is None else cochleae

    dic = {}
    for cochlea in cochleae:
        table = read_table(f"{cochlea}/tables/{_seg_name(cochlea)}/default.tsv", s3=True)
        subset = table[table["subtype_label"].isin(["Type I", "Type II"])]
        dic[cochlea] = {
            "Type I": list(subset.loc[subset["subtype_label"] == "Type I", "offset"]),
            "Type II": list(subset.loc[subset["subtype_label"] == "Type II", "offset"]),
        }

    prism_style()
    alias = [ALIAS[k] for k in cochleae]
    main_label_size = 20
    sub_label_size = 16

    fig, ax = plt.subplots(figsize=(4, 5))
    if plot_type == "mean":
        values_left = [np.mean(dic[cochlea]["Type I"]) for cochlea in cochleae]
        values_right = [np.mean(dic[cochlea]["Type II"]) for cochlea in cochleae]

        x_left = 1
        x_right = 2
        offset = 0.08

        x_pos_left = paired_x_positions(len(values_left), x_left, offset)
        x_pos_right = paired_x_positions(len(values_right), x_right, offset)

        for num, key in enumerate(COLORS_OFFSET.keys()):
            plt.scatter(x_pos_left[num], values_left[num], label=f"{alias[num]}",
                        color=COLORS_OFFSET[key], marker=MARKER_LEFT, s=80, zorder=1)
            plt.scatter(x_pos_right[num], values_right[num],
                        color=COLORS_OFFSET[key], marker=MARKER_RIGHT, s=80, zorder=1)

        # lines between the two types of one cochlea
        for num, (left, right) in enumerate(zip(values_left, values_right)):
            ax.plot([x_pos_left[num], x_pos_right[num]], [left, right],
                    linestyle="solid", color="grey", alpha=0.4, zorder=0)

        plt.xticks([x_left, x_right], ["Type I", "Type II"], fontsize=sub_label_size)
        for la in plt.gca().get_xticklabels():
            la.set_verticalalignment('center')
        ax.tick_params(axis='x', which='major', pad=16)
        plt.ylabel("Mean offset [µm]", fontsize=main_label_size)
        plt.xlim(0.5, 2.5)

    elif plot_type == "points":
        rng = np.random.RandomState(seed)
        offset = 0.1
        x_left = 1 - offset * (len(cochleae) // 2)
        x_right = 2 - offset * (len(cochleae) // 2)
        jitter = 0.08  # horizontal scatter for visibility

        for num, cochlea in enumerate(cochleae):
            dist_left = dic[cochlea]["Type I"]
            dist_right = dic[cochlea]["Type II"]

            color = list(COLORS_OFFSET.values())[num % len(COLORS_OFFSET)]
            alias_name = alias[num] if num < len(alias) else f"C{num + 1}"

            x_left_jitter = x_left + offset * num + rng.uniform(-jitter, jitter, size=len(dist_left))
            x_right_jitter = x_right + offset * num + rng.uniform(-jitter, jitter, size=len(dist_right))

            ax.scatter(x_left_jitter, dist_left,
                       color=color, marker=MARKER_LEFT, alpha=0.7, s=1, label=alias_name if num == 0 else "", zorder=2)
            ax.scatter(x_right_jitter, dist_right,
                       color=color, marker=MARKER_RIGHT, alpha=0.7, s=1, zorder=2)

        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["Type I", "Type II"], fontsize=sub_label_size)
        ax.tick_params(axis='x', which='major', pad=16)
        ax.set_ylabel("Offset [µm]", fontsize=main_label_size)
        ax.set_xlim(0.5, 2.5)

    else:
        raise ValueError("Choose either 'mean' or 'points' for parameter plot_type.")

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_threshold_legend(
    save_path: str,
    stain: str,
    ncol: int = 1,
):
    """Plot the legend of plot_marker_threshold_histograms for one stain.

    Args:
        save_path: Path to save legend.
        stain: Stain for thresholding.
        ncol: Number of columns in legend.
    """
    labels = ["Positive", "Negative"]
    handles = [get_flatline_handle(COLORS_THRESHOLD[stain][label]) for label in labels]
    save_legend(handles, labels, save_path, ncol=len(labels) if ncol is None else ncol)


def plot_marker_threshold_histograms(
    save_paths: Dict[str, str],
    cochlea: str,
    plot: bool = False,
    sharex: bool = True,
    sharey: bool = True,
    title_type: str = "generic",
):
    """Plot histograms for positive and negative populations of subtype markers based on thresholding.

    The function writes one figure per stain of the cochlea, with one histogram per annotated crop
    in two rows.

    Args:
        save_paths: Save path per stain. Must hold every stain with a threshold file of the
            cochlea in THRESHOLD_DIR.
        cochlea: Key of sgn_subtype_utils.COCHLEAE.
        plot: Plot figure.
        sharex: Shared x-axis for all subplots.
        sharey: Shared y-axis for all subplots.
        title_type: Type for showing subplot titles. Either "generic" - Crop x - or "center_str".
    """
    prism_style()
    om_paths = [entry.path for entry in os.scandir(THRESHOLD_DIR) if "_om.json" in entry.name or
                "_crop-intensity.json" in entry.name]
    cochlea_str = cochlea.replace('_', '-')
    om_paths = [p for p in om_paths if cochlea_str in p]
    stains = [os.path.basename(p).split(f"{cochlea_str}_")[1].split("_")[0] for p in om_paths]

    assert sorted(save_paths) == sorted(stains), f"Expected save paths for the stains {stains}."

    data_name = COCHLEAE[cochlea]["seg_data"]
    stain_str = " and ".join(stains)
    print(f"Evaluating {cochlea} with stains {stain_str}.")
    for stain, json_file in zip(stains, om_paths):
        save_path = save_paths[stain]

        try:
            with open(json_file, "r") as myfile:
                param_dicts = json.loads(myfile.read())
        except PermissionError as e:
            print(e)
            continue
        center_strs = param_dicts["center_strings"]
        center_strs.sort()
        intensity_mode = param_dicts["intensity_mode"]
        number_plots = len(center_strs)
        if number_plots == 0:
            print(f"No columns for cochlea {cochlea}.")
            continue

        seg_str = data_name.replace('_', '-')
        if intensity_mode == "ratio":
            table_measurement_path = f"{cochlea}/tables/{data_name}/subtype_ratio.tsv"
            column = f"{stain}_ratio_PV"
        elif intensity_mode == "absolute":
            table_measurement_path = f"{cochlea}/tables/{data_name}/{stain}_{seg_str}_object-measures.tsv"
            column = "median"

        if cochlea in CUSTOM_THRESHOLDS and stain in CUSTOM_THRESHOLDS[cochlea]:
            threshold_type = "custom"
        else:
            threshold_type = "manual"

        rows = 2
        columns = number_plots // rows

        main_label_size = 20
        x_label_size = 16
        main_tick_size = 16
        fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.5), sharex=sharex, sharey=sharey)
        plt.xlim([-0.1, 8])
        ax = axes.flatten()
        table_measurement = read_table(table_measurement_path, s3=True)

        for num, center_str in enumerate(center_strs):
            seg_ids = param_dicts[center_str]["seg_ids"]
            if threshold_type == "manual":
                threshold = float(param_dicts[center_str]["median_intensity"][0])
            else:
                threshold_dic = CUSTOM_THRESHOLDS[cochlea][stain]
                if isinstance(threshold_dic, (int, float)):
                    threshold = threshold_dic
                else:
                    threshold = threshold_dic[center_str]["manual"]

            subset = table_measurement[table_measurement["label_id"].isin(seg_ids)]
            neg_values = list(subset[(subset[column] < threshold)][column])
            pos_values = list(subset[(subset[column] > threshold)][column])

            bins = np.linspace(min(neg_values + pos_values), max(neg_values + pos_values), 30)

            ax[num].hist(pos_values, bins=bins, alpha=0.6,
                         label='Positive', color=COLORS_THRESHOLD[stain]["Positive"])
            ax[num].hist(neg_values, bins=bins, alpha=0.6,
                         label='Negative', color=COLORS_THRESHOLD[stain]["Negative"])
            if num % columns == 0:
                ax[num].set_ylabel('Count', fontsize=main_label_size)
            if num >= columns:
                ax[num].set_xlabel('Intensity [a.u.]', fontsize=x_label_size)
            ax[num].tick_params(axis='x', labelsize=main_tick_size)
            ax[num].tick_params(axis='y', labelsize=main_tick_size)
            if title_type == "center_str":
                ax[num].set_title(center_str, fontsize=main_label_size)
            else:
                ax[num].set_title(f"Crop {str(num + 1).zfill(1)}")

        plt.tight_layout()
        save_figure(save_path, plot)


def plot_subtype_confusion_matrix(
    save_path: str,
    plot: bool = False,
    include_typeii: bool = False,
):
    """Plot the confusion matrix of the subtype assignment with CR + Ntng1 against CR + Lypd1.

    Args:
        save_path: Path for saving figure.
        plot: Plot figure.
        include_typeii: Include the boxes for Type II, shown as "indet.".
    """
    main_label_size = 32
    color_tick_size = 24

    if include_typeii:
        subtypes = ["Type Ia", "Type Ib", "Type Ic", "Type II"]
        text_size = 16
        tick_size = 16
    else:
        subtypes = ["Type Ia", "Type Ib", "Type Ic"]
        text_size = 20
        tick_size = 20

    table = read_table("M_AMD_N180_L/tables/SGN_merged/default.tsv", s3=True)

    subset = table[table["subtype_label"].isin(subtypes)]
    if include_typeii:
        subtypes = ["Type Ia", "Type Ib", "Type Ic", "indet."]
        subset.loc[subset["subtype_label"] == "Type II", "subtype_label"] = "indet."
        subset.loc[subset["subtype_label_Lypd1"] == "Type II", "subtype_label_Lypd1"] = "indet."

    cm_counts = confusion_matrix(
        subset['subtype_label'], subset['subtype_label_Lypd1'], labels=subtypes, normalize=None
    )
    cm_norm = confusion_matrix(
        subset['subtype_label'], subset['subtype_label_Lypd1'], labels=subtypes, normalize='true'
    )

    disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=subtypes)
    disp.plot(cmap='Oranges', xticks_rotation=0, values_format=".2f")
    disp.ax_.get_images()[0].set_clim(0, 1)

    # Replace the default annotations with the normalized value and the count.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            disp.text_[i, j].remove()
            if cm_counts[i, j] != 0:
                text_str = f"{cm_norm[i, j]:.2f}\nn={cm_counts[i, j]}"
            else:
                text_str = f"{cm_norm[i, j]:.2f}"
            plt.text(j, i, text_str, ha="center", va="center", fontsize=text_size)

    cbar = plt.gca().images[0].colorbar
    cbar.ax.tick_params(labelsize=color_tick_size)

    plt.xticks(fontsize=tick_size)
    plt.yticks(fontsize=tick_size)
    plt.xlabel("CR + Lypd1", fontsize=main_label_size)
    plt.ylabel("CR + Ntng1", fontsize=main_label_size)
    save_figure(save_path, plot)
