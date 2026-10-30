import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.analysis.central_path_utils import mean_by_offset
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT
from flamingo_tools.s3_utils import create_s3_target

from util import (
    literature_reference_values,
    prism_style, prism_cleanup_axes,
    ax_prism_boxplot, average_by_fraction, save_figure,
    VALUE_DICT, COHORT_DICT, cohort_cochleae, cochleae_for, OUTLIER_DICT,
)
from data import add_synapses_per_ihc, get_mapped_table
from tonotopic_mapping import add_frequency_axis, plot_synapses_along_cochlea

FILE_EXTENSION = "png"

COHORTS = ["mwfls", "idisco"]
# The SGN segmentation of each cohort. The IHC segmentation and the synapse detection of a
# cochlea come from SYNAPSE_DICT.
SGN_VERSION = {"idisco": "SGN_v2", "mwfls": "SGN_v3"}
STAINS = {"SGN": "PV", "IHC": "Vglut3"}

COLOR_LITERATURE = "#9C2E53"
COLOR_OUTLIER = "red"
COHORT_MARKERS = {
    "idisco": "o",
    "mwfls": "s",
}
MEYER_COLORS = ["#10CC96", "#10CC56", "#57CC10", "#A6CC10"]

MAIN_LABEL_SIZE = 20
MAIN_TICK_SIZE = 16
SUB_LABEL_SIZE = 16


def load_tables(cohorts=COHORTS, intensity=("SGN", "IHC")):
    """Load the SGN and IHC instances of every cochlea of the cohorts.

    The tables are mapped again in memory, see data.get_mapped_table. The IHC tables get the
    column 'syn_per_IHC'.

    Args:
        cohorts: Keys of COHORT_DICT.
        intensity: Structures that get the column 'intensity' from the object measures of their stain.

    Returns:
        Mapping of structure ("SGN" / "IHC") to a mapping of cochlea name to table.
    """
    s3 = create_s3_target()
    tables = {"SGN": {}, "IHC": {}}
    for cohort in cohorts:
        for cochlea in cohort_cochleae(cohort):
            versions = {"SGN": SGN_VERSION[cohort], "IHC": SYNAPSE_DICT[cochlea]["ihc_table_name"]}
            for structure, version in versions.items():
                print("Loading", cochlea, version)
                stain = STAINS[structure] if structure in intensity else None
                table = get_mapped_table(s3, cochlea, structure, version, stain)
                expected = VALUE_DICT[cochlea][structure][version]["count"]
                if len(table) != expected:
                    raise ValueError(
                        f"{cochlea} {version}: {len(table)} mapped instances, VALUE_DICT documents {expected}."
                    )
                tables[structure][cochlea] = table
            tables["IHC"][cochlea] = add_synapses_per_ihc(
                s3, cochlea, tables["IHC"][cochlea], SYNAPSE_DICT[cochlea]["synapse_table_name"],
            )
    return tables


def _set_cohort_xaxis(ax, x_positions):
    ax.set_xlim(x_positions[0] - 0.5, x_positions[-1] + 0.5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([COHORT_DICT[c]["label"] for c in COHORTS], fontsize=SUB_LABEL_SIZE)
    ax.tick_params(axis="x", which="major", pad=8)


def fig_cohort_boxplot(tables: dict, save_path: str, plot: bool = False, remove_outliers: bool = False):
    """Cohort-comparison boxplot: iDISCO vs MWfLS for SGN, IHC, and synapses per IHC.

    The SGN and IHC boxes hold one count per cochlea; load_tables checks these against VALUE_DICT.
    The synapse box pools the synapse counts of all IHCs.

    Args:
        tables: Output of load_tables.
        save_path: Output file path.
        plot: Show interactive plot.
        remove_outliers: Draw the cochleae of OUTLIER_DICT as a red cross instead of in the box.
            The cross shows the count, or the mean synapse count per IHC.
    """
    prism_style()
    x_positions = [1, 2]

    def _values(structure, cohort, outlier):
        values = []
        for cochlea in cohort_cochleae(cohort):
            if (cochlea in OUTLIER_DICT.get(structure, []) and remove_outliers) != outlier:
                continue
            if structure == "synapse":
                syn_per_ihc = tables["IHC"][cochlea]["syn_per_IHC"].values
                values.extend([syn_per_ihc.mean()] if outlier else syn_per_ihc)
            else:
                values.append(len(tables[structure][cochlea]))
        return values

    def _draw_subplot(ax, structure, ylim0, ylim1, y_ticks, ylabel=None, title=None, show_literature=False):
        for pos, cohort in zip(x_positions, COHORTS):
            ax_prism_boxplot(ax, _values(structure, cohort, False), positions=[pos], color=COHORT_DICT[cohort]["color"])
            outliers = _values(structure, cohort, True)
            ax.scatter([pos] * len(outliers), outliers, color=COLOR_OUTLIER, zorder=2, marker="x", s=30, alpha=0.7)

        lower_y, upper_y = literature_reference_values(structure)
        _set_cohort_xaxis(ax, x_positions)
        xmin, xmax = ax.get_xlim()
        ax.hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
        ax.fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)
        if show_literature:
            ax.text(1., lower_y + (ylim1 - ylim0) * 0.01, "literature",
                    color=COLOR_LITERATURE, fontsize=MAIN_LABEL_SIZE, ha="center")

        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=MAIN_TICK_SIZE)
        ax.set_ylim(ylim0, ylim1)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=MAIN_LABEL_SIZE)
        ax.set_title(structure if title is None else title, fontsize=MAIN_LABEL_SIZE, pad=10)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    _draw_subplot(
        axes[0], "SGN", ylim0=1000, ylim1=12500, y_ticks=list(range(2000, 12001, 2000)),
        ylabel="Count per cochlea", show_literature=True,
    )
    _draw_subplot(axes[1], "IHC", ylim0=400, ylim1=800, y_ticks=list(range(400, 801, 100)))
    _draw_subplot(
        axes[2], "synapse", title="Synapses per IHC", ylim0=-1, ylim1=41, y_ticks=[0, 10, 20, 30, 40],
    )

    prism_cleanup_axes(axes)
    plt.tight_layout()
    save_figure(save_path, plot)


def fig_mwfls_synapses(
    tables: dict,
    save_path: str,
    n_bins: int = 25,
    top_axis: bool = False,
    trendline: bool = False,
    trendline_std: bool = False,
    errorbar: bool = False,
    plot: bool = False,
    show_legend: bool = False,
):
    """Synapse-per-IHC vs. length-fraction for both iDISCO and MWfLS cohorts.

    iDISCO cochleae are shown with marker 'o'; MWfLS with marker 's'. Colors differ within each
    cohort (per cochlea). The synapse outliers of OUTLIER_DICT are drawn as a red cross and stay
    out of the cohort trendline.

    Args:
        tables: Output of load_tables.
        save_path: Output file path.
        n_bins: Number of fractional bins along the cochlea length.
        top_axis: Add frequency axis on top.
        trendline: Overlay the mean trendline of each cohort.
        trendline_std: Also show ±1 SD band around the trendline.
        errorbar: Show errorbar instead of trendline fill.
        plot: Show interactive plot.
        show_legend: Show legend within plot.
    """
    cohort_dict = {}
    for cohort in COHORTS:
        for num, name in enumerate(cohort_cochleae(cohort)):
            entry = cochleae_for([name], "IHC", SYNAPSE_DICT[name]["ihc_table_name"])[name]
            entry["marker"] = COHORT_MARKERS[cohort]
            entry["cohort"] = COHORT_DICT[cohort]["label"]
            if cohort == "idisco":
                entry["color"] = MEYER_COLORS[num % len(MEYER_COLORS)]
            if name in OUTLIER_DICT["synapse"]:
                # A cohort name outside trendline_colors keeps the cochlea out of every trendline.
                entry.update(color=COLOR_OUTLIER, marker="x", cohort="outlier")
            cohort_dict[name] = entry

    plot_synapses_along_cochlea(
        tonotopic_data={name: tables["IHC"][name] for name in cohort_dict},
        save_path=save_path,
        cohort_dict=cohort_dict,
        trendline_colors={COHORT_DICT[c]["label"]: COHORT_DICT[c]["color"] for c in COHORTS},
        use_alias=True,
        plot=plot,
        n_bins=n_bins,
        top_axis=top_axis,
        trendline=trendline,
        trendline_std=trendline_std,
        errorbar=errorbar,
        show_legend=show_legend,
    )


def plot_intensity_boxplot(
    structure_tables: dict, title: str, save_path: str, per_cochlea: bool = True, plot: bool = False,
):
    """Box plot of the stain intensity, one box per cohort.

    By default, a box holds the mean intensity of each cochlea of the cohort. Without per_cochlea, a
    box pools all instances of all cochleae of the cohort. The fliers of this variant are hidden,
    because a box holds thousands of instances.

    Args:
        structure_tables: Tables of one structure, a value of load_tables.
        title: Title of the plot.
        save_path: Output file path.
        per_cochlea: Use the mean intensity of each cochlea instead of every instance.
        plot: Show interactive plot.
    """
    prism_style()
    x_positions = [1, 2]
    fig, ax = plt.subplots(figsize=(4, 4.5))
    for pos, cohort in zip(x_positions, COHORTS):
        intensities = [structure_tables[c]["intensity"].values for c in cohort_cochleae(cohort)]
        values = [i.mean() for i in intensities] if per_cochlea else np.concatenate(intensities)
        box = ax_prism_boxplot(ax, values, positions=[pos], color=COHORT_DICT[cohort]["color"])
        if not per_cochlea:
            for flier in box["fliers"]:
                flier.set_visible(False)
    ax.relim(visible_only=True)
    ax.autoscale_view()
    ax.set_ylim(bottom=0)

    _set_cohort_xaxis(ax, x_positions)
    ax.tick_params(axis="y", labelsize=MAIN_TICK_SIZE)
    ax.set_ylabel("Mean intensity [a.u.]" if per_cochlea else "Intensity [a.u.]", fontsize=MAIN_LABEL_SIZE)
    ax.set_title(title, fontsize=MAIN_LABEL_SIZE, pad=10)
    prism_cleanup_axes(ax)
    plt.tight_layout()
    save_figure(save_path, plot)


def _plot_cohort_profiles(ax, profiles: dict):
    """Draw the per-cochlea profiles as points and the cohort mean +/- 1 SD over the cochleae as a line.

    Args:
        ax: The axis.
        profiles: Mapping of cochlea name to a series of the value per x position.
    """
    for cohort in COHORTS:
        color = COHORT_DICT[cohort]["color"]
        values = pd.DataFrame({name: profiles[name] for name in cohort_cochleae(cohort)})
        for name in values:
            ax.scatter(values.index, values[name], color=color, marker=COHORT_MARKERS[cohort], alpha=0.4, s=20)
        mean, std = values.mean(axis=1), values.std(axis=1, ddof=0)
        ax.plot(values.index, mean, color=color, linewidth=3, label=COHORT_DICT[cohort]["label"])
        ax.fill_between(values.index, mean - std, mean + std, color=color, alpha=0.15)
    ax.tick_params(axis="both", labelsize=MAIN_TICK_SIZE)
    ax.set_ylabel("Intensity [a.u.]", fontsize=MAIN_LABEL_SIZE)
    ax.legend(frameon=False, fontsize=SUB_LABEL_SIZE)


def plot_intensity_by_offset(
    structure_tables: dict, save_path: str, offset_step: float = 10.0, max_offset: float = 100.0,
    min_count: int = 10, plot: bool = False,
):
    """Plot the mean stain intensity against the offset to the central path of Rosenthal's canal.

    Each point is the mean of one cochlea in one offset bin, drawn at the bin center. The last bin
    holds every offset of `max_offset` or more.

    Args:
        structure_tables: Tables of one structure, a value of load_tables.
        save_path: Output file path.
        offset_step: Width of an offset bin in micrometers.
        max_offset: Lower edge of the last, open offset bin in micrometers.
        min_count: Minimal number of instances of a cochlea in a bin.
        plot: Show interactive plot.
    """
    profiles = {}
    for name, table in structure_tables.items():
        binned = mean_by_offset(table, ["intensity"], n_length_bins=1, offset_step=offset_step, max_offset=max_offset)
        binned = binned[binned["n"] >= min_count]
        profiles[name] = pd.Series(
            binned["intensity"].values, index=binned["offset_bin"].astype(float).values + offset_step / 2,
        )

    prism_style()
    fig, ax = plt.subplots(figsize=(6.7, 5))
    _plot_cohort_profiles(ax, profiles)
    ax.set_xlim(0, max_offset + offset_step)
    ax.set_xlabel("Offset to RC center [µm]", fontsize=MAIN_LABEL_SIZE)
    prism_cleanup_axes(ax)
    plt.tight_layout()
    save_figure(save_path, plot)


def plot_intensity_along_cochlea(structure_tables: dict, save_path: str, n_bins: int = 20, plot: bool = False):
    """Plot the mean stain intensity against the length fraction, with the frequency as top axis.

    Args:
        structure_tables: Tables of one structure, a value of load_tables.
        save_path: Output file path.
        n_bins: Number of fractional bins along the cochlea length.
        plot: Show interactive plot.
    """
    profiles = {}
    for name, table in structure_tables.items():
        binned = average_by_fraction(table["length_fraction"].values, table["intensity"].values, n_bins=n_bins)
        profiles[name] = pd.Series(
            binned["mean_syn_per_IHC"].values, index=binned["fraction_midpoint"].values.astype(float),
        )

    prism_style()
    fig, ax = plt.subplots(figsize=(6.7, 5))
    _plot_cohort_profiles(ax, profiles)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Length fraction", fontsize=MAIN_LABEL_SIZE)
    add_frequency_axis(ax, MAIN_TICK_SIZE, MAIN_LABEL_SIZE)
    prism_cleanup_axes(ax)
    plt.tight_layout()
    save_figure(save_path, plot)


def main():
    parser = argparse.ArgumentParser(description="Generate MWfLS vs iDISCO comparison plots.")
    parser.add_argument("--figure_dir", "-f", type=str, default="./panels/mwfls",
                        help="Output directory for plots.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--intensity", nargs="*", choices=["SGN", "IHC"], default=["SGN", "IHC"],
                        help="Structures whose intensity panels to plot. Leave out a structure whose object "
                             "measures are not on S3.")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    def out(name):
        return os.path.join(args.figure_dir, f"{name}.{FILE_EXTENSION}")

    tables = load_tables(intensity=args.intensity)

    if "SGN" in args.intensity:
        plot_intensity_boxplot(tables["SGN"], "PV@SGNs", out("fig_02d_sgn_intensity_by_cohort"), plot=args.plot)
        plot_intensity_boxplot(
            tables["SGN"], "PV@SGNs", out("fig_02d_sgn_intensity_by_cohort_all_instances"), per_cochlea=False,
            plot=args.plot,
        )
        plot_intensity_by_offset(tables["SGN"], out("fig_02e_sgn_intensity_by_rc_offset"), plot=args.plot)
    if "IHC" in args.intensity:
        plot_intensity_boxplot(tables["IHC"], "VGlut3@IHCs", out("fig_02f_ihc_intensity_by_cohort"), plot=args.plot)
        plot_intensity_boxplot(
            tables["IHC"], "VGlut3@IHCs", out("fig_02f_ihc_intensity_by_cohort_all_instances"), per_cochlea=False,
            plot=args.plot,
        )
        plot_intensity_along_cochlea(tables["IHC"], out("fig_02g_ihc_intensity_by_tonotopy"), plot=args.plot)

    fig_cohort_boxplot(tables, out("fig_03a_counts_by_cohort"), plot=args.plot)
    fig_cohort_boxplot(tables, out("fig_03a_counts_by_cohort_outlier"), plot=args.plot, remove_outliers=True)
    fig_mwfls_synapses(tables, out("fig_03b_synapses_by_tonotopy"), plot=args.plot, top_axis=True)
    fig_mwfls_synapses(
        tables, out("fig_03b_synapses_by_tonotopy_trendline"), plot=args.plot, trendline=True, top_axis=True,
    )
    fig_mwfls_synapses(
        tables, out("fig_03b_synapses_by_tonotopy_trendline_legend"), plot=args.plot, trendline=True,
        top_axis=True, show_legend=True,
    )


if __name__ == "__main__":
    main()
