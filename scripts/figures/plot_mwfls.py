import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from flamingo_tools.analysis.seg_table_utils import add_object_measures_to_table
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT

from util import (
    literature_reference_values,
    prism_style, prism_cleanup_axes,
    ax_prism_boxplot,
    VALUE_DICT, COHORT_DICT,
    MWFLS_COCHLEAE_DICT, OUTLIER_DICT
)
from plot_fig2 import _load_ribbon_synapse_counts
from plot_fig3 import supp_fig_03a_meyer, COCHLEAE_DICT, get_tonotopic_data, SYNAPSE_DIR

png_dpi = 300
FILE_EXTENSION = "png"

COLOR_LITERATURE = "#9C2E53"
COHORT_COLORS = {
    "iDISCO": "#10CC17",
    "MWfLS": "#3F69FF",
}
COHORT_MARKERS = {
    "iDISCO": "o",
    "MWfLS": "s",
}


def fig_cohort_boxplot(
    save_path: str,
    idisco_syn_version: str = "v4c",
    mwfls_syn_version: str = "v9",
    plot: bool = False,
    remove_outliers: bool = False,
):
    """Cohort-comparison boxplot: iDISCO vs MWfLS for SGN, IHC, and synapses per IHC.

    Each of the three subplots shows two boxplots (one per cohort) with the instance
    name as subtitle and cohort labels on the x-axis.

    Args:
        save_path: Output file path.
        idisco_syn_version: Synapse table version suffix for iDISCO (default "v4c").
        mwfls_syn_version: Synapse table version suffix for MWfLS (default "v9").
        plot: Show interactive plot.
    """
    prism_style()
    main_label_size = 20
    main_tick_size = 16
    sub_label_size = 16

    cohorts = ["iDISCO", "MWfLS"]
    syn_versions = {"iDISCO": idisco_syn_version, "MWfLS": mwfls_syn_version}

    # Collect SGN / IHC counts per cohort.
    sgn_by_cohort = {}
    ihc_by_cohort = {}
    sgn_outlier = {}
    for cohort in cohorts:
        sgn_outlier[cohort] = [
            VALUE_DICT[name]["SGN"][0]["count"]
            for name in COHORT_DICT[cohort] if
            name in OUTLIER_DICT["SGN"] and
            remove_outliers
        ]
        sgn_by_cohort[cohort] = [
            VALUE_DICT[name]["SGN"][0]["count"]
            for name in COHORT_DICT[cohort] if
            name not in OUTLIER_DICT["SGN"] or
            not remove_outliers
        ]
        ihc_by_cohort[cohort] = [
            VALUE_DICT[name]["IHC"][0]["count"]
            for name in COHORT_DICT[cohort]
        ]

    # Collect synapse counts per cohort.
    syn_by_cohort = {}
    for cohort in cohorts:
        syn_by_cohort[cohort] = _load_ribbon_synapse_counts(
            ihc_version=syn_versions[cohort],
            cochleae=COHORT_DICT[cohort],
        )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))

    x_positions = [1, 2]

    def _draw_subplot(
        ax, data_by_cohort, structure, ylim0, ylim1, y_ticks,
        outlier=None, ylabel=None, title=None, show_literature=False,
    ):
        if title is None:
            title = structure
        for pos, cohort in zip(x_positions, cohorts):
            ax_prism_boxplot(ax, data_by_cohort[cohort], positions=[pos], color=COHORT_COLORS[cohort])
        if outlier is not None:
            for pos, cohort in zip(x_positions, cohorts):
                ax.scatter(
                    [pos] * len(outlier[cohort]),
                    outlier[cohort],
                    color="red",
                    zorder=2,
                    marker="x",
                    s=30,
                    alpha=0.7,
                )

        lower_y, upper_y = literature_reference_values(structure)
        xmin, xmax = 0.5, 2.5
        ax.set_xlim(xmin, xmax)
        ax.hlines([lower_y, upper_y], xmin, xmax, color=COLOR_LITERATURE)
        ax.fill_between([xmin, xmax], lower_y, upper_y, color=COLOR_LITERATURE, alpha=0.05, interpolate=True)

        if show_literature:
            ax.text(1., lower_y + (ylim1 - ylim0) * 0.01, "literature",
                    color=COLOR_LITERATURE, fontsize=main_label_size, ha="center")

        ax.set_xticks(x_positions)
        ax.set_xticklabels(cohorts, fontsize=sub_label_size)
        ax.tick_params(axis="x", which="major", pad=8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
        ax.set_ylim(ylim0, ylim1)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=main_label_size)

        ax.set_title(title, fontsize=main_label_size, pad=10)

    _draw_subplot(
        axes[0], sgn_by_cohort, "SGN",
        ylim0=1000, ylim1=12500,
        y_ticks=list(range(2000, 12001, 2000)),
        ylabel="Count per cochlea",
        outlier=sgn_outlier,
        show_literature=True,
    )
    _draw_subplot(
        axes[1], ihc_by_cohort, "IHC",
        ylim0=400, ylim1=800,
        y_ticks=list(range(400, 801, 100)),
    )
    _draw_subplot(
        axes[2], syn_by_cohort, "synapse", title="Synapses per IHC",
        ylim0=-1, ylim1=41,
        y_ticks=[0, 10, 20, 30, 40],
    )

    prism_cleanup_axes(axes)
    plt.tight_layout()

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def fig_mwfls_synapses(
    save_path: str,
    idisco_syn_version: str = "v4c",
    mwfls_syn_version: str = "v9",
    n_bins: int = 25,
    top_axis: bool = False,
    trendline: bool = False,
    trendline_std: bool = False,
    errorbar: bool = False,
    plot: bool = False,
    show_legend: bool = False,
):
    """Synapse-per-IHC vs. length-fraction for both iDISCO and MWfLS cohorts.

    iDISCO cochleae are shown with marker 'o'; MWfLS with marker 's'.
    Colors differ within each cohort (per cochlea).

    Args:
        save_path: Output file path.
        idisco_syn_version: Synapse table version suffix for iDISCO (default "v4c").
        mwfls_syn_version: Synapse table version suffix for MWfLS (default "v9").
        n_bins: Number of fractional bins along the cochlea length.
        top_axis: Add frequency axis on top.
        trendline: Overlay mean trendline across iDISCO cochleae.
        trendline_std: Also show ±1 SD band around the trendline.
        errorbar: Show errorbar instead of trendline fill.
        plot: Show interactive plot.
    """
    idisco_data = get_tonotopic_data(
        cochleae_dict=COCHLEAE_DICT,
        source_name=f"IHC_{idisco_syn_version}",
        synapse_dir=SYNAPSE_DIR,
    )
    mwfls_data = get_tonotopic_data(
        cochleae_dict=MWFLS_COCHLEAE_DICT,
        source_name=f"IHC_{mwfls_syn_version}",
        synapse_dir=SYNAPSE_DIR,
    )

    MEYER_COLORS = [
        "#10CC96",
        "#10CC56",
        "#57CC10",
        "#A6CC10",
    ]

    # Build a unified cohort_dict consumed by the generalized supp_fig_03a_meyer.
    merged_cohort_dict = {}
    for num, name in enumerate(COHORT_DICT["iDISCO"]):
        entry = dict(COCHLEAE_DICT[name])
        entry["color"] = MEYER_COLORS[num % len(MEYER_COLORS)]
        entry["marker"] = COHORT_MARKERS["iDISCO"]
        entry["cohort"] = "iDISCO"
        merged_cohort_dict[name] = entry
    for name in COHORT_DICT["MWfLS"]:
        entry = dict(MWFLS_COCHLEAE_DICT[name])
        entry["marker"] = COHORT_MARKERS["MWfLS"]
        entry["cohort"] = "MWfLS"
        merged_cohort_dict[name] = entry

    tonotopic_data = {**idisco_data, **mwfls_data}

    supp_fig_03a_meyer(
        tonotopic_data=tonotopic_data,
        save_path=save_path,
        cohort_dict=merged_cohort_dict,
        trendline_colors=COHORT_COLORS,
        use_alias=True,
        plot=plot,
        n_bins=n_bins,
        top_axis=top_axis,
        trendline=trendline,
        trendline_std=trendline_std,
        errorbar=errorbar,
        show_legend=show_legend,
    )


def plot_intensity(save_path, plot=False):
    """Plot intensities of stains at segmentation.
    Two cohorts, IDISCO and mwfls, are compared with each other in respect to the staining intensity.
    The intensities of PC and Vglut3 are compared at the position of SGN and IHC segmentation, respectively.
    """

    main_label_size = 20
    main_tick_size = 16
    sub_label_size = 16
    idisco = [c for c in COCHLEAE_DICT.keys()]
    mwfls = [c for c in MWFLS_COCHLEAE_DICT.keys()]

    def get_median_intensity(cochlea, mode="IHC"):
        if mode == "IHC":
            component_list = [1]
            if "component_list" in SYNAPSE_DICT[cochlea].keys():
                component_list = SYNAPSE_DICT[cochlea]["component_list"]
            stain = "Vglut3"
            seg_version = SYNAPSE_DICT[cochlea]["ihc_table_name"]
        else:
            component_list = [1]
            stain = "PV"
            seg_version = "SGN_v2"

        keyword = f"{stain}_bg-mask_median"
        seg_version_str = "-".join(seg_version.split("_"))
        table_seg_path = f"{cochlea}/tables/{seg_version}/default.tsv"
        table_meas_path = f"{cochlea}/tables/{seg_version}/{stain}_{seg_version_str}_object-measures-bg-mask.tsv"

        tsv_path, fs = get_s3_path(table_seg_path)
        with fs.open(tsv_path, "r") as f:
            table_seg = pd.read_csv(f, sep="\t")

        if keyword not in list(table_seg.columns):
            local_out = os.path.join(os.getcwd(), f"{cochlea}_seg_table_{mode}.tsv")
            if not os.path.exists(local_out):
                print("Creating local table")
                add_object_measures_to_table(table_seg_path, table_meas_path, local_out, s3_seg=True, s3_meas=True)
            table_seg = pd.read_csv(local_out, sep="\t")


        subset = table_seg[table_seg["component_labels"].isin(component_list)]
        intensities = subset[keyword].values
        return sum(intensities) / len(intensities)

    # get intensities for SGN segmentation
    sgn_idisco = [get_median_intensity(c, mode="SGN") for c in idisco]
    sgn_mwfls = [get_median_intensity(c, mode="SGN") for c in mwfls]

    # get intensities for IHC segmentation
    ihc_idisco = [get_median_intensity(c, mode="IHC") for c in idisco]
    ihc_mwfls = [get_median_intensity(c, mode="IHC") for c in mwfls]

    cohorts = ["iDISCO", "MWfLS"]
    sgn_by_cohort = {"iDISCO": sgn_idisco, "MWfLS": sgn_mwfls}
    ihc_by_cohort = {"iDISCO": ihc_idisco, "MWfLS": ihc_mwfls}

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5))

    x_positions = [1, 2]

    def _draw_subplot(
        ax, data_by_cohort, structure, ylim0, ylim1, y_ticks,
        outlier=None, ylabel=None, title=None,
    ):
        if title is None:
            title = structure
        for pos, cohort in zip(x_positions, cohorts):
            ax_prism_boxplot(ax, data_by_cohort[cohort], positions=[pos], color=COHORT_COLORS[cohort])
        if outlier is not None:
            for pos, cohort in zip(x_positions, cohorts):
                ax.scatter(
                    [pos] * len(outlier[cohort]),
                    outlier[cohort],
                    color="red",
                    zorder=2,
                    marker="x",
                    s=30,
                    alpha=0.7,
                )

        xmin, xmax = 0.5, 2.5
        ax.set_xlim(xmin, xmax)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(cohorts, fontsize=sub_label_size)
        ax.tick_params(axis="x", which="major", pad=8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
        ax.set_ylim(ylim0, ylim1)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=main_label_size)

        ax.set_title(title, fontsize=main_label_size, pad=10)

    _draw_subplot(
        axes[0], sgn_by_cohort, "SGN",
        ylim0=0, ylim1=200,
        y_ticks=list(range(000, 201, 50)),
        ylabel="Intensity [a.u.]",
        title="PV@SGNs"
    )
    _draw_subplot(
        axes[1], ihc_by_cohort, "IHC",
        ylim0=400, ylim1=900,
        y_ticks=list(range(400, 901, 100)),
        title="VGlut3@IHCs"
    )

    prism_cleanup_axes(axes)
    plt.tight_layout()

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate MWfLS vs iDISCO comparison plots.")
    parser.add_argument("--figure_dir", "-f", type=str, default="./panels/mwfls",
                        help="Output directory for plots.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--skip_synapses", action="store_true",
                        help="Skip synapse plots (use when synapse tables are not mounted).")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    if not args.skip_synapses:
        fig_cohort_boxplot(
            save_path=os.path.join(args.figure_dir, f"fig_cohort_boxplot.{FILE_EXTENSION}"),
            plot=args.plot,
        )
        fig_cohort_boxplot(
            save_path=os.path.join(args.figure_dir, f"fig_cohort_boxplot_outlier.{FILE_EXTENSION}"),
            plot=args.plot, remove_outliers=True,
        )
        fig_mwfls_synapses(
            save_path=os.path.join(args.figure_dir, f"fig_mwfls_synapses.{FILE_EXTENSION}"),
            plot=args.plot, top_axis=True,
        )
        fig_mwfls_synapses(
            save_path=os.path.join(args.figure_dir, f"fig_mwfls_synapses_trendline.{FILE_EXTENSION}"),
            plot=args.plot, trendline=True, top_axis=True,
        )
        fig_mwfls_synapses(
            save_path=os.path.join(args.figure_dir, f"fig_mwfls_synapses_trendline_legend.{FILE_EXTENSION}"),
            plot=args.plot, trendline=True, top_axis=True, show_legend=True,
        )
        plot_intensity(
            save_path=os.path.join(args.figure_dir, f"fig_stain_intensity.{FILE_EXTENSION}"),
            plot=args.plot,
        )
    else:
        print("Skipping synapse figures (--skip_synapses).")


if __name__ == "__main__":
    main()
