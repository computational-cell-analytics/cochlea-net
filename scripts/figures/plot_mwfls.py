import argparse
import os

import matplotlib.pyplot as plt

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

COLOR_LITERATURE = "#27339C"
COHORT_COLORS = {
    "iDISCO": "#3F69FF",
    "MWfLS": "#10CC17",
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

    def _draw_subplot(ax, data_by_cohort, structure, ylim0, ylim1, y_ticks, outlier=None, ylabel=None, title=None):
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

        ax.set_xticks(x_positions)
        ax.set_xticklabels(cohorts, fontsize=sub_label_size)
        ax.tick_params(axis="x", which="major", pad=8)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(y_ticks, rotation=0, fontsize=main_tick_size)
        ax.set_ylim(ylim0, ylim1)

        if ylabel:
            ax.set_ylabel(ylabel, fontsize=main_label_size)

        ax.set_title(structure, fontsize=main_label_size, pad=10)

    _draw_subplot(
        axes[0], sgn_by_cohort, "SGN",
        ylim0=1000, ylim1=12500,
        y_ticks=list(range(2000, 12001, 2000)),
        ylabel="Count per cochlea",
        outlier=sgn_outlier,
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


def supp_fig_mwfls_synapses(
    save_path: str,
    idisco_syn_version: str = "v4c",
    mwfls_syn_version: str = "v9",
    n_bins: int = 25,
    top_axis: bool = False,
    trendline: bool = False,
    trendline_std: bool = False,
    errorbar: bool = False,
    plot: bool = False,
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
        merged_cohort_dict[name] = entry
    for name in COHORT_DICT["MWfLS"]:
        entry = dict(MWFLS_COCHLEAE_DICT[name])
        entry["marker"] = COHORT_MARKERS["MWfLS"]
        merged_cohort_dict[name] = entry

    tonotopic_data = {**idisco_data, **mwfls_data}

    supp_fig_03a_meyer(
        tonotopic_data=tonotopic_data,
        save_path=save_path,
        cohort_dict=merged_cohort_dict,
        use_alias=True,
        plot=plot,
        n_bins=n_bins,
        top_axis=top_axis,
        trendline=trendline,
        trendline_std=trendline_std,
        errorbar=errorbar,
    )


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
        supp_fig_mwfls_synapses(
            save_path=os.path.join(args.figure_dir, f"supp_fig_mwfls_synapses.{FILE_EXTENSION}"),
            plot=args.plot,
        )
    else:
        print("Skipping synapse figures (--skip_synapses).")


if __name__ == "__main__":
    main()
