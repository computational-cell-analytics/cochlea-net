import argparse
import os
import sys
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MaxNLocator

from flamingo_tools.analysis.central_path_utils import mean_by_offset
from flamingo_tools.json_util import load_processing_params
from flamingo_tools.postprocessing.label_components import label_components_ihc
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT
from flamingo_tools.s3_utils import create_s3_target

from util import (
    literature_reference_values,
    prism_style, prism_cleanup_axes,
    ax_prism_boxplot, average_by_fraction, save_figure,
    VALUE_DICT, COHORT_DICT, cohort_cochleae, cochleae_for, OUTLIER_DICT,
)
from data import add_synapses_per_ihc, get_mapped_table, read_source_table
from tonotopic_mapping import add_frequency_axis, plot_synapses_along_cochlea

FILE_EXTENSION = "png"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSING_DIR = os.path.join(SCRIPT_DIR, "..", "..", "reproducibility", "processing")

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

# The co-stain of each cosynapse cochlea. CtBP2 is the reference stain of both.
COLOCALIZATION = {"M_AMD_000117_L": "Homer1", "M_AMD_000118_L": "Bsn"}
# The segments of a co-localization bar: the score key of evaluate_colocalization, the label and the color.
COLOCALIZATION_SEGMENTS = [
    ("fn", "CtBP2 only", "#2a78d6"), ("tp", "Co-localized", "#1baf7a"), ("fp", "Co-stain only", "#eb6834"),
]
# Darker shades of the segment colors for text on white, with a contrast of at least 4.5:1.
COLOCALIZATION_TEXT_COLORS = {"reference": "#1f5fae", "matched": "#12805a", "test": "#b8461a"}
COLOR_TEXT = "black"
COLOR_TEXT_MUTED = "#52514e"

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


def print_cohort_averages(tables: dict):
    """Print the mean and the sample SD over the cochleae of each cohort.

    A cochlea contributes its SGN count, its IHC count and its mean number of synapses per IHC. The
    pooled mean weights every IHC equally, as the synapse box of fig_cohort_boxplot does.

    Args:
        tables: Output of load_tables.
    """
    for cohort in COHORTS:
        cochleae = cohort_cochleae(cohort)
        syn_per_ihc = {name: tables["IHC"][name]["syn_per_IHC"] for name in cochleae}
        rows = [
            ("SGN count", [len(tables["SGN"][name]) for name in cochleae], ".0f"),
            ("IHC count", [len(tables["IHC"][name]) for name in cochleae], ".0f"),
            ("Synapses per IHC", [values.mean() for values in syn_per_ihc.values()], ".2f"),
        ]
        outliers = [name for name in cochleae if name in OUTLIER_DICT["synapse"]]
        if outliers:
            rows.append((f"Synapses per IHC without {', '.join(outliers)}",
                         [values.mean() for name, values in syn_per_ihc.items() if name not in outliers], ".2f"))

        print(f"\n{COHORT_DICT[cohort]['label']}")
        for label, values, fmt in rows:
            print(f"  {label}: {np.mean(values):{fmt}} +/- {np.std(values, ddof=1):{fmt}} (n = {len(values)} cochleae)")
        pooled = pd.concat(syn_per_ihc.values())
        print(f"  Synapses per IHC, pooled over {len(pooled)} IHCs: {pooled.mean():.2f}")


def load_colocalization(max_dist: float = 2.0, read_table=None) -> dict:
    """Match the co-stain synapses of each COLOCALIZATION cochlea against its CtBP2 synapses.

    The IHC_v11 tables of these cochleae have no component labels. This function labels the
    components in memory with the parameters of the processing file. Only the synapses of the IHCs in
    its 'component_list' count.

    Args:
        max_dist: Maximum distance of a match in micrometers.
        read_table: Function (cochlea, source name) -> table. The default reads the S3 table of the source.

    Returns:
        Mapping of cochlea name to a tuple of the scores and the per-IHC synapse counts of
        evaluate_colocalization. The scores also hold 'max_dist'.
    """
    sys.path.insert(0, os.path.join(SCRIPT_DIR, "..", "measurements"))
    from synapse_colocalization import evaluate_colocalization

    if read_table is None:
        read_table = partial(read_source_table, create_s3_target())
    colocalization = {}
    for cochlea, stain in COLOCALIZATION.items():
        params, = load_processing_params(os.path.join(PROCESSING_DIR, f"{cochlea}_IHC.json"), "label_components")
        label_params = {key: params[key] for key in ("min_size", "min_component_length", "max_edge_distance")
                        if key in params}
        ihc = label_components_ihc(read_table(cochlea, "IHC_v11"), **label_params)
        valid_ihcs = ihc["label_id"][ihc["component_labels"].isin(params["component_list"])].values
        reference = read_table(cochlea, "synapse_v3_ihc_v11")
        test = read_table(cochlea, f"{stain}_synapse_v3_ihc_v11")
        scores, ihc_counts = evaluate_colocalization(reference, test, max_dist, valid_ihcs=valid_ihcs)
        colocalization[cochlea] = ({"max_dist": max_dist, **scores}, ihc_counts)
        print(f"\n{cochlea}, CtBP2 vs {stain}: {scores}")
    return colocalization


def _colored_text(parts, fontsize: float, fontweight: str = "normal") -> HPacker:
    """Return one line of text from (text, color) parts."""
    return HPacker(children=[
        TextArea(text, textprops=dict(color=color, fontsize=fontsize, fontweight=fontweight)) for text, color in parts
    ], align="baseline", pad=0, sep=0)


def plot_synapse_colocalization(colocalization: dict, save_path: str, plot: bool = False):
    """Plot how the CtBP2 synapses and the co-stain synapses of each cochlea co-localize.

    The bar of a cochlea splits its synapses into three parts: the CtBP2 synapses without a co-stain
    partner, the co-localized pairs and the co-stain synapses without a CtBP2 partner. A pair is a
    one-to-one match within 'max_dist'.

    The table right of the bars gives the share of the synapses of each stain that have a partner,
    with the pair count and the synapse count of the stain.

    Args:
        colocalization: Output of load_colocalization.
        save_path: Output file path.
        plot: Show interactive plot.
    """
    prism_style()
    fontsize = SUB_LABEL_SIZE - 2
    names = list(colocalization)
    scores = {name: colocalization[name][0] for name in names}
    max_dist, = {scores[name]["max_dist"] for name in names}

    fig = plt.figure(figsize=(13, 3.8))
    grid = fig.add_gridspec(1, 2, width_ratios=[3, 1.7], wspace=0.1)
    ax = fig.add_subplot(grid[0])
    ax_table = fig.add_subplot(grid[1], sharey=ax)
    ax_table.axis("off")
    # The right edge of the share column of the table, in axes fraction.
    share_x = 0.5

    x_max = max(sum(scores[name][key] for key, _, _ in COLOCALIZATION_SEGMENTS) for name in names)
    for y, name in enumerate(names):
        left = 0
        for key, label, color in COLOCALIZATION_SEGMENTS:
            width = scores[name][key]
            ax.barh(y, width, left=left, height=0.6, color=color, edgecolor="white", linewidth=2,
                    label=label if y == 0 else None)
            # A count goes into its segment only if it fits.
            if width >= 0.06 * x_max:
                ax.text(left + width / 2, y, f"{width}", ha="center", va="center", color="white", fontsize=fontsize)
            left += width

        stain = COLOCALIZATION[name]
        y_label = VPacker(children=[
            _colored_text([(name, COLOR_TEXT)], fontsize),
            _colored_text([("CtBP2", COLOCALIZATION_TEXT_COLORS["reference"]), (" / ", COLOR_TEXT),
                           (stain, COLOCALIZATION_TEXT_COLORS["test"])], fontsize),
        ], align="right", pad=0, sep=4)
        # The offset clears the tick mark and its pad.
        ax.add_artist(AnnotationBbox(
            y_label, (0, y), xycoords=ax.get_yaxis_transform(), xybox=(-10, 0), boxcoords="offset points",
            box_alignment=(1, 0.5), frameon=False, annotation_clip=False,
        ))

        # The scores hold rounded fractions, so take the fractions from the counts.
        tp = scores[name]["tp"]
        for dy, stain_label, n_synapses, text_color in [
            (-0.16, "CtBP2", scores[name]["n_synapses_reference"], COLOCALIZATION_TEXT_COLORS["reference"]),
            (0.16, stain, scores[name]["n_synapses_test"], COLOCALIZATION_TEXT_COLORS["test"]),
        ]:
            for x, ha, text, color in [
                (0, "left", stain_label, text_color),
                (share_x, "right", f"{tp / n_synapses:.0%}", COLOR_TEXT),
                (1, "right", f"{tp} / {n_synapses}", COLOR_TEXT),
            ]:
                ax_table.text(x, y + dy, text, ha=ha, va="center", color=color, fontsize=fontsize)

    for x, ha, label in [(0, "left", "stain"), (share_x, "right", "share"), (1, "right", "pairs / synapses")]:
        ax_table.text(x, -0.5, label, ha=ha, va="bottom", color=COLOR_TEXT_MUTED, fontsize=fontsize - 2)
    ax_table.annotate(
        f"Co-localized (within {max_dist:g} µm)", (0, 1), xycoords="axes fraction", xytext=(0, 22),
        textcoords="offset points", ha="left", va="bottom", fontsize=fontsize, fontweight="bold",
        color=COLOCALIZATION_TEXT_COLORS["matched"],
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(0, 1.02 * x_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2.5, 5, 10]))
    ax.tick_params(axis="x", labelsize=MAIN_TICK_SIZE)
    ax.set_xlabel("Synapses on the IHC row", fontsize=MAIN_LABEL_SIZE)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=fontsize,
              handlelength=1.5, columnspacing=1.2)
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def colocalization_table(colocalization: dict) -> pd.DataFrame:
    """Summarize the synapse counts of load_colocalization.

    Each cochlea gets one row for CtBP2, one row for its co-stain and one row for the co-localized
    pairs. A pair counts for the IHC of its CtBP2 synapse. The per-IHC values cover all IHCs of the
    IHC row, so an IHC without a synapse counts as 0.

    Args:
        colocalization: Output of load_colocalization.

    Returns:
        The table. 'colocalized_share' is the fraction of the synapses of the stain that have a
        partner. It is NaN for the pair rows.
    """
    rows = []
    for cochlea, (_, ihc_counts) in colocalization.items():
        n_pairs = ihc_counts["n_matched"].sum()
        for stain, column in [
            ("CtBP2", "n_reference"), (COLOCALIZATION[cochlea], "n_test"), ("Co-localized pairs", "n_matched"),
        ]:
            counts = ihc_counts[column]
            rows.append({
                "cochlea": cochlea,
                "stain": stain,
                "n_ihcs": len(counts),
                "n_ihcs_with_synapses": int((counts > 0).sum()),
                "n_synapses": int(counts.sum()),
                "mean_per_ihc": counts.mean(),
                "sd_per_ihc": counts.std(),
                "median_per_ihc": counts.median(),
                "colocalized_share": np.nan if column == "n_matched" else n_pairs / counts.sum(),
            })
    return pd.DataFrame(rows)


def write_colocalization_table(colocalization: dict, save_path: str):
    """Write the output of colocalization_table as a Markdown file.

    Args:
        colocalization: Output of load_colocalization.
        save_path: Output file path.
    """
    header = [
        "Cochlea", "Stain", "IHCs", "IHCs with synapses", "Synapses", "Synapses per IHC (mean +/- SD)",
        "Synapses per IHC (median)", "Co-localized share",
    ]
    lines = [
        "# Synapse co-localization",
        "",
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * 2 + ["---:"] * (len(header) - 2)) + " |",
    ]
    for row in colocalization_table(colocalization).itertuples(index=False):
        share = "-" if np.isnan(row.colocalized_share) else f"{row.colocalized_share:.1%}"
        cells = [
            row.cochlea, row.stain, row.n_ihcs, row.n_ihcs_with_synapses, row.n_synapses,
            f"{row.mean_per_ihc:.2f} +/- {row.sd_per_ihc:.2f}", f"{row.median_per_ihc:g}", share,
        ]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines += [""] + [f"- {note}" for note in _colocalization_notes(colocalization)] + [""]
    with open(save_path, "w") as f:
        f.write("\n".join(lines))


def write_colocalization_excel(colocalization: dict, save_path: str):
    """Write the output of colocalization_table as an Excel file.

    The first sheet holds the table with numeric values. The second sheet holds the definitions of
    the Markdown file.

    Args:
        colocalization: Output of load_colocalization.
        save_path: Output file path.
    """
    columns = {
        "cochlea": "Cochlea", "stain": "Stain", "n_ihcs": "IHCs", "n_ihcs_with_synapses": "IHCs with synapses",
        "n_synapses": "Synapses", "mean_per_ihc": "Synapses per IHC (mean)", "sd_per_ihc": "Synapses per IHC (SD)",
        "median_per_ihc": "Synapses per IHC (median)", "colocalized_share": "Co-localized share",
    }
    number_formats = {"mean_per_ihc": "0.00", "sd_per_ihc": "0.00", "median_per_ihc": "0.0",
                      "colocalized_share": "0.0%"}
    table = colocalization_table(colocalization)[list(columns)]
    notes = pd.DataFrame({"Notes": [note.replace("`", "") for note in _colocalization_notes(colocalization)]})
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        table.rename(columns=columns).to_excel(writer, sheet_name="colocalization", index=False)
        notes.to_excel(writer, sheet_name="notes", index=False)
        sheet = writer.sheets["colocalization"]
        for sheet_column, (key, name) in zip(sheet.iter_cols(), columns.items()):
            sheet.column_dimensions[sheet_column[0].column_letter].width = max(len(name), 16) + 2
            for cell in sheet_column[1:]:
                cell.number_format = number_formats.get(key, "General")
        writer.sheets["notes"].column_dimensions["A"].width = 120


def _colocalization_notes(colocalization: dict) -> list:
    """Return the definitions of the co-localization table, with Markdown code marks."""
    max_dist, = {scores["max_dist"] for scores, _ in colocalization.values()}
    co_stain_sources = ", ".join(f"`{COLOCALIZATION[cochlea]}_synapse_v3_ihc_v11`" for cochlea in colocalization)
    return [
        f"Sources: the S3 tables `IHC_v11`, `synapse_v3_ihc_v11` (CtBP2) and {co_stain_sources}.",
        "IHCs: the IHCs of the IHC row, i.e. the `component_list` components of "
        "`reproducibility/processing/<cochlea>_IHC.json`.",
        "Synapses: the synapses within 3 um of one of these IHCs.",
        f"Co-localized pairs: one-to-one matches of a CtBP2 synapse and a co-stain synapse within {max_dist:g} um. "
        "A pair counts for the IHC of its CtBP2 synapse.",
        "Co-localized share: the fraction of the synapses of the stain that have a partner.",
        "Synapses per IHC: over all IHCs of the row. An IHC without a synapse counts as 0. SD is the sample SD.",
    ]


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
    print_cohort_averages(tables)
    # The intensity panels leave out the instances without an object measure. The count and
    # synapse panels keep them.
    measured = {
        structure: {name: table.dropna(subset=["intensity"]) for name, table in tables[structure].items()}
        for structure in args.intensity
    }

    if "SGN" in args.intensity:
        plot_intensity_boxplot(measured["SGN"], "PV@SGNs", out("fig_02d_sgn_intensity_by_cohort"), plot=args.plot)
        plot_intensity_boxplot(
            measured["SGN"], "PV@SGNs", out("fig_02d_sgn_intensity_by_cohort_all_instances"), per_cochlea=False,
            plot=args.plot,
        )
        plot_intensity_by_offset(measured["SGN"], out("fig_02e_sgn_intensity_by_rc_offset"), plot=args.plot)
    if "IHC" in args.intensity:
        plot_intensity_boxplot(
            measured["IHC"], "VGlut3@IHCs", out("fig_02f_ihc_intensity_by_cohort"), plot=args.plot,
        )
        plot_intensity_boxplot(
            measured["IHC"], "VGlut3@IHCs", out("fig_02f_ihc_intensity_by_cohort_all_instances"), per_cochlea=False,
            plot=args.plot,
        )
        plot_intensity_along_cochlea(measured["IHC"], out("fig_02g_ihc_intensity_by_tonotopy"), plot=args.plot)

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
    colocalization = load_colocalization()
    plot_synapse_colocalization(colocalization, out("synapse_colocalization"), plot=args.plot)
    write_colocalization_table(colocalization, os.path.join(args.figure_dir, "colocalization.md"))
    write_colocalization_excel(colocalization, os.path.join(args.figure_dir, "colocalization.xlsx"))


if __name__ == "__main__":
    main()
