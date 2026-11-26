import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MaxNLocator

from flamingo_tools.analysis.central_path_utils import mean_by_offset
from flamingo_tools.json_util import load_processing_params
from flamingo_tools.postprocessing.label_components import label_components_ihc, refine_ihc_row
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT
from flamingo_tools.s3_utils import create_s3_target

from util import (
    literature_reference_values, average_by_fraction, cochlea_label,
    VALUE_DICT, COCHLEA_DICT, COHORT_DICT, cohort_cochleae, OUTLIER_DICT,
)
from data import add_synapses_per_ihc, get_mapped_table, read_source_table
from panel_style import (
    AXIS_LINE_WIDTH, COLORS, DATA_LINE_WIDTH, FONT_SIZE, MARKER_SIZE, MEAN_LINE_WIDTH, OUTLIER,
    box_plot, dot_plot, group_shades, label_points, new_panel, nice_upper, profile_plot, readable, reference_band,
    save_panel, style, text_color_on, text_key,
)
from tonotopic_mapping import add_frequency_axis, meyer_synapse_profile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSING_DIR = os.path.join(SCRIPT_DIR, "..", "..", "reproducibility", "processing")

COHORTS = ["mwfls", "idisco"]
# The SGN segmentation of each cohort. The IHC segmentation and the synapse detection of a
# cochlea come from SYNAPSE_DICT.
SGN_VERSION = {"idisco": "SGN_v2", "mwfls": "SGN_v3"}
STAINS = {"SGN": "PV", "IHC": "Vglut3"}

COHORT_MARKERS = {
    "idisco": "o",
    "mwfls": "s",
}
# The hue turn of the per-cochlea shades, away from the other cohort: MWfLS toward violet,
# iDISCO toward yellow-green. See panel_style.group_shades.
SHADE_HUE_RANGE = {"mwfls": 0.06, "idisco": -0.06}

# The co-stain of each cosynapse cochlea. CtBP2 is the reference stain of both.
COLOCALIZATION = {"M_AMD_000117_L": "Homer1", "M_AMD_000118_L": "Bsn"}
# The segments of a co-localization bar: the score key of evaluate_colocalization, the label and the color.
# Blue and green stand for the cohorts, so the segments use other colors.
COLOCALIZATION_SEGMENTS = [
    ("fn", "CtBP2 only", COLORS["purple"]), ("tp", "Co-localized", COLORS["dark_gray"]),
    ("fp", "Co-stain only", COLORS["orange"]),
]
COLOCALIZATION_TEXT_COLORS = {
    "reference": readable(COLORS["purple"]), "matched": readable(COLORS["dark_gray"]),
    "test": readable(COLORS["orange"]),
}
COLOR_TEXT = COLORS["black"]
COLOR_TEXT_MUTED = COLORS["dark_gray"]

# The print size of each panel type in mm (width, height). Fig 2D-G fill one 180 mm row.
PANEL_SIZES = {
    "cohort": (40, 45),  # 2D, 2F
    "profile": (50, 45),  # 2E, 2G
    "cohort_counts": (115, 50),  # 3A
    "synapses": (65, 50),  # 3B
    "synapses_legend": (120, 50),  # 3B with the cochlea legend
    "colocalization": (180, 40),
    "sgn_intensity_check": (120, 50),
}

# The x positions of the cohorts in the panels that compare them, and the x range around them.
COHORT_POSITIONS = [1, 2]
COHORT_XLIM = (0.4, 2.6)


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

    A cochlea contributes its SGN count, its IHC count and its mean number of synapses per IHC, as in
    plot_counts_by_cohort. The pooled mean weights every IHC equally.

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


def row_ihcs(table: pd.DataFrame, params: dict) -> np.ndarray:
    """Return the label ids of the IHC row that the label_components parameters of a processing file select.

    The components are labeled in memory as flamingo_tools.label_components does it, with the refined row
    if the parameters have 'refine_row'.

    Args:
        table: IHC segmentation table.
        params: One entry of load_processing_params(<cochlea>_IHC.json, "label_components").

    Returns:
        The label ids of the instances in the components of 'component_list'.
    """
    # The defaults of label_components_single. refine_ihc_row has other defaults.
    label_params = {"min_size": 1000, "min_component_length": 50, "max_edge_distance": 30}
    label_params.update({key: params[key] for key in label_params if key in params})
    labeled = label_components_ihc(table, **label_params)
    if params.get("refine_row"):
        path_components = params.get("component_list_path") or params["component_list"]
        max_path_deviation = params.get("max_path_deviation")
        labeled, _, _ = refine_ihc_row(
            labeled, path_components, max_path_deviation=15.0 if max_path_deviation is None else max_path_deviation,
            merge_components=params.get("merge_components", False), **label_params,
        )
    return labeled["label_id"][labeled["component_labels"].isin(params["component_list"])].values


def load_colocalization(max_dist: float = 2.0, read_table=None) -> dict:
    """Match the co-stain synapses of each COLOCALIZATION cochlea against its CtBP2 synapses.

    The IHC components are labeled in memory with the parameters of the processing file, see row_ihcs.
    Only the synapses of the IHCs in its 'component_list' count.

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
        valid_ihcs = row_ihcs(read_table(cochlea, "IHC_v11"), params)
        reference = read_table(cochlea, "synapse_v3_ihc_v11")
        test = read_table(cochlea, f"{stain}_synapse_v3_ihc_v11")
        scores, ihc_counts = evaluate_colocalization(reference, test, max_dist, valid_ihcs=valid_ihcs)
        colocalization[cochlea] = ({"max_dist": max_dist, **scores}, ihc_counts)
        print(f"\n{cochlea}, CtBP2 vs {stain}: {scores}")
    return colocalization


def _colored_text(parts, fontweight: str = "normal") -> HPacker:
    """Return one line of text from (text, color) parts."""
    return HPacker(children=[
        TextArea(text, textprops=dict(color=color, fontsize=FONT_SIZE, fontweight=fontweight)) for text, color in parts
    ], align="baseline", pad=0, sep=0)


@style()
def plot_synapse_colocalization(colocalization: dict, save_path: str, plot: bool = False):
    """Plot how the CtBP2 synapses and the co-stain synapses of each cochlea co-localize.

    The bar of a cochlea splits its synapses into three parts: the CtBP2 synapses without a co-stain
    partner, the co-localized pairs and the co-stain synapses without a CtBP2 partner. A pair is a
    one-to-one match within 'max_dist'.

    The table right of the bars gives the share of the synapses of each stain that have a partner,
    with the pair count and the synapse count of the stain.

    Args:
        colocalization: Output of load_colocalization.
        save_path: Output path without an extension.
        plot: Show interactive plot.
    """
    names = list(colocalization)
    scores = {name: colocalization[name][0] for name in names}
    max_dist, = {scores[name]["max_dist"] for name in names}

    fig, (ax, ax_table) = new_panel(*PANEL_SIZES["colocalization"], ncols=2, sharey=True,
                                    gridspec_kw={"width_ratios": [3, 1.7]})
    ax_table.axis("off")
    # The right edge of the share column of the table, in axes fraction.
    share_x = 0.5

    x_max = max(sum(scores[name][key] for key, _, _ in COLOCALIZATION_SEGMENTS) for name in names)
    for y, name in enumerate(names):
        left = 0
        for key, label, color in COLOCALIZATION_SEGMENTS:
            width = scores[name][key]
            ax.barh(y, width, left=left, height=0.6, color=color, edgecolor="white", linewidth=1,
                    label=label if y == 0 else None)
            # A count goes into its segment only if it fits.
            if width >= 0.06 * x_max:
                ax.text(left + width / 2, y, f"{width}", ha="center", va="center", color=text_color_on(color))
            left += width

        stain = COLOCALIZATION[name]
        y_label = VPacker(children=[
            _colored_text([(name, COLOR_TEXT)]),
            _colored_text([("CtBP2", COLOCALIZATION_TEXT_COLORS["reference"]), (" / ", COLOR_TEXT),
                           (stain, COLOCALIZATION_TEXT_COLORS["test"])]),
        ], align="right", pad=0, sep=2)
        # The offset clears the tick mark and its pad.
        ax.add_artist(AnnotationBbox(
            y_label, (0, y), xycoords=ax.get_yaxis_transform(), xybox=(-5, 0), boxcoords="offset points",
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
                ax_table.text(x, y + dy, text, ha=ha, va="center", color=color)

    for x, ha, label in [(0, "left", "stain"), (share_x, "right", "share"), (1, "right", "pairs / synapses")]:
        ax_table.text(x, -0.5, label, ha=ha, va="bottom", color=COLOR_TEXT_MUTED)
    ax_table.annotate(
        f"Co-localized (within {max_dist:g} µm)", (0, 1), xycoords="axes fraction", xytext=(0, 12),
        textcoords="offset points", ha="left", va="bottom", fontweight="bold",
        color=COLOCALIZATION_TEXT_COLORS["matched"],
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([])
    ax.set_ylim(len(names) - 0.5, -0.5)
    ax.set_xlim(0, 1.02 * x_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2.5, 5, 10]))
    ax.set_xlabel("Synapses on the IHC row")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3)
    save_panel(fig, save_path, plot)


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


def _set_cohort_xaxis(ax, n_cochleae=None):
    """Put the cohorts at COHORT_POSITIONS, with the number of cochleae below the cohort label if given."""
    ax.set_xlim(*COHORT_XLIM)
    ax.set_xticks(COHORT_POSITIONS)
    labels = [COHORT_DICT[cohort]["label"] for cohort in COHORTS]
    if n_cochleae is not None:
        labels = [f"{label}\nn = {n}" for label, n in zip(labels, n_cochleae)]
    ax.set_xticklabels(labels)


def _draw_cohort_points(ax, values, position: float, cohort: str, outliers=()):
    """Draw one value per cochlea as a dot plot with the mean and the sample SD, see panel_style.dot_plot.

    Set the axis limits first.
    """
    dot_plot(ax, values, position, COHORT_DICT[cohort]["color"], COHORT_MARKERS[cohort], outliers)


def _set_cohort_ylim(ax, value_sets):
    """Set the y range from 0 to the next tick above the largest value or mean + SD of the value sets."""
    upper = max(max(np.max(values), np.mean(values) + np.std(values, ddof=1)) for values in value_sets)
    ax.set_ylim(0, nice_upper(upper))


def cohort_values(tables: dict, structure: str, cohort: str, remove_outliers: bool = False):
    """Return one value per cochlea of a cohort for a panel of plot_counts_by_cohort.

    Args:
        tables: Output of load_tables.
        structure: "SGN" or "IHC" for the count, "synapse" for the mean number of synapses per IHC.
        cohort: Key of COHORT_DICT.
        remove_outliers: Return the values of the OUTLIER_DICT cochleae separately.

    Returns:
        The values of the cochleae and the values of the outliers.
    """
    values, outliers = [], []
    for cochlea in cohort_cochleae(cohort):
        if structure == "synapse":
            value = tables["IHC"][cochlea]["syn_per_IHC"].mean()
        else:
            value = len(tables[structure][cochlea])
        is_outlier = remove_outliers and cochlea in OUTLIER_DICT.get(structure, [])
        (outliers if is_outlier else values).append(value)
    return values, outliers


@style()
def plot_counts_by_cohort(tables: dict, save_path: str, plot: bool = False, remove_outliers: bool = False):
    """Compare the cohorts in the SGN count, the IHC count and the mean number of synapses per IHC.

    Each point is one cochlea, with the mean and the sample SD of the cohort. The SGN and IHC
    counts are the counts that load_tables checks against VALUE_DICT.

    Args:
        tables: Output of load_tables.
        save_path: Output path without an extension.
        plot: Show interactive plot.
        remove_outliers: Draw the cochleae of OUTLIER_DICT as an outlier cross and leave them out
            of the mean and the SD.
    """
    n_cochleae = [len(cohort_cochleae(cohort)) for cohort in COHORTS]
    fig, axes = new_panel(*PANEL_SIZES["cohort_counts"], ncols=3)
    panels = [
        ("SGN", "SGN", (4000, 12000), range(4000, 12001, 2000)),
        ("IHC", "IHC", (400, 800), range(400, 801, 100)),
        ("synapse", "Synapses per IHC", (-0.5, 20), range(0, 21, 5)),
    ]
    for ax, (structure, title, ylim, y_ticks) in zip(axes, panels):
        _set_cohort_xaxis(ax, n_cochleae)
        ax.set_yticks(list(y_ticks))
        ax.set_ylim(*ylim)
        ax.set_title(title)
        for position, cohort in zip(COHORT_POSITIONS, COHORTS):
            values, outliers = cohort_values(tables, structure, cohort, remove_outliers)
            _draw_cohort_points(ax, values, position, cohort, outliers)
        # The SGN label sits below the band at the right, where no SGN count is.
        reference_band(ax, *literature_reference_values(structure), label="literature" if structure == "SGN" else None)

    axes[0].set_ylabel("Count per cochlea")
    axes[2].set_ylabel("Mean per cochlea")
    save_panel(fig, save_path, plot)


def _grouped_legend(fig, groups):
    """Add a legend right of the axes with one column per group and a bold group heading.

    Args:
        fig: The figure.
        groups: Pairs of a heading and a list of (handle, label) pairs.
    """
    n_rows = 1 + max(len(entries) for _, entries in groups)
    handles, labels, headings = [], [], []
    for heading, entries in groups:
        headings.append(len(labels))
        column = [(Line2D([], [], linestyle="none"), heading)] + entries
        # Blank entries fill the column, so that the next group starts a new column.
        column += [(Line2D([], [], linestyle="none"), "")] * (n_rows - len(column))
        handles += [handle for handle, _ in column]
        labels += [label for _, label in column]
    legend = fig.legend(handles, labels, loc="outside right upper", ncol=len(groups), handletextpad=0.3)
    for index in headings:
        legend.get_texts()[index].set_fontweight("bold")


@style()
def plot_synapses_by_tonotopy(
    tables: dict,
    save_path: str,
    n_bins: int = 25,
    trendline: bool = False,
    plot: bool = False,
    show_legend: bool = False,
):
    """Plot the synapses per IHC along the cochlea for the cochleae of both cohorts.

    Each cochlea has a shade of its cohort color and the cohort symbol. The synapse outliers of
    OUTLIER_DICT are drawn as outlier crosses and stay out of the cohort mean. The counts of
    Meyer et al. 2009 are the black line.

    Args:
        tables: Output of load_tables.
        save_path: Output path without an extension.
        n_bins: Number of bins along the cochlea length.
        trendline: Draw the mean over the cochleae of each cohort as dashed line.
        plot: Show interactive plot.
        show_legend: Add a legend with one column per cohort right of the plot.
    """
    fig, ax = new_panel(*PANEL_SIZES["synapses_legend" if show_legend else "synapses"])
    point_size = (0.75 * MARKER_SIZE) ** 2
    legend_groups = []
    for cohort in COHORTS:
        color = COHORT_DICT[cohort]["color"]
        cochleae = cohort_cochleae(cohort)
        included = [name for name in cochleae if name not in OUTLIER_DICT["synapse"]]
        cochlea_colors = dict(zip(included, group_shades(color, len(included), hue_range=SHADE_HUE_RANGE[cohort])))
        profiles, entries = {}, []
        for name in cochleae:
            table = tables["IHC"][name]
            binned = average_by_fraction(table["length_fraction"].values, table["syn_per_IHC"].values, n_bins=n_bins)
            x = binned["fraction_midpoint"].values.astype(float)
            y = binned["mean_syn_per_IHC"].values
            if name in cochlea_colors:
                handle = ax.scatter(x, y, s=point_size, color=cochlea_colors[name], marker=COHORT_MARKERS[cohort],
                                    linewidths=0, alpha=0.8, zorder=1)
                profiles[name] = pd.Series(y, index=x)
            else:
                handle = ax.scatter(x, y, s=point_size, color=OUTLIER["color"], marker=OUTLIER["marker"],
                                    linewidths=AXIS_LINE_WIDTH, zorder=1)
            entries.append((handle, cochlea_label(name, COCHLEA_DICT[name])))
        if trendline:
            mean = pd.DataFrame(profiles).mean(axis=1)
            line, = ax.plot(mean.index, mean.values, color=color, linestyle="--", linewidth=MEAN_LINE_WIDTH, zorder=2)
            entries.insert(0, (line, "Mean"))
        legend_groups.append((COHORT_DICT[cohort]["label"], entries))

    meyer = meyer_synapse_profile(n_bins)
    if meyer is not None:
        line, = ax.plot(meyer["fraction_midpoint"].values.astype(float), meyer["mean_syn_per_IHC"].values,
                        color=COLORS["black"], marker="D", markersize=0.75 * MARKER_SIZE, linewidth=DATA_LINE_WIDTH,
                        zorder=3)
        legend_groups.append(("Literature", [(line, "Meyer et al. 2009")]))

    ax.set_xlim(0, 1)
    ax.set_xlabel("Length fraction")
    ax.set_ylabel("Synapses per IHC")
    add_frequency_axis(ax, FONT_SIZE, FONT_SIZE, label="Frequency (kHz)")
    if show_legend:
        _grouped_legend(fig, legend_groups)
    save_panel(fig, save_path, plot)


@style()
def plot_intensity_by_cohort(
    structure_tables: dict, title: str, save_path: str, per_cochlea: bool = True, plot: bool = False,
):
    """Compare the stain intensity of the cohorts.

    By default, each point is the mean intensity of one cochlea, with the mean and the sample SD of
    the cohort. Without per_cochlea, a box plot pools all instances of all cochleae of the cohort.
    Its fliers are hidden, because a box holds thousands of instances.

    Args:
        structure_tables: Tables of one structure, a value of load_tables.
        title: Title of the plot.
        save_path: Output path without an extension.
        per_cochlea: Use the mean intensity of each cochlea instead of every instance.
        plot: Show interactive plot.
    """
    fig, ax = new_panel(*PANEL_SIZES["cohort"])
    intensities = {
        cohort: [structure_tables[c]["intensity"].values for c in cohort_cochleae(cohort)] for cohort in COHORTS
    }
    _set_cohort_xaxis(ax, [len(cohort_cochleae(cohort)) for cohort in COHORTS] if per_cochlea else None)
    if per_cochlea:
        means = {cohort: [values.mean() for values in intensities[cohort]] for cohort in COHORTS}
        _set_cohort_ylim(ax, means.values())
        for position, cohort in zip(COHORT_POSITIONS, COHORTS):
            _draw_cohort_points(ax, means[cohort], position, cohort)
    else:
        for position, cohort in zip(COHORT_POSITIONS, COHORTS):
            box_plot(ax, np.concatenate(intensities[cohort]), position, COHORT_DICT[cohort]["color"])
        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(0, nice_upper(ax.get_ylim()[1]))

    ax.set_ylabel("Mean intensity (a.u.)" if per_cochlea else "Intensity (a.u.)")
    ax.set_title(title)
    save_panel(fig, save_path, plot)


@style()
def plot_sgn_intensity_check(
    sgn_tables: dict, measured_tables: dict, save_path: str, plot: bool = False,
) -> pd.DataFrame:
    """Plot the mean PV intensity of the SGNs of each cochlea, and the SGN count against it.

    A cochlea with a weak PV stain can have a low SGN count, because the segmentation misses dim SGNs.
    The left panel shows the mean intensity per cohort, the right panel the SGN count against the mean
    intensity, with the literature band of the SGN count. Each point of the right panel is labeled
    with the cochlea alias.

    Args:
        sgn_tables: The SGN tables of load_tables. The count includes the SGNs without a measure.
        measured_tables: The same tables without the SGNs that have no intensity.
        save_path: Output path without an extension.
        plot: Show interactive plot.

    Returns:
        One row per cochlea with the alias, the SGN count, the mean and median intensity and the
        share of the SGNs with a measure.
    """
    rows = []
    for cohort in COHORTS:
        for name in cohort_cochleae(cohort):
            intensity = measured_tables[name]["intensity"]
            count = len(sgn_tables[name])
            rows.append({
                "cochlea": name, "alias": cochlea_label(name, COCHLEA_DICT[name]), "cohort": cohort,
                "sgn_count": count, "mean_intensity": intensity.mean(), "median_intensity": intensity.median(),
                "measured_share": len(intensity) / count,
            })
    summary = pd.DataFrame(rows)

    fig, (ax_cohort, ax_count) = new_panel(*PANEL_SIZES["sgn_intensity_check"], ncols=2,
                                           gridspec_kw={"width_ratios": [1, 2.4]})
    means = {cohort: summary.loc[summary["cohort"] == cohort, "mean_intensity"].values for cohort in COHORTS}
    _set_cohort_xaxis(ax_cohort, [len(cohort_cochleae(cohort)) for cohort in COHORTS])
    _set_cohort_ylim(ax_cohort, means.values())
    for position, cohort in zip(COHORT_POSITIONS, COHORTS):
        _draw_cohort_points(ax_cohort, means[cohort], position, cohort)
    ax_cohort.set_ylabel("Mean PV intensity (a.u.)")

    for cohort in COHORTS:
        cohort_rows = summary[summary["cohort"] == cohort]
        ax_count.scatter(cohort_rows["mean_intensity"], cohort_rows["sgn_count"], color=COHORT_DICT[cohort]["color"],
                         marker=COHORT_MARKERS[cohort], s=MARKER_SIZE ** 2, linewidths=0, zorder=3,
                         label=COHORT_DICT[cohort]["label"])
    ax_count.set_xlim(0, nice_upper(1.1 * summary["mean_intensity"].max()))
    reference_band(ax_count, *literature_reference_values("SGN"), label="literature")
    ax_count.set_xlabel("Mean PV intensity (a.u.)")
    ax_count.set_ylabel("SGN count")
    ax_count.legend(loc="lower right")
    label_points(ax_count, summary["mean_intensity"], summary["sgn_count"], summary["alias"])
    save_panel(fig, save_path, plot)
    return summary


def _draw_cohort_profiles(ax, profiles: dict):
    """Draw the per-cochlea profiles as points and the cohort mean +/- 1 sample SD over the cochleae as a line.

    Args:
        ax: The axis.
        profiles: Mapping of cochlea name to a series of the value per x position.
    """
    for cohort in COHORTS:
        samples = pd.DataFrame({name: profiles[name] for name in cohort_cochleae(cohort)})
        profile_plot(ax, samples, COHORT_DICT[cohort]["color"], COHORT_MARKERS[cohort])
    text_key(ax, [(COHORT_DICT[cohort]["label"], COHORT_DICT[cohort]["color"]) for cohort in COHORTS])
    ax.set_ylabel("Intensity (a.u.)")


@style()
def plot_intensity_by_rc_offset(
    structure_tables: dict, save_path: str, offset_step: float = 10.0, max_offset: float = 100.0,
    min_count: int = 10, plot: bool = False,
):
    """Plot the mean stain intensity against the offset to the central path of Rosenthal's canal.

    Each point is the mean of one cochlea in one offset bin, drawn at the bin center. The last bin
    holds every offset of `max_offset` or more.

    Args:
        structure_tables: Tables of one structure, a value of load_tables.
        save_path: Output path without an extension.
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

    fig, ax = new_panel(*PANEL_SIZES["profile"])
    _draw_cohort_profiles(ax, profiles)
    ax.set_xlim(0, max_offset + offset_step)
    ax.set_xlabel("Offset to RC center (µm)")
    save_panel(fig, save_path, plot)


@style()
def plot_intensity_by_tonotopy(structure_tables: dict, save_path: str, n_bins: int = 20, plot: bool = False):
    """Plot the mean stain intensity against the length fraction, with the frequency as top axis.

    Args:
        structure_tables: Tables of one structure, a value of load_tables.
        save_path: Output path without an extension.
        n_bins: Number of fractional bins along the cochlea length.
        plot: Show interactive plot.
    """
    profiles = {}
    for name, table in structure_tables.items():
        binned = average_by_fraction(table["length_fraction"].values, table["intensity"].values, n_bins=n_bins)
        profiles[name] = pd.Series(
            binned["mean_syn_per_IHC"].values, index=binned["fraction_midpoint"].values.astype(float),
        )

    fig, ax = new_panel(*PANEL_SIZES["profile"])
    _draw_cohort_profiles(ax, profiles)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Length fraction")
    add_frequency_axis(ax, FONT_SIZE, FONT_SIZE, label="Frequency (kHz)")
    save_panel(fig, save_path, plot)


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
        # save_panel writes a PDF and a PNG preview.
        return os.path.join(args.figure_dir, name)

    tables = load_tables(intensity=args.intensity)
    print_cohort_averages(tables)
    # The intensity panels leave out the instances without an object measure. The count and
    # synapse panels keep them.
    measured = {
        structure: {name: table.dropna(subset=["intensity"]) for name, table in tables[structure].items()}
        for structure in args.intensity
    }

    if "SGN" in args.intensity:
        plot_intensity_by_cohort(measured["SGN"], "PV@SGNs", out("fig_02d_sgn_intensity_by_cohort"), plot=args.plot)
        plot_intensity_by_cohort(
            measured["SGN"], "PV@SGNs", out("fig_02d_sgn_intensity_by_cohort_all_instances"), per_cochlea=False,
            plot=args.plot,
        )
        plot_intensity_by_rc_offset(measured["SGN"], out("fig_02e_sgn_intensity_by_rc_offset"), plot=args.plot)
        summary = plot_sgn_intensity_check(tables["SGN"], measured["SGN"], out("sgn_intensity_check"), plot=args.plot)
        print("\nSGN count and PV intensity per cochlea")
        print(summary.to_string(index=False, float_format="{:.2f}".format))
    if "IHC" in args.intensity:
        plot_intensity_by_cohort(
            measured["IHC"], "VGlut3@IHCs", out("fig_02f_ihc_intensity_by_cohort"), plot=args.plot,
        )
        plot_intensity_by_cohort(
            measured["IHC"], "VGlut3@IHCs", out("fig_02f_ihc_intensity_by_cohort_all_instances"), per_cochlea=False,
            plot=args.plot,
        )
        plot_intensity_by_tonotopy(measured["IHC"], out("fig_02g_ihc_intensity_by_tonotopy"), plot=args.plot)

    plot_counts_by_cohort(tables, out("fig_03a_counts_by_cohort"), plot=args.plot)
    plot_counts_by_cohort(tables, out("fig_03a_counts_by_cohort_outlier"), plot=args.plot, remove_outliers=True)
    plot_synapses_by_tonotopy(tables, out("fig_03b_synapses_by_tonotopy"), plot=args.plot)
    plot_synapses_by_tonotopy(tables, out("fig_03b_synapses_by_tonotopy_trendline"), plot=args.plot, trendline=True)
    plot_synapses_by_tonotopy(
        tables, out("fig_03b_synapses_by_tonotopy_trendline_legend"), plot=args.plot, trendline=True,
        show_legend=True,
    )
    colocalization = load_colocalization()
    plot_synapse_colocalization(colocalization, out("synapse_colocalization"), plot=args.plot)
    write_colocalization_table(colocalization, os.path.join(args.figure_dir, "colocalization.md"))
    write_colocalization_excel(colocalization, os.path.join(args.figure_dir, "colocalization.xlsx"))


if __name__ == "__main__":
    main()
