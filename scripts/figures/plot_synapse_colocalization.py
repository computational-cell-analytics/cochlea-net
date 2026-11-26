"""Plot how the CtBP2 synapses and the co-stain synapses of the cosynapse cochleae co-localize.

The script writes the panel and a Markdown and an Excel table of the synapse counts.
"""
import argparse
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea, VPacker
from matplotlib.ticker import MaxNLocator

from flamingo_tools.json_util import STEP_KEYS, load_processing_params
from flamingo_tools.postprocessing.label_components import label_components_table
from flamingo_tools.s3_utils import create_s3_target

from data import read_source_table
from panel_style import COLORS, FONT_SIZE, new_panel, readable, save_panel, style, text_color_on

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSING_DIR = os.path.join(SCRIPT_DIR, "..", "..", "reproducibility", "processing")

# The co-stain of each cosynapse cochlea. CtBP2 is the reference stain of both.
COLOCALIZATION = {"M_AMD_000117_L": "Homer1", "M_AMD_000118_L": "Bsn"}
# The segments of a co-localization bar: the score key of evaluate_colocalization, the label and the color.
# Blue and green stand for the MWfLS and iDISCO cohorts in plot_mwfls.py, so the segments use
# other colors.
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

# The print size of the panel in mm (width, height).
PANEL_SIZE = (180, 40)


def row_ihcs(table: pd.DataFrame, params: dict) -> np.ndarray:
    """Return the label ids of the IHC row that the label_components parameters of a processing file select.

    `label_components_table` labels the components in memory, as flamingo_tools.label_components does it.

    Args:
        table: IHC segmentation table.
        params: One entry of load_processing_params(<cochlea>_IHC.json, "label_components").

    Returns:
        The label ids of the instances in the components of 'component_list'.
    """
    keys = STEP_KEYS["label_components"] | {"cell_type", "component_list"}
    labeled = label_components_table(table, **{key: value for key, value in params.items() if key in keys}).table
    return labeled["label_id"][labeled["component_labels"].isin(params["component_list"])].values


def load_colocalization(max_dist: float = 2.0, read_table=None) -> dict:
    """Match the co-stain synapses of each COLOCALIZATION cochlea against its CtBP2 synapses.

    The IHC components are labeled in memory with the IHC_v11 entry of the processing file, see row_ihcs.
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
        entries = load_processing_params(os.path.join(PROCESSING_DIR, f"{cochlea}_IHC.json"), "label_components")
        params, = [entry for entry in entries if entry["segmentation_channel"] == "IHC_v11"]
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

    fig, (ax, ax_table) = new_panel(*PANEL_SIZE, ncols=2, sharey=True,
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


def write_colocalization_markdown(colocalization: dict, save_path: str):
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


def main():
    parser = argparse.ArgumentParser(
        description="Plot the synapse co-localization of the cosynapse cochleae and write its tables."
    )
    parser.add_argument("--figure_dir", "-f", type=str, default="./panels/synapse_colocalization",
                        help="Output directory for the panel and the tables.")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--max_dist", type=float, default=2.0,
                        help="Maximum distance of a co-localized pair in micrometers.")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)
    colocalization = load_colocalization(args.max_dist)
    plot_synapse_colocalization(colocalization, os.path.join(args.figure_dir, "synapse_colocalization"),
                                plot=args.plot)
    write_colocalization_markdown(colocalization, os.path.join(args.figure_dir, "colocalization.md"))
    write_colocalization_excel(colocalization, os.path.join(args.figure_dir, "colocalization.xlsx"))


if __name__ == "__main__":
    main()
