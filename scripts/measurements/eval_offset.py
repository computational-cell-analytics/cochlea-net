"""Compare the PV intensity of SGNs against their offset to the central path of Rosenthal's canal.

The script reads the SGN_v2 segmentation table of each cochlea from S3. The table holds the median
PV intensity with background mask in the column 'PV_bg-mask_median'. The script maps the table in
memory with the default central path method of `tonotopic_mapping` and the component list of the
processing JSON, because the mapping on S3 is missing or out of date for these cochleae.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.analysis.central_path_utils import mean_by_offset
from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping
from flamingo_tools.s3_utils import get_s3_path

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "figures"))
from util import COHORT_DICT, OUTLIER_DICT, cohort_cochleae  # noqa: E402

PROCESSING_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "reproducibility", "processing")
PNG_DPI = 300


def load_cochlea(cochlea: str) -> pd.DataFrame:
    """Load the mapped SGNs of a cochlea with their absolute and normalized PV intensity."""
    s3_path, fs = get_s3_path(f"{cochlea}/tables/SGN_v2/default.tsv")
    with fs.open(s3_path, "r") as f:
        table = pd.read_csv(f, sep="\t")
    with open(os.path.join(PROCESSING_DIR, f"{cochlea}_SGN.json")) as f:
        component_list = json.load(f)["component_list"]
    table, _ = tonotopic_mapping(table, component_label=component_list, cell_type="sgn")

    table = table[table["offset"] >= 0].rename(columns={"PV_bg-mask_median": "intensity"})
    n_missing = table["intensity"].isna().sum()
    if n_missing > 0:
        raise ValueError(f"{cochlea}: {n_missing} mapped instances have no PV intensity.")

    table["intensity_norm"] = table["intensity"] / table["intensity"].median()
    return table


def plot_heatmaps(cells: pd.DataFrame, cohorts: list, min_count: int, save_path: str):
    """Plot the mean normalized intensity per window and offset bin, one heatmap per cohort.

    Each cochlea has equal weight in the cohort mean. Outliers and cells with fewer than
    `min_count` instances are left out.
    """
    cells = cells[~cells["outlier"] & (cells["n"] >= min_count)]
    grids = {
        cohort: cells[cells["cohort"] == cohort].groupby(
            ["offset_bin", "length_bin"], observed=False
        )["intensity_norm"].mean().unstack()
        for cohort in cohorts
    }
    deviation = np.nanmax([np.nanmax(np.abs(grid.values - 1)) for grid in grids.values()])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#bdbdbd")

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 4), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, (cohort, grid) in zip(axes[0], grids.items()):
        n_offset_bins = len(grid.index)
        image = ax.imshow(grid.values, origin="lower", aspect="auto", extent=(0, 1, 0, n_offset_bins),
                          cmap=cmap, vmin=1 - deviation, vmax=1 + deviation)
        n_cochleae = cells.loc[cells["cohort"] == cohort, "cochlea"].nunique()
        ax.set_title(f"{COHORT_DICT[cohort]['label']} ({n_cochleae} cochleae)")
        ax.set_xlabel("Length fraction")
        ax.set_yticks(range(n_offset_bins), [f"{edge:g}" for edge in grid.index])
    axes[0, 0].set_ylabel("Offset [µm]")
    fig.colorbar(image, ax=axes[0], label="PV intensity / cochlea median")
    fig.suptitle(f"Gray: fewer than {min_count} SGNs", fontsize="small", x=0.01, ha="left")

    plt.savefig(save_path, dpi=PNG_DPI)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for tables and plot.")
    parser.add_argument("--cohorts", type=str, nargs="+", default=["idisco", "mwfls"],
                        help="Cohorts to compare. Keys of COHORT_DICT in scripts/figures/util.py.")
    parser.add_argument("--min_count", type=int, default=10,
                        help="Minimal number of SGNs in a cell of a cochlea for the plot.")
    args = parser.parse_args()

    summary, cells = [], []
    for cohort in args.cohorts:
        for cochlea in cohort_cochleae(cohort):
            table = load_cochlea(cochlea)
            outlier = cochlea in OUTLIER_DICT["SGN"]
            summary.append({"cochlea": cochlea, "cohort": cohort, "outlier": outlier, "n": len(table),
                            "median_intensity": table["intensity"].median()})
            binned = mean_by_offset(table, ["intensity", "intensity_norm"])
            binned.insert(0, "outlier", outlier)
            binned.insert(0, "cohort", cohort)
            binned.insert(0, "cochlea", cochlea)
            cells.append(binned)
    summary = pd.DataFrame(summary)
    cells = pd.concat(cells, ignore_index=True)

    os.makedirs(args.output_dir, exist_ok=True)
    summary.to_csv(os.path.join(args.output_dir, "SGN_cochlea_intensity.tsv"), sep="\t", index=False)
    cells.to_csv(os.path.join(args.output_dir, "SGN_offset_intensity.tsv"), sep="\t", index=False)
    plot_heatmaps(cells, args.cohorts, args.min_count, os.path.join(args.output_dir, "SGN_offset_intensity.png"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
