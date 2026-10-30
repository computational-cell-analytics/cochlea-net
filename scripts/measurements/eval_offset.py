"""Compare the PV intensity of SGNs against their offset to the central path of Rosenthal's canal.

The script reads the SGN segmentation table of each cochlea from S3, in the version of
plot_mwfls.SGN_VERSION, and the median PV intensity with background mask from its object measures.
It maps the table in memory with the default central path method of `tonotopic_mapping` and the
component list of util.COCHLEA_DICT, because the mapping on S3 can be missing or out of date.
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from flamingo_tools.analysis.central_path_utils import mean_by_offset
from flamingo_tools.s3_utils import create_s3_target

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "figures"))
from util import COHORT_DICT, OUTLIER_DICT, cohort_cochleae  # noqa: E402
from data import get_mapped_table  # noqa: E402
from plot_mwfls import SGN_VERSION  # noqa: E402

PNG_DPI = 300


def load_cochlea(s3, cochlea: str, sgn_version: str) -> pd.DataFrame:
    """Load the mapped SGNs of a cochlea with their absolute and normalized PV intensity."""
    table = get_mapped_table(s3, cochlea, "SGN", sgn_version, stain="PV")
    table["intensity_norm"] = table["intensity"] / table["intensity"].median()
    return table


def plot_heatmaps(cells: pd.DataFrame, cohorts: list, min_count: int, min_cochleae: int, save_path: str):
    """Plot the mean normalized intensity per window and offset bin, one heatmap per cohort.

    Each cochlea has equal weight in the cohort mean. Outliers and cells with fewer than
    `min_count` instances are left out. A grid cell needs `min_cochleae` cochleae.

    The color scale is logarithmic and symmetric around 1. Its limits are the 98th percentile
    of the deviation from 1, so that a few extreme cells do not compress the scale.
    """
    cells = cells[~cells["outlier"] & (cells["n"] >= min_count)]
    grids = {}
    for cohort in cohorts:
        grouped = cells[cells["cohort"] == cohort].groupby(["offset_bin", "length_bin"], observed=False)
        grid = grouped["intensity_norm"].mean().where(grouped["cochlea"].nunique() >= min_cochleae)
        grids[cohort] = grid.unstack()
    ratios = np.concatenate([grid.values.ravel() for grid in grids.values()])
    ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
    deviation = np.percentile(np.maximum(ratios, 1 / ratios), 98)
    norm = LogNorm(vmin=1 / deviation, vmax=deviation)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#bdbdbd")

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5 * len(cohorts), 4), sharey=True,
                             constrained_layout=True, squeeze=False)
    for ax, (cohort, grid) in zip(axes[0], grids.items()):
        n_offset_bins = len(grid.index)
        image = ax.imshow(grid.values, origin="lower", aspect="auto", extent=(0, 1, 0, n_offset_bins),
                          cmap=cmap, norm=norm)
        n_cochleae = cells.loc[cells["cohort"] == cohort, "cochlea"].nunique()
        ax.set_title(f"{COHORT_DICT[cohort]['label']} ({n_cochleae} cochleae)")
        ax.set_xlabel("Length fraction")
        labels = [f"{edge:g}" for edge in grid.index]
        labels[-1] = f"≥{labels[-1]}"
        ax.set_yticks(range(n_offset_bins), labels)
    axes[0, 0].set_ylabel("Offset [µm]")
    colorbar = fig.colorbar(image, ax=axes[0], label="PV intensity / cochlea median", extend="both")
    ticks = [t for t in (0.5, 0.67, 0.8, 1, 1.25, 1.5, 2) if 1 / deviation <= t <= deviation]
    colorbar.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
    colorbar.minorticks_off()
    fig.suptitle(f"Gray: fewer than {min_count} SGNs per cochlea or fewer than {min_cochleae} cochleae",
                 fontsize="small", x=0.01, ha="left")

    plt.savefig(save_path, dpi=PNG_DPI)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory for tables and plot.")
    parser.add_argument("--cohorts", type=str, nargs="+", default=["idisco", "mwfls"], choices=list(SGN_VERSION),
                        help="Cohorts to compare. Keys of SGN_VERSION in scripts/figures/plot_mwfls.py.")
    parser.add_argument("--min_count", type=int, default=10,
                        help="Minimal number of SGNs in a cell of a cochlea for the plot.")
    parser.add_argument("--min_cochleae", type=int, default=2,
                        help="Minimal number of cochleae in a cell of the cohort heatmap.")
    args = parser.parse_args()

    s3 = create_s3_target()
    summary, cells = [], []
    for cohort in args.cohorts:
        for cochlea in cohort_cochleae(cohort):
            table = load_cochlea(s3, cochlea, SGN_VERSION[cohort])
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
    plot_heatmaps(cells, args.cohorts, args.min_count, args.min_cochleae,
                  os.path.join(args.output_dir, "SGN_offset_intensity.png"))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
