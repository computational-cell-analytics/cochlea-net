"""Summarize the stain comparisons of several cochleae in tables and in one figure per intensity measure.

The inputs are output folders of `flamingo_tools.synapse_detection.stain_comparison`. All of them must compare
the same reference and test channels.

Run `python -m flamingo_tools.synapse_detection.stain_summary -h` for the command line.
"""

import argparse
import json
import os
import warnings
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from flamingo_tools.synapse_detection.stain_comparison import intensity_summary

# A categorical palette in a fixed order. Only the first three colors stay separable when all lines of a
# panel overlap, so more cochleae trigger a warning.
COCHLEA_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
_SEPARABLE_COLORS = 3
_GRID_COLOR = "#e4e3df"
_TEXT_COLOR = "#52514e"


def load_comparison(folder: str) -> dict:
    """Read the output folder of `stain_comparison`.

    Args:
        folder: The output folder, with 'summary.json' and the two `<name>_matched.tsv` files.

    Returns:
        The summary dictionary and the matched tables under the keys 'summary', 'reference' and 'test'.
    """
    with open(os.path.join(folder, "summary.json")) as f:
        summary = json.load(f)
    tables = {
        role: pd.read_csv(os.path.join(folder, f"{summary[f'{role}_name']}_matched.tsv"), sep="\t")
        for role in ("reference", "test")
    }
    return {"summary": summary, **tables}


def _channels(comparisons: Dict[str, dict]) -> Tuple[str, str]:
    channels = {(c["summary"]["reference_channel"], c["summary"]["test_channel"]) for c in comparisons.values()}
    if len(channels) != 1:
        raise ValueError(f"The comparisons must share the reference and the test channel, got {sorted(channels)}.")
    return channels.pop()


def summary_table(comparisons: Dict[str, dict]) -> pd.DataFrame:
    """Collect the spot counts and the colocalization scores of each cochlea.

    Args:
        comparisons: The outputs of `load_comparison`, keyed by the cochlea name.

    Returns:
        One row per cochlea. The per-IHC columns are NaN for a comparison without valid IHCs.
    """
    rows = []
    for name, comparison in comparisons.items():
        summary = comparison["summary"]
        n_ihcs = summary["n_ihcs"]
        rows.append({
            "cochlea": name,
            "n_ihcs": n_ihcs,
            "n_reference": summary["n_reference"],
            "n_test": summary["n_test"],
            "reference_per_ihc": summary["n_reference"] / n_ihcs if n_ihcs else np.nan,
            "test_per_ihc": summary["n_test"] / n_ihcs if n_ihcs else np.nan,
            "tp": summary["tp"],
            "precision": summary["precision"],
            "recall": summary["recall"],
        })
    return pd.DataFrame(rows)


def _style(ax) -> None:
    ax.grid(True, color=_GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _grouped_bars(ax, groups: Sequence[str], values: Dict[str, Sequence[float]], colors: Dict[str, str],
                  number_format: str) -> None:
    n = len(values)
    width = 0.8 / n
    x = np.arange(len(groups))
    for i, (name, group_values) in enumerate(values.items()):
        positions = x - 0.4 + width * (i + 0.5)
        bars = ax.bar(positions, np.nan_to_num(group_values), width=width * 0.92, color=colors[name], label=name)
        ax.bar_label(bars, labels=[number_format.format(v) if np.isfinite(v) else "n/a" for v in group_values],
                     fontsize=8, color=_TEXT_COLOR, padding=2)
    ax.set_xticks(x, groups)


def _density(ax, values: Dict[str, pd.Series], colors: Dict[str, str], xlabel: str, title: str,
             value_range: Optional[Tuple[float, float]] = None,
             medians: Optional[Dict[str, float]] = None, median_format: str = "{:.2f}") -> None:
    values = {name: series.dropna() for name, series in values.items()}
    pooled = pd.concat(values.values())
    if value_range is None:
        value_range = tuple(np.nanquantile(pooled, [0.005, 0.995])) if len(pooled) else (0.0, 1.0)
    if medians is None:
        medians = {name: series.median() for name, series in values.items()}
    bins = np.linspace(*value_range, 51)
    for name, series in values.items():
        label = f"{name} (n={len(series)}, median {median_format.format(medians[name])})" if len(series) \
            else f"{name} (n=0)"
        if len(series):
            ax.hist(series, bins=bins, density=True, histtype="step", linewidth=2, color=colors[name], label=label)
        else:
            ax.plot([], [], color=colors[name], linewidth=2, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    ax.set_title(title, loc="left")
    ax.legend(frameon=False, fontsize=8)


def plot_stain_summary(comparisons: Dict[str, dict], output_path: str, measure: str = "snr") -> None:
    """Compare the stain intensities of several cochleae in one figure.

    The first row shows the spots per IHC, the colocalization and the absolute intensities. The second row
    shows the test spots and the third row the reference spots, each in both stains and with the
    integrated intensity of the own stain. The color marks the cochlea.

    Args:
        comparisons: The outputs of `load_comparison`, keyed by the cochlea name.
        output_path: The output image file.
        measure: The relative intensity measure of the distribution panels, e.g. 'snr' or 'ratio'.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(comparisons) > len(COCHLEA_COLORS):
        raise ValueError(f"At most {len(COCHLEA_COLORS)} cochleae fit in one figure, got {len(comparisons)}.")
    if len(comparisons) > _SEPARABLE_COLORS:
        warnings.warn(f"More than {_SEPARABLE_COLORS} cochleae: the overlapping histograms are hard to separate.")
    reference_channel, test_channel = _channels(comparisons)
    colors = dict(zip(comparisons, COCHLEA_COLORS))
    table = summary_table(comparisons).set_index("cochlea")

    fig, axes = plt.subplots(3, 3, figsize=(17, 14))

    ax = axes[0, 0]
    per_ihc = table[["test_per_ihc", "reference_per_ihc"]].notna().all(axis=1).all()
    columns = ("test_per_ihc", "reference_per_ihc") if per_ihc else ("n_test", "n_reference")
    _grouped_bars(ax, [test_channel, reference_channel],
                  {name: table.loc[name, list(columns)].to_numpy(dtype="float64") for name in comparisons},
                  colors, "{:.1f}" if per_ihc else "{:.0f}")
    ax.set_ylabel("spots per valid IHC" if per_ihc else "spots")
    ax.set_title("(a) Spots per IHC" if per_ihc else "(a) Spots", loc="left")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    _grouped_bars(ax, [f"{test_channel} spots\nwith {reference_channel} partner",
                       f"{reference_channel} spots\nwith {test_channel} partner"],
                  {name: table.loc[name, ["precision", "recall"]].to_numpy(dtype="float64") for name in comparisons},
                  colors, "{:.2f}")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("fraction")
    max_dist = next(iter(comparisons.values()))["summary"]["max_dist"]
    ax.set_title(f"(b) Colocalization (max. {max_dist:g} µm)", loc="left")

    ax = axes[0, 2]
    groups = [(test_channel, "test", "peak"), (test_channel, "test", "background"),
              (reference_channel, "reference", "peak"), (reference_channel, "reference", "background")]
    width = 0.8 / len(comparisons)
    for i, (name, comparison) in enumerate(comparisons.items()):
        for j, (channel, role, measure_name) in enumerate(groups):
            values = comparison[role][f"{channel}_{measure_name}"].dropna()
            if not len(values):
                continue
            q25, q50, q75 = np.quantile(values, [0.25, 0.5, 0.75])
            position = j - 0.4 + width * (i + 0.5)
            ax.plot([position, position], [q25, q75], color=colors[name], linewidth=2)
            ax.plot(position, q50, "o", color=colors[name], markersize=8, label=name if j == 0 else None)
    ax.set_xticks(np.arange(len(groups)), [f"{channel}\n{measure_name}" for channel, _, measure_name in groups])
    ax.set_ylim(bottom=0)
    ax.set_ylabel("intensity at the own spots (median, IQR)")
    ax.set_title("(c) Absolute intensity", loc="left")
    ax.legend(frameon=False, fontsize=8)

    letters = iter("defghi")
    for row, role, own, other in ((1, "test", test_channel, reference_channel),
                                  (2, "reference", reference_channel, test_channel)):
        for col, channel in enumerate((own, other)):
            column = f"{channel}_{measure}"
            _density(axes[row, col], {name: c[role][column] for name, c in comparisons.items()}, colors,
                     f"{column} (0.5 - 99.5 % range)", f"({next(letters)}) {own} spots, {channel} stain")
        integrated, medians = {}, {}
        for name, c in comparisons.items():
            values = c[role][f"{own}_integrated"]
            medians[name] = values.median()
            integrated[name] = values / medians[name]
        _density(axes[row, 2], integrated, colors, "integrated intensity / median of the cochlea (-1 to 6 shown)",
                 f"({next(letters)}) {own} spots, integrated {own}", value_range=(-1.0, 6.0),
                 medians=medians, median_format="{:.0f}")

    for ax in axes.ravel():
        _style(ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def summarize_comparisons(
    comparisons: Dict[str, dict],
    output_folder: str,
    measures: Sequence[str] = ("snr", "ratio"),
) -> pd.DataFrame:
    """Write the summary tables and one summary figure per measure.

    The output folder receives 'summary.tsv' (counts and scores per cochlea), 'intensity_summary.tsv'
    (the quantiles of `stain_comparison.intensity_summary` with a 'cochlea' column) and
    `summary_<measure>.png` for each measure.

    Args:
        comparisons: The outputs of `load_comparison`, keyed by the cochlea name.
        output_folder: The output folder.
        measures: The relative intensity measures, one figure each.

    Returns:
        The summary table.
    """
    reference_channel, test_channel = _channels(comparisons)
    os.makedirs(output_folder, exist_ok=True)
    table = summary_table(comparisons)
    table.to_csv(os.path.join(output_folder, "summary.tsv"), sep="\t", index=False)

    intensity = []
    for name, comparison in comparisons.items():
        for role in ("reference", "test"):
            intensity.append(intensity_summary(comparison[role], [reference_channel, test_channel]).assign(
                cochlea=name, stain=comparison["summary"][f"{role}_name"],
            ))
    pd.concat(intensity, ignore_index=True).to_csv(
        os.path.join(output_folder, "intensity_summary.tsv"), sep="\t", index=False,
    )

    for measure in measures:
        plot_stain_summary(comparisons, os.path.join(output_folder, f"summary_{measure}.png"), measure=measure)
    return table


def _default_name(folder: str) -> str:
    folder = os.path.abspath(folder)
    name = os.path.basename(folder)
    return os.path.basename(os.path.dirname(folder)) if name == "comparison" else name


def main():
    parser = argparse.ArgumentParser(
        description="Summarize the stain comparisons of several cochleae: counts and colocalization per cochlea, "
        "and a figure per intensity measure. The inputs are output folders of "
        "flamingo_tools.synapse_detection.stain_comparison."
    )
    parser.add_argument("-i", "--input_folders", required=True, nargs="+", help="Output folders of stain_comparison.")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Cochlea names, one per folder. Default: the folder name, or the name of its parent "
                        "for a folder called 'comparison'.")
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder.")
    parser.add_argument("--measures", nargs="+", default=["snr", "ratio"],
                        help="Relative intensity measures, one figure each. Default: snr ratio")
    args = parser.parse_args()

    names = args.names if args.names is not None else [_default_name(folder) for folder in args.input_folders]
    if len(names) != len(args.input_folders):
        parser.error("Give one name per input folder.")
    if len(set(names)) != len(names):
        parser.error(f"The names must be unique, got {names}.")

    comparisons = {name: load_comparison(folder) for name, folder in zip(names, args.input_folders)}
    table = summarize_comparisons(comparisons, args.output_folder, measures=args.measures)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
