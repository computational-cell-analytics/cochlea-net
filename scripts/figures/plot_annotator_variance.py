"""Plot the annotator variance of manually thresholded marker annotations.

The input are the `<cochlea>_<marker>_<seg>_variance.json` files written by
`scripts/measurements/eval_marker_annotations.py` and `eval_subtype_annotations.py`.
Each file holds one scenario per annotator plus a "median" consensus scenario.
"""

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from util import (cochlea_label, export_legend, get_flatline_handle, get_marker_handle, prism_cleanup_axes,
                  prism_palette, prism_style)
from plot_fig4 import COCHLEAE_DICT

FILE_EXTENSION = "png"
png_dpi = 300

# Display names for the annotators. The order also fixes the marker shape and the x offset.
# The marker annotations and the subtype annotations use a different key for the same annotator.
ANNOTATOR_ALIAS = {
    "ResultsAMD": "Annotator 1",
    "Result_AMD": "Annotator 1",
    "ResultsEK": "Annotator 2",
    "Result_EK": "Annotator 2",
    "ResultsLR": "Annotator 3",
}

ANNOTATOR_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*"]

# Colors per plotted series. A series is a marker class, e.g. "positive", or "<stain> <class>"
# if the input contains more than one stain.
CLASS_COLORS = {
    "positive": "#279C52",
    "negative": "#9C5027",
    "Type Ia": "#4E79A7",
    "Type Ib": "#F28E2B",
    "Type Ic": "#E15759",
    "Type IbIc": "#76B7B2",
    "Type II": "#B07AA1",
    "inconclusive": "#BAB0AC",
}

MEDIAN_KEY = "median"
MEDIAN_LABEL = "Median"


def _class_names(scenario: dict) -> List[str]:
    """Get the marker classes of a scenario, e.g. ["positive", "negative"]."""
    return [key[len("percent_"):] for key in scenario if key.startswith("percent_")]


def _cochlea_label(cochlea: str, use_alias: bool) -> str:
    # An unregistered cochlea has no alias, so it always falls back to the shortened name.
    use_alias = use_alias and cochlea in COCHLEAE_DICT
    return cochlea_label(cochlea, COCHLEAE_DICT.get(cochlea, {}), use_alias)


def load_variance_records(
    input_dir: str,
    marker: Optional[str] = None,
    pattern: str = "*_variance.json",
    use_alias: bool = True,
    min_crops: int = 5,
) -> pd.DataFrame:
    """Read the variance JSON files of a directory into a long-form table.

    Args:
        input_dir: Directory containing the variance JSON files.
        marker: Optional marker name. Only files of this marker are loaded.
        pattern: Glob pattern for the variance files.
        use_alias: Use the cochlea alias for the x axis label.
        min_crops: Minimal number of annotated crops to include annotator.

    Returns:
        Table with one row per cochlea, series, and annotator. The "median" scenario is marked
        with `is_median`.
    """
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if len(paths) == 0:
        raise ValueError(f"No file matching '{pattern}' in {input_dir}.")

    contents = []
    for path in paths:
        with open(path, "r") as f:
            content = json.load(f)
        if marker is not None and content.get("marker") != marker:
            continue
        contents.append(content)
    if len(contents) == 0:
        raise ValueError(f"No file for marker {marker} in {input_dir}.")

    # The series name only carries the stain if the input mixes several stains.
    markers = {content.get("marker") for content in contents}
    prefix_marker = len(markers) > 1

    records = []
    for content in contents:
        cochlea = content["cochlea"]
        marker_name = content.get("marker")
        for scenario_name, scenario in content["scenarios"].items():
            is_median = scenario_name == MEDIAN_KEY
            for class_name in _class_names(scenario):
                percent = scenario[f"percent_{class_name}"]
                if percent is None:
                    continue
                if scenario["n_crops"] < min_crops:
                    continue
                records.append({
                    "cochlea": cochlea,
                    "label": _cochlea_label(cochlea, use_alias),
                    "marker": marker_name,
                    "class_name": class_name,
                    "series": f"{marker_name} {class_name}" if prefix_marker else class_name,
                    "annotator": scenario_name,
                    "annotator_label": MEDIAN_LABEL if is_median
                    else ANNOTATOR_ALIAS.get(scenario_name, scenario_name),
                    "percent": float(percent),
                    "is_median": is_median,
                })

    return pd.DataFrame(records)


def find_break_ranges(
    values: Sequence[float],
    n_breaks: int = 1,
    min_gap: float = 15.0,
    pad: float = 2.0,
) -> List[Tuple[float, float]]:
    """Find y ranges without data points, which can be left out of the plot.

    Args:
        values: The plotted values.
        n_breaks: Maximal number of breaks.
        min_gap: Minimal size of a gap in percentage points for it to become a break.
        pad: Margin in percentage points kept between a data point and the break.

    Returns:
        Sorted list of (lower, upper) ranges to leave out. The list is empty if no gap qualifies.
    """
    values = np.unique(np.asarray(values, dtype=float))
    if len(values) < 2 or n_breaks < 1:
        return []

    gaps = np.diff(values)
    # A break must stay smaller than the gap, so the padding cannot eat it up.
    valid = np.nonzero(gaps > max(min_gap, 2 * pad + 1e-6))[0]
    if len(valid) == 0:
        return []

    # Sort by gap size, keeping the first occurrence in case of ties.
    order = sorted(valid, key=lambda idx: (-gaps[idx], idx))[:n_breaks]
    return sorted((float(values[idx] + pad), float(values[idx + 1] - pad)) for idx in sorted(order))


def _segments_from_breaks(
    ylim: Tuple[float, float],
    break_ranges: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    """Split the y range into segments, ordered from the top segment to the bottom segment."""
    edges = [ylim[0]]
    for lower, upper in break_ranges:
        edges.extend([lower, upper])
    edges.append(ylim[1])
    segments = [(edges[i], edges[i + 1]) for i in range(0, len(edges), 2)]
    return segments[::-1]


def _draw_break_marks(axes: Sequence[plt.Axes], height_ratios: Sequence[float]) -> None:
    """Draw the diagonal marks between the panels of a broken axis."""
    size = 0.015
    mean_ratio = float(np.mean(height_ratios))
    for num in range(len(axes) - 1):
        upper, lower = axes[num], axes[num + 1]
        # The marks are given in axes coordinates, so they are scaled to look equally large.
        dy_upper = size * mean_ratio / height_ratios[num]
        dy_lower = size * mean_ratio / height_ratios[num + 1]
        kwargs = dict(color="black", clip_on=False, linewidth=1.2)
        upper.plot((-size, +size), (-dy_upper, +dy_upper), transform=upper.transAxes, **kwargs)
        lower.plot((-size, +size), (1 - dy_lower, 1 + dy_lower), transform=lower.transAxes, **kwargs)


def _annotator_order(df: pd.DataFrame, show_median: bool) -> List[str]:
    """Order the scenarios, so that an annotator keeps its shape and x offset for every cochlea."""
    present = set(df["annotator"])
    annotators = [name for name in ANNOTATOR_ALIAS if name in present]
    annotators += sorted(name for name in present if name not in ANNOTATOR_ALIAS and name != MEDIAN_KEY)
    if show_median and MEDIAN_KEY in present:
        annotators = [MEDIAN_KEY] + annotators
    return annotators


def _annotator_shapes(annotators: Sequence[str]) -> Dict[str, str]:
    """Assign a marker shape per scenario.

    The shape follows the alias, not the position within `annotators`. An annotator keeps its shape
    for every cochlea and cohort, also if only a subset of the annotators is plotted.
    """
    order = list(dict.fromkeys(ANNOTATOR_ALIAS.values()))
    shapes = {}
    for name in annotators:
        if name == MEDIAN_KEY:
            shapes[name] = "_"
            continue
        label = ANNOTATOR_ALIAS.get(name)
        if label in order:
            index = order.index(label)
        else:
            index = len(order)
            order.append(label if label is not None else name)
        shapes[name] = ANNOTATOR_MARKERS[index % len(ANNOTATOR_MARKERS)]
    return shapes


def _x_offsets(
    series_names: Sequence[str],
    annotators: Sequence[str],
    annotator_offset: float = 0.06,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Get the x offsets. The series are spread over the slot of a cochlea, the annotators within
    a series."""
    series_offset = max(annotator_offset * len(annotators) + 0.06, 0.2)
    series_shift = {name: (num - (len(series_names) - 1) / 2) * series_offset
                    for num, name in enumerate(series_names)}
    annotator_shift = {name: (num - (len(annotators) - 1) / 2) * annotator_offset
                       for num, name in enumerate(annotators)}
    return series_shift, annotator_shift


def _series_colors(series_names: Sequence[str]) -> Dict[str, str]:
    colors = {}
    fallback = [color for color in prism_palette if color not in CLASS_COLORS.values()]
    for name in series_names:
        if name in CLASS_COLORS:
            colors[name] = CLASS_COLORS[name]
        else:
            colors[name] = fallback[len(colors) % len(fallback)]
    return colors


def _legend_entries(
    series_names: Sequence[str],
    colors: Dict[str, str],
    annotators: Sequence[str],
    shapes: Dict[str, str],
) -> Tuple[list, List[str]]:
    handles = [get_flatline_handle(colors[name]) for name in series_names]
    labels = [name.capitalize() if name.islower() else name for name in series_names]
    for annotator in annotators:
        if annotator == MEDIAN_KEY:
            handles.append(get_marker_handle("black", shapes[annotator]))
            labels.append(MEDIAN_LABEL)
        else:
            handles.append(get_marker_handle("black", shapes[annotator]))
            labels.append(ANNOTATOR_ALIAS.get(annotator, annotator))
    return handles, labels


def plot_annotator_variance(
    df: pd.DataFrame,
    save_path: str,
    classes: Optional[Sequence[str]] = None,
    break_ranges: Union[str, None, Tuple[float, float], Sequence[Tuple[float, float]]] = "auto",
    n_breaks: int = 1,
    min_gap: float = 15.0,
    show_median: bool = True,
    show_range: bool = False,
    show_legend: bool = True,
    ylim: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (12, 6),
    plot: bool = False,
) -> None:
    """Plot the marker percentage per cochlea, with one point per annotator.

    Args:
        df: Long-form table created by `load_variance_records`.
        save_path: File path for the figure.
        classes: Marker classes to plot, e.g. ["positive"]. All classes are plotted by default.
        break_ranges: Y ranges to leave out. "auto" finds the largest gaps without data points,
            None plots a single axis, a tuple or list of tuples sets the ranges explicitly.
        n_breaks: Maximal number of breaks for `break_ranges="auto"`.
        min_gap: Minimal gap in percentage points for `break_ranges="auto"`.
        show_median: Show the median scenario, which is the threshold used for the segmentation table.
        show_range: Draw a line from the lowest to the highest annotator value.
        show_legend: Draw the legend in the figure.
        ylim: Y limits. They are derived from the data by default.
        figsize: Figure size.
        plot: Show the figure.
    """
    prism_style()

    main_label_size = 20
    main_tick_size = 16
    xtick_size = 14

    if classes is not None:
        df = df[df["class_name"].isin(classes)]
        if len(df) == 0:
            raise ValueError(f"No data for the classes {classes}.")
    if not show_median:
        df = df[~df["is_median"]]

    cochleae = list(dict.fromkeys(df["cochlea"]))
    known = [name for name in COCHLEAE_DICT if name in cochleae]
    cochleae = known + sorted(name for name in cochleae if name not in COCHLEAE_DICT)
    labels = [df.loc[df["cochlea"] == name, "label"].iloc[0] for name in cochleae]

    series_names = list(dict.fromkeys(df["series"]))
    colors = _series_colors(series_names)
    annotators = _annotator_order(df, show_median)
    shapes = _annotator_shapes(annotators)
    series_shift, annotator_shift = _x_offsets(series_names, annotators)

    values = df["percent"].to_numpy()
    if ylim is None:
        pad = max(0.05 * (values.max() - values.min()), 1.0)
        ylim = (max(0.0, values.min() - pad), min(100.0, values.max() + pad))

    if break_ranges == "auto":
        break_ranges = find_break_ranges(values, n_breaks=n_breaks, min_gap=min_gap)
    elif break_ranges is None:
        break_ranges = []
    elif len(break_ranges) == 2 and np.isscalar(break_ranges[0]):
        break_ranges = [tuple(break_ranges)]
    break_ranges = sorted(tuple(rng) for rng in break_ranges)

    hidden = df[df["percent"].apply(lambda val: any(low < val < high for low, high in break_ranges))]
    for _, row in hidden.iterrows():
        print(f"Warning: {row['label']} / {row['annotator_label']} / {row['series']} "
              f"at {row['percent']:.2f}% falls into a break and is not shown.")

    segments = _segments_from_breaks(ylim, break_ranges)
    height_ratios = [high - low for low, high in segments]
    fig, axes = plt.subplots(
        len(segments), 1, sharex=True, figsize=figsize,
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.08},
    )
    axes = np.atleast_1d(axes)

    for ax in axes:
        for num, cochlea in enumerate(cochleae):
            for series in series_names:
                subset = df[(df["cochlea"] == cochlea) & (df["series"] == series)]
                if len(subset) == 0:
                    continue
                x_center = num + series_shift[series]

                annotated = subset[~subset["is_median"]]
                if show_range and len(annotated) > 1:
                    ax.plot(
                        [x_center, x_center], [annotated["percent"].min(), annotated["percent"].max()],
                        color=colors[series], alpha=0.35, linewidth=1.5, zorder=0,
                    )

                for _, row in subset.iterrows():
                    x_pos = x_center + annotator_shift[row["annotator"]]
                    if row["is_median"]:
                        ax.scatter(
                            x_pos, row["percent"], marker=shapes[MEDIAN_KEY], s=260, linewidths=2.5,
                            color=colors[series], alpha=1.0, zorder=3,
                        )
                    else:
                        ax.scatter(
                            x_pos, row["percent"], marker=shapes[row["annotator"]], s=90,
                            color=colors[series], alpha=0.6, zorder=2,
                        )

    for ax, (low, high) in zip(axes, segments):
        ax.set_ylim(low, high)
        n_ticks = max(2, int(round(4 * (high - low) / max(height_ratios))))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=n_ticks, steps=[1, 2, 5, 10]))
        ax.tick_params(axis="y", labelsize=main_tick_size)

    prism_cleanup_axes(axes)
    for ax in axes[:-1]:
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
    if len(axes) > 1:
        _draw_break_marks(axes, height_ratios)

    axes[-1].set_xlim(-0.5, len(cochleae) - 0.5)
    axes[-1].set_xticks(range(len(cochleae)))
    axes[-1].set_xticklabels(labels, rotation=45, ha="right", fontsize=xtick_size)

    y_label = "Marker positive [%]" if series_names == ["positive"] else "Fraction of segmented cells [%]"
    fig.supylabel(y_label, fontsize=main_label_size, fontweight="bold")

    if show_legend:
        handles, legend_labels = _legend_entries(series_names, colors, annotators, shapes)
        axes[0].legend(
            handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_labels), frameon=False, fontsize=main_tick_size, handletextpad=0.4,
            columnspacing=1.2,
        )

    if save_path.endswith(".png"):
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close(fig)


def plot_annotator_offset(
    df: pd.DataFrame,
    save_path: str,
    reference_class: str = "positive",
    ylim: Optional[Tuple[float, float]] = None,
    default_ylim: Tuple[float, float] = (-5.0, 5.0),
    show_legend: bool = True,
    figsize: Tuple[float, float] = (12, 5),
    plot: bool = False,
) -> None:
    """Plot the deviation of the annotators from the median, with the median as the zero reference.

    This function is specific for a marker that divides the cells into positive and negative.
    The percentages of both classes add up to 100, so the deviation of the negative class is the
    mirror image of the positive one and only `reference_class` is plotted.

    Args:
        df: Long-form table created by `load_variance_records`.
        save_path: File path for the figure.
        reference_class: Marker class used for the deviation, either "positive" or "negative".
        ylim: Y limits. They are derived from the data by default.
        default_ylim: Y limits used if the deviations stay within this range.
        show_legend: Draw the legend in the figure.
        figsize: Figure size.
        plot: Show the figure.
    """
    prism_style()

    main_label_size = 20
    main_tick_size = 16
    xtick_size = 14

    classes = set(df["class_name"])
    if not classes <= {"positive", "negative"}:
        raise ValueError(
            f"plot_annotator_offset needs a positive / negative marker, but the input has {sorted(classes)}."
        )
    df = df[df["class_name"] == reference_class]
    if len(df) == 0:
        raise ValueError(f"No data for the class {reference_class}.")

    # Subtract the median per cochlea and series, which puts every cochlea on the same reference.
    offsets = []
    for (cochlea, series), group in df.groupby(["cochlea", "series"], sort=False):
        reference = group.loc[group["is_median"], "percent"]
        if len(reference) == 0:
            print(f"Warning: skipping {cochlea} / {series}. No median scenario available.")
            continue
        annotated = group[~group["is_median"]].copy()
        annotated["offset"] = annotated["percent"] - float(reference.iloc[0])
        offsets.append(annotated)
    if len(offsets) == 0:
        raise ValueError("No cochlea with a median scenario.")
    df = pd.concat(offsets, ignore_index=True)

    cochleae = list(dict.fromkeys(df["cochlea"]))
    known = [name for name in COCHLEAE_DICT if name in cochleae]
    cochleae = known + sorted(name for name in cochleae if name not in COCHLEAE_DICT)
    labels = [df.loc[df["cochlea"] == name, "label"].iloc[0] for name in cochleae]

    series_names = list(dict.fromkeys(df["series"]))
    colors = _series_colors(series_names)
    annotators = _annotator_order(df, show_median=False)
    shapes = _annotator_shapes(annotators)
    series_shift, annotator_shift = _x_offsets(series_names, annotators)

    if ylim is None:
        # The default range is kept unless a deviation exceeds it, so figures stay comparable.
        limit = max(abs(default_ylim[0]), abs(default_ylim[1]),
                    float(np.ceil(1.1 * df["offset"].abs().max())))
        ylim = (-limit, limit)

    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(0, color="black", linewidth=1.2, zorder=1)

    for num, cochlea in enumerate(cochleae):
        for series in series_names:
            subset = df[(df["cochlea"] == cochlea) & (df["series"] == series)]
            for _, row in subset.iterrows():
                x_pos = num + series_shift[series] + annotator_shift[row["annotator"]]
                ax.scatter(
                    x_pos, row["offset"], marker=shapes[row["annotator"]], s=90,
                    color=colors[series], alpha=0.6, zorder=2,
                )

    ax.set_ylim(*ylim)
    ax.tick_params(axis="y", labelsize=main_tick_size)
    ax.set_xlim(-0.5, len(cochleae) - 0.5)
    ax.set_xticks(range(len(cochleae)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=xtick_size)
    ax.set_ylabel("Deviation from median [% points]", fontsize=main_label_size)
    prism_cleanup_axes(ax)

    if show_legend:
        # The color only carries information if the input mixes several stains.
        legend_series = series_names if len(series_names) > 1 else []
        handles, legend_labels = _legend_entries(legend_series, colors, annotators, shapes)
        ax.legend(
            handles, legend_labels, loc="lower center", bbox_to_anchor=(0.5, 1.02),
            ncol=len(legend_labels), frameon=False, fontsize=main_tick_size, handletextpad=0.4,
            columnspacing=1.2,
        )

    if save_path.endswith(".png"):
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close(fig)


def plot_annotator_variance_legend(
    df: pd.DataFrame,
    save_path: str,
    classes: Optional[Sequence[str]] = None,
    show_median: bool = True,
) -> None:
    """Save the legend of `plot_annotator_variance` as a separate figure."""
    prism_style()

    if classes is not None:
        df = df[df["class_name"].isin(classes)]
    if not show_median:
        df = df[~df["is_median"]]

    series_names = list(dict.fromkeys(df["series"]))
    colors = _series_colors(series_names)
    annotators = _annotator_order(df, show_median)
    shapes = _annotator_shapes(annotators)

    handles, labels = _legend_entries(series_names, colors, annotators, shapes)
    legend = plt.legend(handles, labels, loc=(0, 0), ncol=len(labels), framealpha=1, frameon=False)
    export_legend(legend, save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot the annotator variance of marker annotations.")
    parser.add_argument("--input_dir", "-i", type=str, required=True,
                        help="Directory containing the variance JSON files.")
    parser.add_argument("--figure_dir", "-f", type=str, default="./panels/annotator_variance",
                        help="Output directory for plots.")
    parser.add_argument("--pattern", type=str, default="*_variance.json",
                        help="Glob pattern for the variance files.")
    parser.add_argument("--marker", type=str, default=None,
                        help="Only plot the variance of this marker, e.g. GFP.")
    parser.add_argument("--classes", type=str, nargs="+", default=None,
                        help="Marker classes to plot, e.g. positive negative.")
    parser.add_argument("--break_range", type=float, nargs=2, default=None, metavar=("LOWER", "UPPER"),
                        help="Y range to leave out of the plot, e.g. 20 70.")
    parser.add_argument("--no_break", action="store_true", help="Plot a single unbroken y axis.")
    parser.add_argument("--n_breaks", type=int, default=0, help="Maximal number of automatic breaks.")
    parser.add_argument("--min_gap", type=float, default=15.0,
                        help="Minimal gap in percentage points for an automatic break.")
    parser.add_argument("--no_median", action="store_true", help="Do not plot the median scenario.")
    parser.add_argument("--min_crops", type=int, default=5,
                        help="Minimal number of annotated crops to include an annotator.")
    parser.add_argument("--ylim", type=float, nargs=2, default=None, metavar=("LOWER", "UPPER"),
                        help="Y limits of the offset plot, e.g. -3 3.")
    parser.add_argument("--no_alias", action="store_true", help="Do not use the cochlea alias.")
    parser.add_argument("--plot", action="store_true", help="Show the figures.")
    parser.add_argument("--offset", action="store_true", help="Plot offset figures.")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    df = load_variance_records(
        args.input_dir, marker=args.marker, pattern=args.pattern, use_alias=not args.no_alias,
        min_crops=args.min_crops,
    )
    marker_name = args.marker if args.marker is not None else "-".join(sorted(set(df["marker"].dropna())))
    show_median = not args.no_median

    if args.no_break:
        break_ranges = None
    elif args.break_range is not None:
        break_ranges = [tuple(args.break_range)]
    else:
        break_ranges = "auto"

    suffix = "" if args.classes is None else "_" + "-".join(args.classes)
    plot_annotator_variance(
        df,
        save_path=os.path.join(args.figure_dir, f"annotator_variance_{marker_name}{suffix}.{FILE_EXTENSION}"),
        classes=args.classes, break_ranges=break_ranges, n_breaks=args.n_breaks, min_gap=args.min_gap,
        show_median=show_median, plot=args.plot,
    )
    plot_annotator_variance_legend(
        df,
        save_path=os.path.join(args.figure_dir, f"annotator_variance_{marker_name}{suffix}_legend.{FILE_EXTENSION}"),
        classes=args.classes, show_median=show_median,
    )

    if args.offset:
        plot_annotator_offset(
            df,
            save_path=os.path.join(args.figure_dir, f"annotator_offset_{marker_name}.{FILE_EXTENSION}"),
            ylim=None if args.ylim is None else tuple(args.ylim), plot=args.plot,
        )
        plot_annotator_variance_legend(
            df,
            save_path=os.path.join(args.figure_dir, f"annotator_offset_{marker_name}_legend.{FILE_EXTENSION}"),
            classes=["positive"], show_median=False,
        )


if __name__ == "__main__":
    main()
