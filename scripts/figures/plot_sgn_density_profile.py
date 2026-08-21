"""Plot the SGN density along Rosenthal's canal as a continuous function of the length fraction.

The density is a linear density in cells/µm along the central path through Rosenthal's canal.
The length fraction is used for the x-axis, because the cochleae differ in absolute length and
only the normalized position is comparable across animals and species.
"""

import argparse
import json
import os
import pickle
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import (
    COCHLEA_DICT,
    cochlea_components,
    cochlea_label,
    get_line_marker_handle,
    density_by_fraction_bins,
    density_by_sliding_window,
    get_flatline_handle,
    get_marker_handle,
    length_column,
    png_dpi,
    prism_palette,
    prism_style,
    total_run_length,
)

SOURCE_NAME = "SGN_v2"
FILE_EXTENSION = "png"

# Default selection of cochleae per cohort. The metadata lives in util.COCHLEA_DICT.
IDISCO = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]

MWFLS = ["M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R"]

CHREEF_MOUSE = [
    "M_LR_000144_L",
    "M_LR_000144_R",
    "M_LR_000145_L",
    "M_LR_000145_R",
    "M_LR_000153_L",
    "M_LR_000153_R",
    "M_LR_000155_L",
    "M_LR_000155_R",
    "M_LR_000189_L",
    "M_LR_000189_R",
]

FCHRIMSON_GERBIL = [
    "G_EK_000049_L",
    "G_EK_000049_R",
    "G_EK_000071_L",
    "G_EK_000071_R",
    "G_EK_000074_L",
    "G_EK_000074_R",
    "G_EK_000076_L",
    "G_EK_000076_R",
]

WT_GERBIL = [
    "G_EK_000233_L",
    "G_LR_000301_R",
    "G_LR_000302_R",
]

COHORTS = {
    "idisco": IDISCO,
    "mwfls": MWFLS,
    "chreef_mouse": CHREEF_MOUSE,
    "fchrimson_gerbil": FCHRIMSON_GERBIL,
    "wt_gerbil": WT_GERBIL,
}

COHORT_LABELS = {
    "idisco": "iDISCO",
    "mwfls": "MWfLS",
    "chreef_mouse": "ChReef mouse",
    "fchrimson_gerbil": "f-Chrimson gerbil",
    "wt_gerbil": "WT gerbil",
}

COHORT_COLORS = {
    "idisco": "#10CC17",
    "mwfls": "#3F69FF",
    "chreef_mouse": "#DB0063",
    "fchrimson_gerbil": "#8E00DB",
}

# Everything that is not listed here is a mouse.
COHORT_ANIMALS = {"fchrimson_gerbil": "gerbil", "wt_gerbil": "gerbil"}

MARKER_LEFT = "o"
MARKER_RIGHT = "^"

# Measurements to exclude from the trendline, as length fraction ranges per cochlea. The range
# rather than a bin index keeps the mask valid when n_bins changes.
# M_LR_000227_L has a gap in Rosenthal's canal near the middle of the canal, which produces a
# density far below the other cochleae of the cohort.
OUTLIERS = {
    "M_LR_000227_L": [(0.40, 0.45)],
}
OUTLIER_COLOR = "red"

# Style of a wild type trendline that is carried into an optogenetic therapy figure.
REFERENCE_STYLE = {"color": "black", "linestyle": "solid", "marker": "o", "label": "WT trendline"}

# Trendline per cochlea side for the ChReef cohorts, following fig_04e and fig_e_gerbil.
# The left cochlea is injected, the right one is not.
SIDE_TRENDLINES = {
    "L": {"label": "Injected", "color": "grey", "linestyle": "dashed", "alpha": 0.6},
    "R": {"label": "Non-Injected", "color": "grey", "linestyle": "dotted", "alpha": 0.7},
}

# Greenwood parameters f(x) = A * (10 ** (a * x) - k), see cochlea_mapping.map_frequency.
GREENWOOD = {
    "mouse": {"A": 1.46, "a": 1.77, "k": 0.0, "ticks": [2, 4, 8, 16, 32, 64]},
    "gerbil": {"A": 0.35, "a": 2.1, "k": 0.7, "ticks": [0.5, 1, 2, 4, 8, 16, 32]},
}


def get_sgn_length_data(
    cochleae: List[str],
    source_name: str = SOURCE_NAME,
    cache_path: Optional[str] = None,
    force_download: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Load the run length of every SGN of the given cochleae from the S3 segmentation tables.

    Args:
        cochleae: Names of the cochleae to load.
        source_name: Name of the SGN segmentation source in the MoBIE dataset.
        cache_path: Path for the pickle cache. Defaults to "./sgn_length_data_<source_name>.pkl".
        force_download: Ignore the cached pickle and fetch the tables from S3 again.

    Returns:
        Mapping of cochlea name to a table with the columns label_id, length_fraction and
        length[µm].
    """
    if cache_path is None:
        cache_path = f"./sgn_length_data_{source_name}.pkl"

    cached = {}
    if not force_download and os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if all(cochlea in cached for cochlea in cochleae):
            return {cochlea: cached[cochlea] for cochlea in cochleae if cached[cochlea] is not None}

    s3 = create_s3_target()
    for cochlea in cochleae:
        if cochlea in cached:
            continue
        print("Processing cochlea:", cochlea)
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        if source_name not in info["sources"]:
            print(f"Cochlea {cochlea} has no source {source_name}, it will be skipped.")
            cached[cochlea] = None
            continue
        source = info["sources"][source_name]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]

        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # The tonotopic mapping sets the length fraction of instances outside of the mapped
        # components to 0. Without this filter they pile up in the first bin at the apex.
        table = table[table.component_labels.isin(cochlea_components(cochlea, "SGN", source_name))]

        try:
            length_col = length_column(table)
            values = table[["label_id", "length_fraction", length_col]]
        except KeyError:
            print(f"Cochlea {cochlea} has no run length columns, it will be skipped. "
                  "Run flamingo_tools.tonotopic_mapping for it first.")
            cached[cochlea] = None
            continue

        cached[cochlea] = values.rename(columns={length_col: "length[µm]"})

    with open(cache_path, "wb") as f:
        pickle.dump(cached, f)
    return {cochlea: cached[cochlea] for cochlea in cochleae if cached[cochlea] is not None}


def build_plot_metadata(cochleae: List[str], cohort: Optional[str] = None) -> dict:
    """Build the per-cochlea plot metadata with an alias, a color and a marker.

    The color of the central registry is used when it is present. Otherwise a color is assigned by
    index. The marker distinguishes the left from the right cochlea.

    Args:
        cochleae: Names of the cochleae to plot.
        cohort: Name of the cohort the cochleae belong to. Used for the per-cohort trendlines.

    Returns:
        Mapping of cochlea name to a dict with the keys alias, color, marker, side and cohort.
    """
    # The prism palette only holds 10 colors. A larger cohort would reuse a color, which makes two
    # cochleae indistinguishable in the sliding mode, where no marker separates them.
    palette = prism_palette
    if len(cochleae) > len(prism_palette):
        cmap = plt.get_cmap("tab20")
        palette = [cmap(i / 20) for i in range(20)]

    metadata = {}
    for num, cochlea in enumerate(cochleae):
        entry = COCHLEA_DICT[cochlea]
        metadata[cochlea] = {
            "alias": entry["alias"],
            "color": entry.get("color", palette[num % len(palette)]),
            "marker": MARKER_RIGHT if cochlea.endswith("_R") else MARKER_LEFT,
            "side": "R" if cochlea.endswith("_R") else "L",
        }
        if cohort is not None:
            metadata[cochlea]["cohort"] = cohort
    return metadata


def _density_profile(values: pd.DataFrame, mode: str, n_bins: int, window: float, n_points: int):
    """Compute the density profile of a single cochlea in the given mode."""
    total_length = total_run_length(values)
    length_fraction = values["length_fraction"].to_numpy()
    if mode == "bins":
        fraction, density = density_by_fraction_bins(length_fraction, total_length, n_bins=n_bins)
    elif mode == "sliding":
        fraction, density = density_by_sliding_window(
            length_fraction, total_length, window=window, n_points=n_points
        )
    else:
        raise ValueError(f"Unrecognized mode: {mode}. Choose either 'bins' or 'sliding'.")
    return fraction, density, total_length


def fig_sgn_density_profile(
    length_data: dict,
    save_path: str,
    mode: str = "bins",
    n_bins: int = 10,
    window: float = 0.05,
    n_points: int = 200,
    cochleae_dict: dict = None,
    use_alias: bool = True,
    plot: bool = False,
    trendline: bool = False,
    trendline_std: bool = False,
    trendline_colors: dict = None,
    trendline_by_side: bool = False,
    reference_trendline=None,
    reference_style: dict = None,
    mask_outlier: bool = False,
    top_axis: bool = False,
    animal: str = "mouse",
    show_legend: bool = False,
    length_info: bool = False,
    ylabel: str = "Cells / µm",
):
    """Plot the SGN density in cells/µm over the length fraction of Rosenthal's canal.

    Args:
        length_data: Mapping of cochlea name to a table with length_fraction and length[µm].
        save_path: Save path for figure.
        mode: Density calculation. Either 'bins' for equally spaced length fraction bins, or
            'sliding' for a centered sliding window.
        n_bins: Number of bins to divide the length fraction into. Only used for mode 'bins'.
        window: Width of the sliding window as a length fraction. Only used for mode 'sliding'.
        n_points: Number of points of the sliding window grid. Only used for mode 'sliding'.
        cochleae_dict: Per-cochlea plot metadata with the keys alias, color, marker and cohort,
            as built by build_plot_metadata.
        use_alias: Use cochleae aliases.
        plot: Plot figure.
        trendline: Visualize the trendline as the average over the cochleae.
        trendline_std: Visualize the standard deviation of the trendline.
        trendline_colors: Mapping of cohort name to color. One trendline is drawn per cohort.
            A single gray trendline is drawn over all cochleae when this is None.
        trendline_by_side: Draw one trendline for the left and one for the right cochleae, instead
            of a single trendline. Use it for the ChReef cohorts, where the left cochlea is
            injected and the right one is not. Takes precedence over trendline_colors.
        reference_trendline: Trendline(s) returned by another call, drawn as a reference. Accepts a
            single dict or a list of dicts. Use it to show the wild type density inside an
            optogenetic therapy figure.
        reference_style: Overrides for REFERENCE_STYLE, which the reference is drawn with.
        mask_outlier: Draw the measurements listed in OUTLIERS in OUTLIER_COLOR and leave them out
            of the trendline. In mode 'sliding' the per-cochlea curve stays unbroken and only the
            trendline excludes the range.
        top_axis: Plot the top x-axis as the frequency range.
        animal: Species for the frequency mapping of the top axis. Either 'mouse' or 'gerbil'.
        show_legend: Show legend below the plot.
        length_info: Print the length of the cochleae and of the bin or window used for plotting.
        ylabel: Label of the y-axis.

    Returns:
        The drawn trendlines, each a dict with the keys x, y, std, color, linestyle and label.
        Empty when no trendline was drawn.
    """
    main_label_size = 24
    tick_size = 16
    if trendline_std:
        line_alphas = {"center": 0.6, "upper": 0.08, "lower": 0.08, "fill": 0.05}
    else:
        line_alphas = {"center": 1, "upper": 0., "lower": 0., "fill": 0.}
    alpha = 0.5 if trendline_colors else 1

    prism_style()

    result = {"cochlea": [], "fraction": [], "density": [], "outlier": []}
    color_dict = {}
    marker_dict = {}
    alias_to_cohort = {}
    alias_to_side = {}
    cochleae_length = []
    for name, values in length_data.items():
        meta = cochleae_dict[name]
        alias = cochlea_label(name, meta, use_alias)

        color_dict[alias] = meta["color"]
        marker_dict[alias] = meta.get("marker", "o")
        if "cohort" in meta:
            alias_to_cohort[alias] = meta["cohort"]
        alias_to_side[alias] = meta["side"]

        fraction, density, total_length = _density_profile(values, mode, n_bins, window, n_points)
        cochleae_length.append(total_length)

        outlier = np.zeros(len(fraction), dtype=bool)
        if mask_outlier:
            for low, high in OUTLIERS.get(name, []):
                outlier |= (fraction >= low) & (fraction <= high)

        result["cochlea"].extend([alias] * len(fraction))
        result["fraction"].extend(fraction)
        result["density"].extend(density)
        result["outlier"].extend(outlier)

    if length_info:
        avg_length = sum(cochleae_length) / len(cochleae_length)
        print(f"Average total length: {round(avg_length, 2)} µm")
        if mode == "bins":
            print(f"Average length per bin: {round(avg_length / n_bins, 2)} µm")
        else:
            print(f"Average length per window: {round(avg_length * window, 2)} µm")

    result = pd.DataFrame(result)

    # Lay out the legend before the figure, so that the plot keeps its height when a cohort with
    # many cochleae needs several legend rows.
    if trendline and trendline_by_side:
        n_trend = len({side for side in alias_to_side.values() if side in SIDE_TRENDLINES})
    elif trendline and trendline_colors:
        n_trend = len(trendline_colors)
    else:
        n_trend = 0
    n_reference = 1 if isinstance(reference_trendline, dict) else len(reference_trendline or [])
    n_entries = len(length_data) + n_trend + n_reference
    n_col = min((n_entries + 1) // 2, 7)
    n_row = int(np.ceil(n_entries / n_col)) if show_legend else 0
    # One row of legend entries plus the space that the x-axis label needs below the axes.
    legend_height = 0.32 * n_row + 0.7 if show_legend else 0
    fig, ax = plt.subplots(figsize=(6.7, 5 + legend_height))

    for name, grp in result.groupby("cochlea"):
        fraction = grp["fraction"].to_numpy()
        density = grp["density"].to_numpy()
        outlier = grp["outlier"].to_numpy()
        valid = ~np.isnan(density)
        if mode == "sliding":
            # Keep the curve unbroken. A gap at a masked range would misrepresent the density.
            ax.plot(fraction[valid], density[valid], label=name, color=color_dict[name], alpha=alpha)
        else:
            keep = valid & ~outlier
            ax.scatter(fraction[keep], density[keep], label=name,
                       color=color_dict[name], marker=marker_dict[name], alpha=alpha)
            masked = valid & outlier
            if masked.any():
                ax.scatter(fraction[masked], density[masked],
                           color=OUTLIER_COLOR, marker=marker_dict[name], alpha=alpha)

    # Build trend dict(s): one per cohort when trendline_colors is set, otherwise one combined.
    # Every cochlea shares the same evaluation grid, so the positions align exactly.
    def _build_trend_dict(aliases):
        td = {}
        for name, grp in result.groupby("cochlea"):
            if name not in aliases:
                continue
            kept = grp[~grp["outlier"]]
            for fraction, density in zip(kept["fraction"], kept["density"]):
                td.setdefault(fraction, []).append(density)
        return dict(sorted(td.items()))

    def _draw_trendline(ax, trend_dict, color, linestyle="dashed", alpha=None, label=None):
        x_pos = list(trend_dict.keys())
        center_line = [np.nanmean(v) for v in trend_dict.values()]
        val_std = [np.nanstd(v) for v in trend_dict.values()]
        lower_std = [m - s for m, s in zip(center_line, val_std)]
        upper_std = [m + s for m, s in zip(center_line, val_std)]

        ax.plot(x_pos, center_line, linestyle=linestyle, color=color,
                alpha=line_alphas["center"] if alpha is None else alpha, linewidth=3, zorder=2)
        ax.plot(x_pos, upper_std, linestyle="solid", color=color,
                alpha=line_alphas["upper"], zorder=0)
        ax.plot(x_pos, lower_std, linestyle="solid", color=color,
                alpha=line_alphas["lower"], zorder=0)
        ax.fill_between(x_pos, lower_std, upper_std,
                        color=color, alpha=line_alphas["fill"], interpolate=True)

        return {"x": x_pos, "y": center_line, "std": val_std,
                "color": color, "linestyle": linestyle, "label": label}

    sides_drawn = []
    trendlines = []
    if trendline:
        if trendline_by_side:
            for side, style in SIDE_TRENDLINES.items():
                side_aliases = {a for a in color_dict if alias_to_side.get(a) == side}
                if not side_aliases:
                    continue
                trendlines.append(_draw_trendline(
                    ax, _build_trend_dict(side_aliases), style["color"],
                    linestyle=style["linestyle"], alpha=style["alpha"], label=style["label"],
                ))
                sides_drawn.append(side)
        elif trendline_colors and alias_to_cohort:
            for cohort, color in trendline_colors.items():
                cohort_aliases = {a for a in color_dict if alias_to_cohort.get(a) == cohort}
                if cohort_aliases:
                    trendlines.append(_draw_trendline(
                        ax, _build_trend_dict(cohort_aliases), color, label=cohort))
        else:
            trendlines.append(_draw_trendline(ax, _build_trend_dict(set(color_dict)), "gray"))

    # Draw the wild type reference on top of the cohort data.
    references = reference_trendline or []
    if isinstance(references, dict):
        references = [references]
    reference_styles = []
    for reference in references:
        style = {**REFERENCE_STYLE, **(reference_style or {})}
        ax.plot(reference["x"], reference["y"], color=style["color"], linestyle=style["linestyle"],
                marker=style["marker"], linewidth=3, zorder=3)
        reference_styles.append(style)

    if top_axis:
        params = GREENWOOD[animal]
        freq_ticks = np.array(params["ticks"], dtype=float)
        # Inverse Greenwood mapping length_fraction = log10(f / A + k) / a.
        length_positions = np.log10(freq_ticks / params["A"] + params["k"]) / params["a"]

        ax_top = ax.twiny()
        ax_top.set_xlim(ax.get_xlim())
        ax_top.set_xticks(length_positions)
        ax_top.set_xticklabels([f"{f:g}" for f in freq_ticks], fontsize=tick_size)
        ax_top.set_xlabel("Frequency [kHz]", fontsize=main_label_size)

    ax.tick_params(axis="x", labelsize=tick_size)
    ax.tick_params(axis="y", labelsize=tick_size)
    ax.set_xlabel("Length fraction", fontsize=main_label_size)
    ax.set_ylabel(ylabel, fontsize=main_label_size)
    ax.set_ylim(bottom=0)

    plt.tight_layout()

    if show_legend:
        label = list(marker_dict.keys())
        if mode == "sliding":
            handles = [get_flatline_handle(color_dict[key]) for key in label]
        else:
            handles = [get_marker_handle(color_dict[key], marker_dict[key]) for key in label]

        if trendline and trendline_by_side:
            for side in sides_drawn:
                style = SIDE_TRENDLINES[side]
                handles.append(get_flatline_handle(style["color"], linestyle=style["linestyle"]))
                label.append(style["label"])
        elif trendline and trendline_colors:
            for cohort_name, trendline_color in trendline_colors.items():
                handles.append(get_flatline_handle(trendline_color, linestyle="dashed"))
                label.append(cohort_name)

        for style in reference_styles:
            handles.append(get_line_marker_handle(
                style["color"], linestyle=style["linestyle"], marker=style["marker"]))
            label.append(style["label"])

        fig.subplots_adjust(bottom=legend_height / (5 + legend_height))
        fig.legend(handles, label, loc="lower center", ncol=n_col, framealpha=1, frameon=False)

    if save_path.endswith(".png"):
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()

    return trendlines


def main():
    parser = argparse.ArgumentParser(
        description="Plot the SGN density in cells/µm over the length fraction of Rosenthal's canal."
    )
    parser.add_argument(
        "--figure_dir", "-f", type=str, default="./panels/sgn_density_profile",
        help="Output directory for plots.",
    )
    parser.add_argument(
        "--cohort", "-c", type=str, nargs="+", choices=list(COHORTS), default=list(COHORTS),
        help="Cohorts to plot. One figure is created per cohort, plus a combined figure.",
    )
    parser.add_argument(
        "--mode", "-m", type=str, choices=["bins", "sliding", "both"], default="both",
        help="Density calculation. 'bins' uses equally spaced bins, 'sliding' a centered window.",
    )
    parser.add_argument("--n_bins", type=int, default=10, help="Number of length fraction bins.")
    parser.add_argument("--window", type=float, default=0.05,
                        help="Width of the sliding window as a length fraction.")
    parser.add_argument("--n_points", type=int, default=200,
                        help="Number of points of the sliding window grid.")
    parser.add_argument("--no_alias", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--refresh_cache", action="store_true",
                        help="Ignore the cached pickle and fetch the tables from S3 again.")
    args = parser.parse_args()

    use_alias = not args.no_alias
    modes = ["bins", "sliding"] if args.mode == "both" else [args.mode]
    os.makedirs(args.figure_dir, exist_ok=True)

    length_data = {}
    metadata = {}
    for cohort in args.cohort:
        data = get_sgn_length_data(COHORTS[cohort], force_download=args.refresh_cache)
        if not data:
            # The SGN tables of a cohort are only usable after the tonotopic mapping was run.
            print(f"Cohort {cohort} has no cochlea with run length columns, it will be skipped.")
            continue
        length_data[cohort] = data
        metadata[cohort] = build_plot_metadata(COHORTS[cohort], cohort=COHORT_LABELS[cohort])

    cohorts = [cohort for cohort in args.cohort if cohort in length_data]
    if not cohorts:
        raise RuntimeError("None of the selected cohorts has a tonotopically mapped SGN table.")

    # wild type mouse
    cohort = "idisco"
    mode = "bins"
    n_bins = 20
    idisco_trend = fig_sgn_density_profile(
        length_data[cohort],
        save_path=os.path.join(args.figure_dir, f"sgn_density_{cohort}_{mode}.{FILE_EXTENSION}"),
        mode=mode, n_bins=n_bins, window=args.window, n_points=args.n_points,
        cochleae_dict=metadata[cohort], use_alias=use_alias, plot=args.plot,
        trendline=True, trendline_std=True, mask_outlier=True, top_axis=True,
        animal=COHORT_ANIMALS.get(cohort, "mouse"),
        show_legend=True, length_info=False,
    )

    # wild type gerbil
    cohort = "wt_gerbil"
    mode = "bins"
    n_bins = 10
    wt_gerbil_trend = fig_sgn_density_profile(
        length_data[cohort],
        save_path=os.path.join(args.figure_dir, f"sgn_density_{cohort}_{mode}.{FILE_EXTENSION}"),
        mode=mode, n_bins=n_bins, window=args.window, n_points=args.n_points,
        cochleae_dict=metadata[cohort], use_alias=use_alias, plot=args.plot,
        trendline=True, trendline_std=False, top_axis=True,
        animal=COHORT_ANIMALS.get(cohort, "mouse"),
        show_legend=True, length_info=False,
    )

    # ChReef mouse with bins
    cohort = "chreef_mouse"
    mode = "bins"
    n_bins = 10
    fig_sgn_density_profile(
        length_data[cohort],
        save_path=os.path.join(args.figure_dir, f"sgn_density_{cohort}_{mode}.{FILE_EXTENSION}"),
        mode=mode, n_bins=n_bins, window=args.window, n_points=args.n_points,
        cochleae_dict=metadata[cohort], use_alias=use_alias, plot=args.plot,
        trendline=True, trendline_std=False, trendline_by_side=True, top_axis=True,
        reference_trendline=idisco_trend,
        animal=COHORT_ANIMALS.get(cohort, "mouse"),
        show_legend=True, length_info=False,
    )

    # f-Chrimson gerbil
    cohort = "fchrimson_gerbil"
    mode = "bins"
    n_bins = 10
    fig_sgn_density_profile(
        length_data[cohort],
        save_path=os.path.join(args.figure_dir, f"sgn_density_{cohort}_{mode}.{FILE_EXTENSION}"),
        mode=mode, n_bins=n_bins, window=args.window, n_points=args.n_points,
        cochleae_dict=metadata[cohort], use_alias=use_alias, plot=args.plot,
        trendline=True, trendline_std=False, trendline_by_side=True, top_axis=True,
        reference_trendline=wt_gerbil_trend,
        animal=COHORT_ANIMALS.get(cohort, "mouse"),
        show_legend=True, length_info=False,
    )


#    for mode in modes:
#        for cohort in cohorts:
#            fig_sgn_density_profile(
#                length_data[cohort],
#                save_path=os.path.join(args.figure_dir, f"sgn_density_{cohort}_{mode}.{FILE_EXTENSION}"),
#                mode=mode, n_bins=args.n_bins, window=args.window, n_points=args.n_points,
#                cochleae_dict=metadata[cohort], use_alias=use_alias, plot=args.plot,
#                trendline=True, trendline_std=True, top_axis=True,
#                animal=COHORT_ANIMALS.get(cohort, "mouse"),
#                show_legend=True, length_info=True,
#            )

        # The combined figure uses one trendline per cohort. The top frequency axis is dropped,
        # because the Greenwood mapping differs between mouse and gerbil.
#        combined_data = {c: v for cohort in cohorts for c, v in length_data[cohort].items()}
#        combined_meta = {c: v for cohort in cohorts for c, v in metadata[cohort].items()}
#        fig_sgn_density_profile(
#            combined_data,
#            save_path=os.path.join(args.figure_dir, f"sgn_density_combined_{mode}.{FILE_EXTENSION}"),
#            mode=mode, n_bins=args.n_bins, window=args.window, n_points=args.n_points,
#            cochleae_dict=combined_meta, use_alias=use_alias, plot=args.plot,
#            trendline=True, trendline_std=True,
#            trendline_colors={COHORT_LABELS[c]: COHORT_COLORS[c] for c in cohorts},
#            show_legend=True,
#        )


if __name__ == "__main__":
    main()
