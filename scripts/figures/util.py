from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

# Directory with synapse measurement tables
SYNAPSE_DIR_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses"
# SYNAPSE_DIR_ROOT = "./synapses"
png_dpi = 300

# Documented instance counts per cochlea, structure and segmentation version. The component
# lists that produced these counts live in COCHLEA_DICT.
VALUE_DICT = {
    # iDISCO
    "M_LR_000226_L": {
        "IHC": {
            "IHC_v4c": {"count": 712},
            "IHC_v11": {"count": 687},
        },
        "SGN": {
            "SGN_v2": {"count": 11153},
        },
    },
    "M_LR_000226_R": {
        "IHC": {
            "IHC_v4c": {"count": 710},
            "IHC_v11": {"count": 648},
        },
        "SGN": {
            "SGN_v2": {"count": 11398},
        },
    },
    "M_LR_000227_L": {
        "IHC": {
            "IHC_v4c": {"count": 721},
            "IHC_v11": {"count": 617},
        },
        "SGN": {
            "SGN_v2": {"count": 10333},
        },
    },
    "M_LR_000227_R": {
        "IHC": {
            "IHC_v4c": {"count": 675},
            "IHC_v11": {"count": 640},
        },
        "SGN": {
            "SGN_v2": {"count": 11820},
        },
    },
    # PELCOfHC2longnoDCM
    "M_AMD_000126_L": {
        "IHC": {
            "IHC_v9": {"count": 665},
        },
        "SGN": {
            "SGN_v2": {"count": 11360},
        },
    },
    "M_AMD_000126_R": {
        "IHC": {
            "IHC_v9": {"count": 669},
        },
        "SGN": {
            "SGN_v2": {"count": 10751},
        },
    },
    "M_AMD_000127_L": {
        "IHC": {
            "IHC_v9": {"count": 617},
        },
        "SGN": {
            "SGN_v2": {"count": 1665},
        },
    },
    "M_AMD_000127_R": {
        "IHC": {
            "IHC_v9": {"count": 647},
        },
        "SGN": {
            "SGN_v2": {"count": 7860},
        },
    },
    "G_EK_000233_L": {
        "IHC": {
            "IHC_v11": {"count": 1018},
        },
        "SGN": {
            "SGN_v2": {"count": 18541},
        },
    },
    "G_LR_000301_R": {
        "IHC": {
            "IHC_v11": {"count": 975},
        },
        "SGN": {
            "SGN_v2": {"count": 21801},
        },
    },
    "G_LR_000302_R": {
        "IHC": {
            "IHC_v11": {"count": 935},
        },
        "SGN": {
            "SGN_v2": {"count": 23717},
        },
    },
}


def ax_prism_boxplot(ax, data, positions=None, color="tab:blue"):
    """
    Draw a Prism-style boxplot on the given Axes.
    """
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,     # to allow facecolor
        boxprops=dict(color="black", linewidth=1.2),
        whiskerprops=dict(color="black", linewidth=1.2),
        capprops=dict(color="black", linewidth=1.2),
        medianprops=dict(color="black", linewidth=1.2),
        flierprops=dict(marker="o", markersize=4, markerfacecolor="black", alpha=0.5)
    )

    # Optional: light fill color (like Prism pastels)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.2)

    return bp


prism_palette = [
    "#4E79A7",  # blue
    "#F28E2B",  # orange
    "#E15759",  # red
    "#76B7B2",  # teal
    "#59A14F",  # green
    "#EDC948",  # yellow
    "#B07AA1",  # purple
    "#FF9DA7",  # pink
    "#9C755F",  # brown
    "#BAB0AC"   # gray
]

# Cochlea side colors for the ChReef analysis: the left cochlea is injected, the right is not.
COLOR_LEFT = "#8E00DB"
COLOR_RIGHT = "#DB0063"

# Color of the untreated reference, drawn as a band or a pair of bounds.
COLOR_UNTREATED = "#DB7B00"


def custom_formatter(precision=1):
    """Get a tick formatter that prints 0 and 1 without decimals.

    The bounds are returned as literals rather than formatted with '.0f', because matplotlib
    passes a tiny negative value for the zero tick, which would render as '-0'.

    Args:
        precision: Number of decimals for every other tick.

    Returns:
        Tick formatter to pass to Axis.set_major_formatter.
    """
    def _format(x, pos):
        if np.isclose(x, 1.0):
            return "1"
        if np.isclose(x, 0.0):
            return "0"
        return f"{x:.{precision}f}"

    return mticker.FuncFormatter(_format)


def export_legend(legend, filename="legend.png", extra_artists=None):
    """Save a legend as its own image file.

    Args:
        legend: Legend to save. Its axes are turned off, so the host figure holds only the legend.
        filename: Output path.
        extra_artists: Artists drawn beside the legend, e.g. a box around a group of entries.
            They are included in the saved crop, which would otherwise cut them off.
    """
    legend.axes.axis("off")
    fig = legend.figure
    fig.canvas.draw()
    boxes = [legend.get_window_extent()]
    boxes += [artist.get_window_extent() for artist in extra_artists or []]
    bbox = Bbox.union(boxes).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, bbox_inches=bbox, dpi=png_dpi)


def get_marker_handle(color, marker, edgecolors=None):
    """Get function handle for plotting external legend without plot.
    """
    if edgecolors is None:
        return plt.plot([], [], marker=marker, color=color, ls="none")[0]
    else:
        return plt.plot([], [], marker=marker, markerfacecolor='none', markeredgecolor=edgecolors, ls="none")[0]


def cochlea_label(cochlea_name, meta, use_alias=True):
    """Get the plot label of a cochlea: its alias, or the shortened cochlea name."""
    return meta["alias"] if use_alias else cochlea_name.replace("_", "").replace("0", "")


def animal_colors(cochleae_dict, use_alias=True):
    """Map every animal to the color that its left and right cochlea share.

    The animal is the plot label without the side suffix, which is the grouping that
    plot_fig4.group_lr returns. A cochlea without a color in the registry falls back to
    prism_palette.

    Args:
        cochleae_dict: Mapping of cochlea name to its COCHLEA_DICT entry.
        use_alias: Use the alias instead of the shortened cochlea name.

    Returns:
        Mapping of animal to color.
    """
    colors = {}
    for name, meta in cochleae_dict.items():
        animal = cochlea_label(name, meta, use_alias)[:-1]
        if animal in colors:
            continue
        colors[animal] = meta.get("color", prism_palette[len(colors) % len(prism_palette)])
    return colors


def cochlea_colors(cochleae_dict, names=None, use_alias=True):
    """Map every cochlea to the color to plot it with.

    A cochlea without a color in the registry falls back to prism_palette, like animal_colors.

    Args:
        cochleae_dict: Mapping of cochlea name to its metadata.
        names: Cochleae to include, in the order they should appear. Defaults to all of them.
        use_alias: Use the alias instead of the shortened cochlea name.

    Returns:
        Mapping of plot label to color.
    """
    colors = {}
    for name in (cochleae_dict if names is None else names):
        meta = cochleae_dict[name]
        label = cochlea_label(name, meta, use_alias)
        colors[label] = meta.get("color", prism_palette[len(colors) % len(prism_palette)])
    return colors


def iteration_statistics(
    metrics: Dict[str, dict],
    keys: Sequence[str],
    metric_names: Sequence[str] = ("precision", "recall", "f1-score"),
) -> Tuple[Dict[str, Tuple[float, float]], List[str]]:
    """Average a metric over several training iterations of the same network.

    Args:
        metrics: Accuracy entries of one accuracy JSON file, keyed by network iteration.
        keys: Iteration entries to average. Keys that are absent from metrics are skipped.
        metric_names: Metrics to average. A metric is skipped for an entry that stores None.

    Returns:
        The mean and the population standard deviation per metric, and the keys found in metrics.
    """
    present = [key for key in keys if key in metrics]
    if not present:
        return {}, present

    stats = {}
    for metric in metric_names:
        values = [metrics[key][metric] for key in present if metrics[key].get(metric) is not None]
        if values:
            stats[metric] = (float(np.mean(values)), float(np.std(values)))
    return stats, present


def get_flatline_handle(color, linestyle="solid"):
    return Line2D([], [], lw=3, color=color, linestyle=linestyle)


def get_line_marker_handle(color, linestyle="solid", marker="o"):
    """Get a legend handle that shows a line and its markers."""
    return Line2D([], [], lw=3, color=color, linestyle=linestyle, marker=marker)


def get_trendline_handle(linestyle, linewidth):
    return Line2D(
        [], [], lw=3, color="gray", linestyle=linestyle,
        alpha=0.6, linewidth=linewidth,
    )


def prism_style():
    plt.style.use("default")  # reset any active styles
    plt.rcParams.update({
        # Fonts
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        # "font.sans-serif": ["Arial"],  # Prism uses Arial by default
        "font.size": 12,

        # Axes
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.labelweight": "bold",
        "axes.prop_cycle": plt.cycler("color", prism_palette),

        # Ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.major.size": 5,
        "ytick.major.size": 5,

        # Grid
        "axes.grid": False,

        # Legend
        "legend.frameon": True,
        "legend.fontsize": 10,

        # Error bars (Prism-style)
        "errorbar.capsize": 3,   # short caps

        # Markers
        "lines.markersize": 6,
        "lines.linewidth": 1.5,

        # Savefig
        "savefig.dpi": 300,
        "savefig.bbox": "tight"
    })


def prism_cleanup_axes(ax):
    """
    Apply Prism-style cleanup to one or multiple axes.
    """
    # If ax is an array (from plt.subplots), flatten it
    if isinstance(ax, (np.ndarray, list)):
        for a in np.ravel(ax):
            prism_cleanup_axes(a)  # recurse
        return

    # Otherwise ax is a single matplotlib Axes
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


# Define the animal specific octave bands.
def _get_mapping(animal):
    if animal == "mouse":
        bin_edges = [0, 2, 4, 8, 16, 32, 64, np.inf]
        bin_labels = [
            "<2", "2-4", "4-8", "8-16", "16-32", "32-64", ">64"
        ]
    elif animal == "gerbil":
        bin_edges = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
        bin_labels = [
            "<0.5", "0.5-1", "1-2", "2-4", "4-8", "8-16", "16-32", ">32"
        ]
    else:
        raise ValueError
    assert len(bin_edges) == len(bin_labels) + 1
    return bin_edges, bin_labels


def frequency_mapping(
    frequencies, values, animal="mouse", transduction_efficiency=False,
    bin_edges=None, bin_labels=None, aggregation="mean",
):
    # Get the mapping of frequencies to octave bands for the given species.
    if bin_edges is None:
        assert bin_labels is None
        bin_edges, bin_labels = _get_mapping(animal)

    # Construct the data frame with octave bands.
    df = pd.DataFrame({"freq_khz": frequencies, "value": values})
    df["octave_band"] = pd.cut(
        df["freq_khz"], bins=bin_edges, labels=bin_labels, right=False
    )

    if transduction_efficiency:  # We compute the transduction efficiency per band.
        num_pos = df[df["value"] == 1].groupby("octave_band", observed=False).size()
        num_tot = df[df["value"].isin([1, 2])].groupby("octave_band", observed=False).size()
        value_by_band = (num_pos / num_tot).reindex(bin_labels)
    else:  # Otherwise, aggregate the values over the octave band using the mean.
        aggregator = getattr(df.groupby("octave_band", observed=True)["value"], aggregation)
        value_by_band = aggregator().reindex(bin_labels)  # keep octave order even if a bin is empty
    return value_by_band


def average_by_fraction(length_fraction, syn_count, n_bins=10):
    """Average syn_per_IHC within equally spaced fractional bins."""
    # Define bins and labels
    bins = np.linspace(0, 1, n_bins + 1)
    labels = (bins[:-1] + bins[1:]) / 2  # midpoint of each bin

    # Put data into a DataFrame for convenience
    df = pd.DataFrame({
        "fraction": length_fraction,
        "syn_per_IHC": syn_count
    })

    # Bin the data
    df["bin"] = pd.cut(df["fraction"], bins=bins, labels=labels, include_lowest=True)

    # Compute mean per bin
    avg_per_bin = df.groupby("bin", observed=False)["syn_per_IHC"].mean().reset_index()
    avg_per_bin.columns = ["fraction_midpoint", "mean_syn_per_IHC"]

    return avg_per_bin


def length_column(table):
    """Get the name of the absolute run length column, which holds a non-ASCII character.

    Some tables are stored with a mangled 'length[µm]' header, so the column is matched by prefix.
    """
    for column in table.columns:
        if column.startswith("length["):
            return column
    raise KeyError(f"No 'length[µm]' column in {list(table.columns)}.")


def total_run_length(table):
    """Get the total length of the central path through Rosenthal's canal in µm.

    The tonotopic mapping writes 'length[µm]' as 'length_fraction' times the total path length,
    so the ratio of the two columns is constant. Rows with a length fraction of 0 carry no
    information, and rows outside of the mapped components are set to 0 by the tonotopic mapping.

    Args:
        table: Segmentation table with the columns 'length_fraction' and 'length[µm]'.

    Returns:
        Total length of the central path in µm.
    """
    fraction = np.asarray(table["length_fraction"], dtype=float)
    length = np.asarray(table[length_column(table)], dtype=float)
    valid = fraction > 0
    if not valid.any():
        raise ValueError("The table does not contain any instance with a length fraction above 0.")
    return float(np.median(length[valid] / fraction[valid]))


def density_by_fraction_bins(length_fraction, total_length, n_bins=10):
    """Compute the linear density in cells/µm for equally spaced length fraction bins.

    Args:
        length_fraction: Length fraction of every instance along Rosenthal's canal.
        total_length: Total length of the central path in µm.
        n_bins: Number of bins to divide the length fraction into.

    Returns:
        Midpoint of every bin.
        Density in cells/µm per bin.
    """
    edges = np.linspace(0, 1, n_bins + 1)
    midpoints = (edges[:-1] + edges[1:]) / 2
    counts, _ = np.histogram(np.asarray(length_fraction, dtype=float), bins=edges)
    return midpoints, counts / (total_length / n_bins)


def density_by_sliding_window(length_fraction, total_length, window=0.05, n_points=200):
    """Compute the linear density in cells/µm with a centered sliding window.

    The window is given as a length fraction. The evaluation grid is limited to
    [window / 2, 1 - window / 2], so that every point uses a full window and no roll-off
    occurs at the ends.

    Args:
        length_fraction: Length fraction of every instance along Rosenthal's canal.
        total_length: Total length of the central path in µm.
        window: Width of the sliding window as a length fraction.
        n_points: Number of points to evaluate the window at.

    Returns:
        Positions of the evaluation grid as a length fraction.
        Density in cells/µm at every position.
    """
    if not 0 < window <= 1:
        raise ValueError(f"The window must be a length fraction in (0, 1], got {window}.")
    values = np.sort(np.asarray(length_fraction, dtype=float))
    positions = np.linspace(window / 2, 1 - window / 2, n_points)
    lower = np.searchsorted(values, positions - window / 2, side="left")
    upper = np.searchsorted(values, positions + window / 2, side="right")
    return positions, (upper - lower) / (window * total_length)


# Reference intervals from the literature, per species and structure. SGN and IHC are instance
# counts per cochlea, synapse is the number of synapses per IHC.
_LITERATURE_REFERENCE_VALUES = {
    "mouse": {"SGN": (9141, 11736), "IHC": (656, 681), "synapse": (9.1, 20.7)},
    "gerbil": {"SGN": (22933, 26267), "IHC": (1081, 1081), "synapse": (15.8, 25.6)},
}


def literature_reference_values(structure, animal="mouse"):
    """Get the reference interval of one structure from the literature.

    Args:
        structure: "SGN", "IHC" or "synapse".
        animal: "mouse" or "gerbil".

    Returns:
        Lower bound of the interval.
        Upper bound of the interval.
    """
    if animal not in _LITERATURE_REFERENCE_VALUES:
        raise ValueError(f"animal must be one of {list(_LITERATURE_REFERENCE_VALUES)}, got '{animal}'.")
    values = _LITERATURE_REFERENCE_VALUES[animal]
    if structure not in values:
        raise ValueError(f"structure must be one of {list(values)}, got '{structure}'.")
    return values[structure]


# Central registry of the cochlea cohorts. "cochleae" is the member list, "label" the text shown
# on a plot, and "animal" selects the Greenwood parameters and the octave bands. The keys are the
# --cohort values of plot_sgn_density_profile.py. Read the member list through cohort_cochleae();
# "label" and "animal" are single level, so indexing them directly is fine. "color" is optional,
# because not every cohort is drawn as one group.
COHORT_DICT = {
    "idisco": {
        "label": "iDISCO", "animal": "mouse", "color": "#10CC17",
        "cochleae": ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"],
    },
    "mwfls": {
        "label": "MWfLS", "animal": "mouse", "color": "#3F69FF",
        "cochleae": ["M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R"],
    },
    # M_LR_000143 is the pilot animal. It has no color in COCHLEA_DICT, so a figure that colors
    # per cochlea rather than per animal leaves it out.
    "chreef_mouse": {
        "label": "ChReef mouse", "animal": "mouse", "color": "#DB0063",
        "cochleae": [
            "M_LR_000143_L", "M_LR_000144_L", "M_LR_000145_L",
            "M_LR_000153_L", "M_LR_000155_L", "M_LR_000189_L",
            "M_LR_000143_R", "M_LR_000144_R", "M_LR_000145_R",
            "M_LR_000153_R", "M_LR_000155_R", "M_LR_000189_R",
        ],
    },
    # These have no SGN segmentation, so they carry no density profile.
    "otof_mouse": {
        "label": "OTOF mouse", "animal": "mouse",
        "cochleae": ["M_AMD_OTOF27_L", "M_AMD_OTOF27_R", "M_AMD_OTOF28_L", "M_AMD_OTOF28_R"],
    },
    "fchrimson_gerbil": {
        "label": "f-Chrimson gerbil", "animal": "gerbil", "color": "#8E00DB",
        "cochleae": [
            "G_EK_000049_L", "G_EK_000049_R", "G_EK_000071_L", "G_EK_000071_R",
            "G_EK_000074_L", "G_EK_000074_R", "G_EK_000076_L", "G_EK_000076_R",
        ],
        # G_EK_000049 received the injection postnatally, the other three animals as adults.
        # The two groups are not comparable, so a figure can plot and average them apart.
        "postnatal": ["G_EK_000049_L", "G_EK_000049_R"],
    },
    "wt_gerbil": {
        "label": "WT gerbil", "animal": "gerbil", "color": COLOR_UNTREATED,
        "cochleae": ["G_EK_000233_L", "G_LR_000301_R", "G_LR_000302_R"],
    },
}


def cohort_cochleae(cohort):
    """Get the cochleae of one cohort.

    Args:
        cohort: Key in COHORT_DICT.

    Returns:
        List of cochlea names.
    """
    if cohort not in COHORT_DICT:
        raise KeyError(f"Unknown cohort {cohort}, expected one of {list(COHORT_DICT)}.")
    return COHORT_DICT[cohort]["cochleae"]


def cohort_postnatal(cohort):
    """Get the cochleae of one cohort that were injected postnatally.

    Args:
        cohort: Key in COHORT_DICT.

    Returns:
        List of cochlea names, empty if the cohort holds no postnatal injection.
    """
    if cohort not in COHORT_DICT:
        raise KeyError(f"Unknown cohort {cohort}, expected one of {list(COHORT_DICT)}.")
    return COHORT_DICT[cohort].get("postnatal", [])


# Central registry of the cochleae used by the figure scripts. A component list belongs to one
# segmentation of one cochlea, so it is nested per structure ("SGN" / "IHC") and per segmentation
# version, mirroring VALUE_DICT. "alias" and "color" are structure independent. Read the registry
# through cochlea_components() or cochleae_for(), never by indexing the nested dicts directly.
# Only the (structure, version) pairs that a figure script needs are listed, so an unintended
# lookup raises instead of filtering on a guessed component list.
# TODO: plot_fig6.py still defines its own COCHLEAE_DICT_LaVision. Its LaVision cochleae are not
# in this registry, and plot_lavision.py holds a third, conflicting component list for them.
COCHLEA_DICT = {
    # iDISCO reference cochleae. The IHC component lists were validated on IHC_v11. IHC_v4c
    # reuses them, which is what plot_mwfls.py already did through the old plot_fig3 registry.
    "M_LR_000226_L": {
        "alias": "M_01L", "color": "#9C5027",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v11": {"component": [1, 3]}, "IHC_v4c": {"component": [1, 3]}},
    },
    "M_LR_000226_R": {
        "alias": "M_01R", "color": "#279C52",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v11": {"component": [1]}, "IHC_v4c": {"component": [1]}},
    },
    "M_LR_000227_L": {
        "alias": "M_02L", "color": "#67279C",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v11": {"component": [1]}, "IHC_v4c": {"component": [1]}},
    },
    "M_LR_000227_R": {
        "alias": "M_02R", "color": "#27339C",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v11": {"component": [1]}, "IHC_v4c": {"component": [1]}},
    },
    # MWfLS cochleae.
    "M_AMD_000126_L": {
        "alias": "M_03L", "color": "#5B1CE8",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v9": {"component": [1]}},
    },
    "M_AMD_000126_R": {
        "alias": "M_03R", "color": "#1C1FE8",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v9": {"component": [1]}},
    },
    "M_AMD_000127_L": {
        "alias": "M_04L", "color": "#1C60E9",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v9": {"component": [1]}},
    },
    "M_AMD_000127_R": {
        "alias": "M_04R", "color": "#1CA0E8",
        "SGN": {"SGN_v2": {"component": [1]}},
        "IHC": {"IHC_v9": {"component": [1]}},
    },
    # Mouse cochleae for the OTOF gene therapy. These have no SGN segmentation. The components
    # match the component_list used to generate each
    # reproducibility/object_measures/MAMDOTOF*_IHC.json.
    "M_AMD_OTOF27_L": {
        "alias": "M_30L", "color": "#9C5027",
        "IHC": {"IHC_v11": {"component": [1]}},
    },
    "M_AMD_OTOF27_R": {
        "alias": "M_30R", "color": "#9C5027",
        "IHC": {"IHC_v11": {"component": [2, 4, 10]}},
    },
    "M_AMD_OTOF28_L": {
        "alias": "M_31L", "color": "#279C52",
        "IHC": {"IHC_v11": {"component": [5, 9, 1, 3, 4, 14, 8, 15]}},
    },
    "M_AMD_OTOF28_R": {
        "alias": "M_31R", "color": "#279C52",
        "IHC": {"IHC_v11": {"component": [2, 1, 3, 4]}},
    },
    # Mouse cochleae for the ChReef analysis.
    "M_LR_000143_L": {"alias": "M0L", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000144_L": {"alias": "M_05L", "color": "#9C5027", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000145_L": {"alias": "M_06L", "color": "#279C52", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000153_L": {"alias": "M_07L", "color": "#67279C", "SGN": {"SGN_v2": {"component": [1, 2, 3]}}},
    "M_LR_000155_L": {"alias": "M_08L", "color": "#27339C", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000189_L": {"alias": "M_09L", "color": "#9C276F", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000143_R": {"alias": "M0R", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000144_R": {"alias": "M_05R", "color": "#9C5027", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000145_R": {"alias": "M_06R", "color": "#279C52", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000153_R": {"alias": "M_07R", "color": "#67279C", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000155_R": {"alias": "M_08R", "color": "#27339C", "SGN": {"SGN_v2": {"component": [1]}}},
    "M_LR_000189_R": {"alias": "M_09R", "color": "#9C276F", "SGN": {"SGN_v2": {"component": [1]}}},
    # Gerbil cochleae for the f-Chrimson analysis. The components match the component_list used to
    # generate each SGN_density_2d.json.
    "G_EK_000049_L": {"alias": "G_1L", "color": "#9C5027", "SGN": {"SGN_v2": {"component": [1, 3, 4, 5]}}},
    "G_EK_000071_L": {"alias": "G_2L", "color": "#279C52", "SGN": {"SGN_v2": {"component": [1]}}},
    "G_EK_000074_L": {"alias": "G_3L", "color": "#67279C", "SGN": {"SGN_v2": {"component": [1]}}},
    "G_EK_000076_L": {"alias": "G_4L", "color": "#27339C", "SGN": {"SGN_v2": {"component": [1, 2, 3]}}},
    "G_EK_000049_R": {"alias": "G_1R", "color": "#9C5027", "SGN": {"SGN_v2": {"component": [1, 2]}}},
    "G_EK_000071_R": {"alias": "G_2R", "color": "#279C52", "SGN": {"SGN_v2": {"component": [1]}}},
    "G_EK_000074_R": {"alias": "G_3R", "color": "#67279C", "SGN": {"SGN_v2": {"component": [1]}}},
    "G_EK_000076_R": {"alias": "G_4R", "color": "#27339C", "SGN": {"SGN_v2": {"component": [1]}}},
    # Untreated gerbil cochleae. G_LR_000302_R keeps component 3, which holds 234 of its 23717
    # SGNs; the SGN_density_2d_extended.json on S3 was recalculated with [1, 3].
    "G_EK_000233_L": {"alias": "G_5L", "color": "#279C52", "SGN": {"SGN_v2": {"component": [1]}}},
    "G_LR_000301_R": {"alias": "G_6R", "color": "#67279C", "SGN": {"SGN_v2": {"component": [1]}}},
    "G_LR_000302_R": {"alias": "G_7R", "color": "#27339C", "SGN": {"SGN_v2": {"component": [1, 3]}}},
}


def cochlea_components(cochlea_name, structure, version):
    """Get the component labels to keep for one segmentation of one cochlea.

    A component list is only valid for the labeling run it was derived from, so the version is
    required and never defaulted.

    Args:
        cochlea_name: Key in COCHLEA_DICT.
        structure: "SGN" or "IHC".
        version: Segmentation version, e.g. "SGN_v2" or "IHC_v11".

    Returns:
        List of component labels.
    """
    entry = COCHLEA_DICT[cochlea_name]
    if structure not in entry:
        raise KeyError(f"Cochlea {cochlea_name} has no {structure} segmentation in COCHLEA_DICT.")
    if version not in entry[structure]:
        raise KeyError(
            f"Cochlea {cochlea_name} has no {structure} component list for {version}, "
            f"only for {sorted(entry[structure])}."
        )
    return entry[structure][version]["component"]


def cochleae_for(cochlea_names, structure, version):
    """Get a flat per-cochlea view of COCHLEA_DICT for one segmentation.

    The view holds the keys the figure scripts read from a metadata mapping, so a consumer keeps
    using meta["component"], meta["alias"] and meta["color"] without knowing about the nesting.
    A consumer may add its own keys to an entry, so every entry is a fresh dict.

    Args:
        cochlea_names: Iterable of keys in COCHLEA_DICT.
        structure: "SGN" or "IHC".
        version: Segmentation version, e.g. "SGN_v2" or "IHC_v11".

    Returns:
        Mapping of cochlea name to a flat metadata dict.
    """
    view = {}
    for name in cochlea_names:
        entry = COCHLEA_DICT[name]
        meta = {"alias": entry["alias"], "component": cochlea_components(name, structure, version)}
        if "color" in entry:
            meta["color"] = entry["color"]
        view[name] = meta
    return view


# The only consumer, plot_mwfls.py, uses these for the IHC_v9 tonotopic data.
MWFLS_COCHLEAE_DICT = cochleae_for(cohort_cochleae("mwfls"), "IHC", "IHC_v9")

OUTLIER_DICT = {"SGN": ["M_AMD_000127_L"]}
