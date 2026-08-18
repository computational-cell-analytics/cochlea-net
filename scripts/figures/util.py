import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

# Directory with synapse measurement tables
SYNAPSE_DIR_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses"
# SYNAPSE_DIR_ROOT = "./synapses"
png_dpi = 300

VALUE_DICT = {
    # iDISCO
    "M_LR_000226_L": {
        "IHC": {
            "IHC_v4c": {
                "count": 712, "version": "IHC_v4c",
            },
            "IHC_v11": {
                "count": 687, "component_list": [1, 3], "version": "IHC_v11",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 11153, "version": "SGN_v2",
            },
        },
    },
    "M_LR_000226_R": {
        "IHC": {
            "IHC_v4c": {
                "count": 710, "version": "IHC_v4c",
            },
            "IHC_v11": {
                "count": 648, "version": "IHC_v11",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 11398, "version": "SGN_v2",
            },
        },
    },
    "M_LR_000227_L": {
        "IHC": {
            "IHC_v4c": {
                "count": 721, "version": "IHC_v4c",
            },
            "IHC_v11": {
                "count": 617, "version": "IHC_v11",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 10333, "version": "SGN_v2",
            },
        },
    },
    "M_LR_000227_R": {
        "IHC": {
            "IHC_v4c": {
                "count": 675, "version": "IHC_v4c",
            },
            "IHC_v11": {
                "count": 640, "version": "IHC_v11",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 11820, "version": "SGN_v2",
            },
        },
    },
    # PELCOfHC2longnoDCM
    "M_AMD_000126_L": {
        "IHC": {
            "IHC_v9": {
                "count": 665, "version": "IHC_v9",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 11360, "version": "SGN_v2",
            },
        },
    },
    "M_AMD_000126_R": {
        "IHC": {
            "IHC_v9": {
                "count": 669, "version": "IHC_v9",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 10751, "version": "SGN_v2",
            },
        },
    },
    "M_AMD_000127_L": {
        "IHC": {
            "IHC_v9": {
                "count": 617, "version": "IHC_v9",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 1665, "version": "SGN_v2",
            },
        },
    },
    "M_AMD_000127_R": {
        "IHC": {
            "IHC_v9": {
                "count": 647, "version": "IHC_v9",
            },
        },
        "SGN": {
            "SGN_v2": {
                "count": 7860, "version": "SGN_v2",
            },
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


def custom_formatter_1(x, pos):
    if np.isclose(x, 1.0):
        return '1'  # no decimal
    elif np.isclose(x, 0.0):
        return '0'  # no decimal
    else:
        return f"{x:.1f}"


def custom_formatter_2(x, pos):
    if np.isclose(x, 1.0):
        return '1'  # no decimal
    elif np.isclose(x, 0.0):
        return '0'  # no decimal
    else:
        return f"{x:.2f}"


def export_legend(legend, filename="legend.png"):
    legend.axes.axis("off")
    fig = legend.figure
    fig.canvas.draw()
    bbox = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(filename, bbox_inches=bbox, dpi=png_dpi)


def get_marker_handle(color, marker, edgecolors=None):
    """Get function handle for plotting external legend without plot.
    """
    if edgecolors is None:
        return plt.plot([], [], marker=marker, color=color, ls="none")[0]
    else:
        return plt.plot([], [], marker=marker, markerfacecolor='none', markeredgecolor=edgecolors, ls="none")[0]


def get_flatline_handle(color, linestyle="solid"):
    return Line2D([], [], lw=3, color=color, linestyle=linestyle)


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


def sliding_runlength_sum(run_length, values, width):
    assert len(run_length) == len(values)
    # Create data frame and sort it.
    df = pd.DataFrame({"run_length": run_length, "value": values})
    df = df.sort_values("run_length").reset_index(drop=True).copy()

    x = df["run_length"].to_numpy()
    y = df["value"].to_numpy()

    cumsum = np.cumsum(y)
    start_idx = np.searchsorted(x, x - width, side="left")
    window_sum = cumsum - np.concatenate(([0], cumsum[:-1]))[start_idx]
    assert len(window_sum) == len(x)

    return x, window_sum


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


# For mouse
def literature_reference_values(structure):
    if structure == "SGN":
        lower_bound, upper_bound = 9141, 11736
    elif structure == "IHC":
        lower_bound, upper_bound = 656, 681
    elif structure == "synapse":
        lower_bound, upper_bound = 9.1, 20.7
    else:
        raise ValueError
    return lower_bound, upper_bound


# For gerbil
def literature_reference_values_gerbil(structure):
    if structure == "SGN":
        lower_bound, upper_bound = 22933, 26267
    elif structure == "IHC":
        lower_bound, upper_bound = 1081, 1081
    elif structure == "synapse":
        lower_bound, upper_bound = 15.8, 25.6
    else:
        raise ValueError
    return lower_bound, upper_bound


COHORT_DICT = {
    "iDISCO": ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"],
    "MWfLS": ["M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R"],
}

# Central registry of the cochleae used by the figure scripts. "component" holds the Rosenthal's
# canal component label(s) to keep when filtering a segmentation table. "color" is optional,
# because some scripts color by animal instead of by cochlea.
# TODO: plot_fig3.py and plot_fig6.py still define their own COCHLEAE_DICT. Deriving them from
# this registry also touches plot_mwfls.py, which imports COCHLEAE_DICT from plot_fig3.
COCHLEA_DICT = {
    # iDISCO reference cochleae.
    "M_LR_000226_L": {"alias": "M_01L", "component": [1], "color": "#9C5027"},
    "M_LR_000226_R": {"alias": "M_01R", "component": [1], "color": "#279C52"},
    "M_LR_000227_L": {"alias": "M_02L", "component": [1], "color": "#67279C"},
    "M_LR_000227_R": {"alias": "M_02R", "component": [1], "color": "#27339C"},
    # MWfLS cochleae.
    "M_AMD_000126_L": {"alias": "M_03L", "component": [1], "color": "#5B1CE8"},
    "M_AMD_000126_R": {"alias": "M_03R", "component": [1], "color": "#1C1FE8"},
    "M_AMD_000127_L": {"alias": "M_04L", "component": [1], "color": "#1C60E9"},
    "M_AMD_000127_R": {"alias": "M_04R", "component": [1], "color": "#1CA0E8"},
    # Mouse cochleae for the ChReef analysis.
    "M_LR_000143_L": {"alias": "M0L", "component": [1]},
    "M_LR_000144_L": {"alias": "M_05L", "component": [1], "color": "#9C5027"},
    "M_LR_000145_L": {"alias": "M_06L", "component": [1], "color": "#279C52"},
    "M_LR_000153_L": {"alias": "M_07L", "component": [1, 2, 3], "color": "#67279C"},
    "M_LR_000155_L": {"alias": "M_08L", "component": [1], "color": "#27339C"},
    "M_LR_000189_L": {"alias": "M_09L", "component": [1], "color": "#9C276F"},
    "M_LR_000143_R": {"alias": "M0R", "component": [1]},
    "M_LR_000144_R": {"alias": "M_05R", "component": [1], "color": "#9C5027"},
    "M_LR_000145_R": {"alias": "M_06R", "component": [1], "color": "#279C52"},
    "M_LR_000153_R": {"alias": "M_07R", "component": [1], "color": "#67279C"},
    "M_LR_000155_R": {"alias": "M_08R", "component": [1], "color": "#27339C"},
    "M_LR_000189_R": {"alias": "M_09R", "component": [1], "color": "#9C276F"},
    # Gerbil cochleae for the ChReef analysis. The components match the component_list used to
    # generate each SGN_density_2d.json.
    "G_EK_000049_L": {"alias": "G_1L", "component": [1, 3, 4, 5], "color": "#9C5027"},
    "G_EK_000071_L": {"alias": "G_2L", "component": [1], "color": "#279C52"},
    "G_EK_000074_L": {"alias": "G_3L", "component": [1], "color": "#67279C"},
    "G_EK_000076_L": {"alias": "G_4L", "component": [1, 2, 3], "color": "#27339C"},
    "G_EK_000049_R": {"alias": "G_1R", "component": [1, 2], "color": "#9C5027"},
    "G_EK_000071_R": {"alias": "G_2R", "component": [1], "color": "#279C52"},
    "G_EK_000074_R": {"alias": "G_3R", "component": [1], "color": "#67279C"},
    "G_EK_000076_R": {"alias": "G_4R", "component": [1], "color": "#27339C"},
}

MWFLS_COCHLEAE = ["M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R"]

MWFLS_COCHLEAE_DICT = {name: COCHLEA_DICT[name] for name in MWFLS_COCHLEAE}

OUTLIER_DICT = {"SGN": ["M_AMD_000127_L"]}


def to_alias(cochlea_name):
    name_short = cochlea_name.replace("_", "").replace("0", "")
    name_to_alias = {
        "MLR226L": "M_01L",
        "MLR226R": "M_01R",
        "MLR227L": "M_02L",
        "MLR227R": "M_02R",
        "MAMD126L": "M_03L",
        "MAMD126R": "M_03R",
        "MAMD127L": "M_04L",
        "MAMD127R": "M_04R",
    }
    return name_to_alias[name_short]
