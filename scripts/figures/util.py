import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D

# Directory with synapse measurement tables
SYNAPSE_DIR_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses"
# SYNAPSE_DIR_ROOT = "./synapses"
png_dpi = 300


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


def get_flatline_handle(color):
    return Line2D([], [], lw=3, color=color)


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
            "<2", "2–4", "4–8", "8–16", "16–32", "32–64", ">64"
        ]
    elif animal == "gerbil":
        bin_edges = [0, 0.5, 1, 2, 4, 8, 16, 32, np.inf]
        bin_labels = [
            "<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", "16–32", ">32"
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
        lower_bound, upper_bound = 24700, 28450
    elif structure == "IHC":
        lower_bound, upper_bound = 1081, 1081
    elif structure == "synapse":
        lower_bound, upper_bound = 12.5, 25
    else:
        raise ValueError
    return lower_bound, upper_bound


def to_alias(cochlea_name):
    name_short = cochlea_name.replace("_", "").replace("0", "")
    name_to_alias = {
        "MLR226L": "M_01L",
        "MLR226R": "M_01R",
        "MLR227L": "M_02L",
        "MLR227R": "M_02R",
    }
    return name_to_alias[name_short]
