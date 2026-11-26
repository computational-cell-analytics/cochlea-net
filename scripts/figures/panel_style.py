"""Panel style for new figures.

The style combines two references:
- The look of GraphPad Prism graphs: left and bottom axes only, outward ticks, bold axis titles,
  scatter dot plots with mean and SD, open box plots.
- The panel rules of the Nature Reviews artwork guide: one file per panel, a vector PDF with
  editable text, the final print size (at most 180 x 215 mm), 8 pt text, units in parentheses
  in the axis titles, and few colors that have the same meaning in all panels.

Use it as follows:
- Run the panel code inside `style()`, as a decorator or as a `with` block. The rcParams then
  apply only to that code and do not change the global state.
- Create the figure at its print size with `new_panel` and save it with `save_panel`.
- Place the saved PDF at 100 % in the figure layout. At any other scale the text is not 8 pt.
- Write the sample size upright ("n = 6"), not as math text.

The published CochleaNet panels use `util.prism_style`. Do not mix the two styles.

Run this module as a script to render the palette and one example panel of each type.
"""
import argparse
import colorsys
import os

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib import colors as mcolors
from matplotlib import rc_context
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

FONT_SIZE = 8  # pt
MAX_WIDTH_MM = 180
MAX_HEIGHT_MM = 215
PREVIEW_DPI = 300
MM_PER_INCH = 25.4

# Line widths in pt. The axis lines and ticks are thinner than the data lines.
AXIS_LINE_WIDTH = 0.75
DATA_LINE_WIDTH = 1.0
MEAN_LINE_WIDTH = 1.25
MARKER_SIZE = 4.0  # pt, the diameter of a symbol

# The qualitative ColorBrewer scheme "Paired" plus neutral grays. ColorBrewer made its schemes
# for print and defines them in CMYK, so these colors keep their look in a CMYK conversion.
COLORS = {
    "blue": "#1F78B4",
    "light_blue": "#A6CEE3",
    "green": "#33A02C",
    "light_green": "#B2DF8A",
    "orange": "#FF7F00",
    "light_orange": "#FDBF6F",
    "purple": "#6A3D9A",
    "light_purple": "#CAB2D6",
    "red": "#E31A1C",
    "light_red": "#FB9A99",
    "black": "#000000",
    "dark_gray": "#525252",
    "gray": "#969696",
    "light_gray": "#D9D9D9",
}
# A literature or other external reference is neutral.
REFERENCE_COLOR = COLORS["gray"]
# A sample that is shown but left out of the statistics.
OUTLIER = {"marker": "x", "color": COLORS["red"]}

# The minimum contrast of text against white, see WCAG 2.1 (normal text).
MIN_TEXT_CONTRAST = 4.5

RC_PARAMS = {
    "font.family": "sans-serif",
    # Liberation Sans has the metrics of Arial, Nimbus Sans those of Helvetica.
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE,
    "axes.titleweight": "bold",
    "axes.labelsize": FONT_SIZE,
    "axes.labelweight": "bold",
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "legend.title_fontsize": FONT_SIZE,
    "figure.titlesize": FONT_SIZE,
    "figure.titleweight": "bold",
    # Math text, e.g. a Greek letter or a superscript, uses the text font.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.sf": "sans",
    "mathtext.cal": "sans:italic",
    "axes.linewidth": AXIS_LINE_WIDTH,
    "axes.edgecolor": COLORS["black"],
    "axes.labelcolor": COLORS["black"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "axes.titlepad": 4.0,
    "axes.labelpad": 3.0,
    "axes.prop_cycle": cycler(color=[
        COLORS["blue"], COLORS["green"], COLORS["orange"], COLORS["purple"], COLORS["red"], COLORS["gray"],
    ]),
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.width": AXIS_LINE_WIDTH,
    "ytick.major.width": AXIS_LINE_WIDTH,
    "xtick.minor.width": AXIS_LINE_WIDTH,
    "ytick.minor.width": AXIS_LINE_WIDTH,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "xtick.color": COLORS["black"],
    "ytick.color": COLORS["black"],
    "lines.linewidth": DATA_LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "lines.markeredgewidth": 0.5,
    "patch.linewidth": AXIS_LINE_WIDTH,
    "errorbar.capsize": 2.0,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.4,
    "legend.labelspacing": 0.3,
    "legend.columnspacing": 1.0,
    "legend.borderaxespad": 0.3,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": PREVIEW_DPI,
    # The saved file must have the figure size, so never crop it.
    "savefig.bbox": "standard",
    # Embed TrueType fonts, so that the text stays editable in Illustrator or Inkscape.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def style():
    """Return a context for the panel rcParams. It also works as a function decorator."""
    return rc_context(RC_PARAMS)


def new_panel(width_mm: float, height_mm: float, nrows: int = 1, ncols: int = 1, **kwargs):
    """Create a figure at its print size, with constrained layout.

    Args:
        width_mm: Print width of the panel in mm.
        height_mm: Print height of the panel in mm.
        nrows: Number of subplot rows.
        ncols: Number of subplot columns.
        kwargs: Further arguments for plt.subplots.

    Returns:
        The figure and the axes of plt.subplots.
    """
    if width_mm > MAX_WIDTH_MM or height_mm > MAX_HEIGHT_MM:
        raise ValueError(
            f"A panel of {width_mm} x {height_mm} mm is larger than the print area of "
            f"{MAX_WIDTH_MM} x {MAX_HEIGHT_MM} mm."
        )
    figsize = (width_mm / MM_PER_INCH, height_mm / MM_PER_INCH)
    return plt.subplots(nrows, ncols, figsize=figsize, layout="constrained", **kwargs)


def save_panel(fig, save_path: str, plot: bool = False):
    """Save a panel as a PDF and as a PNG preview, both at the figure size.

    The PDF has no creation date, so that a rerun with the same data gives the same file.

    Args:
        fig: The figure.
        save_path: Output path without an extension. A ".pdf" or ".png" extension is removed.
        plot: Show the figure after saving instead of closing it.
    """
    stem, extension = os.path.splitext(save_path)
    if extension.lower() not in (".pdf", ".png"):
        stem = save_path
    with style():
        fig.savefig(f"{stem}.pdf", metadata={"CreationDate": None})
        fig.savefig(f"{stem}.png", dpi=PREVIEW_DPI)
    if plot:
        plt.show()
    else:
        plt.close(fig)


def tint(color, amount: float) -> tuple:
    """Blend a color with white. An amount of 0 keeps the color, 1 gives white."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1 - rgb) * amount)


def shade(color, amount: float) -> tuple:
    """Blend a color with black. An amount of 0 keeps the color, 1 gives black."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * (1 - amount))


def group_shades(color, n: int, hue_range: float = 0.0, lightness_range: float = 0.4) -> list:
    """Return n distinguishable shades around the hue of a color, from dark to light.

    Use them for the samples of one group, e.g. the cochleae of one cohort, so that the group
    color stays visible.

    Args:
        color: The base color.
        n: Number of shades.
        hue_range: Hue change from the darkest to the lightest shade, as a fraction of the color
            circle. The darkest shade has the hue of the color. A negative value turns the hue the
            other way. Turn it away from the hues of the other groups.
        lightness_range: Lightness change from the darkest to the lightest shade.

    Returns:
        n RGB tuples. For an odd n, the middle shade has the lightness of the color.
    """
    if n == 1:
        return [mcolors.to_rgb(color)]
    hue, lightness, saturation = colorsys.rgb_to_hls(*mcolors.to_rgb(color))
    result = []
    for t in np.linspace(-0.5, 0.5, n):
        shade_lightness = float(np.clip(lightness + t * lightness_range, 0.15, 0.85))
        result.append(colorsys.hls_to_rgb((hue + (t + 0.5) * hue_range) % 1.0, shade_lightness, saturation))
    return result


def _relative_luminance(color) -> float:
    """Return the relative luminance of WCAG 2.1."""
    rgb = np.array(mcolors.to_rgb(color))
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return float(linear @ [0.2126, 0.7152, 0.0722])


def contrast_ratio(color_a, color_b) -> float:
    """Return the WCAG 2.1 contrast ratio of two colors, from 1 to 21."""
    lighter, darker = sorted([_relative_luminance(color_a), _relative_luminance(color_b)], reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def text_color_on(fill) -> str:
    """Return black or white, whichever has the higher contrast on the fill color."""
    black, white = COLORS["black"], "#FFFFFF"
    return black if contrast_ratio(fill, black) >= contrast_ratio(fill, white) else white


def readable(color) -> tuple:
    """Darken a color until text in it has a contrast of at least MIN_TEXT_CONTRAST on white."""
    for amount in np.linspace(0, 1, 41):
        darker = shade(color, amount)
        if contrast_ratio(darker, "#FFFFFF") >= MIN_TEXT_CONTRAST:
            return darker
    return mcolors.to_rgb(COLORS["black"])


def swarm_offsets(values, y_tolerance: float, step: float) -> np.ndarray:
    """Return x offsets that place close values side by side, as in a Prism scatter dot plot.

    The sorted values form groups, in which each value is closer than y_tolerance to the
    previous one. A value without close neighbours stays at offset 0. The k values of a group
    get the offsets (i - (k - 1) / 2) * step, so that the group is centered on 0.

    Args:
        values: The y values.
        y_tolerance: The y distance below which two symbols overlap, in data units.
        step: The x distance between neighbouring symbols, in data units.

    Returns:
        One offset per value, in the order of the values.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="stable")
    offsets = np.zeros(len(values))
    group = []
    for index in order:
        if group and values[index] - values[group[-1]] >= y_tolerance:
            offsets[group] = (np.arange(len(group)) - (len(group) - 1) / 2) * step
            group = []
        group.append(index)
    if group:
        offsets[group] = (np.arange(len(group)) - (len(group) - 1) / 2) * step
    return offsets


def nice_upper(value: float, nbins: int = 5) -> float:
    """Return the first tick at or above value of a MaxNLocator scale that starts at 0.

    Use it as the upper axis limit, so that the axis ends on a tick, as in Prism.
    """
    ticks = MaxNLocator(nbins=nbins).tick_values(0, value)
    return float(ticks[ticks >= value][0])


def _axis_size_pt(ax) -> tuple:
    """Return the width and height of an axis in pt, after the layout of its figure ran.

    The layout runs with the artists that exist now. Later tick labels or axis titles can still
    make the axis a little smaller.
    """
    fig = ax.figure
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.execute(fig)
    position = ax.get_position()
    return position.width * fig.get_figwidth() * 72, position.height * fig.get_figheight() * 72


def _symbol_extent(ax) -> tuple:
    """Return the size of one symbol in x and y data units."""
    width_pt, height_pt = _axis_size_pt(ax)
    x_span = np.diff(ax.get_xlim())[0]
    y_span = np.diff(ax.get_ylim())[0]
    return MARKER_SIZE / width_pt * x_span, MARKER_SIZE / height_pt * y_span


def dot_plot(
    ax, values, position: float, color, marker: str = "o", outliers=(), mean_half_width: float = 0.25,
    show_sd: bool = True,
):
    """Draw a Prism scatter dot plot: one symbol per value, the mean line and sample SD whiskers.

    Set the x and y limits of the axis first. The spread of the symbols depends on them.

    Args:
        ax: The axis.
        values: The values of the samples.
        position: The x position of the group.
        color: The symbol color.
        marker: The symbol.
        outliers: Values that are drawn with the OUTLIER symbol. They are not part of the mean or the SD.
        mean_half_width: Half width of the mean line in x data units.
        show_sd: Draw the sample SD whiskers. They need at least 2 values.

    Returns:
        The mean, the sample SD (NaN for less than 2 values) and the x positions of the symbols.
    """
    values = np.asarray(values, dtype=float)
    x_extent, y_extent = _symbol_extent(ax)
    # The factor 1.2 keeps a small gap between the symbols.
    step, y_tolerance = 1.2 * x_extent, 1.2 * y_extent
    x = position + swarm_offsets(values, y_tolerance, step)
    ax.scatter(x, values, s=MARKER_SIZE ** 2, color=color, marker=marker, linewidths=0, zorder=2)
    if len(outliers) > 0:
        outliers = np.asarray(outliers, dtype=float)
        ax.scatter(position + swarm_offsets(outliers, y_tolerance, step), outliers, s=MARKER_SIZE ** 2,
                   color=OUTLIER["color"], marker=OUTLIER["marker"], linewidths=DATA_LINE_WIDTH, zorder=2)

    mean = values.mean()
    sd = values.std(ddof=1) if len(values) > 1 else np.nan
    ax.hlines(mean, position - mean_half_width, position + mean_half_width, color=COLORS["black"],
              linewidth=MEAN_LINE_WIDTH, zorder=3)
    if show_sd and len(values) > 1:
        ax.errorbar(position, mean, yerr=sd, fmt="none", ecolor=COLORS["black"], elinewidth=AXIS_LINE_WIDTH,
                    capthick=AXIS_LINE_WIDTH, capsize=3.0, zorder=3)
    return mean, sd, x


def box_plot(ax, values, position: float, color, width: float = 0.5):
    """Draw a Prism box-and-whiskers plot: an open box with the outline, median and whiskers in color.

    The whiskers reach the last value within 1.5 IQR of the box. Fliers are hidden, because the
    box is meant for samples with many values.

    Args:
        ax: The axis.
        values: The values.
        position: The x position of the box.
        color: The line color.
        width: Box width in x data units.

    Returns:
        The artists of Axes.boxplot.
    """
    line = dict(color=color, linewidth=DATA_LINE_WIDTH)
    return ax.boxplot(
        [np.asarray(values)], positions=[position], widths=width, patch_artist=True, showfliers=False,
        manage_ticks=False, capwidths=width / 2,
        boxprops=dict(facecolor="white", edgecolor=color, linewidth=DATA_LINE_WIDTH),
        medianprops=dict(color=color, linewidth=MEAN_LINE_WIDTH), whiskerprops=line, capprops=line,
    )


def reference_band(ax, lower: float, upper: float, label: str = None, label_side: str = "below"):
    """Draw a reference range, e.g. a literature range, as a gray band over the full axis width.

    Args:
        ax: The axis.
        lower: Lower bound of the range.
        upper: Upper bound of the range.
        label: Text at the right end of the band.
        label_side: Put the label "below" or "above" the band.
    """
    ax.axhspan(lower, upper, color=tint(REFERENCE_COLOR, 0.85), linewidth=0, zorder=0)
    for bound in (lower, upper):
        ax.axhline(bound, color=REFERENCE_COLOR, linewidth=AXIS_LINE_WIDTH, zorder=1)
    if label is not None:
        below = label_side == "below"
        ax.annotate(label, (1, lower if below else upper), xycoords=("axes fraction", "data"),
                    xytext=(-1, -2 if below else 2), textcoords="offset points", ha="right",
                    va="top" if below else "bottom", color=COLORS["dark_gray"])


def profile_plot(ax, samples, color, marker: str = "o", show_points: bool = True, show_sd: bool = True,
                 label: str = None):
    """Draw the samples of one group along x, with the group mean as line and the sample SD as band.

    Args:
        ax: The axis.
        samples: DataFrame with the x positions as index and one column per sample.
        color: The group color.
        marker: The symbol of the sample values.
        show_points: Draw the value of each sample.
        show_sd: Draw the sample SD over the samples as band.
        label: Legend label of the mean line.

    Returns:
        The mean and the sample SD per x position.
    """
    x = samples.index.values
    if show_points:
        for column in samples:
            ax.scatter(x, samples[column], s=(0.6 * MARKER_SIZE) ** 2, color=tint(color, 0.45), marker=marker,
                       linewidths=0, zorder=1)
    mean, sd = samples.mean(axis=1), samples.std(axis=1)
    if show_sd:
        ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.2, linewidth=0, zorder=0)
    ax.plot(x, mean, color=color, linewidth=MEAN_LINE_WIDTH, zorder=2, label=label)
    return mean, sd


def _overlap_area(box_a, box_b) -> float:
    """Return the overlap area of two boxes (left, bottom, right, top)."""
    width = min(box_a[2], box_b[2]) - max(box_a[0], box_b[0])
    height = min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])
    return max(width, 0) * max(height, 0)


def label_points(ax, x, y, labels, pad: float = 2.0) -> list:
    """Label points with a text next to each, so that no label covers another label or a point.

    Each label takes the first free place of: right, left, above, below and the four diagonals
    of its point. A place outside the axis is not free. If no place is free, the label takes the
    place with the smallest overlap. The test uses an estimated text width, so check the result.

    Set the axis limits, tick labels and axis titles first, because the test uses the axis size
    after the layout.

    Args:
        ax: The axis.
        x: The x values of the points.
        y: The y values of the points.
        labels: One text per point.
        pad: Distance between a symbol and its label in pt.

    Returns:
        The annotations.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    width_pt, height_pt = _axis_size_pt(ax)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px, py = (x - x0) / (x1 - x0) * width_pt, (y - y0) / (y1 - y0) * height_pt
    radius = MARKER_SIZE / 2
    boxes = [(a - radius, b - radius, a + radius, b + radius) for a, b in zip(px, py)]
    offset = radius + pad
    places = [
        (offset, 0, "left", "center"), (-offset, 0, "right", "center"),
        (0, offset, "center", "bottom"), (0, -offset, "center", "top"),
        (offset, offset, "left", "bottom"), (-offset, offset, "right", "bottom"),
        (offset, -offset, "left", "top"), (-offset, -offset, "right", "top"),
    ]
    # Part of the text box that lies left of or below the anchor, per alignment.
    left_share = {"left": 0.0, "center": 0.5, "right": 1.0}
    bottom_share = {"bottom": 0.0, "center": 0.5, "top": 1.0}
    annotations = []
    for i, label in enumerate(labels):
        # The average glyph width of Arial is about 0.6 em. The text box with ascent and descent is
        # about 1.2 em high.
        text_width, text_height = 0.6 * FONT_SIZE * len(label), 1.2 * FONT_SIZE
        best = None
        for dx, dy, ha, va in places:
            left = px[i] + dx - left_share[ha] * text_width
            bottom = py[i] + dy - bottom_share[va] * text_height
            box = (left, bottom, left + text_width, bottom + text_height)
            inside = box[0] >= 0 and box[1] >= 0 and box[2] <= width_pt and box[3] <= height_pt
            # The test box is 1 pt larger on each side, so that neighbouring texts keep a gap.
            test_box = (box[0] - 1, box[1] - 1, box[2] + 1, box[3] + 1)
            score = sum(_overlap_area(test_box, other) for other in boxes) + (0 if inside else width_pt * height_pt)
            if best is None or score < best[0]:
                best = (score, dx, dy, ha, va, box)
            if score == 0:
                break
        _, dx, dy, ha, va, box = best
        boxes.append(box)
        annotations.append(ax.annotate(label, (x[i], y[i]), xytext=(dx, dy), textcoords="offset points",
                                       ha=ha, va=va))
    return annotations


def text_key(ax, entries, loc: str = "best", outside: bool = False, ncol: int = 1, fontweight: str = "bold"):
    """Add a key of colored text without symbols, as a direct label of the groups.

    The key is a legend without handles, so it replaces a legend of the axis. With loc="best",
    matplotlib puts it where it covers the fewest lines and points. Bands are not taken into
    account. Use outside=True if the axis has no free space: the key then sits right of the axis,
    and the constrained layout makes room for it.

    A color with too little contrast for text is darkened, see readable.

    Args:
        ax: The axis.
        entries: Pairs of text and color.
        loc: Location in the axis, as for Axes.legend. Not used with outside=True.
        outside: Put the key right of the axis, aligned with its top.
        ncol: Number of key columns.
        fontweight: Font weight of the text.

    Returns:
        The legend.
    """
    handles = [Line2D([], [], linestyle="none") for _ in entries]
    placement = dict(loc="upper left", bbox_to_anchor=(1.0, 1.0)) if outside else dict(loc=loc)
    legend = ax.legend(handles, [text for text, _ in entries], handlelength=0, handletextpad=0, ncol=ncol,
                       prop=dict(size=FONT_SIZE, weight=fontweight), **placement)
    for text, (_, color) in zip(legend.get_texts(), entries):
        text.set_color(readable(color))
    return legend


def _demo_palette(save_path):
    fig, ax = new_panel(120, 70)
    for i, (name, color) in enumerate(COLORS.items()):
        x, y = (i // 7) * 6, -(i % 7)
        ax.add_patch(plt.Rectangle((x, y - 0.4), 0.8, 0.8, color=color))
        ax.text(x + 1, y, f"{name} {color}", va="center", color=readable(color))
    for j, (name, n, hue_range) in enumerate([("blue", 6, 0.06), ("green", 4, -0.06)]):
        y = -7.5 - j
        for k, color in enumerate(group_shades(COLORS[name], n, hue_range=hue_range)):
            ax.add_patch(plt.Rectangle((k * 0.9, y - 0.4), 0.8, 0.8, color=color))
        ax.text(6, y, f"group_shades({name}, {n}, hue_range={hue_range})", va="center")
    ax.set_xlim(0, 12)
    ax.set_ylim(-9, 0.6)
    ax.axis("off")
    save_panel(fig, save_path)


def _demo_dot_plot(save_path, rng):
    groups = [("A", COLORS["blue"], "s", rng.normal(600, 60, 6)), ("B", COLORS["green"], "o", rng.normal(640, 20, 4))]
    fig, ax = new_panel(35, 45)
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(400, 800)
    for position, (_, color, marker, values) in enumerate(groups, start=1):
        dot_plot(ax, values, position, color, marker, outliers=[450] if position == 1 else ())
    reference_band(ax, 660, 700, label="literature", label_side="above")
    ax.set_xticks([1, 2], [f"{name}\nn = {len(values)}" for name, _, _, values in groups])
    ax.set_ylabel("Count per cochlea")
    ax.set_title("IHC")
    save_panel(fig, save_path)


def _demo_box_plot(save_path, rng):
    fig, ax = new_panel(35, 45)
    box_plot(ax, rng.gamma(4, 20, 5000), 1, COLORS["blue"])
    box_plot(ax, rng.gamma(9, 15, 5000), 2, COLORS["green"])
    ax.set_xlim(0.4, 2.6)
    ax.set_ylim(bottom=0)
    ax.set_xticks([1, 2], ["A", "B"])
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_title("PV@SGNs")
    save_panel(fig, save_path)


def _demo_profile_plot(save_path, rng):
    import pandas as pd

    x = np.linspace(0.025, 0.975, 20)
    fig, ax = new_panel(55, 45)
    for name, color, marker, level in [("A", COLORS["blue"], "s", 600), ("B", COLORS["green"], "o", 700)]:
        samples = pd.DataFrame({i: level + 100 * np.sin(3 * x) + rng.normal(0, 60, len(x)) for i in range(4)}, index=x)
        profile_plot(ax, samples, color, marker)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    text_key(ax, [("A", COLORS["blue"]), ("B", COLORS["green"])])
    ax.set_xlabel("Length fraction")
    ax.set_ylabel("Intensity (a.u.)")
    save_panel(fig, save_path)


def _demo_stacked_bar(save_path):
    segments = [("only A", COLORS["purple"]), ("both", COLORS["dark_gray"]), ("only B", COLORS["orange"])]
    counts = [[525, 7150, 7545], [675, 248, 10444]]
    fig, ax = new_panel(90, 30)
    for y, row in enumerate(counts):
        left = 0
        for (label, color), width in zip(segments, row):
            ax.barh(y, width, left=left, height=0.6, color=color, edgecolor="white", linewidth=1,
                    label=label if y == 0 else None)
            if width > 0.06 * sum(row):
                ax.text(left + width / 2, y, str(width), ha="center", va="center", color=text_color_on(color))
            left += width
    ax.set_yticks([0, 1], ["Sample 1", "Sample 2"])
    ax.set_ylim(1.5, -0.5)
    ax.set_xlabel("Synapses")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1), ncol=3)
    save_panel(fig, save_path)


@style()
def demo(out_dir: str):
    """Render the palette and one example panel of each type with synthetic data.

    Args:
        out_dir: Output directory.
    """
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(0)
    _demo_palette(os.path.join(out_dir, "palette"))
    _demo_dot_plot(os.path.join(out_dir, "dot_plot"), rng)
    _demo_box_plot(os.path.join(out_dir, "box_plot"), rng)
    _demo_profile_plot(os.path.join(out_dir, "profile_plot"), rng)
    _demo_stacked_bar(os.path.join(out_dir, "stacked_bar"))


def main():
    parser = argparse.ArgumentParser(description="Render the palette and example panels of the panel style.")
    parser.add_argument("out_dir", help="Output directory.")
    demo(parser.parse_args().out_dir)


if __name__ == "__main__":
    main()
