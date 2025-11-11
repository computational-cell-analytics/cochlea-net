import argparse
import os

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

from flamingo_tools.s3_utils import get_s3_path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from flamingo_tools.segmentation.sgn_subtype_utils import COCHLEAE
from util import prism_style, prism_cleanup_axes, export_legend

png_dpi = 300
FILE_EXTENSION = "png"

COCHLEAE_DICT = {
    "M_LR_000184_L": {"alias": "S01"},
    "M_LR_000184_R": {"alias": "S02"},
    "M_LR_000260_L": {"alias": "S03"},
}

COLORS_ANIMAL = {
    "S01": "#9C5027",
    "S02": "#279C52",
    "S03": "#67279C",
}

MARKER_LEFT = "o"
MARKER_RIGHT = "^"


def plot_legend(save_path):
    """Plot common legend for supplementary figure 3.

    Args:
        chreef_data: Data of ChReef cochleae.
        save_path: save path to save legend.
        grouping: Grouping for cochleae.
            "side_mono" for division in Injected and Non-Injected.
            "side_multi" for division per cochlea.
            "animal" for division per animal.
        use_alias: Use alias.
    """
    cochleae = [c for c in COCHLEAE.keys() if
                len(COCHLEAE[c]["subtype_stains"]) == 1 and
                "Prph" in COCHLEAE[c]["subtype_stains"]]
    alias = [COCHLEAE_DICT[k]["alias"] for k in cochleae]

    colors = []
    labels = []
    markers = []
    keys_animal = list(COLORS_ANIMAL.keys())
    ncol = len(keys_animal)
    for num in range(len(COLORS_ANIMAL)):
        colors.append(COLORS_ANIMAL[keys_animal[num]])
        colors.append(COLORS_ANIMAL[keys_animal[num]])
        labels.append(f"{alias[num]}L")
        labels.append(f"{alias[num]}R")
        markers.append(MARKER_LEFT)
        markers.append(MARKER_RIGHT)
        f = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    handles = [f(marker, color) for (color, marker) in zip(colors, markers)]
    legend = plt.legend(handles, labels, loc=3, ncol=ncol, framealpha=1, frameon=False)

    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def supp_fig_03_cm(save_path, plot=False):
    cochlea = "M_AMD_N180_L"
    seg_name = "SGN_merged"
    table_path = f"{cochlea}/tables/{seg_name}/default.tsv"
    table_path_s3, fs = get_s3_path(table_path)
    with fs.open(table_path_s3, "r") as f:
        table = pd.read_csv(f, sep="\t")

    subtypes = ["Type Ia", "Type Ib", "Type Ic"]
    subset = table[table["subtype_label"].isin(subtypes)]

    cm = confusion_matrix(subset['subtype_label'], subset['subtype_label_Lypd1'], labels=subtypes, normalize='true')
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=subtypes)
    disp.plot(cmap='Blues', xticks_rotation=45, values_format=".2f")
    # plt.title("Confusion Matrix: CR+Lypd1 vs CR+Ntng1 (True label)")
    plt.xlabel("CR, Lypd1")
    plt.ylabel("CR, Ntng1")

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def supp_fig_03_offset(save_path, plot=False, plot_type="mean"):
    cochleae = [c for c in COCHLEAE.keys() if
                len(COCHLEAE[c]["subtype_stains"]) == 1 and
                "Prph" in COCHLEAE[c]["subtype_stains"]]

    print(cochleae)
    dic = {}
    for cochlea in cochleae:
        offset_dic = {}
        if "output_seg" in list(COCHLEAE[cochlea].keys()):
            seg_name = COCHLEAE[cochlea]["output_seg"]
        else:
            seg_name = COCHLEAE[cochlea]["seg_data"]

        table_path = f"{cochlea}/tables/{seg_name}/default.tsv"
        table_path_s3, fs = get_s3_path(table_path)
        with fs.open(table_path_s3, "r") as f:
            table = pd.read_csv(f, sep="\t")

        subtypes = ["Type I", "Type II"]
        subset = table[table["subtype_label"].isin(subtypes)]
        offset1 = list(subset.loc[subset["subtype_label"] == "Type I", "offset"])
        offset2 = list(subset.loc[subset["subtype_label"] == "Type II", "offset"])
        offset_dic["Type I"] = offset1
        offset_dic["Type II"] = offset2
        dic[cochlea] = offset_dic

    prism_style()
    alias = [COCHLEAE_DICT[k]["alias"] for k in cochleae]

    # Plot
    fig, ax = plt.subplots(figsize=(4, 5))
    if plot_type == "mean":
        values_left = [np.mean(dic[cochlea]["Type I"]) for cochlea in cochleae]
        values_right = [np.mean(dic[cochlea]["Type II"]) for cochlea in cochleae]

        main_label_size = 20
        sub_label_size = 16

        label = "Mean offset [µm]"
        x_left = 1
        x_right = 2
        offset = 0.08

        x_pos_inj = [x_left - len(values_left) // 2 * offset + offset * i for i in range(len(values_left))]
        x_pos_non = [x_right - len(values_right) // 2 * offset + offset * i for i in range(len(values_right))]

        for num, key in enumerate(COLORS_ANIMAL.keys()):
            plt.scatter(x_pos_inj[num], values_left[num], label=f"{alias[num]}",
                        color=COLORS_ANIMAL[key], marker=MARKER_LEFT, s=80, zorder=1)
            plt.scatter(x_pos_non[num], values_right[num],
                        color=COLORS_ANIMAL[key], marker=MARKER_RIGHT, s=80, zorder=1)

        # lines between cochleae of same animal
        for num, (left, right) in enumerate(zip(values_left, values_right)):
            ax.plot(
                [x_pos_inj[num], x_pos_non[num]],
                [left, right],
                linestyle="solid",
                color="grey",
                alpha=0.4,
                zorder=0
            )
        # plt.ylim(0.65, 1.05)
        # plt.yticks(np.arange(0.7, 1, 0.1), fontsize=main_tick_size)

        # Labels and formatting
        plt.xticks([x_left, x_right], ["Type I", "Type II"], fontsize=sub_label_size)
        for la in plt.gca().get_xticklabels():
            la.set_verticalalignment('center')
        ax.tick_params(axis='x', which='major', pad=16)
        plt.ylabel(label, fontsize=main_label_size)
        # ax.yaxis.set_major_formatter(mticker.FuncFormatter(custom_formatter_1))

        xmin = 0.5
        xmax = 2.5
        plt.xlim(xmin, xmax)

    elif plot_type == "points":
        main_label_size = 20
        sub_label_size = 16

        label = "Offset [µm]"
        offset = 0.1
        x_left = 1 - offset * (len(cochleae) // 2)
        x_right = 2 - offset * (len(cochleae) // 2)
        jitter = 0.08  # horizontal scatter for visibility

        # Instead of just means, plot all distance values
        for num, cochlea in enumerate(cochleae):
            # distances for this cochlea
            dist_left = dic[cochlea]["Type I"]
            dist_right = dic[cochlea]["Type II"]

            color = list(COLORS_ANIMAL.values())[num % len(COLORS_ANIMAL)]
            alias_name = alias[num] if num < len(alias) else f"C{num+1}"

            # random horizontal jitter so points don’t overlap
            x_left_jitter = x_left + offset*num + np.random.uniform(-jitter, jitter, size=len(dist_left))
            x_right_jitter = x_right + offset*num + np.random.uniform(-jitter, jitter, size=len(dist_right))

            # plot all individual points
            ax.scatter(x_left_jitter, dist_left,
                       color=color, marker=MARKER_LEFT, alpha=0.7, s=1, label=alias_name if num == 0 else "", zorder=2)
            ax.scatter(x_right_jitter, dist_right,
                       color=color, marker=MARKER_RIGHT, alpha=0.7, s=1, zorder=2)

        # Labels and formatting
        ax.set_xticks([x_left, x_right])
        ax.set_xticklabels(["Type I", "Type II"], fontsize=sub_label_size)
        ax.tick_params(axis='x', which='major', pad=16)
        ax.set_ylabel(label, fontsize=main_label_size)
        ax.set_xlim(0.5, 2.5)

    else:
        raise ValueError("Choose either 'mean' or 'points' for parameter plot_type.")

    plt.tight_layout()
    prism_cleanup_axes(ax)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Supplementary Fig 3 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, required=True, help="Output directory for plots.")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    supp_fig_03_cm(save_path=os.path.join(args.figure_dir, f"figsupp_03_cm.{FILE_EXTENSION}"))
    plot_type = "mean"
    supp_fig_03_offset(save_path=os.path.join(args.figure_dir, f"figsupp_03_offset_{plot_type}.{FILE_EXTENSION}"),
                       plot_type=plot_type)
    plot_type = "points"
    supp_fig_03_offset(save_path=os.path.join(args.figure_dir, f"figsupp_03_offset_{plot_type}.{FILE_EXTENSION}"),
                       plot_type=plot_type)

    plot_legend(save_path=os.path.join(args.figure_dir, f"figsupp_03_legend.{FILE_EXTENSION}"))


if __name__ == "__main__":
    main()
