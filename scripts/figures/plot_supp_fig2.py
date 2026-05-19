import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from util import get_flatline_handle
from util import prism_cleanup_axes, export_legend

png_dpi = 300
FILE_EXTENSION = "png"

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"

# Plotting metadata (label and marker) per segmentation type and baseline key.
# Key order determines the left-to-right order on the x-axis.
PLOT_METADATA = {
    "SGN": {
        "stardist": {"label": "Stardist", "marker": "*"},
        "micro-sam": {"label": "µSAM", "marker": "D"},
        "micro-sam_finetuned": {"label": "µSAM finetuned", "marker": "D"},
        "cellpose3": {"label": "Cellpose 3", "marker": "v"},
        "cellpose3_finetuned": {"label": "Cellpose 3 finetuned", "marker": "v"},
        "cellpose-sam": {"label": "Cellpose-SAM", "marker": "^"},
        "spiner2D": {"label": "Spiner", "marker": "o"},
        "distance_unet": {"label": "CochleaNet", "marker": "s"},
    },
    "IHC": {
        "micro-sam": {"label": "µSAM", "marker": "D"},
        "cellpose3": {"label": "Cellpose 3", "marker": "v"},
        "cellpose-sam": {"label": "Cellpose-SAM", "marker": "^"},
        "distance_unet": {"label": "CochleaNet", "marker": "s"},
    },
}


def plot_legend_supp_fig02(save_path):
    """Plot common legend for figure 2c.

    Args:
        save_path: save path to save legend.
    """
    # Colors
    color = [COLOR_P, COLOR_R, COLOR_F, COLOR_T]
    label = ["Precision", "Recall", "F1-score", "Processing time"]

    handles = [get_flatline_handle(c) for c in color]
    legend = plt.legend(handles, label, loc=3, ncol=len(label), framealpha=1, frameon=False)
    export_legend(legend, save_path)
    legend.remove()
    plt.close()


def supp_fig_02(
    save_path: str,
    plot: bool = False,
    segm: str = "SGN",
    mode: str = "precision",
    data_dir: str = None,
    use_folds: bool = False,
):
    """Plot panels for Supplementary Figure 2.

    Args:
        save_path: Path for saving figure.
        plot:
        segm: Segmentation type. Either "SGN" or "IHC".
        mode: Mode for plotting. Either "precision" or "runtime".
        data_dir: Directory containing SGN.json and IHC.json accuracy files produced by
            eval_baseline.py. When provided, metrics are loaded from these files instead of
            the hardcoded fallback values.
        use_folds: If True, replace the direct distance_unet value with the mean and standard
            deviation computed across all cross-validation fold variants (keys matching
            '<base>_f<digit>') found in the JSON. Applies to any key that has fold variants.
    """
    json_path = os.path.join(data_dir, f"{segm}.json")
    with open(json_path, "r") as f:
        metrics = json.load(f)

    # Pre-compute fold mean ± std for any key that has fold variants in the JSON.
    fold_stats = {}
    if use_folds:
        for key in PLOT_METADATA[segm]:
            fold_keys = [k for k in metrics if k.startswith(f"{key}_f") and k[len(key) + 2:].isdigit()]
            if not fold_keys:
                continue
            fs = {}
            for metric in ("precision", "recall", "f1-score"):
                vals = [metrics[k][metric] for k in fold_keys]
                fs[metric] = (float(np.mean(vals)), float(np.std(vals)))
            rt_vals = [metrics[k]["runtime"] for k in fold_keys if metrics[k]["runtime"] is not None]
            fs["runtime"] = (float(np.mean(rt_vals)), float(np.std(rt_vals))) if rt_vals else (None, None)
            fold_stats[key] = fs

    segm_dict = {}
    for key, meta in PLOT_METADATA[segm].items():
        if key not in metrics:
            continue
        segm_dict[key] = {"label": meta["label"], "marker": meta["marker"], **metrics[key]}
    value_dict = {segm: segm_dict}

    # Convert setting labels to numerical x positions
    offset = 0.08  # horizontal shift for scatter separation

    # Plot
    tick_rotation = 45

    main_label_size = 20
    main_tick_size = 16
    marker_size = 200
    capsize = 4

    labels = [value_dict[segm][key]["label"] for key in value_dict[segm].keys()]

    if mode == "precision":
        fig, ax = plt.subplots(figsize=(10, 5))
        offset = 0.08
        for num, key in enumerate(list(value_dict[segm].keys())):
            marker = value_dict[segm][key]["marker"]
            x_pos = num + 1

            if use_folds and key in fold_stats:
                precision, precision_std = fold_stats[key]["precision"]
                recall, recall_std = fold_stats[key]["recall"]
                f1score, f1score_std = fold_stats[key]["f1-score"]
            else:
                precision = value_dict[segm][key]["precision"]
                recall = value_dict[segm][key]["recall"]
                f1score = value_dict[segm][key]["f1-score"]
                precision_std = recall_std = f1score_std = None

            plt.scatter([x_pos - offset], [precision], color=COLOR_P, marker=marker, s=marker_size)
            if precision_std is not None:
                plt.errorbar([x_pos - offset], [precision], yerr=[precision_std],
                             fmt="none", color="black", capsize=capsize)
            plt.scatter([x_pos], [recall], color=COLOR_R, marker=marker, s=marker_size)
            if recall_std is not None:
                plt.errorbar([x_pos], [recall], yerr=[recall_std],
                             fmt="none", color="black", capsize=capsize)
            plt.scatter([x_pos + offset], [f1score], color=COLOR_F, marker=marker, s=marker_size)
            if f1score_std is not None:
                plt.errorbar([x_pos + offset], [f1score], yerr=[f1score_std],
                             fmt="none", color="black", capsize=capsize)

        # Labels and formatting
        x_pos = np.arange(1, len(labels) + 1)
        plt.xticks(x_pos, labels, fontsize=main_tick_size, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        plt.ylim(-0.1, 1)
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif mode == "runtime":
        fig, ax = plt.subplots(figsize=(8.5, 5))
        if "Spiner" in labels:
            labels.remove("Spiner")

        x_pos = 1
        for num, key in enumerate(list(value_dict[segm].keys())):
            if use_folds and key in fold_stats:
                runtime, runtime_std = fold_stats[key]["runtime"]
            else:
                runtime = value_dict[segm][key]["runtime"]
                runtime_std = None
            if runtime is None:
                continue
            marker = value_dict[segm][key]["marker"]
            plt.scatter([x_pos], [runtime], color=COLOR_T, marker=marker, s=marker_size)
            if runtime_std is not None:
                plt.errorbar([x_pos], [runtime], yerr=[runtime_std],
                             fmt="none", color="black", capsize=capsize)
            x_pos += 1

        # Labels and formatting
        x_pos = np.arange(1, len(labels) + 1)
        plt.xticks(x_pos, labels, fontsize=16, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Processing time [s]", fontsize=main_label_size)
        plt.ylim(10, 2600)
        plt.yscale('log')
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    else:
        raise ValueError("Unsupported mode for plotting.")

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


def plot_fold_accuracy(
    save_path: str,
    data_dir: str = None,
    plot: bool = False,
):
    """Plot precision, recall, F1-score, and processing time per cross-validation fold.

    Reads fold entries (keys matching 'distance_unet_f*') from SGN.json and plots all four
    metrics in one figure: precision, recall, and F1-score on the left y-axis; processing
    time on a log-scale right y-axis.

    Args:
        save_path: Path for saving figure.
        data_dir: Directory containing SGN.json produced by eval_baseline.py.
        plot: Whether to display the plot interactively.
    """
    if data_dir is None:
        raise ValueError("Please provide a data directory containing SGN.json.")

    json_path = os.path.join(data_dir, "SGN.json")
    with open(json_path, "r") as f:
        metrics = json.load(f)

    fold_keys = sorted(k for k in metrics if k.startswith("distance_unet_f"))
    labels = [k.replace("distance_unet_f", "Fold ") for k in fold_keys]
    x_positions = np.arange(1, len(fold_keys) + 1)

    marker = "s"
    marker_size = 200
    main_label_size = 20
    main_tick_size = 16
    offset = 0.08

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    for i, key in enumerate(fold_keys):
        d = metrics[key]
        x = x_positions[i]
        ax1.scatter([x - offset], [d["precision"]], color=COLOR_P, marker=marker, s=marker_size)
        ax1.scatter([x], [d["recall"]], color=COLOR_R, marker=marker, s=marker_size)
        ax1.scatter([x + offset], [d["f1-score"]], color=COLOR_F, marker=marker, s=marker_size)
        if d["runtime"] is not None:
            ax2.scatter([x], [d["runtime"]], color=COLOR_T, marker=marker, s=marker_size)

    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(labels, fontsize=main_tick_size)
    ax1.set_ylim(-0.1, 1)
    ax1.set_ylabel("Value", fontsize=main_label_size)
    ax1.tick_params(axis="y", labelsize=main_tick_size)
    # ax1.grid(axis="y", linestyle="solid", alpha=0.5)

    ax2.set_ylabel("Processing time [s]", fontsize=main_label_size, color=COLOR_T)
    ax2.set_ylim(95, 200)
    # ax2.set_yscale("log")
    ax2.tick_params(axis="y", labelsize=main_tick_size, labelcolor=COLOR_T)

    color = [COLOR_P, COLOR_R, COLOR_F, COLOR_T]
    label = ["Precision", "Recall", "F1-score", "Processing time"]

    handles = [get_flatline_handle(c) for c in color]
    plt.legend(handles, label, loc=(0, 1), ncol=len(label), framealpha=1, frameon=False)

    plt.tight_layout()
    prism_cleanup_axes(ax1)

    if ".png" in save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=png_dpi)
    else:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)

    if plot:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate plots for Supplementary Fig 2 of the cochlea paper.")
    parser.add_argument("--figure_dir", "-f", type=str, help="Output directory for plots.",
                        default="./panels/supp_fig2")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument(
        "--use_folds", action="store_true",
        help="Replace the CochleaNet point with the mean ± std across the five cross-validation folds.",
    )
    _default_data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "reproducibility", "model_accuracy",
    )
    parser.add_argument(
        "--data_dir", "-d", type=str, default=_default_data_dir,
        help="Directory containing SGN.json and IHC.json accuracy files. "
             f"Defaults to {_default_data_dir}.",
    )
    args = parser.parse_args()

    os.makedirs(args.figure_dir, exist_ok=True)

    data_dir = args.data_dir if os.path.isdir(args.data_dir) else None

    # Supplementary Figure 2: Comparing other methods in terms of segmentation accuracy and runtime
    plot_legend_supp_fig02(save_path=os.path.join(args.figure_dir, f"supp_fig_02_legend_colors.{FILE_EXTENSION}"))
    if data_dir is not None:
        supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02a_sgn_accuracy.{FILE_EXTENSION}"),
                    segm="SGN", data_dir=data_dir, use_folds=args.use_folds)
        supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02b_ihc_accuracy.{FILE_EXTENSION}"),
                    segm="IHC", data_dir=data_dir, use_folds=args.use_folds)
        supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02a_sgn_time.{FILE_EXTENSION}"),
                    segm="SGN", mode="runtime", data_dir=data_dir, use_folds=args.use_folds)
        supp_fig_02(save_path=os.path.join(args.figure_dir, f"supp_fig_02b_ihc_time.{FILE_EXTENSION}"),
                    segm="IHC", mode="runtime", data_dir=data_dir, use_folds=args.use_folds)
        plot_fold_accuracy(
            save_path=os.path.join(args.figure_dir, f"supp_fig_sgn_folds.{FILE_EXTENSION}"),
            data_dir=data_dir,
            plot=args.plot,
        )
    else:
        raise ValueError("Please provide a data directory containing dictionaries produced by eval_baseline.py")


if __name__ == "__main__":
    main()
