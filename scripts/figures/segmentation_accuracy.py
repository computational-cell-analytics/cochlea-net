"""Plots of segmentation accuracy and processing time, read from the model accuracy JSON files."""
import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

from util import custom_formatter, get_flatline_handle, get_marker_handle, iteration_statistics
from util import prism_cleanup_axes, prism_style, save_figure, save_legend

COLOR_P = "#9C5027"
COLOR_R = "#67279C"
COLOR_F = "#9C276F"
COLOR_T = "#279C52"

# Scatter marker area, shared by every panel so the figures stay consistent. Kept small enough
# that a marker does not hide the error bar drawn behind it: the SGN and IHC seed standard
# deviations are ~0.005, which is only a few points tall on these axes.
MARKER_SIZE = 28

METRICS = ("precision", "recall", "f1-score")

# Accepted values of the 'error_bars' argument of plot_accuracy_manual_vs_automatic.
ERROR_BAR_MODES = ("none", "production", "mean")

# Plotting metadata (label and marker) per segmentation type and baseline key.
# Key order determines the left-to-right order on the x-axis.
PLOT_METADATA = {
    "SGN": {
        "stardist": {"label": "Stardist", "marker": "*"},
        "micro-sam": {"label": "µSAM", "marker": "D"},
        "micro-sam_finetuned": {"label": "µSAM ft", "marker": "D"},
        "cellpose3": {"label": "Cellpose 3", "marker": "v"},
        "cellpose3_finetuned": {"label": "Cellpose 3 ft", "marker": "v"},
        "cellpose-sam": {"label": "Cellpose-SAM", "marker": "^"},
        "spiner2D": {"label": "Spiner", "marker": "o"},
        "distance_unet": {"label": "CochleaNet", "marker": "s"},
        "distance_unet_f0": {"label": "Fold 1", "marker": "s"},
        "distance_unet_f1": {"label": "Fold 2", "marker": "v"},
        "distance_unet_f2": {"label": "Fold 3", "marker": "D"},
        "distance_unet_f3": {"label": "Fold 4", "marker": "^"},
        "distance_unet_f4": {"label": "Fold 5", "marker": "o"},
    },
    "IHC": {
        "micro-sam": {"label": "µSAM", "marker": "D"},
        "micro-sam_finetuned": {"label": "µSAM ft", "marker": "D"},
        "cellpose3": {"label": "Cellpose 3", "marker": "v"},
        "cellpose3_finetuned": {"label": "Cellpose 3 ft", "marker": "v"},
        "cellpose-sam": {"label": "Cellpose-SAM", "marker": "^"},
        "distance_unet_v11": {"label": "CochleaNet", "marker": "s"},
        "distance_unet_v11-f0": {"label": "Fold 1", "marker": "s"},
        "distance_unet_v11-f1": {"label": "Fold 2", "marker": "v"},
        "distance_unet_v11-f2": {"label": "Fold 3", "marker": "D"},
        "distance_unet_v11-f3": {"label": "Fold 4", "marker": "^"},
        "distance_unet_v11-f4": {"label": "Fold 5", "marker": "o"},
    },
    "IHC_3D": {
        "v4b": {"label": "v4b", "marker": "D"},
        "v4c": {"label": "v4c", "marker": "v"},
        "v9": {"label": "v9", "marker": "^"},
        "v10": {"label": "v10", "marker": "s"},
        "v11": {"label": "v11", "marker": "o"},
    },
    "synapses": {
        "v3": {"label": "v3", "marker": "D"},
        "v5-f0": {"label": "Fold 1", "marker": "s"},
        "v5-f1": {"label": "Fold 2", "marker": "v"},
        "v5-f2": {"label": "Fold 3", "marker": "D"},
        "v5-f3": {"label": "Fold 4", "marker": "^"},
        "v5-f4": {"label": "Fold 5", "marker": "o"},
        "v5": {"label": "v5", "marker": "^"},
        "v3_05t": {"label": "v3_old", "marker": "s"},
        "v5_f1val_threshold": {"label": "v5_optimal", "marker": "o"},
        "v5_train_threshold": {"label": "v5_train", "marker": "*"},
        "v5_05t": {"label": "v5_0.5t", "marker": "P"},
        "v3_for_consensus_annotations_synapses_AMD": {"label": "v3_AMD", "marker": "v"},
        "v3_for_consensus_annotations_synapses_EK": {"label": "v3_EK", "marker": "<"},
        "v3_for_consensus_annotations_synapses_LR": {"label": "v3_LR", "marker": ">"},
        "v3_05t_for_consensus_annotations_synapses_AMD": {"label": "v3_old_AMD", "marker": "<"},
        "v3_05t_for_consensus_annotations_synapses_EK": {"label": "v3_old_EK", "marker": "<"},
        "v3_05t_for_consensus_annotations_synapses_LR": {"label": "v3_old_LR", "marker": ">"},
        "v5_for_consensus_annotations_synapses_AMD": {"label": "v5_AMD", "marker": "v"},
        "v5_for_consensus_annotations_synapses_EK": {"label": "v5_EK", "marker": "<"},
        "v5_for_consensus_annotations_synapses_LR": {"label": "v5_LR", "marker": ">"},
        "v5_f1val_threshold_for_consensus_annotations_synapses_AMD": {"label": "v5_opti_AMD", "marker": "v"},
        "v5_f1val_threshold_for_consensus_annotations_synapses_EK": {"label": "v5_opti_EK", "marker": "<"},
        "v5_f1val_threshold_for_consensus_annotations_synapses_LR": {"label": "v5_opti_LR", "marker": ">"},
        "v5_train_threshold_for_consensus_annotations_synapses_AMD": {"label": "v5_train_AMD", "marker": "v"},
        "v5_train_threshold_for_consensus_annotations_synapses_EK": {"label": "v5_train_EK", "marker": "<"},
        "v5_train_threshold_for_consensus_annotations_synapses_LR": {"label": "v5_train_LR", "marker": ">"},
        "v5_05t_for_consensus_annotations_synapses_AMD": {"label": "v5_0.5t_AMD", "marker": "v"},
        "v5_05t_for_consensus_annotations_synapses_EK": {"label": "v5_0.5t_EK", "marker": "<"},
        "v5_05t_for_consensus_annotations_synapses_LR": {"label": "v5_0.5t_LR", "marker": ">"},
    },
}

# X-axis labels per accuracy file of plot_replicate_variation.
GROUP_LABELS = {"SGN": "SGN", "IHC": "IHC", "synapses": "Synapses"}


def _read_metrics(data_dir: str, file_name: str) -> dict:
    """Read one accuracy JSON file.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.

    Returns:
        The accuracy entries of the file, keyed by network version or annotator scenario.
    """
    with open(os.path.join(data_dir, file_name), "r") as f:
        return json.load(f)


def _read_scores(data_dir: str, file_name: str, key: str) -> list:
    """Read the precision, recall, and F1-score of one entry from an accuracy JSON file.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.
        key: Top-level entry of the file, e.g. a network version or an annotator scenario.

    Returns:
        The precision, recall, and F1-score of the entry.
    """
    metrics = _read_metrics(data_dir, file_name)
    if key not in metrics:
        source = os.path.join(data_dir, file_name)
        raise KeyError(f"{source} has no entry '{key}'. Available entries: {sorted(metrics)}.")
    return [metrics[key][metric] for metric in METRICS]


def _read_replicate_stats(data_dir: str, file_name: str, keys: list) -> tuple:
    """Read the mean and standard deviation of precision, recall and F1 over several entries.

    Args:
        data_dir: Directory containing the accuracy JSON files.
        file_name: Name of the accuracy JSON file.
        keys: Top-level entries to aggregate over, e.g. training-seed replicates or a selected
            model set.

    Returns:
        The per-metric means and the per-metric standard deviations, each in the order
        precision, recall, F1-score.
    """
    scores = np.array([_read_scores(data_dir, file_name, key) for key in keys])
    # np.std's default normalization, to match plot_replicate_variation.
    return scores.mean(axis=0), scores.std(axis=0)


def plot_accuracy_legend(
    save_path: str,
    plot_mode: str = "shapes",
):
    """Plot the legend of plot_accuracy_manual_vs_automatic.

    Args:
        save_path: save path to save legend.
        plot_mode: Plot either 'shapes' or 'colors' of data points.
    """
    if plot_mode == "shapes":
        labels = ["Manual", "Automatic"]
        handles = [get_marker_handle(c, m) for (c, m) in zip(["black", "black"], ["o", "s"])]
    elif plot_mode == "colors":
        labels = ["Precision", "Recall", "F1-score"]
        handles = [get_flatline_handle(c) for c in [COLOR_P, COLOR_R, COLOR_F]]
    else:
        raise ValueError("Choose either 'shapes' or 'colors' as plot_mode.")
    save_legend(handles, labels, save_path, ncol=len(labels))


def plot_accuracy_manual_vs_automatic(
    save_path: str,
    data_dir: str,
    model_sets: Dict[str, dict],
    plot: bool = False,
    annotator_keyword: str = "all",
    error_bars: str = "none",
):
    """Scatter plot of the precision, recall, and F1-score of manual and automatic segmentation.

    Args:
        save_path: Path for saving the figure.
        data_dir: Directory containing the accuracy JSON files.
        model_sets: One entry per structure, in x-axis order. An entry holds the x-axis "label",
            the network "accuracy_file", the annotator "consensus_file", the "released" model key
            and the "replicates" behind the error bar.
        plot: Whether to display the plot interactively.
        annotator_keyword: Entry to read from the annotator accuracy files.
        error_bars: How to show variability over the replicates. 'none' plots the released model
            without an error bar, 'production' keeps the released model as the marker and adds
            the replicate standard deviation as the error bar, and 'mean' plots the replicate mean
            instead. The manual results have no replicates and never get an error bar.
    """
    if error_bars not in ERROR_BAR_MODES:
        raise ValueError(f"Invalid error_bars '{error_bars}'. Expected one of {ERROR_BAR_MODES}.")
    prism_style()

    setting = [entry["label"] for entry in model_sets.values()]

    manual = [
        _read_scores(data_dir, entry["consensus_file"], annotator_keyword)
        for entry in model_sets.values()
    ]
    automatic = [
        _read_scores(data_dir, entry["accuracy_file"], entry["released"])
        for entry in model_sets.values()
    ]

    precision_manual = [i[0] for i in manual]
    recall_manual = [i[1] for i in manual]
    f1score_manual = [i[2] for i in manual]

    precision_automatic = [i[0] for i in automatic]
    recall_automatic = [i[1] for i in automatic]
    f1score_automatic = [i[2] for i in automatic]

    # Convert setting labels to numerical x positions
    x_manual = np.array([0.8, 1.8, 2.8])
    x_automatic = np.array([1.2, 2.2, 3.2])
    offset = 0.08  # horizontal shift for scatter separation

    fig, ax = plt.subplots(figsize=(8, 4.5))

    main_label_size = 20
    main_tick_size = 16
    capsize = 4

    plt.scatter(x_manual - offset, precision_manual, label="Precision manual", color=COLOR_P,
                marker="o", s=MARKER_SIZE)
    plt.scatter(x_manual, recall_manual, label="Recall manual", color=COLOR_R,
                marker="o", s=MARKER_SIZE)
    plt.scatter(x_manual + offset, f1score_manual, label="F1-score manual", color=COLOR_F,
                marker="o", s=MARKER_SIZE)

    # The automatic values, optionally as mean +- standard deviation over the replicates.
    # 'production' keeps the released model on the marker and only takes the spread from the
    # replicates, so the plotted values stay the ones reported elsewhere.
    automatic_std = None
    if error_bars != "none":
        means, stds = [], []
        for entry in model_sets.values():
            mean, std = _read_replicate_stats(data_dir, entry["accuracy_file"], entry["replicates"])
            means.append(mean)
            stds.append(std)
        automatic_std = np.array(stds)
        if error_bars == "mean":
            means = np.array(means)
            precision_automatic = means[:, 0].tolist()
            recall_automatic = means[:, 1].tolist()
            f1score_automatic = means[:, 2].tolist()

    for values, color, shift, label, column in (
        (precision_automatic, COLOR_P, -offset, "Precision automatic", 0),
        (recall_automatic, COLOR_R, 0.0, "Recall automatic", 1),
        (f1score_automatic, COLOR_F, offset, "F1-score automatic", 2),
    ):
        if automatic_std is not None:
            plt.errorbar(x_automatic + shift, values, yerr=automatic_std[:, column],
                         fmt="none", color="black", capsize=capsize, zorder=1)
        plt.scatter(x_automatic + shift, values, label=label, color=color, marker="s",
                    s=MARKER_SIZE, zorder=2)

    plt.xticks([1, 2, 3], setting, fontsize=main_label_size)
    plt.yticks(fontsize=main_tick_size)
    ax.yaxis.set_major_formatter(custom_formatter(2))
    plt.ylabel("Value", fontsize=main_label_size)
    plt.ylim(0.69, 1)
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_method_comparison_legend(save_path):
    """Plot the legend of plot_method_comparison, including the processing time.

    Args:
        save_path: save path to save legend.
    """
    labels = ["Precision", "Recall", "F1-score", "Processing time"]
    handles = [get_flatline_handle(c) for c in [COLOR_P, COLOR_R, COLOR_F, COLOR_T]]
    save_legend(handles, labels, save_path, ncol=len(labels))


def plot_method_comparison(
    save_path: str,
    plot: bool = False,
    segm: str = "SGN",
    key_list: Optional[List[str]] = None,
    mode: str = "precision",
    data_dir: Optional[str] = None,
    use_folds: bool = False,
    ref_dict: Optional[Dict[str, List[str]]] = None,
    show_legend: bool = False,
    ylim: Optional[List[float]] = None,
):
    """Plot the accuracy or the processing time of several segmentation methods or versions.

    The plot uses the matplotlib defaults, not prism_style.

    Args:
        save_path: Path for saving figure.
        plot: Whether to display the plot interactively.
        segm: Segmentation type, a key of PLOT_METADATA. Selects the accuracy file <segm>.json.
        key_list: Entries to plot. Defaults to every entry of PLOT_METADATA[segm].
        mode: Mode for plotting. Either "precision" or "runtime".
        data_dir: Directory containing the accuracy files produced by eval_baseline.py.
        use_folds: If True, replace the value of every key that has cross-validation fold
            variants (keys matching '<base>_f<digit>') with their mean and standard deviation.
        ref_dict: Map from a reference key to the list of variation keys to average over, e.g.
            {"distance_unet": ["distance_unet_f0", ..., "distance_unet_f4"]}. When given, it
            enables the fold-variation display on its own, without also setting use_folds, and
            takes precedence over the automatic '<base>_f<digit>' detection for any key it
            covers.
        show_legend: Plot legend.
        ylim: Lower and upper y-axis limit of the precision mode.
    """
    metrics = _read_metrics(data_dir, f"{segm}.json")

    # Pre-compute fold mean ± std for any key that has fold variants in the JSON.
    fold_stats = {}
    if use_folds or ref_dict is not None:
        if ref_dict is not None:
            fold_groups = {key: [k for k in variants if k in metrics] for key, variants in ref_dict.items()}
        else:
            fold_groups = {
                key: [k for k in metrics if k.startswith(f"{key}_f") and k[len(key) + 2:].isdigit()]
                for key in PLOT_METADATA[segm]
            }
        for key, fold_keys in fold_groups.items():
            if not fold_keys:
                continue
            fs, _ = iteration_statistics(
                metrics, fold_keys, metric_names=("precision", "recall", "f1-score", "runtime")
            )
            fs.setdefault("runtime", (None, None))
            fold_stats[key] = fs

    if key_list is None:
        key_list = list(PLOT_METADATA[segm].keys())

    entries = {}
    for key, meta in PLOT_METADATA[segm].items():
        if key not in key_list or key not in metrics:
            continue
        entries[key] = {"label": meta["label"], "marker": meta["marker"], **metrics[key]}

    tick_rotation = 45
    main_label_size = 20
    main_tick_size = 16
    marker_size = MARKER_SIZE
    capsize = 4

    labels = [entry["label"] for entry in entries.values()]

    if mode == "precision":
        fig, ax = plt.subplots(figsize=(10, 5))
        offset = 0.08
        for num, (key, entry) in enumerate(entries.items()):
            marker = entry["marker"]
            x_pos = num + 1

            if key in fold_stats:
                precision, precision_std = fold_stats[key]["precision"]
                recall, recall_std = fold_stats[key]["recall"]
                f1score, f1score_std = fold_stats[key]["f1-score"]
            else:
                precision = entry["precision"]
                recall = entry["recall"]
                f1score = entry["f1-score"]
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

        x_pos = np.arange(1, len(labels) + 1)
        plt.xticks(x_pos, labels, fontsize=main_tick_size, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Value", fontsize=main_label_size)
        if ylim is None:
            plt.ylim(-0.1, 1)
        else:
            plt.ylim(ylim[0], ylim[1])
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    elif mode == "runtime":
        fig, ax = plt.subplots(figsize=(8.5, 5))
        if "Spiner" in labels:
            labels.remove("Spiner")

        x_pos = 1
        for key, entry in entries.items():
            if key in fold_stats:
                runtime, runtime_std = fold_stats[key]["runtime"]
            else:
                runtime = entry["runtime"]
                runtime_std = None
            if runtime is None:
                continue
            plt.scatter([x_pos], [runtime], color=COLOR_T, marker=entry["marker"], s=marker_size)
            if runtime_std is not None:
                plt.errorbar([x_pos], [runtime], yerr=[runtime_std],
                             fmt="none", color="black", capsize=capsize)
            x_pos += 1

        x_pos = np.arange(1, len(labels) + 1)
        plt.xticks(x_pos, labels, fontsize=16, rotation=tick_rotation)
        plt.yticks(fontsize=main_tick_size)
        plt.ylabel("Processing time [s]", fontsize=main_label_size)
        plt.ylim(10, 2600)
        plt.yscale('log')
        plt.grid(axis="y", linestyle="solid", alpha=0.5)

    else:
        raise ValueError("Unsupported mode for plotting.")

    if show_legend and mode == "precision":
        label = ["Precision", "Recall", "F1-score"]
        handles = [get_flatline_handle(c) for c in [COLOR_P, COLOR_R, COLOR_F]]
        plt.legend(handles, label, loc=(0, 1), ncol=len(label), framealpha=1, frameon=False)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)


def plot_fold_accuracy(
    save_path: str,
    data_dir: str = None,
    plot: bool = False,
):
    """Plot precision, recall, F1-score, and processing time per cross-validation fold.

    Reads fold entries (keys matching 'distance_unet_f*') from SGN.json and plots all four
    metrics in one figure: precision, recall, and F1-score on the left y-axis; processing
    time on the right y-axis.

    Args:
        save_path: Path for saving figure.
        data_dir: Directory containing SGN.json produced by eval_baseline.py.
        plot: Whether to display the plot interactively.
    """
    if data_dir is None:
        raise ValueError("Please provide a data directory containing SGN.json.")
    metrics = _read_metrics(data_dir, "SGN.json")

    fold_keys = sorted(k for k in metrics if k.startswith("distance_unet_f"))
    labels = [k.replace("distance_unet_f", "Fold ") for k in fold_keys]
    x_positions = np.arange(1, len(fold_keys) + 1)

    marker = "s"
    marker_size = MARKER_SIZE
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

    ax2.set_ylabel("Processing time [s]", fontsize=main_label_size, color=COLOR_T)
    ax2.set_ylim(95, 200)
    ax2.tick_params(axis="y", labelsize=main_tick_size, labelcolor=COLOR_T)

    label = ["Precision", "Recall", "F1-score", "Processing time"]
    handles = [get_flatline_handle(c) for c in [COLOR_P, COLOR_R, COLOR_F, COLOR_T]]
    plt.legend(handles, label, loc=(0, 1), ncol=len(label), framealpha=1, frameon=False)

    plt.tight_layout()
    prism_cleanup_axes(ax1)
    save_figure(save_path, plot)


def plot_replicate_variation(
    save_path: str,
    replicate_dict: Dict[str, List[str]],
    data_dir: str,
    group_labels: Optional[Dict[str, str]] = None,
    ylim: Optional[List[float]] = None,
    show_legend: bool = True,
    plot: bool = False,
):
    """Plot the accuracy variation across replicates, e.g. folds or training seeds, per structure.

    Each structure gets one x position with precision, recall, and F1-score shown as the mean
    across its replicates, with the standard deviation as error bar.

    Args:
        save_path: Path for saving figure.
        replicate_dict: Replicate keys per structure. Each outer key selects the accuracy file
            '<key>.json' in data_dir and its position on the x-axis. Each value lists the
            entries in that file to average over.
        data_dir: Directory containing the accuracy files produced by eval_baseline.py.
        group_labels: X-axis label per structure. Defaults to GROUP_LABELS.
        ylim: Lower and upper y-axis limit.
        show_legend: Plot legend.
        plot: Whether to display the plot interactively.
    """
    if group_labels is None:
        group_labels = GROUP_LABELS

    stats = {}
    for group, keys in replicate_dict.items():
        json_path = os.path.join(data_dir, f"{group}.json")
        metrics = _read_metrics(data_dir, f"{group}.json")

        stats[group], present = iteration_statistics(metrics, keys, metric_names=METRICS)
        if not present:
            raise ValueError(f"None of the keys {keys} are present in {json_path}.")
        missing = [key for key in keys if key not in present]
        if missing:
            print(f"Warning: {group}: {', '.join(missing)} not found in {json_path}. "
                  f"Averaging over {len(present)} of {len(keys)} entries.")

        for metric in METRICS:
            mean, std = stats[group][metric]
            print(f"{group} {metric}: {mean:.3f} +- {std:.3f} (n={len(present)})")

    main_label_size = 20
    main_tick_size = 16
    marker = "s"
    capsize = 4
    offset = 0.15

    fig, ax = plt.subplots(figsize=(6, 5))

    for num, group in enumerate(stats):
        x_pos = num + 1
        for metric, color, shift in zip(METRICS, (COLOR_P, COLOR_R, COLOR_F), (-offset, 0, offset)):
            mean, std = stats[group][metric]
            plt.errorbar([x_pos + shift], [mean], yerr=[std], fmt="none", color="black",
                         capsize=capsize, zorder=1)
            plt.scatter([x_pos + shift], [mean], color=color, marker=marker, s=MARKER_SIZE, zorder=2)

    x_pos = np.arange(1, len(stats) + 1)
    plt.xlim(0.5, len(stats) + 0.5)
    plt.xticks(x_pos, [group_labels.get(group, group) for group in stats], fontsize=main_tick_size)
    plt.yticks(fontsize=main_tick_size)
    plt.ylabel("Value", fontsize=main_label_size)
    if ylim is None:
        plt.ylim(0.5, 1.05)
    else:
        plt.ylim(ylim[0], ylim[1])
    plt.grid(axis="y", linestyle="solid", alpha=0.5)

    if show_legend:
        label = ["Precision", "Recall", "F1-score"]
        handles = [get_flatline_handle(c) for c in [COLOR_P, COLOR_R, COLOR_F]]
        plt.legend(handles, label, loc=(0, 1), ncol=len(label), framealpha=1, frameon=False)

    plt.tight_layout()
    prism_cleanup_axes(ax)
    save_figure(save_path, plot)
