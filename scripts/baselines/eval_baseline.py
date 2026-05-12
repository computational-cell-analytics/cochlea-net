import os
import json
import argparse

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path

from flamingo_tools.validation import compute_matches_for_annotated_slice

SGN_BASELINES = [
    "spiner2D",
    "cellpose3",
    "cellpose3_finetuned",
    "cellpose-sam",
    "distance_unet",
    "distance_unet_f0",
    "distance_unet_f1",
    "distance_unet_f2",
    "distance_unet_f3",
    "distance_unet_f4",
    "micro-sam",
    "micro-sam_finetuned",
    "stardist",
]

IHC_BASELINES = [
    "cellpose3",
    "cellpose-sam",
    "distance_unet_v4b",
    "micro-sam",
]

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"


def filter_seg(
    seg_arr: np.typing.ArrayLike,
    min_count: int = 3000,
    max_count: int = 50000,
) -> np.ndarray:
    """Filter segmentation based on minimal and maximal number of pixels of the segmented object.

    Args:
        seg_arr: Input segmentation
        min_count: Minimal number of pixels
        max_count: Maximal number of pixels

    Returns:
        Filtered segmentation
    """
    labels, counts = np.unique(seg_arr, return_counts=True)
    valid = labels[(counts >= min_count) & (counts <= max_count)]
    mask = np.isin(seg_arr, valid)
    seg_arr[~mask] = 0
    return seg_arr


def eval_seg_dict(
    dic: dict,
    out_path: str,
) -> None:
    """Format dictionary entries and write dictionary to output path.

    Args:
        dic: Parameter dictionary for baseline evaluation.
        out_path: Output path for json file.
    """
    dic["tp_objects"] = list(dic["tp_objects"])
    dic["tp_annotations"] = list(dic["tp_annotations"])
    dic["fp"] = list(dic["fp"])
    dic["fn"] = list(dic["fn"])

    dic["tp_objects"] = [int(i) for i in dic["tp_objects"]]
    dic["tp_annotations"] = [int(i) for i in dic["tp_annotations"]]
    dic["fp"] = [int(i) for i in dic["fp"]]
    dic["fn"] = [int(i) for i in dic["fn"]]

    json_out = os.path.join(out_path)
    with open(json_out, "w") as f:
        json.dump(dic, f, indent='\t', separators=(',', ': '))


def eval_all_sgn(
    baselines: list[str] = SGN_BASELINES,
    cochlea_dir: str = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet",
) -> None:
    """Evaluate all SGN baselines.
    """
    seg_dir = os.path.join(cochlea_dir, "predictions/val_sgn")
    annotation_dir = os.path.join(cochlea_dir,
                                  "AnnotatedImageCrops",
                                  "F1ValidationSGNs",
                                  "final_annotations",
                                  "final_consensus_annotations")

    for baseline in baselines:
        if "spiner" in baseline:
            eval_segmentation_spiner(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)
            # eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir, filter=False)
        else:
            eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)


def eval_all_ihc(
    baselines: list[str] = IHC_BASELINES,
    cochlea_dir: str = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet",
) -> None:
    """Evaluate all IHC baselines.
    """
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")
    annotation_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationIHCs/consensus_annotation")

    for baseline in baselines:
        eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)


def eval_segmentation(
    seg_dir: str,
    annotation_dir: str,
    filter: bool = True,
    verbose: bool = False,
) -> None:
    """Evaluate 3D segmentation from baseline methods.

    Args:
        seg_dir: Directory containing segmentation output of baseline methods.
        annotation_dir: Directory containing annotations in CSV format.
        filter: Bool for filtering segmentation based on size. Per default size between 3000 and 50000 pixels.
        verbise: Output already existing dictionaries.
    """
    print(f"\nEvaluating segmentation in directory {seg_dir}")
    segs = [entry.path for entry in os.scandir(seg_dir) if entry.is_file() and ".tif" in entry.path]

    seg_dicts = []
    for seg in segs:

        basename = os.path.basename(seg)
        basename = ".".join(basename.split(".")[:-1])
        basename = "".join(basename.split("_seg")[0])
        # print("Annotation_dir", annotation_dir)
        dic_out = os.path.join(seg_dir, f"{basename}_dic.json")
        if not os.path.isfile(dic_out):
            print(basename)

            df_path = os.path.join(annotation_dir, f"{basename}.csv")
            df = pd.read_csv(df_path, sep=",")

            seg_arr = imageio.imread(seg)
            print(f"shape {seg_arr.shape}")
            if filter:
                seg_arr = filter_seg(seg_arr=seg_arr)

            seg_dic = compute_matches_for_annotated_slice(segmentation=seg_arr,
                                                          annotations=df,
                                                          matching_tolerance=5)
            seg_dic["annotation_length"] = len(df)
            seg_dic["crop_name"] = basename
            timer_file = os.path.join(seg_dir, f"{basename}_timer.json")
            if os.path.isfile(timer_file):
                with open(timer_file) as f:
                    timer_dic = json.load(f)
                seg_dic["time"] = float(timer_dic["total_duration[s]"])
            else:
                seg_dic["time"] = None

            eval_seg_dict(seg_dic, dic_out)

            seg_dicts.append(seg_dic)
        elif verbose:
            print(f"Dictionary for {basename} already exists")

    json_out = os.path.join(seg_dir, "eval_seg.json")
    with open(json_out, "w") as f:
        json.dump(seg_dicts, f, indent='\t', separators=(',', ': '))


def eval_segmentation_spiner(
    seg_dir: str,
    annotation_dir: str,
) -> None:
    """Evaluate 2D spiner segmentation: https://github.com/reubenrosenNCSU/cellannotation
    The segmentation tool, which is used inside a browser, exports bounding boxes in the CSV format.
    An instance segmentation is created based on the CSV data and used for evaluation.
    The size filter is left out because the segmentation is only 2D and has limited sizes per default.

    Args:
        seg_dir: Directory containing segmentation output of baseline methods.
        annotation_dir: Directory containing annotations in CSV format.
    """
    print(f"Evaluating segmentation in directory {seg_dir}")
    annots = [entry.path for entry in os.scandir(seg_dir)
              if entry.is_file() and ".csv" in entry.path]

    seg_dicts = []
    for annot in annots:

        basename = os.path.basename(annot)
        basename = ".".join(basename.split(".")[:-1])
        basename = "".join(basename.split("_annot")[0])
        dic_out = os.path.join(seg_dir, f"{basename}_dic.json")
        if not os.path.isfile(dic_out):

            df_path = os.path.join(annotation_dir, f"{basename}.csv")
            df = pd.read_csv(df_path, sep=",")

            image_spiner = os.path.join(seg_dir, f"{basename}.tif")
            img = imageio.imread(image_spiner)
            seg_arr = np.zeros(img.shape)

            df_annot = pd.read_csv(annot, sep=",")
            for num, row in df_annot.iterrows():
                # coordinate switch to account for image orientation in Python
                x1 = int(row["y1"])
                x2 = int(row["y2"])
                y1 = int(row["x1"])
                y2 = int(row["x2"])
                seg_arr[x1:x2, y1:y2] = num + 1

            seg_dic = compute_matches_for_annotated_slice(segmentation=seg_arr,
                                                          annotations=df,
                                                          matching_tolerance=5)
            seg_dic["annotation_length"] = len(df)
            seg_dic["crop_name"] = basename
            seg_dic["time"] = None

            eval_seg_dict(seg_dic, dic_out)

            seg_dicts.append(seg_dic)
        else:
            print(f"Dictionary for {basename} already exists")

    json_out = os.path.join(seg_dir, "eval_seg.json")
    with open(json_out, "w") as f:
        json.dump(seg_dicts, f, indent='\t', separators=(',', ': '))


def compute_accuracy(eval_dir: str) -> dict:
    """Compute average precision, recall, F1-score, and runtime from per-crop dictionaries.

    Args:
        eval_dir: Directory containing per-crop *_dic.json files.

    Returns:
        Dict with keys 'precision', 'recall', 'f1-score', 'runtime', 'runtime_std'.
    """
    eval_dicts = [entry.path for entry in os.scandir(eval_dir) if entry.is_file() and "dic.json" in entry.path]
    precision_list, recall_list, f1_list, time_list = [], [], [], []
    for eval_dic in eval_dicts:
        with open(eval_dic, "r") as f:
            d = json.load(f)
        tp = len(d["tp_objects"])
        fp = len(d["fp"])
        fn = len(d["fn"])

        precision = tp / (tp + fp) if tp + fp != 0 else 0
        recall = tp / (tp + fn) if tp + fn != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        if d["time"] is not None:
            time_list.append(d["time"])

    runtime_mean = float(np.mean(time_list)) if time_list else None
    runtime_std = float(np.std(time_list)) if time_list else None

    return {
        "precision": round(float(np.mean(precision_list)), 3),
        "recall": round(float(np.mean(recall_list)), 3),
        "f1-score": round(float(np.mean(f1_list)), 3),
        "runtime": round(runtime_mean, 3) if runtime_mean is not None else None,
        "runtime_std": round(runtime_std, 3) if runtime_std is not None else None,
    }


def save_accuracy_json(
    segm_type: str,
    baselines: list[str],
    seg_dir: str,
    output_dir: str,
) -> None:
    """Compute and save accuracy metrics for all baselines to a JSON file.

    Args:
        segm_type: Segmentation type, 'SGN' or 'IHC'.
        baseline_keys: Mapping from baseline directory name to plot key.
        seg_dir: Directory containing per-baseline subdirectories.
        output_dir: Output directory for the JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    result = {}
    for baseline in baselines:
        baseline_dir = os.path.join(seg_dir, baseline)
        if not os.path.isdir(baseline_dir):
            print(f"Warning: {baseline_dir} not found, skipping.")
            continue
        result[baseline] = compute_accuracy(baseline_dir)

    out_path = os.path.join(output_dir, f"{segm_type}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent="\t", separators=(",", ": "))
    print(f"Saved accuracy metrics to {out_path}")


def print_accuracy(
    eval_dir: str,
) -> None:
    """Print 'Precision', 'Recall', and 'F1-score' for dictionaries in a given directory.
    The directory is scanned for files containing ".dic.json" and evaluates segmentation accuracy and runtime.

    Args:
        eval_dir: Print average accuracy of dictionary files in evaluation directory.
    """
    eval_dicts = [entry.path for entry in os.scandir(eval_dir) if entry.is_file() and "dic.json" in entry.path]
    precision_list = []
    recall_list = []
    f1_score_list = []
    time_list = []
    for eval_dic in eval_dicts:
        with open(eval_dic, "r") as f:
            d = json.load(f)
        tp = len(d["tp_objects"])
        fp = len(d["fp"])
        fn = len(d["fn"])
        time = d["time"]
        if time is None:
            show_time = False
        else:
            show_time = True

        if tp + fp != 0:
            precision = tp / (tp + fp)
        else:
            precision = 0
        if tp + fn != 0:
            recall = tp / (tp + fn)
        else:
            recall = 0
        if precision + recall != 0:
            f1_score = 2 * precision * recall / (precision + recall)
        else:
            f1_score = 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        time_list.append(time)

    if show_time:
        param_list = [precision_list, recall_list, f1_score_list, time_list]
        names = ["Precision", "Recall", "F1 score", "Time"]
    else:
        param_list = [precision_list, recall_list, f1_score_list]
        names = ["Precision", "Recall", "F1 score"]
    for num, lis in enumerate(param_list):
        print(names[num], sum(lis) / len(lis))


def print_accuracy_sgn(
    baselines: list[str] = SGN_BASELINES,
    cochlea_dir: str = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet",
) -> None:
    """Print 'Precision', 'Recall', and 'F1-score' for all SGN baselines.
    """
    print("Evaluating SGN segmentation")
    seg_dir = os.path.join(cochlea_dir, "predictions/val_sgn")

    for baseline in baselines:
        print(f"Evaluating baseline {baseline}")
        print_accuracy(os.path.join(seg_dir, baseline))


def print_accuracy_ihc(
    baselines: list[str] = IHC_BASELINES,
    cochlea_dir: str = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet",
) -> None:
    """Print 'Precision', 'Recall', and 'F1-score' for all IHC baselines.
    """
    print("Evaluating IHC segmentation")
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")

    for baseline in baselines:
        print(f"Evaluating baseline {baseline}")
        print_accuracy(os.path.join(seg_dir, baseline))


def runtimes_sgn(
    cochlea_dir: str = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet",
) -> None:
    for_comparison = [
        "distance_unet",
        "micro-sam",
        "micro-sam_finetuned",
        "cellpose3",
        "cellpose3_finetuned",
        "cellpose-sam",
        "stardist",
    ]

    val_sgn_dir = f"{cochlea_dir}/predictions/val_sgn"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation"

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))

    runtimes = {name: [] for name in for_comparison}

    for path in image_paths:
        eval_fname = f"{Path(path).stem}_dic.json"
        for seg_name in for_comparison:
            eval_path = os.path.join(val_sgn_dir, seg_name, eval_fname)
            with open(eval_path, "r") as f:
                result = json.load(f)
            rt = result["time"]
            runtimes[seg_name].append(rt)

    for name, rts in runtimes.items():
        print(name, ":", np.mean(rts), "+-", np.std(rts))


def runtimes_ihc(
    cochlea_dir: str = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet",
) -> None:
    for_comparison = ["distance_unet_v3", "micro-sam", "cellpose3", "cellpose-sam"]

    val_sgn_dir = f"{cochlea_dir}/predictions/val_ihc"
    image_dir = f"{cochlea_dir}/AnnotatedImageCrops/F1ValidationIHCs"

    image_paths = sorted(glob(os.path.join(image_dir, "*.tif")))

    runtimes = {name: [] for name in for_comparison}

    for path in image_paths:
        eval_fname = f"{Path(path).stem}_dic.json"
        for seg_name in for_comparison:
            eval_path = os.path.join(val_sgn_dir, seg_name, eval_fname)
            if not os.path.exists(eval_path):
                continue
            with open(eval_path, "r") as f:
                result = json.load(f)
            rt = result["time"]
            runtimes[seg_name].append(rt)

    for name, rts in runtimes.items():
        print(name, ":", np.mean(rts), "+-", np.std(rts))


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline segmentation methods.")
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None,
        help="Optional directory to save accuracy JSON files (e.g. flamingo_tools/reproducibility/model_accuracy).",
    )
    args = parser.parse_args()

    eval_all_sgn()
    eval_all_ihc()
    print_accuracy_sgn()
    print_accuracy_ihc()

    # average runtimes and standard deviation
    print("\nAverage runtimes and their standard deviation.")
    print("SGNs:")
    runtimes_sgn()
    print()
    print("IHCs:")
    runtimes_ihc()

    if args.output_dir is not None:
        sgn_seg_dir = os.path.join(COCHLEA_DIR, "predictions/val_sgn")
        ihc_seg_dir = os.path.join(COCHLEA_DIR, "predictions/val_ihc")
        save_accuracy_json("SGN", SGN_BASELINES, sgn_seg_dir, args.output_dir)
        save_accuracy_json("IHC", IHC_BASELINES, ihc_seg_dir, args.output_dir)


if __name__ == "__main__":
    main()
