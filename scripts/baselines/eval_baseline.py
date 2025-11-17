import os
import json

import imageio.v3 as imageio
import numpy as np
import pandas as pd

from flamingo_tools.validation import compute_matches_for_annotated_slice


def filter_seg(seg_arr: np.typing.ArrayLike, min_count: int = 3000, max_count: int = 50000):
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


def eval_seg_dict(dic: dict, out_path: str):
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


def eval_all_sgn():
    """Evaluate all SGN baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_sgn")
    annotation_dir = os.path.join(cochlea_dir,
                                  "AnnotatedImageCrops",
                                  "F1ValidationSGNs",
                                  "final_annotations",
                                  "final_consensus_annotations")

    baselines = [
        "spiner2D",
        "cellpose3",
        "cellpose-sam",
        "distance_unet",
        "micro-sam",
        "stardist",
    ]

    for baseline in baselines:
        if "spiner" in baseline:
            eval_segmentation_spiner(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)
            # eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir, filter=False)
        else:
            eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)


def eval_all_ihc():
    """Evaluate all IHC baselines.
    """
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")
    annotation_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationIHCs/consensus_annotation")
    baselines = [
        "cellpose3",
        "cellpose-sam",
        "distance_unet_v4b",
        "micro-sam",
    ]

    for baseline in baselines:
        eval_segmentation(os.path.join(seg_dir, baseline), annotation_dir=annotation_dir)


def eval_segmentation(seg_dir, annotation_dir, filter=True):
    """Evaluate 3D segmentation from baseline methods.

    Args:
        seg_dir: Directory containing segmentation output of baseline methods.
        annotation_dir: Directory containing annotations in CSV format.
        filter: Bool for filtering segmentation based on size. Per default size between 3000 and 50000 pixels.
    """
    print(f"Evaluating segmentation in directory {seg_dir}")
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
        else:
            print(f"Dictionary for {basename} already exists")

    json_out = os.path.join(seg_dir, "eval_seg.json")
    with open(json_out, "w") as f:
        json.dump(seg_dicts, f, indent='\t', separators=(',', ': '))


def eval_segmentation_spiner(seg_dir, annotation_dir):
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


def print_accuracy(eval_dir):
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


def print_accuracy_sgn():
    """Print 'Precision', 'Recall', and 'F1-score' for all SGN baselines.
    """
    print("Evaluating SGN segmentation")
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_sgn")
    baselines = [
        "spiner2D",
        "cellpose3",
        "cellpose-sam",
        "distance_unet",
        "micro-sam",
        "stardist"]
    for baseline in baselines:
        print(f"Evaluating baseline {baseline}")
        print_accuracy(os.path.join(seg_dir, baseline))


def print_accuracy_ihc():
    """Print 'Precision', 'Recall', and 'F1-score' for all IHC baselines.
    """
    print("Evaluating IHC segmentation")
    cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
    seg_dir = os.path.join(cochlea_dir, "predictions/val_ihc")
    baselines = [
        "cellpose3",
        "cellpose-sam",
        "distance_unet_v4b",
        "micro-sam"]

    for baseline in baselines:
        print(f"Evaluating baseline {baseline}")
        print_accuracy(os.path.join(seg_dir, baseline))


def main():
    eval_all_sgn()
    eval_all_ihc()
    print_accuracy_sgn()
    print_accuracy_ihc()


if __name__ == "__main__":
    main()
