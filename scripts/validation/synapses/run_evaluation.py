import os
from glob import glob
from pathlib import Path

import pandas as pd
from elf.io import open_file

from flamingo_tools.validation import match_detections

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
SYNAPSE_DICT = {
    "v3": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v3"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v4/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v4/images"),
    },
    "v4": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v4"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v4/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v4/images"),
    },
    "v5": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v5"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v4/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v4/images"),
    },
}


def evaluate_synapse_detections(
    pred_path: str,
    gt_path: str,
    voxel_size: str = 0.38,
) -> pd.DataFrame:
    """Evaluate synapse detections by comparing a prediction from the spot detection to reference labels.

    Args:
        pred_path: File containing the "synapse_detection.tsv" output from the spot detection.
        gt_path: File containing the reference label from a consensus annotation.
        voxel_size: Voxel size of the image volume with isotropic resolution.

    Returns:
        Printed output of model performance.
    """
    fname = os.path.basename(gt_path)

    pred = pd.read_csv(pred_path, sep="\t")[["z", "y", "x"]].values

    # scale gt to physical coordinates
    gt = pd.read_csv(gt_path, sep=",")
    gt["axis-0"] *= voxel_size
    gt["axis-1"] *= voxel_size
    gt["axis-2"] *= voxel_size
    gt = gt[["axis-0", "axis-1", "axis-2"]].values

    tps_pred, tps_gt, fps, fns = match_detections(pred, gt, max_dist=3)

    return pd.DataFrame({
        "name": [fname], "tp": [len(tps_pred)], "fp": [len(fps)], "fn": [len(fns)],
    })


def run_evaluation(pred_files, gt_files):
    results = []
    for pred, gt in zip(pred_files, gt_files):
        res = evaluate_synapse_detections(pred, gt)
        results.append(res)
    results = pd.concat(results)

    tp = results.tp.sum()
    fp = results.fp.sum()
    fn = results.fn.sum()

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * precision * recall / (precision + recall)

    print("All results:")
    print(results)
    print("Evaluation:")
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1_score)


def visualize_synapse_detections(pred, gt, heatmap_path=None, ctbp2_path=None):
    import napari

    fname = os.path.basename(gt)

    pred = pd.read_csv(pred, sep="\t")[["z", "y", "x"]].values
    gt = pd.read_csv(gt, sep="\t")[["z", "y", "x"]].values
    tps_pred, tps_gt, fps, fns = match_detections(pred, gt, max_dist=4)

    tps = pred[tps_pred]
    fps = pred[fps]
    fns = gt[fns]

    if heatmap_path is None:
        heatmap = None
    else:
        heatmap = open_file(heatmap_path)["prediction"][:]

    if ctbp2_path is None:
        ctbp2 = None
    else:
        ctbp2 = open_file(ctbp2_path)["raw"][:]

    v = napari.Viewer()
    if ctbp2 is not None:
        v.add_image(ctbp2)
    if heatmap is not None:
        v.add_image(heatmap)
    v.add_points(pred, visible=False)
    v.add_points(gt, visible=False)
    v.add_points(tps, name="TPS", face_color="green")
    v.add_points(fps, name="FPs", face_color="orange")
    v.add_points(fns, name="FNs", face_color="yellow")
    v.title = f"{fname}: tps={len(tps)}, fps={len(fps)}, fns={len(fns)}"
    napari.run()


def visualize_evaluation(pred_files, gt_files, ctbp2_files):
    for pred, gt, ctbp2 in zip(pred_files, gt_files, ctbp2_files):
        pred_folder = os.path.split(pred)[0]
        heatmap = os.path.join(pred_folder, "predictions.zarr")
        visualize_synapse_detections(pred, gt, heatmap, ctbp2)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate a synapse detection model.")

    parser.add_argument("-v", "--version", type=str, default=None,
                        help="Use pre-defined directories for a specific network version, e.g. v3, v4, ...")
    parser.add_argument("-p", "--pred_root", type=str, default=None,
                        help="Directory containing sub-directories with predicted data.")
    parser.add_argument("-r", "--ref_root", type=str, default=None,
                        help="Directory containing reference labels in CSV format.")
    parser.add_argument("-c", "--image_root", type=str,
                        help="Directory containing image data in ZARR format.")
    parser.add_argument("--sgn_version", type=str, default="SGN_v2",
                        help="SGN segmentation version.")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if args.version is not None:
        valid_versions = list(SYNAPSE_DICT.keys())
        if args.version not in valid_versions:
            raise ValueError(f"Version {args.version} is not supported. Supported versions: {valid_versions}")

        image_root = SYNAPSE_DICT[args.version]["image_root"]
        ref_root = SYNAPSE_DICT[args.version]["ref_root"]
        pred_root = SYNAPSE_DICT[args.version]["pred_root"]

    else:
        image_root = args.image_root
        ref_root = args.ref_root
        pred_root = args.pred_root

    ctbp2_files = sorted(glob(os.path.join(image_root, "*.zarr")))
    gt_files = sorted(glob(os.path.join(ref_root, "*.csv")))

    pred_files = []
    for ff in ctbp2_files:
        fname = Path(ff).stem
        pred_file = os.path.join(pred_root, fname, "filtered_synapse_detection.tsv")
        if not os.path.isfile(pred_file):
            pred_file = os.path.join(pred_root, fname, "synapse_detection.tsv")

        assert os.path.exists(pred_file), pred_file
        pred_files.append(pred_file)

    if args.visualize:
        visualize_evaluation(pred_files, gt_files, ctbp2_files)
    else:
        run_evaluation(pred_files, gt_files)


if __name__ == "__main__":
    main()
