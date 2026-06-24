import json
import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from elf.io import open_file

from flamingo_tools.validation import match_detections

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
SYNAPSE_DICT = {
    "v3": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v3"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
    "v3_05t": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v3_05t"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
    "v4": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v4"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
    "v5": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v5"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
    "v5_f1val_threshold": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v5_f1val_threshold"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
    "v5_train_threshold": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v5_train_threshold"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
    "v5_05t": {
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v5_05t"),
        "ref_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels"),
        "image_root": os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images"),
    },
}

ANNOTATION_DIR = os.path.join(COCHLEA_DIR, "AnnotatedImageCrops/Synapses_2026-04")
INDIVIDUAL_ANNOTATORS = {
    "AMD": os.path.join(ANNOTATION_DIR, "for_consensus_annotations_synapses_AMD/labels"),
    "EK": os.path.join(ANNOTATION_DIR, "for_consensus_annotations_synapses_EK/labels"),
    "LR": os.path.join(ANNOTATION_DIR, "for_consensus_annotations_synapses_LR/labels"),
}


def _save_match_array(pred, gt, tps_pred, fps, fns, voxel_size, save_path):
    """Save a labelled uint8 array marking TP (1), FP (2), and FN (3) synapse positions.

    Array shape is determined by the maximum voxel coordinate across all marked points.
    Each marked point occupies a single pixel.
    """

    def to_voxel(coords):
        return np.round(np.asarray(coords) / voxel_size).astype(int)

    pred_vox = to_voxel(pred)
    gt_vox = to_voxel(gt)

    coord_groups = []
    if len(tps_pred):
        coord_groups.append(pred_vox[tps_pred])
    if len(fps):
        coord_groups.append(pred_vox[fps])
    if len(fns):
        coord_groups.append(gt_vox[fns])

    if not coord_groups:
        return

    all_coords = np.vstack(coord_groups)
    shape = tuple(all_coords.max(axis=0) + 1)
    arr = np.zeros(shape, dtype=np.uint32)

    for idx in pred_vox[tps_pred]:
        arr[idx[0], idx[1], idx[2]] = 1
    for idx in pred_vox[fps]:
        arr[idx[0], idx[1], idx[2]] = 2
    for idx in gt_vox[fns]:
        arr[idx[0], idx[1], idx[2]] = 3

    out_dir = os.path.dirname(save_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    imageio.imwrite(save_path, arr, compression="zlib")


def evaluate_synapse_detections(
    pred_path: str,
    gt_path: str,
    voxel_size: float = 0.38,
    match_array_path: str = None,
) -> pd.DataFrame:
    """Evaluate synapse detections by comparing a prediction from the spot detection to reference labels.

    Args:
        pred_path: File containing the "synapse_detection.tsv" output from the spot detection.
        gt_path: File containing the reference label from a consensus annotation.
        voxel_size: Voxel size of the image volume with isotropic resolution.
        match_array_path: Optional path to save a uint8 TIF array marking TP (1), FP (2),
            and FN (3) positions. Array size is determined by the maximum coordinate value.

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

    if match_array_path is not None:
        _save_match_array(pred, gt, tps_pred, fps, fns, voxel_size, match_array_path)

    return pd.DataFrame({
        "name": [fname], "tp": [len(tps_pred)], "fp": [len(fps)], "fn": [len(fns)],
    })


def run_evaluation(pred_files, gt_files, output_file=None, version_key=None, match_array_dir=None):
    results = []
    for pred, gt in zip(pred_files, gt_files):
        match_array_path = None
        if match_array_dir is not None:
            crop_stem = Path(gt).stem
            match_array_path = os.path.join(match_array_dir, f"{crop_stem}_match_array.tif")
        res = evaluate_synapse_detections(pred, gt, match_array_path=match_array_path)
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

    if output_file is not None and version_key is not None:
        data = {}
        if os.path.isfile(output_file):
            with open(output_file, "r") as f:
                data = json.load(f)

        data[version_key] = {
            "crops": results["name"].tolist(),
            "tp": [int(v) for v in results["tp"].tolist()],
            "fp": [int(v) for v in results["fp"].tolist()],
            "fn": [int(v) for v in results["fn"].tolist()],
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1-score": round(float(f1_score), 3),
        }

        out_dir = os.path.dirname(output_file)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(data, f, indent="\t", separators=(",", ": "))
        print(f"Saved results to {output_file}")


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
    parser.add_argument("-r", "--ref_root", type=str, default=None, nargs="+",
                        help="Directory containing reference labels in CSV format.")
    parser.add_argument("-c", "--image_root", type=str,
                        help="Directory containing image data in ZARR format.")
    parser.add_argument("--sgn_version", type=str, default="SGN_v2",
                        help="SGN segmentation version.")
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Optional directory to save accuracy JSON file (synapses.json).")
    parser.add_argument("--match_array_dir", type=str, default=None,
                        help="Optional directory to save per-crop uint8 TIF arrays marking "
                             "TP (1), FP (2), and FN (3) synapse positions.")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()

    if args.version is not None:
        valid_versions = list(SYNAPSE_DICT.keys())
        if args.version not in valid_versions:
            raise ValueError(f"Version {args.version} is not supported. Supported versions: {valid_versions}")

        image_root = SYNAPSE_DICT[args.version]["image_root"]
        if args.ref_root is None:
            ref_roots = [SYNAPSE_DICT[args.version]["ref_root"]]
        else:
            ref_roots = args.ref_root
        pred_root = SYNAPSE_DICT[args.version]["pred_root"]

    else:
        image_root = args.image_root
        ref_roots = args.ref_root
        pred_root = args.pred_root

    for ref_root in ref_roots:
        print(f"Analysing prediction with respect to labels in {ref_root}.")
        if len(ref_roots) == 1:
            version_key = args.version if args.version is not None else Path(pred_root).name
        else:
            version_key = f"{Path(pred_root).name}_{ref_root.split(os.path.sep)[-2]}"

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
            output_file = None
            if args.output_dir is not None:
                output_file = os.path.join(args.output_dir, "synapses.json")

            run_evaluation(
                pred_files, gt_files,
                output_file=output_file, version_key=version_key,
                match_array_dir=args.match_array_dir,
            )


if __name__ == "__main__":
    main()
