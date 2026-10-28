"""Selection of the peak-detection settings for the synapse heatmap.

Run it on the validation crops of a training run, never on the test crops that
scripts/validation/synapses/run_evaluation.py reports.
"""

import argparse
import json
import os
import tempfile

import numpy as np
import pandas as pd
import zarr
from skimage.feature import peak_local_max

from flamingo_tools.segmentation.synapse_detection import (
    _PREDICTION_BLOCK_SHAPE,
    _PREDICTION_HALO,
    _get_model_out_channels,
    map_and_filter_detections,
    pad_to_block_shape,
)
from flamingo_tools.segmentation.unet_prediction import prediction_impl
from flamingo_tools.synapse_detection.detection_dataset import read_csv_points
from flamingo_tools.validation import match_detections

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
DATA_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data/v11")
TRAIN_ROOT = os.path.join(DATA_ROOT, "images")
LABEL_ROOT = os.path.join(DATA_ROOT, "labels")
VOXEL_SIZE = 0.38  # µm per voxel, isotropic.

# The selection matches as the reported metric does: match_detections within 3 µm
# (_MATCH_DISTANCE in scripts/validation/synapses/run_evaluation.py), here in voxels.
MATCH_DISTANCE = 3.0 / VOXEL_SIZE
# The evaluation drops every detection further than this many µm from the IHC segmentation
# (PRODUCTION_MAX_DISTANCE in scripts/validation/synapses/prediction.py).
IHC_MAX_DISTANCE = 3.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _predict_heatmap(image_path, raw_key, model_path, out_channels):
    """Return the heatmap of the padded crop and the shape of the unpadded crop."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        padded_path, shape = pad_to_block_shape(image_path, raw_key, tmp_dir)
        _, pred = prediction_impl(
            padded_path, raw_key, None, model_path,
            scale=None, block_shape=_PREDICTION_BLOCK_SHAPE, halo=_PREDICTION_HALO,
            apply_postprocessing=False, output_channels=out_channels,
        )
    heatmap = pred[0] if pred.ndim == 4 else pred
    return heatmap, shape


def _filter_by_ihc(peaks, segmentation_path):
    """Keep the peaks within IHC_MAX_DISTANCE of the IHC segmentation, as the evaluation does."""
    f = zarr.open(segmentation_path, mode="r")
    segmentation = f["segmentation_filtered" if "segmentation_filtered" in f else "segmentation"][:]
    detections = pd.DataFrame(peaks * VOXEL_SIZE, columns=["z", "y", "x"])
    detections = map_and_filter_detections(
        segmentation, detections, max_distance=IHC_MAX_DISTANCE, voxel_size=(VOXEL_SIZE,) * 3, verbose=False,
    )
    return peaks[detections.index.values]


# ---------------------------------------------------------------------------
# Gridsearch
# ---------------------------------------------------------------------------

def gridsearch(
    model_path,
    ihc_dir,
    json_val_path=None,
    image_dir=TRAIN_ROOT,
    label_dir=LABEL_ROOT,
    raw_key="raw",
    out_channels=None,
    min_distances=(1, 2),
    thresholds=None,
):
    """Find the peak-detection threshold and min_distance that maximise F1 on the validation set.

    The JSON at *json_val_path* must contain a ``"val"`` key whose value is a
    list of image names without extension, as `train_synapse_detection.py` writes it.  Images
    are looked up in *image_dir* with a ``.zarr`` extension, labels in *label_dir* with a
    ``.csv`` extension. If no JSON file is supplied, all images in ZARR format in the directory
    are evaluated.

    The scores are summed over the crops, as in the reported metric. Each crop is padded and
    predicted once with the production block shape and halo, as in
    scripts/validation/synapses/prediction.py; the peaks of every setting come from that one
    heatmap.

    Args:
        model_path: Path to the model checkpoint or exported model file.
        ihc_dir: Folder with the IHC segmentation of each crop at `<crop>_ihc/segmentation.zarr`,
            the layout of scripts/validation/synapses/prediction.py. The detections are filtered
            by their distance to the IHCs, as in the evaluation. A radius filter around the
            annotations selects too low a threshold on the consensus crops, and no filter too
            high a threshold.
        json_val_path: Path to the JSON train/val split file.
        image_dir: Directory containing the validation zarr files.
        label_dir: Directory containing the matching CSV label files.
        raw_key: Zarr key for the raw image data.
        out_channels: Number of model output channels.  Auto-detected if None.
        min_distances: The minimum voxel distances between detected peaks to compare.
        thresholds: The absolute heatmap thresholds to compare. Default: 0.05 to 2.95 in steps
            of 0.05.

    Returns:
        The best threshold, the best min_distance, and a table with the precision, recall and F1
        of every setting.
    """
    if out_channels is None:
        out_channels = _get_model_out_channels(model_path)
    if thresholds is None:
        thresholds = np.round(np.arange(0.05, 3.0, 0.05), 2)

    if json_val_path is not None:
        with open(json_val_path) as f:
            val_list = json.load(f)["val"]
    else:
        # use every image in directory
        val_list = [entry.name.split(".zarr")[0] for entry in os.scandir(image_dir) if ".zarr" in entry.name]

    records = []
    for val_name in val_list:
        image_path = os.path.join(image_dir, f"{val_name}.zarr")
        label_path = os.path.join(label_dir, f"{val_name}.csv")

        print(f"Running prediction on {image_path}")
        heatmap, shape = _predict_heatmap(image_path, raw_key, model_path, out_channels)
        label_coords = read_csv_points(label_path)
        print(f"  {len(label_coords)} ground-truth points loaded")

        for min_distance in min_distances:
            # A peak above a higher threshold is also a peak above the lowest one, so one
            # detection per min_distance serves every threshold. The border handling and the
            # padding are those of the production peak detection; peaks in the padding are dropped.
            peaks = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=min(thresholds))
            peaks = peaks[np.all(peaks < np.array(shape), axis=1)]
            peaks = _filter_by_ihc(peaks, os.path.join(ihc_dir, f"{val_name}_ihc", "segmentation.zarr"))
            values = heatmap[tuple(peaks.T)]

            for threshold in thresholds:
                pred_coords = peaks[values >= threshold]
                records.append({
                    "min_distance": min_distance, "threshold": float(threshold),
                    "n_pred": len(pred_coords), "n_gt": len(label_coords),
                    "tp": len(match_detections(pred_coords, label_coords, MATCH_DISTANCE)[0]),
                })

    scores = pd.DataFrame(records).groupby(["min_distance", "threshold"]).sum()
    scores["precision"] = scores["tp"] / scores["n_pred"].clip(lower=1)
    scores["recall"] = scores["tp"] / scores["n_gt"].clip(lower=1)
    scores["f1"] = (
        2 * scores["precision"] * scores["recall"] / (scores["precision"] + scores["recall"])
    ).fillna(0.0)

    best_min_distance, best_threshold = scores["f1"].idxmax()
    print(scores[["precision", "recall", "f1"]].round(3).to_string())
    print(f"Best: threshold {best_threshold:.2f}, min_distance {best_min_distance} "
          f"(F1={scores['f1'].max():.3f})")
    return float(best_threshold), int(best_min_distance), scores


def main():
    parser = argparse.ArgumentParser(
        description="Select the synapse detection threshold and min_distance on the validation crops."
    )
    parser.add_argument("-m", "--model", required=True, help="The model file or trainer checkpoint.")
    parser.add_argument("-s", "--split", default=os.path.join(DATA_ROOT, "train_val_split.json"),
                        help="The train_val_split*.json of the training run. Default: the split of the v11 data.")
    parser.add_argument("-i", "--image_dir", default=TRAIN_ROOT)
    parser.add_argument("-l", "--label_dir", default=LABEL_ROOT)
    parser.add_argument("--ihc_dir", required=True,
                        help="Folder with <crop>_ihc/segmentation.zarr for each crop, as prediction.py writes "
                             "it. Filters the detections by their distance to the IHCs, as the evaluation does.")
    args = parser.parse_args()
    gridsearch(
        args.model, args.ihc_dir, json_val_path=args.split, image_dir=args.image_dir, label_dir=args.label_dir,
    )


if __name__ == "__main__":
    main()
