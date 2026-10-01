"""Selection of the peak-detection settings for the synapse heatmap.

Replaces the czii-protein-challenge gridsearch with flamingo-tools-compatible
equivalents: CSV label files instead of CZII JSON, prediction_impl instead of
get_prediction_torch_em, and inlined metric_coords.

Run it on the validation crops of a training run, never on the test crops that
scripts/validation/synapses/run_evaluation.py reports.
"""

import argparse
import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max

from flamingo_tools.segmentation.synapse_detection import (
    _PREDICTION_BLOCK_SHAPE,
    _PREDICTION_HALO,
    pad_to_block_shape,
)
from flamingo_tools.segmentation.unet_prediction import prediction_impl

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
TRAIN_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data/v7/images")
LABEL_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data/v7/labels")
VOXEL_SIZE = 0.38  # µm per voxel, isotropic.
_DEFAULT_PARAMS = {"threshold": 0.5, "min_distance": 2}

# The reported metric matches a detection to an annotation within 3 µm
# (_MATCH_DISTANCE in scripts/validation/synapses/run_evaluation.py), so the selection does too.
MATCH_DISTANCE = 3.0 / VOXEL_SIZE
# The former selection radius, 1.52 µm. It is printed alongside, because it separates adjacent
# synapses more sharply: 42 % of the annotations have a neighbour within 2 µm.
STRICT_MATCH_DISTANCE = 4
# The training crops are annotated only around the IHCs. A detection further than this from every
# annotation is not scored, which is the "spatially distant" class of the 2026-08-27 diagnostics
# in to-do_revision/synapses.md.
ANNOTATED_RADIUS = 40

# ---------------------------------------------------------------------------
# Coordinate matching metric (inlined from czii evaluation_metrics.py)
# ---------------------------------------------------------------------------


def _true_positives(gts, preds, match_distance):
    """Count the one-to-one matches within *match_distance* voxels (Hungarian matching).

    Args:
        gts: (N, 3) array of ground-truth coordinates [z, y, x].
        preds: (M, 3) array of predicted coordinates [z, y, x].
        match_distance: Maximum voxel distance for a valid match.

    Returns:
        The number of true positives.
    """
    if len(gts) == 0 or len(preds) == 0:
        return 0

    dist = cdist(gts, preds, metric="euclidean")
    if not np.any(dist < match_distance):
        return 0

    max_d = dist.max()
    costs = -(dist < match_distance).astype(float) - (max_d - dist) / (max_d + 1e-8)
    row_ind, col_ind = linear_sum_assignment(costs)
    return int(np.count_nonzero(dist[row_ind, col_ind] < match_distance))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_csv_labels(label_path):
    """Return (N, 3) voxel coordinate array [z, y, x] from a napari CSV file."""
    df = pd.read_csv(label_path)
    return np.stack([df["axis-0"].values, df["axis-1"].values, df["axis-2"].values], axis=1)


def _get_out_channels(model_path):
    """Return the number of output channels from a model file or trainer checkpoint."""
    try:
        import sys
        import flamingo_tools.synapse_detection.detection_dataset as _dd
        sys.modules.setdefault("detection_dataset", _dd)
    except ImportError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import torch
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["init"]["model_kwargs"].get("out_channels", 1)
    return obj.state_dict()["out_conv.bias"].shape[0]


def _predict_heatmap(image_path, raw_key, model_path, out_channels, block_shape, halo):
    with tempfile.TemporaryDirectory() as tmp_dir:
        padded_path, shape = pad_to_block_shape(image_path, raw_key, tmp_dir)
        _, pred = prediction_impl(
            padded_path, raw_key, None, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=out_channels,
        )
    heatmap = pred[0] if pred.ndim == 4 else pred
    return heatmap[: shape[0], : shape[1], : shape[2]]


# ---------------------------------------------------------------------------
# Gridsearch
# ---------------------------------------------------------------------------

def gridsearch(
    model_path,
    json_val_path=None,
    image_dir=TRAIN_ROOT,
    label_dir=LABEL_ROOT,
    raw_key="raw",
    out_channels=None,
    block_shape=_PREDICTION_BLOCK_SHAPE,
    halo=_PREDICTION_HALO,
    min_distances=(1, 2),
    thresholds=None,
    annotated_radius=ANNOTATED_RADIUS,
):
    """Find the peak-detection threshold and min_distance that maximise F1 on the validation set.

    The JSON at *json_val_path* must contain a ``"val"`` key whose value is a
    list of image names without extension, as `train_synapse_detection.py` writes it.  Images
    are looked up in *image_dir* with a ``.zarr`` extension, labels in *label_dir* with a
    ``.csv`` extension. If no JSON file is supplied, all images in ZARR format in the directory
    are evaluated.

    The scores are summed over the crops, as in the reported metric. Each crop is predicted once
    with the production block shape and halo; the peaks of every setting come from that one
    heatmap.

    Args:
        model_path: Path to the model checkpoint or exported model file.
        json_val_path: Path to the JSON train/val split file.
        image_dir: Directory containing the validation zarr files.
        label_dir: Directory containing the matching CSV label files.
        raw_key: Zarr key for the raw image data.
        out_channels: Number of model output channels.  Auto-detected if None.
        block_shape: Spatial block shape for tiled prediction.
        halo: Halo (overlap) for tiled prediction.
        min_distances: The minimum voxel distances between detected peaks to compare.
        thresholds: The absolute heatmap thresholds to compare. Default: 0.3 to 1.9 in steps of 0.1.
        annotated_radius: Detections further than this many voxels from every annotation are not
            scored. None scores every detection.

    Returns:
        The best threshold, the best min_distance, and a table with the precision, recall and F1
        of every setting at the reported and at the strict match distance.
    """
    if out_channels is None:
        out_channels = _get_out_channels(model_path)
    if thresholds is None:
        thresholds = np.round(np.arange(0.3, 2.0, 0.1), 2)

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
        heatmap = _predict_heatmap(image_path, raw_key, model_path, out_channels, block_shape, halo)
        label_coords = _load_csv_labels(label_path)
        print(f"  {len(label_coords)} ground-truth points loaded")

        for min_distance in min_distances:
            # A peak above a higher threshold is also a peak above the lowest one, so one
            # detection per min_distance serves every threshold.
            peaks = peak_local_max(
                heatmap, min_distance=min_distance, threshold_abs=min(thresholds), exclude_border=False,
            )
            if annotated_radius is not None:
                if len(label_coords) == 0:
                    peaks = peaks[:0]
                else:
                    peaks = peaks[cKDTree(label_coords).query(peaks)[0] <= annotated_radius]
            values = heatmap[tuple(peaks.T)]

            for threshold in thresholds:
                pred_coords = peaks[values >= threshold]
                records.append({
                    "min_distance": min_distance, "threshold": float(threshold),
                    "n_pred": len(pred_coords), "n_gt": len(label_coords),
                    "tp": _true_positives(label_coords, pred_coords, MATCH_DISTANCE),
                    "tp_strict": _true_positives(label_coords, pred_coords, STRICT_MATCH_DISTANCE),
                })

    scores = pd.DataFrame(records).groupby(["min_distance", "threshold"]).sum()
    for suffix, tp in (("", scores["tp"]), ("_strict", scores["tp_strict"])):
        precision = tp / scores["n_pred"].clip(lower=1)
        recall = tp / scores["n_gt"].clip(lower=1)
        scores[f"precision{suffix}"] = precision
        scores[f"recall{suffix}"] = recall
        scores[f"f1{suffix}"] = (2 * precision * recall / (precision + recall)).fillna(0.0)

    best_min_distance, best_threshold = scores["f1"].idxmax()
    print(scores[["precision", "recall", "f1", "f1_strict"]].round(3).to_string())
    print(f"Best: threshold {best_threshold:.2f}, min_distance {best_min_distance} "
          f"(F1={scores['f1'].max():.3f})")
    return float(best_threshold), int(best_min_distance), scores


# ---------------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------------

def get_or_compute_detection_params(
    model_path,
    json_val_path=None,
    image_dir=TRAIN_ROOT,
    label_dir=LABEL_ROOT,
    **gridsearch_kwargs,
):
    """Return the cached detection settings for *model_path*, running the gridsearch if needed.

    The settings are stored as ``<model_path_without_extension>_detection_params.json``
    alongside the model file.  Subsequent calls skip the gridsearch and return
    the cached values immediately.

    Args:
        model_path: Path to the model file (used both for prediction and as the
            cache key).
        json_val_path: Path to the JSON train/val split file.
        image_dir: Directory containing validation zarr files.
        label_dir: Directory containing validation CSV label files.
        **gridsearch_kwargs: Forwarded to :func:`gridsearch` (e.g. ``block_shape``,
            ``min_distances``, ``annotated_radius``).

    Returns:
        A dict with the detection ``threshold`` and ``min_distance``. The production defaults
        when the validation data is not available.
    """
    cache_path = os.path.splitext(model_path)[0] + "_detection_params.json"

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            params = json.load(f)
        print(f"Loaded cached detection settings {params} from {cache_path}")
        return params

    if not os.path.isdir(image_dir):
        return dict(_DEFAULT_PARAMS)

    threshold, min_distance, _ = gridsearch(
        model_path, json_val_path, image_dir=image_dir, label_dir=label_dir,
        **gridsearch_kwargs,
    )
    params = {"threshold": threshold, "min_distance": min_distance}
    with open(cache_path, "w") as f:
        json.dump(params, f, indent=2)
    print(f"Detection settings {params} saved to {cache_path}")
    return params


def main():
    parser = argparse.ArgumentParser(
        description="Select the synapse detection threshold and min_distance on the validation crops."
    )
    parser.add_argument("-m", "--model", required=True, help="The model file or trainer checkpoint.")
    parser.add_argument("-s", "--split", default=None,
                        help="The train_val_split*.json of the training run. Default: every crop in --image_dir.")
    parser.add_argument("-i", "--image_dir", default=TRAIN_ROOT)
    parser.add_argument("-l", "--label_dir", default=LABEL_ROOT)
    args = parser.parse_args()
    gridsearch(args.model, args.split, image_dir=args.image_dir, label_dir=args.label_dir)


if __name__ == "__main__":
    main()
