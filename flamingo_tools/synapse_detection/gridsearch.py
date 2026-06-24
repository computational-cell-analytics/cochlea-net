"""Threshold gridsearch for synapse heatmap detection.

Replaces the czii-protein-challenge gridsearch with flamingo-tools-compatible
equivalents: CSV label files instead of CZII JSON, prediction_impl instead of
get_prediction_torch_em, and inlined metric_coords.
"""

import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max
from tqdm import tqdm

from flamingo_tools.segmentation.unet_prediction import prediction_impl

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
TRAIN_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data/v4/images")
LABEL_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data/v4/labels")
DEFAULT_JSON = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data/v4/train_val_split.json")


# ---------------------------------------------------------------------------
# Coordinate matching metric (inlined from czii evaluation_metrics.py)
# ---------------------------------------------------------------------------

def _metric_coords(gts, preds, match_distance=4):
    """Compute precision, recall, and F1 for two sets of 3-D coordinates.

    Uses Hungarian matching; a predicted point counts as a true positive when
    it lies within *match_distance* voxels of its assigned ground-truth point.

    Args:
        gts: (N, 3) array of ground-truth coordinates [z, y, x].
        preds: (M, 3) array of predicted coordinates [z, y, x].
        match_distance: Maximum voxel distance for a valid match.

    Returns:
        Tuple (precision, recall, f1).
    """
    n, m = len(gts), len(preds)
    if n == 0 and m == 0:
        return 1.0, 1.0, 1.0
    if n == 0 or m == 0:
        return 0.0, 0.0, 0.0

    dist = cdist(gts, preds, metric="euclidean")
    if not np.any(dist < match_distance):
        return 0.0, 0.0, 0.0

    max_d = dist.max()
    costs = -(dist < match_distance).astype(float) - (max_d - dist) / (max_d + 1e-8)
    row_ind, col_ind = linear_sum_assignment(costs)
    tp = int(np.count_nonzero(dist[row_ind, col_ind] < match_distance))

    precision = tp / m if m > 0 else 0.0
    recall = tp / n if n > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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
    block_shape=(64, 256, 256),
    halo=(16, 32, 32),
    min_distance=2,
    match_distance=4,
):
    """Find the peak-detection threshold that maximises F1 on the validation set.

    The JSON at *json_val_path* must contain a ``"val"`` key whose value is a
    list of image file names (e.g. ``"sample_001.zarr"``).  Images are looked
    up in *image_dir*; labels are looked up in *label_dir* after replacing the
    ``.zarr`` extension with ``.csv``.  Absolute paths in the val list are used
    directly. If no JSON file is supplied, all images in ZARR format in the directory are evaluated.

    Args:
        model_path: Path to the model checkpoint or exported model file.
        json_val_path: Path to the JSON train/val split file.
        image_dir: Directory containing the validation zarr files.
        label_dir: Directory containing the matching CSV label files.
        raw_key: Zarr key for the raw image data.
        out_channels: Number of model output channels.  Auto-detected if None.
        block_shape: Spatial block shape for tiled prediction.
        halo: Halo (overlap) for tiled prediction.
        min_distance: Minimum voxel distance between detected peaks.
        match_distance: Maximum voxel distance for a GT/pred match (in voxels).

    Returns:
        Best threshold as a float.
    """
    if out_channels is None:
        out_channels = _get_out_channels(model_path)

    if json_val_path is not None:
        with open(json_val_path) as f:
            val_list = json.load(f)["val"]
    else:
        # use every image in directory
        val_list = [entry.name.split(".zarr")[0] for entry in os.scandir(image_dir) if ".zarr" in entry.name]

    threshes = np.round(np.arange(0.3, 2.0, 0.1), 2)
    records = []

    for val_name in val_list:
        # Resolve image path.
        image_path = os.path.join(image_dir, f"{val_name}.zarr")
        # Derive label path: same basename, .csv extension, in label_dir.
        label_path = os.path.join(label_dir, f"{val_name}.csv")

        print(f"Running prediction on {image_path}")
        _, pred = prediction_impl(
            image_path, raw_key, None, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=out_channels,
        )
        # Use heatmap channel only for threshold sweep.
        heatmap = pred[0] if pred.ndim == 4 else pred

        label_coords = _load_csv_labels(label_path)
        print(f"  {len(label_coords)} ground-truth points loaded")

        for thresh in tqdm(threshes, desc="  threshold sweep"):
            pred_coords = peak_local_max(heatmap, min_distance=min_distance, threshold_abs=thresh)
            _, _, f1 = _metric_coords(label_coords, pred_coords, match_distance)
            records.append({"val": val_name, "threshold": float(thresh), "f1": float(f1)})

    # Aggregate F1 across validation samples and find the best threshold.
    df = pd.DataFrame(records)
    mean_f1 = df.groupby("threshold")["f1"].mean()
    best_thresh = float(mean_f1.idxmax())
    print(f"Best threshold: {best_thresh:.2f}  (mean F1={mean_f1.max():.3f})")
    return best_thresh


# ---------------------------------------------------------------------------
# Cached wrapper
# ---------------------------------------------------------------------------

def get_or_compute_threshold(
    model_path,
    json_val_path=DEFAULT_JSON,
    image_dir=TRAIN_ROOT,
    label_dir=LABEL_ROOT,
    **gridsearch_kwargs,
):
    """Return the cached detection threshold for *model_path*, running gridsearch if needed.

    The threshold is stored as ``<model_path_without_extension>_threshold.json``
    alongside the model file.  Subsequent calls skip the gridsearch and return
    the cached value immediately.

    Args:
        model_path: Path to the model file (used both for prediction and as the
            cache key).
        json_val_path: Path to the JSON train/val split file.
        image_dir: Directory containing validation zarr files.
        label_dir: Directory containing validation CSV label files.
        **gridsearch_kwargs: Forwarded to :func:`gridsearch` (e.g. ``block_shape``,
            ``min_distance``, ``match_distance``).

    Returns:
        Optimal detection threshold as a float.
    """
    _DEFAULT_THRESHOLD = 0.5

    cache_path = os.path.splitext(model_path)[0] + "_threshold.json"

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            threshold = json.load(f)["threshold"]
        print(f"Loaded cached threshold {threshold:.3f} from {cache_path}")
        return threshold

    threshold = gridsearch(
        model_path, json_val_path, image_dir=image_dir, label_dir=label_dir,
        **gridsearch_kwargs,
    )
    with open(cache_path, "w") as f:
        json.dump({"threshold": float(threshold)}, f, indent=2)
    print(f"Threshold {threshold:.3f} saved to {cache_path}")
    return threshold
