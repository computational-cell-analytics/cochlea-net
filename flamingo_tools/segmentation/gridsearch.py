"""Gridsearch for IHC distance-watershed segmentation parameters.

Predictions are computed once per image and kept in memory; the watershed
step is then swept over all combinations of ``center_distance_threshold``,
``boundary_distance_threshold``, and ``distance_smoothing``.  Per-image
results are written as JSON so that interrupted runs can be resumed cheaply.
"""
import itertools
import json
import os
from glob import glob

import imageio.v3 as imageio
import numpy as np
from elf.evaluation.dice import symmetric_best_dice_score
from tqdm import tqdm

from flamingo_tools.segmentation.unet_prediction import distance_watershed_implementation, prediction_impl

DEFAULT_GRID_SEARCH_VALUES = {
    "center_distance_threshold": [0.4, 0.5, 0.6, 0.7],
    "boundary_distance_threshold": [0.4, 0.5, 0.6, 0.7],
    "distance_smoothing": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}

_ANNOTATION_SUFFIX = "_annotations"


def _find_image_label_pairs(val_dir, annotation_suffix=_ANNOTATION_SUFFIX):
    """Return sorted list of (image_path, label_path) TIF pairs from *val_dir*."""
    all_tifs = sorted(glob(os.path.join(val_dir, "*.tif")))
    pairs = []
    for path in all_tifs:
        stem = os.path.splitext(os.path.basename(path))[0]
        if stem.endswith(annotation_suffix):
            continue
        label_path = os.path.join(val_dir, stem + annotation_suffix + ".tif")
        if os.path.exists(label_path):
            pairs.append((path, label_path))
    return pairs


def gridsearch(
    val_dir,
    model_path,
    result_dir=None,
    annotation_suffix=_ANNOTATION_SUFFIX,
    grid_search_values=None,
    block_shape=None,
    halo=None,
    min_size=0,
    fg_threshold=0.5,
):
    """Find watershed parameters that maximise symmetric best-Dice on a validation set.

    For each TIF image in *val_dir* (paired with ``<stem>_annotations.tif``),
    the model prediction is computed once and held in memory.  The watershed is
    then applied for every combination of gridsearch parameters, scored with
    :func:`elf.evaluation.dice.symmetric_best_dice_score`, and the result is
    written to ``<result_dir>/<image_stem>.json``.  Existing JSON files are
    re-used so the gridsearch can be resumed after interruption.

    Args:
        val_dir: Directory containing TIF image / annotation pairs.
        model_path: Path to the model checkpoint or exported model file.
        result_dir: Directory for per-image JSON result files.  No files are
            written when ``None``.
        annotation_suffix: Suffix that distinguishes label files from image
            files (default ``"_annotations"``).
        grid_search_values: Dict mapping parameter names to lists of values to
            sweep.  Defaults to :data:`DEFAULT_GRID_SEARCH_VALUES`.
        block_shape: Block shape for tiled prediction (auto-detected when None).
        halo: Halo for tiled prediction (auto-detected when None).
        min_size: Minimum object size passed to the watershed (default 0).
        fg_threshold: Foreground threshold for the watershed mask.

    Returns:
        Tuple ``(best_params, best_score)`` where *best_params* is a dict of
        parameter names to values and *best_score* is the mean
        symmetric-best-Dice across all validation images.
    """
    if grid_search_values is None:
        grid_search_values = DEFAULT_GRID_SEARCH_VALUES

    if result_dir is not None:
        os.makedirs(result_dir, exist_ok=True)

    pairs = _find_image_label_pairs(val_dir, annotation_suffix)
    if not pairs:
        raise ValueError(f"No image/label TIF pairs found in {val_dir!r} (annotation suffix: {annotation_suffix!r})")

    param_names = list(grid_search_values.keys())
    combos = list(itertools.product(*[grid_search_values[k] for k in param_names]))
    print(
        f"Gridsearch: {len(combos)} parameter combinations x {len(pairs)} images "
        f"= {len(combos) * len(pairs)} evaluations"
    )

    # records: list of dicts {param_key_0: v0, ..., "dice": score} per image per combo
    all_records = []

    for image_path, label_path in pairs:
        image_stem = os.path.splitext(os.path.basename(image_path))[0]

        if result_dir is not None:
            cache_path = os.path.join(result_dir, f"{image_stem}.json")
            if os.path.exists(cache_path):
                print(f"  {image_stem}: loading cached results from {cache_path}")
                with open(cache_path) as fh:
                    image_records = json.load(fh)
                all_records.extend(image_records)
                continue

        print(f"\n{image_stem}: running prediction ...")
        _, prediction = prediction_impl(
            image_path, None, None, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=3,
        )

        gt = imageio.imread(label_path)

        image_records = []
        for combo in tqdm(combos, desc=f"  {image_stem} sweep"):
            params = dict(zip(param_names, combo))
            seg = distance_watershed_implementation(
                prediction,
                output_folder=None,
                min_size=min_size,
                fg_threshold=fg_threshold,
                **params,
            )
            score = symmetric_best_dice_score(seg, gt)
            image_records.append({**params, "dice": float(score)})

        if result_dir is not None:
            with open(cache_path, "w") as fh:
                json.dump(image_records, fh, indent=2)

        all_records.extend(image_records)

    # Average dice over images for each parameter combination.
    combo_totals = {}
    combo_counts = {}
    for record in all_records:
        key = tuple(record[k] for k in param_names)
        combo_totals[key] = combo_totals.get(key, 0.0) + record["dice"]
        combo_counts[key] = combo_counts.get(key, 0) + 1

    best_key = max(combo_totals, key=lambda k: combo_totals[k] / combo_counts[k])
    best_params = dict(zip(param_names, best_key))
    best_score = combo_totals[best_key] / combo_counts[best_key]

    print(f"\nBest parameters: {best_params}")
    print(f"Mean symmetric best-Dice: {best_score:.4f}")
    return best_params, float(best_score)


def get_or_compute_best_params(
    model_path,
    val_dir,
    annotation_suffix=_ANNOTATION_SUFFIX,
    grid_search_values=None,
    **gridsearch_kwargs,
):
    """Return the cached best watershed parameters for *model_path*, running gridsearch if needed.

    The result is stored as ``<model_path_without_extension>_best_params.json``
    alongside the model file.  Subsequent calls skip the gridsearch and return
    the cached values immediately.

    Args:
        model_path: Path to the model file (used as the cache key).
        val_dir: Directory containing TIF image / annotation pairs.
        annotation_suffix: Suffix distinguishing label files from image files.
        grid_search_values: Parameter sweep dict forwarded to :func:`gridsearch`.
        **gridsearch_kwargs: Additional keyword arguments forwarded to
            :func:`gridsearch` (e.g. ``block_shape``, ``halo``, ``min_size``).

    Returns:
        Tuple ``(best_params, best_score)``.
    """
    cache_path = os.path.splitext(model_path)[0] + "_best_params.json"

    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            data = json.load(fh)
        print(f"Loaded cached best params from {cache_path}")
        return data["params"], data["score"]

    best_params, best_score = gridsearch(
        val_dir, model_path,
        annotation_suffix=annotation_suffix,
        grid_search_values=grid_search_values,
        **gridsearch_kwargs,
    )

    with open(cache_path, "w") as fh:
        json.dump({"params": best_params, "score": best_score}, fh, indent=2)
    print(f"Best params saved to {cache_path}")
    return best_params, best_score
