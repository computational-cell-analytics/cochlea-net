"""Compute independent pairwise agreement between synapse annotators."""

import argparse
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from flamingo_tools.validation import average_scores_per_row, match_detections


ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/Synapses_2026-04"
DEFAULT_ANNOTATIONS = {
    "AMD": os.path.join(ROOT, "for_consensus_annotations_synapses_AMD/labels"),
    "EK": os.path.join(ROOT, "for_consensus_annotations_synapses_EK/labels"),
    "LR": os.path.join(ROOT, "for_consensus_annotations_synapses_LR/labels"),
}
VOXEL_SIZE = 0.38
COORDINATE_COLUMNS = ["axis-0", "axis-1", "axis-2"]


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else np.nan


def _sample_key(path: str) -> str:
    """Return the shared crop identifier, excluding annotator-specific suffixes."""
    return "_".join(Path(path).stem.split("_")[:3])


def _index_annotations(annotation_dir: str) -> Dict[str, str]:
    """Index annotation CSVs by the crop identifier used across annotators."""
    paths_by_key = {}
    for path in sorted(Path(annotation_dir).glob("*.csv")):
        key = _sample_key(str(path))
        if key in paths_by_key:
            raise ValueError(
                f"Multiple annotation files in {annotation_dir!r} have the crop key {key!r}: "
                f"{paths_by_key[key]!r} and {str(path)!r}"
            )
        paths_by_key[key] = str(path)
    if not paths_by_key:
        raise ValueError(f"No annotation CSV files found in {annotation_dir!r}")
    return paths_by_key


def _read_coordinates(path: str, voxel_size: float) -> np.ndarray:
    """Read voxel coordinates and return physical coordinates in µm."""
    coordinates = pd.read_csv(path, usecols=COORDINATE_COLUMNS).values.astype(float)
    return coordinates * voxel_size


def _agreement_metrics(n_a: int, n_b: int, n_matches: int) -> dict:
    """Compute the agreement scores for one annotator pair.

    Precision and recall are directional: they treat annotator a as the prediction and
    annotator b as the reference. They swap when the two annotators are exchanged. The
    F1-score is the harmonic mean of the two and does not depend on the direction.
    """
    n_unmatched_a = n_a - n_matches
    n_unmatched_b = n_b - n_matches
    return {
        "n_annotations_a": n_a,
        "n_annotations_b": n_b,
        "n_matches": n_matches,
        "n_unmatched_a": n_unmatched_a,
        "n_unmatched_b": n_unmatched_b,
        "precision": _safe_ratio(n_matches, n_a),
        "recall": _safe_ratio(n_matches, n_b),
        "f1-score": _safe_ratio(2 * n_matches, n_a + n_b),
        "jaccard": _safe_ratio(n_matches, n_a + n_b - n_matches),
    }


def compute_pairwise_agreement(
    annotation_dirs: Dict[str, str],
    matching_distance: float = 2.0,
    voxel_size: float = VOXEL_SIZE,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-crop and pooled pairwise agreement for all annotator pairs.

    Args:
        annotation_dirs: Mapping from annotator name to a directory of annotation CSVs.
        matching_distance: Maximum one-to-one matching distance in µm.
        voxel_size: Isotropic voxel size in µm.

    Returns:
        A per-crop results table and a summary table pooled by annotator pair.
    """
    if len(annotation_dirs) < 2:
        raise ValueError("At least two annotators are required")
    if matching_distance < 0:
        raise ValueError("matching_distance must be non-negative")
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")

    indexed_annotations = {
        annotator: _index_annotations(annotation_dir)
        for annotator, annotation_dir in annotation_dirs.items()
    }

    records: List[dict] = []
    for annotator_a, annotator_b in combinations(sorted(annotation_dirs), 2):
        annotations_a = indexed_annotations[annotator_a]
        annotations_b = indexed_annotations[annotator_b]
        shared_crops = sorted(set(annotations_a) & set(annotations_b))
        if not shared_crops:
            raise ValueError(f"Annotators {annotator_a!r} and {annotator_b!r} have no shared crops")

        missing_from_a = sorted(set(annotations_b) - set(annotations_a))
        missing_from_b = sorted(set(annotations_a) - set(annotations_b))
        if missing_from_a:
            print(f"Warning: {annotator_a} is missing {len(missing_from_a)} crops present for {annotator_b}.")
        if missing_from_b:
            print(f"Warning: {annotator_b} is missing {len(missing_from_b)} crops present for {annotator_a}.")

        for crop in shared_crops:
            coordinates_a = _read_coordinates(annotations_a[crop], voxel_size)
            coordinates_b = _read_coordinates(annotations_b[crop], voxel_size)
            matched_a, matched_b, _, _ = match_detections(
                coordinates_a, coordinates_b, max_dist=matching_distance,
            )
            match_distances = np.linalg.norm(
                coordinates_a[matched_a] - coordinates_b[matched_b], axis=1,
            )

            record = {
                "annotator_a": annotator_a,
                "annotator_b": annotator_b,
                "crop": crop,
                "matching_distance_um": matching_distance,
                "voxel_size_um": voxel_size,
                **_agreement_metrics(len(coordinates_a), len(coordinates_b), len(matched_a)),
                "mean_match_distance_um": (
                    float(match_distances.mean()) if len(match_distances) else np.nan
                ),
                "max_match_distance_um": (
                    float(match_distances.max()) if len(match_distances) else np.nan
                ),
            }
            records.append(record)

    per_crop = pd.DataFrame(records)
    summaries = []
    for (annotator_a, annotator_b), group in per_crop.groupby(
        ["annotator_a", "annotator_b"], sort=True,
    ):
        n_a = int(group["n_annotations_a"].sum())
        n_b = int(group["n_annotations_b"].sum())
        n_matches = int(group["n_matches"].sum())
        n_matches_per_crop = group["n_matches"].to_numpy()
        mean_distances = group["mean_match_distance_um"].to_numpy()
        have_matches = n_matches_per_crop > 0
        pooled_mean_distance = (
            float(np.average(mean_distances[have_matches], weights=n_matches_per_crop[have_matches]))
            if have_matches.any()
            else np.nan
        )

        summaries.append({
            "annotator_a": annotator_a,
            "annotator_b": annotator_b,
            "n_crops": len(group),
            "matching_distance_um": matching_distance,
            "voxel_size_um": voxel_size,
            **_agreement_metrics(n_a, n_b, n_matches),
            "macro_precision": float(group["precision"].mean()),
            "macro_recall": float(group["recall"].mean()),
            "macro_f1-score": float(group["f1-score"].mean()),
            "mean_match_distance_um": pooled_mean_distance,
            "max_match_distance_um": float(group["max_match_distance_um"].max()),
        })

    return per_crop, pd.DataFrame(summaries)


def average_pairwise_scores(per_crop: pd.DataFrame) -> Dict[str, Optional[float]]:
    """Average the pairwise agreement over all ordered annotator pairs and crops.

    The per-crop table holds one row per unordered pair, with the counts of the direction
    "a as prediction". This function adds the reverse direction of every row, so that neither
    annotator is treated as the reference. Precision for the pair (a, b) is the recall for the
    pair (b, a), so the returned precision and recall are equal by construction. The F1-score
    differs from them, because it is a harmonic instead of an arithmetic mean.

    Args:
        per_crop: Per-crop table returned by compute_pairwise_agreement.

    Returns:
        Dictionary with the mean precision, recall, and F1-score.
    """
    forward = per_crop.rename(
        columns={"n_matches": "tps", "n_unmatched_a": "fps", "n_unmatched_b": "fns"},
    )
    reverse = per_crop.rename(
        columns={"n_matches": "tps", "n_unmatched_b": "fps", "n_unmatched_a": "fns"},
    )
    return average_scores_per_row(pd.concat([forward, reverse], ignore_index=True))


def _parse_annotation_specs(specs: List[str]) -> Dict[str, str]:
    annotations = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid annotation specification {spec!r}; expected NAME=/path/to/labels"
            )
        name, path = spec.split("=", 1)
        if not name or not path:
            raise ValueError(
                f"Invalid annotation specification {spec!r}; expected NAME=/path/to/labels"
            )
        if name in annotations:
            raise ValueError(f"Annotator {name!r} was specified more than once")
        annotations[name] = path
    return annotations


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute independent pairwise agreement between synapse annotators "
            "using one-to-one point matching."
        )
    )
    parser.add_argument(
        "-d", "--matching_distance", type=float, default=2.0,
        help="Maximum matching distance in µm. Default: 2.0",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=VOXEL_SIZE,
        help="Isotropic voxel size in µm. Default: 0.38",
    )
    parser.add_argument(
        "--annotation", action="append", default=None, metavar="NAME=DIR",
        help=(
            "Annotator name and label directory. Repeat for each annotator. "
            "If omitted, the configured AMD, EK, and LR directories are used."
        ),
    )
    parser.add_argument(
        "-o", "--output_dir", default=ROOT,
        help=f"Directory for output CSV files. Default: {ROOT}",
    )
    args = parser.parse_args()

    annotation_dirs = (
        DEFAULT_ANNOTATIONS
        if args.annotation is None
        else _parse_annotation_specs(args.annotation)
    )
    per_crop, summary = compute_pairwise_agreement(
        annotation_dirs,
        matching_distance=args.matching_distance,
        voxel_size=args.voxel_size,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    per_crop_path = os.path.join(args.output_dir, "pairwise_agreement_per_crop.csv")
    summary_path = os.path.join(args.output_dir, "pairwise_agreement_summary.csv")
    per_crop.to_csv(per_crop_path, index=False)
    summary.to_csv(summary_path, index=False)

    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"Saved per-crop results to {per_crop_path}")
    print(f"Saved summary results to {summary_path}")


if __name__ == "__main__":
    main()
