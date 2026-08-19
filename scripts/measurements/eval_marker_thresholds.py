import argparse
import glob
import itertools
import json
import os
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

from flamingo_tools.intensity_annotation.eval_annotations import percentage
from flamingo_tools.json_util import export_dictionary_as_json

from thresholds_marker import BG_FEATURE, build_features

COCHLEAE = ["M_AMD_OTOF27_L", "M_AMD_OTOF27_R", "M_AMD_OTOF28_L", "M_AMD_OTOF28_R"]

# Features offered as a level test, and features offered as a contrast gate.
LEVEL_FEATURES = ["median", "mean", "percentile-90", "percentile-95", BG_FEATURE]
GATE_FEATURES = ["p95_sub_p5", "p90_sub_p10", "p90_sub_median", "iqr"]


def _threshold_grid(values: np.ndarray, n: int = 200) -> np.ndarray:
    """Candidate thresholds between the observed values, so that every split is reachable."""
    values = np.unique(values[np.isfinite(values)])
    if len(values) < 2:
        return np.array([values[0] + 1e-6]) if len(values) else np.array([0.0])
    quantiles = np.unique(np.quantile(values, np.linspace(0, 1, min(n, len(values)))))
    middles = (quantiles[:-1] + quantiles[1:]) / 2
    return np.concatenate([[quantiles[0] - 1e-6], middles, [quantiles[-1] + 1e-6]])


def build_reference(
    cochlea: str,
    crop_dir: str,
    threshold_dir: str,
    meas_dir: str,
    marker_name: str = "OTOF",
    seg_name: str = "IHC_v11",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Derive reference marker labels for the instances of the annotated crops.

    The crops give which instances an annotator saw, by their global label id. The per-crop
    threshold of the annotators is applied to the background-subtracted median of these
    instances, which gives the reference label that a candidate rule has to reproduce.

    Args:
        cochlea: The name of the cochlea.
        crop_dir: Directory with the segmentation crops.
        threshold_dir: Directory with the per-crop annotator thresholds.
        meas_dir: Directory with the object-measures tables.
        marker_name: Identifier for the marker stain in the threshold file name.
        seg_name: Identifier for the segmentation.

    Returns:
        The reference table for the annotated instances, and the feature table of the cochlea.
    """
    cochlea_str = cochlea.replace("_", "-")
    seg_string = seg_name.replace("_", "-")
    prefix = f"{cochlea_str}_Otof_{seg_string}_object-measures"
    plain_path = os.path.join(meas_dir, f"{prefix}.tsv")
    bg_path = os.path.join(meas_dir, f"{prefix}-bg-mask.tsv")
    features = build_features(
        pd.read_csv(plain_path, sep="\t") if os.path.exists(plain_path) else None,
        pd.read_csv(bg_path, sep="\t") if os.path.exists(bg_path) else None,
    )

    with open(os.path.join(threshold_dir, f"{cochlea_str}_{marker_name}_{seg_string}.json"), "r") as f:
        crop_thresholds = json.load(f)

    lut = features.set_index("label_id")
    rows = []
    for path in sorted(glob.glob(os.path.join(crop_dir, f"{cochlea_str}_crop_*_{seg_string}.tif"))):
        center = re.search(r"_crop_(\d+-\d+-\d+)_", os.path.basename(path)).group(1)
        crop_threshold = crop_thresholds.get(center, {}).get("median_intensity")
        if crop_threshold is None:
            print(f"  {cochlea} {center}: no annotator threshold, crop skipped.")
            continue
        labels = np.unique(tifffile.imread(path))
        for label_id in (int(label) for label in labels[labels > 0]):
            if label_id not in lut.index:
                continue
            row = lut.loc[label_id].to_dict()
            row.update({"label_id": label_id, "crop": center, "crop_threshold": crop_threshold})
            rows.append(row)

    reference = pd.DataFrame(rows).drop_duplicates("label_id").reset_index(drop=True)
    if len(reference) == 0:
        raise ValueError(f"No annotated instance found for cochlea {cochlea}.")
    reference["reference"] = (reference[BG_FEATURE] >= reference["crop_threshold"]).astype(int)
    return reference, features


def evaluate_rule(
    reference: pd.DataFrame,
    level: str,
    gates: Dict[str, float],
) -> dict:
    """Find the best level threshold of a rule, with the gate thresholds held fixed.

    Args:
        reference: Reference table from `build_reference`.
        level: Feature that is thresholded per cochlea.
        gates: Threshold per gate feature, shared over the cochleae.

    Returns:
        The best threshold, its accuracy, the widest optimal plateau, and the confusion counts.
    """
    y = reference["reference"].to_numpy()
    passes_gates = np.ones(len(reference), dtype=bool)
    for feature, value in gates.items():
        passes_gates &= reference[feature].to_numpy(dtype="float64") >= value

    values = reference[level].to_numpy(dtype="float64")
    grid = _threshold_grid(values)
    predictions = (values[:, None] >= grid[None, :]) & passes_gates[:, None]
    accuracy = (predictions == y[:, None]).mean(axis=0)

    best = float(accuracy.max())
    optimal = np.where(accuracy == best)[0]
    runs, start = [], optimal[0]
    for previous, following in zip(optimal, optimal[1:]):
        if following != previous + 1:
            runs.append((start, previous))
            start = following
    runs.append((start, optimal[-1]))
    low, high = max(runs, key=lambda run: run[1] - run[0])
    threshold = float(np.median(grid[low:high + 1]))

    prediction = ((values >= threshold) & passes_gates).astype(int)
    return {
        "level": level,
        "gates": gates,
        "threshold": threshold,
        "accuracy": round(best, 4),
        "plateau": [float(grid[low]), float(grid[high])],
        "unbounded_plateau": bool(high == len(grid) - 1),
        "n_annotated": int(len(y)),
        "n_errors": int((prediction != y).sum()),
        "true_positive": int(((prediction == 1) & (y == 1)).sum()),
        "false_positive": int(((prediction == 1) & (y == 0)).sum()),
        "false_negative": int(((prediction == 0) & (y == 1)).sum()),
        "true_negative": int(((prediction == 0) & (y == 0)).sum()),
    }


def cross_validate(
    reference: pd.DataFrame,
    level: str,
    gates: Dict[str, float],
    n_folds: int = 5,
    n_repeats: int = 20,
    seed: int = 0,
) -> Tuple[float, int]:
    """Estimate how well a rule generalises, by refitting the level threshold on a subset.

    The gate thresholds are shared over the cochleae and stay fixed, so only the level
    threshold is refitted per fold.

    Returns:
        The mean number of errors per repeat, and the number of evaluated instances.
    """
    rng = np.random.default_rng(seed)
    y = reference["reference"].to_numpy()
    passes_gates = np.ones(len(reference), dtype=bool)
    for feature, value in gates.items():
        passes_gates &= reference[feature].to_numpy(dtype="float64") >= value
    values = reference[level].to_numpy(dtype="float64")

    n_errors, n_evaluated = 0, 0
    for _ in range(n_repeats):
        order = rng.permutation(len(y))
        for fold in range(n_folds):
            test = order[fold::n_folds]
            train = np.setdiff1d(order, test)
            grid = _threshold_grid(values[train], 150)
            predictions = (values[train][:, None] >= grid[None, :]) & passes_gates[train][:, None]
            accuracy = (predictions == y[train][:, None]).mean(axis=0)
            optimal = np.where(accuracy == accuracy.max())[0]
            threshold = float(np.median(grid[optimal]))
            prediction = ((values[test] >= threshold) & passes_gates[test]).astype(int)
            n_errors += int((prediction != y[test]).sum())
            n_evaluated += len(test)
    return n_errors / n_repeats, n_evaluated


def search_rules(
    references: Dict[str, pd.DataFrame],
    level_features: List[str],
    gate_features: List[str],
    n_gate_candidates: int = 50,
) -> List[dict]:
    """Rank level and gate feature pairs by their cross-validated accuracy.

    The gate threshold is shared over all cochleae, and the level threshold is fitted per
    cochlea. This keeps the number of free parameters low, which matters because only a few
    hundred instances are annotated.
    """
    results = []
    combinations = [(level, None) for level in level_features]
    combinations += list(itertools.product(level_features, gate_features))

    for level, gate in combinations:
        if gate is None:
            gate_values = [{}]
        else:
            pooled = np.concatenate([ref[gate].to_numpy(dtype="float64") for ref in references.values()])
            gate_values = [{gate: float(value)} for value in _threshold_grid(pooled, n_gate_candidates)]

        best = None
        for gates in gate_values:
            errors = sum(evaluate_rule(ref, level, gates)["n_errors"] for ref in references.values())
            if best is None or errors < best[0]:
                best = (errors, gates)
        in_sample, gates = best

        cv_errors = cv_total = 0
        for ref in references.values():
            errors, evaluated = cross_validate(ref, level, gates)
            cv_errors += errors
            cv_total += evaluated // 20
        results.append({
            "level": level, "gates": gates, "n_errors_in_sample": in_sample,
            "n_errors_cross_validated": round(cv_errors, 1),
            "accuracy_cross_validated": round(1 - cv_errors / cv_total, 4) if cv_total else None,
        })
    results.sort(key=lambda item: item["n_errors_cross_validated"])
    return results


def all_negative_crops(
    cochlea: str,
    threshold_dir: str,
    marker_name: str = "OTOF",
    seg_name: str = "IHC_v11",
) -> set:
    """Find the crops that the annotators marked entirely negative.

    Such a crop carries no annotator threshold. The 1.5 x maximum convention of
    `get_single_annotation_parameters` is not used here, because it depends on a maximum measured
    elsewhere in the cochlea. The crop is given an infinite threshold instead, so that its whole
    length-fraction band stays negative.
    """
    cochlea_str = cochlea.replace("_", "-")
    seg_string = seg_name.replace("_", "-")
    path = os.path.join(threshold_dir, f"{cochlea_str}_{marker_name}_{seg_string}.json")
    with open(path, "r") as f:
        crop_thresholds = json.load(f)

    crops = set()
    for center, entry in crop_thresholds.items():
        sources = {source for source in entry.get("threshold_sources", {}).values() if source}
        if sources and sources <= {"single-annotation-negative"}:
            crops.add(center)
    return crops


def _weights(y: np.ndarray, positive_weight: float) -> np.ndarray:
    """Weight of every instance, so that a reference-positive instance counts more."""
    return np.where(y == 1, positive_weight, 1.0)


def fit_threshold(values: np.ndarray, y: np.ndarray, positive_weight: float = 1.0) -> float:
    """Threshold that maximises the weighted accuracy, at the middle of the optimal plateau.

    A set of instances of a single class gives an infinite threshold for all-negative, so that
    nothing is called positive, and minus infinity for all-positive.
    """
    if len(np.unique(y)) < 2:
        return float("inf") if (len(y) == 0 or y[0] == 0) else float("-inf")
    grid = _threshold_grid(values)
    weights = _weights(y, positive_weight)[:, None]
    score = (((values[:, None] >= grid[None, :]) == y[:, None]) * weights).sum(axis=0) / weights.sum()
    optimal = np.where(score == score.max())[0]
    return float(np.median(grid[optimal]))


def fit_local_thresholds(
    reference: pd.DataFrame,
    level: str,
    all_negative: set,
    positive_weight: float = 1.0,
) -> Dict[str, float]:
    """Fit one threshold per crop, so that local imaging variation is accounted for."""
    thresholds = {}
    for center, group in reference.groupby("crop"):
        if center in all_negative:
            thresholds[center] = float("inf")
        else:
            thresholds[center] = fit_threshold(
                group[level].to_numpy(dtype="float64"), group["reference"].to_numpy(), positive_weight
            )
    return dict(sorted(thresholds.items()))


def _counts(prediction: np.ndarray, y: np.ndarray) -> dict:
    true_positive = int(((prediction == 1) & (y == 1)).sum())
    false_positive = int(((prediction == 1) & (y == 0)).sum())
    false_negative = int(((prediction == 0) & (y == 1)).sum())
    true_negative = int(((prediction == 0) & (y == 0)).sum())
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_errors": false_positive + false_negative,
        "true_positive": true_positive, "false_positive": false_positive,
        "false_negative": false_negative, "true_negative": true_negative,
        "recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4),
    }


def leave_one_out(
    reference: pd.DataFrame,
    level: str,
    scope: str,
    all_negative: set,
    positive_weight: float = 1.0,
) -> dict:
    """Score a rule by refitting its threshold without the instance that it is scored on.

    Args:
        reference: Reference table from `build_reference`.
        level: Feature that is thresholded.
        scope: "global" for one threshold per cochlea, "local" for one threshold per crop.
        all_negative: Crops that the annotators marked entirely negative.
        positive_weight: Weight of a reference-positive instance.

    Returns:
        The confusion counts and the derived recall, precision and F1.
    """
    groups = [reference] if scope == "global" else [group for _, group in reference.groupby("crop")]
    predictions, truths = [], []
    for group in groups:
        y = group["reference"].to_numpy()
        values = group[level].to_numpy(dtype="float64")
        centers = group["crop"].to_numpy()
        for index in range(len(y)):
            if scope == "local" and centers[index] in all_negative:
                predictions.append(0)
            else:
                keep = np.ones(len(y), dtype=bool)
                keep[index] = False
                threshold = fit_threshold(values[keep], y[keep], positive_weight)
                predictions.append(int(values[index] >= threshold))
            truths.append(int(y[index]))
    return _counts(np.array(predictions), np.array(truths))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate marker threshold rules against the annotated crops of the OTOF cochleae."
    )
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=COCHLEAE, help="Cochlea(e) to evaluate.")
    parser.add_argument("--crop_dir", type=str, default="otof_crops",
                        help="Directory with the segmentation crops.")
    parser.add_argument("--threshold_dir", type=str, default="otof_crop_thresholds",
                        help="Directory with the per-crop annotator thresholds.")
    parser.add_argument("--meas_dir", type=str, default="otof_object_measures",
                        help="Directory with the object-measures tables.")
    parser.add_argument("--level", type=str, default=None,
                        help="Evaluate only this level feature, instead of searching.")
    parser.add_argument("--gate", type=str, nargs="*", default=None, metavar="FEATURE=VALUE",
                        help="Evaluate only these gate thresholds, e.g. p95_sub_p5=240.")
    parser.add_argument("--local", action="store_true",
                        help="Fit one threshold per crop instead of one per cochlea.")
    parser.add_argument("--positive_weight", type=float, default=3.0,
                        help="Weight of a reference-positive instance when fitting and scoring.")
    parser.add_argument("--marker_name", type=str, default="OTOF",
                        help="Identifier for the marker stain in the threshold file name.")
    parser.add_argument("--seg_name", type=str, default="IHC_v11", help="Identifier for the segmentation.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output path for a JSON with the results.")
    args = parser.parse_args()

    references, features = {}, {}
    for cochlea in args.cochlea:
        reference, feature_table = build_reference(
            cochlea, args.crop_dir, args.threshold_dir, args.meas_dir,
            marker_name=args.marker_name, seg_name=args.seg_name,
        )
        references[cochlea] = reference
        features[cochlea] = feature_table
        n_positive = int(reference["reference"].sum())
        print(f"{cochlea}: {len(reference)} annotated instances, {n_positive} positive "
              f"({percentage(n_positive, len(reference))} %), {len(feature_table)} instances measured.")

    if args.local:
        level = args.level if args.level is not None else "mean"
        all_negative = {
            cochlea: all_negative_crops(cochlea, args.threshold_dir, args.marker_name, args.seg_name)
            for cochlea in args.cochlea
        }
        print(f"\nLocal thresholds on '{level}', reference-positive instances weighted "
              f"{args.positive_weight}x.")
        print("A crop the annotators marked entirely negative gets an infinite threshold.")

        print(f"\n  {'cochlea':16s} {'crop':>16s} {'threshold':>10s} {'n':>4s} {'ref pos':>8s}")
        local_thresholds = {}
        for cochlea, reference in references.items():
            local_thresholds[cochlea] = fit_local_thresholds(
                reference, level, all_negative[cochlea], args.positive_weight
            )
            for center, threshold in local_thresholds[cochlea].items():
                group = reference[reference["crop"] == center]
                text = "inf" if not np.isfinite(threshold) else f"{threshold:.1f}"
                print(f"  {cochlea:16s} {center:>16s} {text:>10s} {len(group):4d} "
                      f"{int(group['reference'].sum()):8d}")

        print(f"\n  {'scope':7s} {'errors':>7s} {'missed pos':>11s} {'false pos':>10s} "
              f"{'recall':>7s} {'precision':>10s} {'F1':>6s}")
        comparison = {}
        for scope in ("global", "local"):
            totals = [
                leave_one_out(reference, level, scope, all_negative[cochlea], args.positive_weight)
                for cochlea, reference in references.items()
            ]
            merged = {key: sum(item[key] for item in totals) for key in
                      ("n_errors", "true_positive", "false_positive", "false_negative", "true_negative")}
            recall = merged["true_positive"] / max(merged["true_positive"] + merged["false_negative"], 1)
            precision = merged["true_positive"] / max(merged["true_positive"] + merged["false_positive"], 1)
            f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            merged.update({"recall": round(recall, 4), "precision": round(precision, 4), "f1": round(f1, 4)})
            comparison[scope] = merged
            print(f"  {scope:7s} {merged['n_errors']:7d} {merged['false_negative']:11d} "
                  f"{merged['false_positive']:10d} {recall:7.3f} {precision:10.3f} {f1:6.3f}")
        print("  Scored leave-one-instance-out, so a threshold never sees the instance it is scored on.")

        print("\nLOCAL_THRESHOLD_DICT entries for scripts/measurements/thresholds_marker.py:")
        for cochlea, thresholds in local_thresholds.items():
            body = ", ".join(
                f'"{center}": ' + ("float(\"inf\")" if not np.isfinite(value) else f"{value:.1f}")
                for center, value in thresholds.items()
            )
            print(f'    "{cochlea}": {{{body}}},')

        if args.output is not None:
            export_dictionary_as_json(
                {"feature": level, "positive_weight": args.positive_weight,
                 "thresholds": {c: {k: (None if not np.isfinite(v) else v) for k, v in t.items()}
                                for c, t in local_thresholds.items()},
                 "all_negative_crops": {c: sorted(v) for c, v in all_negative.items()},
                 "comparison": comparison},
                args.output, force_overwrite=True,
            )
            print(f"\nSaved the results to {args.output}.")
        return

    if args.level is None:
        print("\nRule search, ranked by cross-validated errors "
              "(gate threshold shared, level threshold fitted per cochlea):")
        results = search_rules(references, LEVEL_FEATURES, GATE_FEATURES)
        print(f"  {'level':16s} {'gate':28s} {'in-sample':>10s} {'cross-val':>10s} {'cv accuracy':>12s}")
        for item in results[:15]:
            gate = ", ".join(f"{k}>={round(v, 1)}" for k, v in item["gates"].items()) or "-"
            print(f"  {item['level']:16s} {gate:28s} {item['n_errors_in_sample']:10d} "
                  f"{item['n_errors_cross_validated']:10.1f} {item['accuracy_cross_validated']:12.4f}")
        chosen = results[0]
        level, gates = chosen["level"], chosen["gates"]
    else:
        level = args.level
        gates = {}
        for entry in (args.gate or []):
            feature, value = entry.split("=")
            gates[feature] = float(value)
        results = None

    gate_text = ", ".join(f"{k} >= {v}" for k, v in gates.items()) or "none"
    print(f"\nRule: level '{level}' per cochlea, gate {gate_text}")
    print(f"  {'cochlea':16s} {'threshold':>10s} {'accuracy':>9s} {'tp':>4s} {'fp':>4s} {'fn':>4s} "
          f"{'tn':>5s} {'plateau':>18s} {'cochlea positive':>17s}")
    per_cochlea = {}
    for cochlea, reference in references.items():
        result = evaluate_rule(reference, level, gates)
        table = features[cochlea]
        positive = np.ones(len(table), dtype=bool)
        for feature, value in gates.items():
            positive &= table[feature].to_numpy(dtype="float64") >= value
        positive &= table[level].to_numpy(dtype="float64") >= result["threshold"]
        result["percent_positive_cochlea"] = percentage(int(positive.sum()), len(table))
        per_cochlea[cochlea] = result
        plateau = f"[{result['plateau'][0]:.0f}, {result['plateau'][1]:.0f}]"
        if result["unbounded_plateau"]:
            plateau += "*"
        print(f"  {cochlea:16s} {result['threshold']:10.0f} {result['accuracy']:9.4f} "
              f"{result['true_positive']:4d} {result['false_positive']:4d} {result['false_negative']:4d} "
              f"{result['true_negative']:5d} {plateau:>18s} {result['percent_positive_cochlea']:16.2f}%")
    total = sum(item["n_errors"] for item in per_cochlea.values())
    annotated = sum(item["n_annotated"] for item in per_cochlea.values())
    print(f"  total errors {total}/{annotated}. A plateau marked * is unbounded, so the data of that "
          "cochlea does not pin the threshold.")
    print("\nTHRESHOLD_DICT entries for scripts/measurements/thresholds_marker.py:")
    for cochlea, item in per_cochlea.items():
        entries = {level: round(item["threshold"])}
        entries.update({k: round(v) for k, v in gates.items()})
        body = ", ".join(f'"{k}": {v}' for k, v in entries.items())
        print(f'    "{cochlea}": {{{body}}},')

    if args.output is not None:
        export_dictionary_as_json(
            {"rule": {"level": level, "gates": gates}, "per_cochlea": per_cochlea, "search": results},
            args.output, force_overwrite=True,
        )
        print(f"\nSaved the results to {args.output}.")


if __name__ == "__main__":
    main()
