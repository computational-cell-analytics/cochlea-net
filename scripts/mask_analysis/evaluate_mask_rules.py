"""Evaluate input mask rules on the frozen train/validation split.

Every rule is a block score s_b on the 128^3 block grid and a threshold T: a block is predicted
if s_b > T. An object is complete if all blocks that its bounding box touches are predicted, so
it is complete exactly if min_b s_b > T over its blocks, and lost if max_b s_b <= T. Objects
that end on a block face (audit_block_cuts.py) get their box extended across that face, because
the true object continues there.

Threshold mappings, each with one parameter theta:
    abs:        T = theta (intensity)
    rel:        T = theta * b, b = median of the non-zero voxels (background)
    contrast_q: T = b + theta * (h_q - b), b = median, h_q = upper quantile of the volume
    noise:      T = b + theta * sigma, sigma = robust width of the background peak
    ratio:      T is chosen per volume so that a fraction theta of the blocks is kept

Target objects must have stain signal: their s3 maximum must exceed signal_floor * b. Objects
below the floor cannot be kept by an intensity rule without the background, and most of them are
U-Net artifacts in empty blocks at the volume border. They are listed in excluded_objects.csv.

Fit on train: per dataset, the critical theta is the most economical value that still keeps the
coverage target. The rule uses the most conservative critical theta over the training datasets.
Validation applies the frozen rule to the validation datasets.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import ndimage

from mask_common import BLOCK, MASK_JOBS_CSV, OUTPUT_ROOT, load_datasets, object_block_ranges, output_path

TARGETS = (0.99, 0.995, 0.999)
DEFAULT_TARGET = 0.995
CONTRAST_QUANTILES = (99.9, 99.99, 99.999)
FOOTPRINTS = {
    "D6": ndimage.generate_binary_structure(3, 1),
    "D26": np.ones((3, 3, 3), dtype=bool),
}


def hist_quantile(hist, q):
    """Quantile of an integer histogram with linear interpolation inside the bin."""
    cdf = np.cumsum(hist).astype("float64")
    target = q / 100 * cdf[-1]
    v = int(np.searchsorted(cdf, target, side="left"))
    below = cdf[v - 1] if v > 0 else 0.0
    return v - 0.5 + (target - below) / max(hist[v], 1)


def normalization(lowres):
    hist = lowres["hist"]
    b = hist_quantile(hist, 50)
    consts = {"b": b, "sigma": b - hist_quantile(hist, 15.87), "max": float(np.max(np.nonzero(hist)[0]))}
    for q in CONTRAST_QUANTILES:
        consts[f"h{q}"] = hist_quantile(hist, q)
    return consts


def extended_boxes(lowres, hits):
    bb_min, bb_max = lowres["bb_min"].copy(), lowres["bb_max"].copy()
    if hits is not None and len(hits):
        index = {label: i for i, label in enumerate(lowres["label_id"])}
        for _, hit in hits.iterrows():
            i = index.get(hit.label_id)
            if i is None:
                continue
            axis = "zyx".index(hit.axis)
            if hit.side == "min":
                bb_min[i, axis] -= 1
            else:
                bb_max[i, axis] += 1
    return bb_min, bb_max


def object_block_pairs(first, last, grid):
    """Flat block indices of every (object, block) pair, plus the start offset of each object."""
    sizes = np.prod(last - first + 1, axis=1)
    obj = np.repeat(np.arange(len(first)), sizes)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    local = np.arange(sizes.sum()) - np.repeat(starts, sizes)
    extent = (last - first + 1)[obj]
    z = first[obj, 0] + local // (extent[:, 1] * extent[:, 2])
    y = first[obj, 1] + (local // extent[:, 2]) % extent[:, 1]
    x = first[obj, 2] + local % extent[:, 2]
    return np.ravel_multi_index((z, y, x), grid), starts


def score_maps(lowres, s0):
    radii = list(lowres["radii_um"])
    block_max = lowres["block_max"].astype("float32")
    maps = {f"s3max_r{r}": block_max[i] for i, r in enumerate(radii)}
    bases = {"s3p95": lowres["block_p95"], "s3p99": lowres["block_p99"]}
    if s0 is not None:
        levels = list(s0["percentile_levels"])
        grid = tuple(s0["grid"])
        for p in (95, 99):
            bases[f"s0p{p}"] = s0["percentiles"][:, levels.index(p)].reshape(grid)
    for name, base in bases.items():
        base = base.astype("float32")
        maps[name] = base
        for dname, footprint in FOOTPRINTS.items():
            maps[f"{name}_{dname}"] = ndimage.maximum_filter(base, footprint=footprint, mode="constant", cval=0)
        # Conditional dilation: a neighbour of a kept block is added if the low-res signal within
        # r of it exceeds the same threshold.
        neighbours = maps[f"{name}_D26"]
        for r in (6, 12, 18):
            maps[f"{name}_Dr{r}"] = np.maximum(base, np.minimum(neighbours, block_max[radii.index(r)]))
    return maps


def mappings_for(score_name):
    if score_name.startswith("s3max"):
        return ["abs", "rel", "noise", "ratio"] + [f"contrast{q}" for q in CONTRAST_QUANTILES]
    return ["abs", "rel", "ratio"]


def to_threshold(mapping, theta, consts, sorted_scores):
    if mapping == "abs":
        return theta
    if mapping == "rel":
        return theta * consts["b"]
    if mapping == "noise":
        return consts["b"] + theta * consts["sigma"]
    if mapping == "ratio":
        # Keep at least the theta fraction of blocks with the highest score; ties at the boundary are kept.
        n = len(sorted_scores)
        k = int(np.clip(np.ceil(theta * n - 1e-9), 1, n))
        boundary = float(sorted_scores[n - k])
        return boundary - 1e-6 * max(1.0, abs(boundary))
    q = float(mapping.replace("contrast", ""))
    return consts["b"] + theta * (consts[f"h{q}"] - consts["b"])


def to_theta(mapping, threshold, consts, sorted_scores):
    """Inverse of to_threshold for the critical threshold of one dataset."""
    if mapping == "abs":
        return threshold
    if mapping == "rel":
        return threshold / consts["b"]
    if mapping == "noise":
        return (threshold - consts["b"]) / consts["sigma"]
    if mapping == "ratio":
        return float(np.mean(sorted_scores > threshold))
    q = float(mapping.replace("contrast", ""))
    return (threshold - consts["b"]) / (consts[f"h{q}"] - consts["b"])


def critical_threshold(obj_min, target):
    """A threshold just below the largest T with a fraction >= target of objects complete (obj_min > T).

    At most m = floor((1 - target) * n) objects have obj_min below the (m + 1)-th smallest value
    t[m], so every T < t[m] keeps the target. Objects in the same blocks share one score, so many
    objects can tie with t[m]; T = t[m] itself would lose all of them.
    """
    t = np.sort(obj_min)
    m = int(np.floor((1 - target) * len(t)))
    boundary = float(t[min(m, len(t) - 1)])
    return boundary - 1e-6 * max(1.0, abs(boundary))


def dataset_scores(lowres, s0, hits, target_objects):
    grid = tuple(lowres["grid"])
    bb_min, bb_max = extended_boxes(lowres, hits)
    first, last = object_block_ranges(bb_min[target_objects], bb_max[target_objects], BLOCK, grid)
    flat, starts = object_block_pairs(first, last, grid)
    results = {}
    for name, score in score_maps(lowres, s0).items():
        values = score.ravel()[flat]
        results[name] = {
            "obj_min": np.minimum.reduceat(values, starts) if len(starts) else np.zeros(0),
            "obj_max": np.maximum.reduceat(values, starts) if len(starts) else np.zeros(0),
            "sorted": np.sort(score.ravel()),
        }
    return results


def evaluate(entry, mapping, theta, consts):
    T = to_threshold(mapping, theta, consts, entry["sorted"])
    n = max(len(entry["obj_min"]), 1)
    complete = np.sum(entry["obj_min"] > T) / n
    lost = np.sum(entry["obj_max"] <= T) / n
    kept = np.mean(entry["sorted"] > T)
    return {"T": T, "complete": complete, "cut": 1 - complete - lost, "lost": lost, "kept": kept}


def excluded_objects(ds, lowres, consts, target_objects):
    shape = lowres["shape_lowres"] * lowres["scale"]
    border = np.minimum(lowres["bb_min"].min(axis=1), (shape - lowres["bb_max"]).min(axis=1))
    signal = lowres["object_max"]
    return [{"key": ds["key"], "class": ds["class"], "split": ds["split"], "label_id": int(lowres["label_id"][i]),
             "component": int(lowres["component"][i]), "n_pixels": int(lowres["n_pixels"][i]),
             "object_max": float(signal[i]), "relative_signal": float(signal[i] / consts["b"]),
             "distance_to_border": int(border[i]),
             "bb_min_zyx": " ".join(map(str, lowres["bb_min"][i])),
             "bb_max_zyx": " ".join(map(str, lowres["bb_max"][i]))}
            for i in np.where(~target_objects)[0]]


def load_all(level, use_s0, signal_floor):
    datasets = load_datasets()["datasets"]
    hits_path = os.path.join(OUTPUT_ROOT, "audit", "face_hits.csv")
    hits = pd.read_csv(hits_path) if os.path.exists(hits_path) else None
    loaded = []
    for ds in datasets:
        path = os.path.join(OUTPUT_ROOT, f"lowres_{level}", f"{ds['key']}.npz")
        if not os.path.exists(path):
            continue
        lowres = dict(np.load(path))
        s0_path = os.path.join(OUTPUT_ROOT, "s0_counts", f"{ds['key']}.npz")
        s0 = dict(np.load(s0_path)) if use_s0 and os.path.exists(s0_path) else None
        ds_hits = hits[hits.key == ds["key"]] if hits is not None else None
        consts = normalization(lowres)
        target_objects = lowres["object_max"] > signal_floor * consts["b"]
        loaded.append({"ds": ds, "consts": consts, "has_s0": s0 is not None, "s0": s0,
                       "scores": dataset_scores(lowres, s0, ds_hits, target_objects),
                       "object_max": lowres["object_max"][target_objects], "lowres": lowres,
                       "excluded": excluded_objects(ds, lowres, consts, target_objects)})
    return loaded


def fit_and_validate(loaded, target):
    rows, crit_rows = [], []
    score_names = sorted(set.intersection(*[set(item["scores"]) for item in loaded]))
    for score_name in score_names:
        for mapping in mappings_for(score_name):
            # Per-dataset critical theta.
            for item in loaded:
                entry = item["scores"][score_name]
                if len(entry["obj_min"]) == 0:
                    continue
                T = critical_threshold(entry["obj_min"], target)
                theta = to_theta(mapping, T, item["consts"], entry["sorted"])
                crit_rows.append({"score": score_name, "mapping": mapping, "key": item["ds"]["key"],
                                  "class": item["ds"]["class"], "split": item["ds"]["split"],
                                  "stratum": item["ds"]["stratum"], "stain": item["ds"]["stain"],
                                  "critical_theta": theta, "critical_T": T,
                                  "kept_at_critical": float(np.mean(entry["sorted"] > T))})
    crit = pd.DataFrame(crit_rows)
    for (score_name, mapping), group in crit.groupby(["score", "mapping"]):
        for scope in ("IHC", "SGN", "pooled"):
            train = group[(group.split == "train") & ((group["class"] == scope) | (scope == "pooled"))]
            if len(train) == 0:
                continue
            # Ratio: a larger kept fraction is more conservative. All others: a lower threshold.
            theta = train.critical_theta.max() if mapping == "ratio" else train.critical_theta.min()
            for split in ("train", "val"):
                for item in loaded:
                    ds = item["ds"]
                    if ds["split"] != split or (scope != "pooled" and ds["class"] != scope):
                        continue
                    result = evaluate(item["scores"][score_name], mapping, theta, item["consts"])
                    oracle = group[group.key == ds["key"]].kept_at_critical
                    rows.append({"score": score_name, "mapping": mapping, "scope": scope, "theta": theta,
                                 "key": ds["key"], "class": ds["class"], "split": split,
                                 "stratum": ds["stratum"], "stain": ds["stain"],
                                 "oracle_kept": float(oracle.iloc[0]) if len(oracle) else np.nan, **result})
    return crit, pd.DataFrame(rows)


def summarize(results, target):
    def agg(g):
        val = g[g.split == "val"]
        train = g[g.split == "train"]
        return pd.Series({
            "theta": g.theta.iloc[0],
            "train_kept_median": train.kept.median(),
            "train_efficiency": (train.oracle_kept / train.kept).median(),
            "val_n": len(val),
            "val_complete_min": val.complete.min(),
            "val_complete_median": val.complete.median(),
            "val_below_target": int((val.complete < target).sum()),
            "val_lost_max": val.lost.max(),
            "val_kept_median": val.kept.median(),
            "val_kept_max": val.kept.max(),
        })
    return results.groupby(["scope", "score", "mapping"]).apply(agg, include_groups=False).reset_index()


def evaluate_fixed(loaded, rules):
    """Evaluate fixed absolute thresholds, e.g. the production values, on every dataset."""
    rows = []
    for item in loaded:
        for score_name, T in rules:
            if score_name not in item["scores"]:
                continue
            result = evaluate(item["scores"][score_name], "abs", T, item["consts"])
            rows.append({"key": item["ds"]["key"], "class": item["ds"]["class"], "split": item["ds"]["split"],
                         "stratum": item["ds"]["stratum"], "stain": item["ds"]["stain"],
                         "score": score_name, **result})
    return pd.DataFrame(rows)


def missed_signal_proxy(loaded):
    """Count blocks that the reference run probably skipped although they hold object signal.

    A block counts if the reference rule excludes it, it is next to (26-neighbourhood) a block
    that relevant objects touch, and its s3 maximum reaches the 5th percentile of the object
    signal of the dataset. The reference rule is evaluated on the 128^3 grid; with the s3
    estimate of the percentile this is an upper bound (smoothing lowers the percentiles, and the
    n5 reference runs used coarser mask blocks).
    """
    rows = []
    for item in loaded:
        ref = item["ds"]["reference_mask"]
        if ref.get("rule_kind") not in ("fixed", "absolute") or ref.get("rule_percentile") is None:
            continue
        p, T = float(ref["rule_percentile"]), float(ref["rule_threshold"])
        if p not in (95.0, 99.0):
            continue
        lowres = item["lowres"]
        grid = tuple(lowres["grid"])
        first, last = object_block_ranges(lowres["bb_min"], lowres["bb_max"], BLOCK, grid)
        touched = np.zeros(grid, dtype=bool)
        for a, b in zip(first, last):
            touched[a[0]:b[0] + 1, a[1]:b[1] + 1, a[2]:b[2] + 1] = True
        near = ndimage.binary_dilation(touched, structure=np.ones((3, 3, 3), dtype=bool))
        signal = lowres["block_max"][0] >= np.percentile(item["object_max"], 5)
        row = {"key": item["ds"]["key"], "class": item["ds"]["class"], "rule": f"p{p:g} > {T:g}",
               "blocks_touched": int(touched.sum())}
        for level in ("s3", "s0"):
            name = f"{level}p{int(p)}"
            if name in item["scores"]:
                score = score_maps_base(lowres, level, int(p), item)
                excluded = score <= T
                row[f"{level}_missed_blocks"] = int((excluded & near & signal).sum())
                row[f"{level}_touched_excluded"] = int((excluded & touched).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def score_maps_base(lowres, level, p, item):
    if level == "s3":
        return lowres[f"block_p{p}"]
    s0 = item["s0"]
    levels = list(s0["percentile_levels"])
    return s0["percentiles"][:, levels.index(p)].reshape(tuple(s0["grid"]))


def calibrate_against_logs(loaded):
    """Compare the kept fraction of logged production mask runs with the s3 (and s0) estimate.

    The NHR runs log "kept/total (x %) used for segmentation". Their OME-Zarr inputs have 64^3
    chunks, so the mask grid is the 128^3 grid of this analysis and the totals must match.
    """
    jobs = pd.read_csv(MASK_JOBS_CSV)
    jobs = jobs[jobs.log_kept_blocks.notna() & (jobs.rule_kind == "absolute")]
    rows = []
    for item in loaded:
        ds = item["ds"]
        for _, job in jobs[jobs.cochlea == ds["cochlea"]].iterrows():
            n_blocks = item["scores"]["s3p99"]["sorted"].size
            if f"/{ds['stain']}." not in str(job.input) or int(job.log_total_blocks) != n_blocks:
                continue
            p = int(job.rule_percentile)
            row = {"key": ds["key"], "job": job.job, "percentile": p, "T": job.rule_threshold,
                   "logged_kept": job.log_kept_blocks / job.log_total_blocks}
            for level in ("s3", "s0"):
                name = f"{level}p{p}"
                if name in item["scores"]:
                    row[f"{level}_kept"] = float(np.mean(item["scores"][name]["sorted"] > job.rule_threshold))
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="s3")
    parser.add_argument("--no_s0", action="store_true", help="Ignore the s0 block counts of the HPC job.")
    parser.add_argument("--signal_floor", type=float, default=1.2,
                        help="Target objects need an s3 maximum above signal_floor * b.")
    parser.add_argument("--output_dir", default="evaluation", help="Output folder inside mask_threshold_analysis.")
    args = parser.parse_args()

    loaded = load_all(args.level, not args.no_s0, args.signal_floor)
    print(f"{len(loaded)} datasets with features ({sum(i['has_s0'] for i in loaded)} with s0 counts)")
    excluded = pd.DataFrame([row for i in loaded for row in i["excluded"]])
    excluded.to_csv(output_path(args.output_dir, "excluded_objects.csv"), index=False)
    print(f"{len(excluded)} objects below the signal floor {args.signal_floor} * b are not targets")
    consts = pd.DataFrame([{"key": i["ds"]["key"], "class": i["ds"]["class"], "split": i["ds"]["split"],
                            "stratum": i["ds"]["stratum"], "stain": i["ds"]["stain"],
                            "n_objects": len(i["object_max"]), "n_excluded": len(i["excluded"]),
                            "object_max_p1": np.percentile(i["object_max"], 1) if len(i["object_max"]) else np.nan,
                            "object_max_p50": np.median(i["object_max"]) if len(i["object_max"]) else np.nan,
                            **i["consts"]} for i in loaded])
    consts.to_csv(output_path(args.output_dir, "normalization.csv"), index=False)

    fixed_rules = [(f"{base}{suffix}", T) for base in ("s3p95", "s3p99", "s0p95", "s0p99")
                   for suffix in ("", "_D6", "_D26", "_Dr12") for T in (100, 150, 200, 300, 400)]
    evaluate_fixed(loaded, fixed_rules).to_csv(output_path(args.output_dir, "fixed_rules.csv"), index=False)
    missed_signal_proxy(loaded).to_csv(output_path(args.output_dir, "missed_signal_proxy.csv"), index=False)
    calibration = calibrate_against_logs(loaded)
    calibration.to_csv(output_path(args.output_dir, "calibration_logged_runs.csv"), index=False)
    if len(calibration):
        print("Logged production masks vs estimates:")
        print(calibration.round(4).to_string(index=False))

    for target in TARGETS:
        valid = [i for i in loaded if i["ds"]["split"] in ("train", "val")]
        crit, results = fit_and_validate(valid, target)
        tag = f"t{target}"
        crit.to_csv(output_path(args.output_dir, f"critical_{tag}.csv"), index=False)
        results.to_csv(output_path(args.output_dir, f"results_{tag}.csv"), index=False)
        summary = summarize(results, target)
        summary.to_csv(output_path(args.output_dir, f"summary_{tag}.csv"), index=False)
        if target == DEFAULT_TARGET:
            pd.set_option("display.width", 250)
            pd.set_option("display.max_rows", 500)
            best = summary.sort_values(["scope", "train_kept_median"])
            print(best.groupby("scope").head(12).round(4).to_string(index=False))

    failures = [i for i in loaded if i["ds"]["split"] == "failure"]
    if failures:
        rows = [{"key": i["ds"]["key"], **{k: round(v, 1) for k, v in i["consts"].items()}} for i in failures]
        pd.DataFrame(rows).to_csv(output_path(args.output_dir, "failure_datasets.csv"), index=False)


if __name__ == "__main__":
    main()
