"""Compare central path methods without a manual reference.

There is no manual annotation of the central line of Rosenthal's canal, so the methods are
compared with measures that only need the segmentation itself. To keep the comparison honest,
the path is built from one half of the cells and every measure is taken on the other half, with
a slab thickness that differs from the one the refinement uses. A method therefore cannot win by
fitting the cells it is scored on.

Measures, per cross-section perpendicular to the local path direction:

- residual: distance from the path node to the area centroid of the convex hull of the cells.
- imbalance: length of the mean unit direction from the node to the cells. It ignores the hull
  and the radial distance, so it does not reward the objective the refinement optimizes.
- clearance: distance from the node to the closest hull edge, over the equivalent hull radius.
  This is maximal at the Chebyshev center, which is a different definition of central.

Measures per path:

- curvature: the cochlear spiral has a radius of curvature of at least about 150 µm, so a value
  above roughly 7 /mm is not anatomical and indicates voxel staircase noise.
- length drift: the change in arc length when the path is resampled. A smooth curve is
  resample-invariant, a staircase is not.

Usage:
    python scripts/validation/central_path/compare_path_methods.py TABLE_DIR OUTPUT.tsv
"""

import argparse
import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, cKDTree

from flamingo_tools.postprocessing.cochlea_mapping import (
    CENTRAL_PATH_METHODS, _arc_length, _hull_area_centroid, _plane_basis, _tangents, resample_path,
)

# Deliberately different from the defaults of refine_central_path.
EVAL_SPACING = 20.0
EVAL_SLAB = 30.0
EVAL_RADIUS = 110.0
MIN_EVAL_POINTS = 12


def cross_sections(path, centroids):
    """Yield the in-plane offsets of the cells of every cross-section that holds enough cells."""
    samples = resample_path(path, EVAL_SPACING)
    tangents = _tangents(samples, 100.0, EVAL_SPACING)
    first, second = _plane_basis(tangents)
    _, assignment = cKDTree(samples).query(centroids, workers=-1)
    order = np.argsort(assignment, kind="stable")
    index = np.arange(len(samples))
    bin_start = np.searchsorted(assignment[order], index, "left")
    bin_end = np.searchsorted(assignment[order], index, "right")
    bins = int(np.ceil(EVAL_SLAB / EVAL_SPACING)) + 1

    for i in range(len(samples)):
        selected = order[bin_start[max(0, i - bins)]:bin_end[min(len(samples) - 1, i + bins)]]
        if selected.size < MIN_EVAL_POINTS:
            continue
        relative = centroids[selected] - samples[i]
        in_plane = np.stack([relative @ first[i], relative @ second[i]], axis=1)
        keep = ((np.abs(relative @ tangents[i]) <= EVAL_SLAB)
                & ((in_plane ** 2).sum(axis=1) <= EVAL_RADIUS ** 2))
        if int(keep.sum()) < MIN_EVAL_POINTS:
            continue
        yield in_plane[keep]


def measure(path, centroids):
    """Measure how central a path runs through a set of cells."""
    residual, imbalance, clearance, outside, n_sections = [], [], [], 0, 0
    for points in cross_sections(path, centroids):
        n_sections += 1
        residual.append(float(np.linalg.norm(_hull_area_centroid(points))))
        radius = np.linalg.norm(points, axis=1)
        good = radius > 1e-9
        if good.any():
            imbalance.append(float(np.linalg.norm((points[good] / radius[good, None]).mean(axis=0))))
        try:
            hull = ConvexHull(points)
        except Exception:
            continue
        edge_distance = -float(hull.equations[:, 2].max())
        equivalent_radius = float(np.sqrt(hull.volume / np.pi))
        outside += int(edge_distance <= 0)
        if equivalent_radius > 0:
            clearance.append(edge_distance / equivalent_radius)

    resampled = [float(_arc_length(resample_path(path, spacing))[-1]) for spacing in (5.0, 10.0, 20.0, 40.0)]
    steps = np.diff(resample_path(path, EVAL_SPACING), axis=0)
    norms = np.linalg.norm(steps, axis=1)
    good = (norms[:-1] > 0) & (norms[1:] > 0)
    cosine = (steps[:-1][good] * steps[1:][good]).sum(axis=1) / (norms[:-1][good] * norms[1:][good])
    curvature = np.arccos(np.clip(cosine, -1, 1)) / EVAL_SPACING * 1000.0

    return {
        "n_sections": n_sections,
        "residual_median": float(np.median(residual)) if residual else float("nan"),
        "residual_p90": float(np.percentile(residual, 90)) if residual else float("nan"),
        "imbalance_median": float(np.median(imbalance)) if imbalance else float("nan"),
        "clearance_median": float(np.median(clearance)) if clearance else float("nan"),
        "outside_fraction": outside / n_sections if n_sections else float("nan"),
        "length": float(_arc_length(path)[-1]),
        "length_drift": (max(resampled) - min(resampled)) / float(np.mean(resampled)),
        "curvature_p95": float(np.percentile(curvature, 95)) if len(curvature) else float("nan"),
    }


def configured_components(cochlea, config_dir):
    """Component list of a cochlea from its processing parameter file, or None."""
    config = os.path.join(config_dir, f"{cochlea}_SGN.json")
    if not os.path.isfile(config):
        return None
    with open(config) as f:
        data = json.load(f)
    entry = data[0] if isinstance(data, list) else data
    return entry.get("component_list", [1]) if entry.get("cell_type") == "sgn" else None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("table_dir", help="Directory of tonotopically labeled SGN tables in TSV format.")
    parser.add_argument("output", help="Output path for the per-component measurements.")
    parser.add_argument("--methods", nargs="+", default=["edt", "edt_refined"],
                        choices=sorted(CENTRAL_PATH_METHODS), help="Central path methods to compare.")
    parser.add_argument("--config_dir", default="reproducibility/processing",
                        help="Directory of processing parameter files, used for the component lists.")
    parser.add_argument("--min_cells", type=int, default=1500,
                        help="Skip components with fewer cells.")
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    rows = []
    for table_path in sorted(glob.glob(os.path.join(args.table_dir, "*.tsv"))):
        cochlea = os.path.basename(table_path)[:-4]
        table = pd.read_csv(table_path, sep="\t")
        if "component_labels" not in table.columns:
            continue
        counts = table.component_labels.value_counts()
        configured = configured_components(cochlea.replace("_v2", ""), args.config_dir)
        labels = [c for c in (configured or sorted(k for k in counts.index if k > 0))
                  if c in counts.index and counts[c] >= args.min_cells]

        for label in labels:
            points = table[table.component_labels == label][
                ["anchor_x", "anchor_y", "anchor_z"]].to_numpy(dtype=float)
            # Build from one half, measure on the other, so no method is scored on its own fit.
            split = np.random.default_rng(0).random(len(points)) < 0.5
            fit, evaluate = points[split], points[~split]
            if min(len(fit), len(evaluate)) < 1000:
                continue

            row = {"cochlea": cochlea, "component": int(label), "n": len(points)}
            try:
                for method in args.methods:
                    path = CENTRAL_PATH_METHODS[method]([list(map(tuple, fit))])[0]
                    for key, value in measure(np.asarray(path, dtype=float), evaluate).items():
                        row[f"{method}_{key}"] = value
            except Exception as error:
                print(f"  skipping {cochlea} component {label}: {type(error).__name__}: {error}")
                continue
            rows.append(row)
            summary = "  ".join(f"{m} {row[f'{m}_residual_median']:5.1f}" for m in args.methods)
            print(f"{cochlea:32s} c{label} n={len(points):6d}  residual [µm]: {summary}", flush=True)

    pd.DataFrame(rows).to_csv(args.output, sep="\t", index=False)
    print(f"\nWrote {len(rows)} components to {args.output}.")


if __name__ == "__main__":
    main()
