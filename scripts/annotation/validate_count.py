"""Validate the cell count of a row stretch from the row spacing, on the refined IHC annotations: leave a stretch out.

The count is k = round(gap / spacing) - 1, with the gap between the positions of the two end cells along the row.
The spacing rules:
- crop: the median gap between neighbouring typical cells outside the stretch (`region._row_spacing`, the rule of
  `--region`);
- local_typical: the median of the 2 nearest gaps on each side of the stretch, between typical cells;
- local_all: the same between all row cells of 5000 voxels or more (also the atypical ones).
Gaps of more than 1.6 times the median gap of the crop hold missing cells and are left out.

The stretches are the middle run of 2 to 5 typical row cells of each crop (`validate_region.stretches`), without
the crops with open issues. The row path and the frame come from the annotation without the stretch.

Usage:
    python validate_count.py -o OUT -j 4
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile

from flamingo_tools.annotation.ids import compact
from flamingo_tools.annotation.issues import MIN_CELL_SIZE, cell_voxels
from flamingo_tools.annotation.region import ANNOTATION, _Frame, _row_spacing, row_context

from validate_region import OPEN_ISSUES, VOXEL_SIZE, gt_crops, stretches

LENGTHS = (2, 3, 4, 5)
RULES = ("crop", "local_typical", "local_all")


def local_spacing(positions, low, high, limit):
    """The median of the 2 nearest gaps on each side of [low, high], without gaps above `limit`."""
    positions = np.sort(positions)
    left, right = positions[positions <= low], positions[positions >= high]
    gaps = np.concatenate([np.diff(left[-3:]), np.diff(right[:3])])
    gaps = gaps[gaps <= limit]
    return float(np.median(gaps)) if len(gaps) else None


def run_crop(crop):
    start = time.time()
    gt, _ = compact(tifffile.imread(crop["path"]))
    try:
        _, row = row_context(gt, {}, VOXEL_SIZE)
    except ValueError as error:
        print(f"{crop['name']}: {error}")
        return []
    rows = []
    for inner, ends in stretches(gt, _Frame(row, gt.shape, VOXEL_SIZE), LENGTHS):
        labels = np.where(np.isin(gt, inner), 0, gt).astype("uint32")
        _, row = row_context(labels, {}, VOXEL_SIZE)
        frame = _Frame(row, gt.shape, VOXEL_SIZE)
        coords = cell_voxels(labels)
        candidates = {label_id: frame.candidate(ANNOTATION, voxels) for label_id, voxels in coords.items()}
        low, high = sorted(candidates[end].position for end in ends)
        crop_spacing = _row_spacing(frame, coords, low, high)
        typical = [c.position for c in candidates.values() if c.eligible and not low < c.position < high]
        everything = [c.position for label_id, c in candidates.items()
                      if len(coords[label_id]) >= MIN_CELL_SIZE and c.distance < 15.0 and not low < c.position < high]
        limit = 1.6 * crop_spacing
        spacings = {"crop": crop_spacing,
                    "local_typical": local_spacing(typical, low, high, limit) or crop_spacing,
                    "local_all": local_spacing(everything, low, high, limit) or crop_spacing}
        for rule, spacing in spacings.items():
            rows.append({"crop": crop["name"], "val": crop["val"], "length": len(inner), "rule": rule,
                         "gap": high - low, "spacing": spacing, "count": int(round((high - low) / spacing)) - 1})
    print(f"{crop['name']}: {len(rows) // len(RULES)} stretches, {time.time() - start:.0f} s", flush=True)
    return rows


def summarize(runs):
    runs = runs.assign(error=runs["count"] - runs["length"])
    table = runs.groupby(["length", "rule"]).agg(
        stretches=("error", "size"), correct=("error", lambda e: float((e == 0).mean())),
        mean_abs_error=("error", lambda e: float(e.abs().mean())), bias=("error", "mean"),
    ).round(3)
    overall = runs.groupby("rule").agg(
        stretches=("error", "size"), correct=("error", lambda e: float((e == 0).mean())),
        mean_abs_error=("error", lambda e: float(e.abs().mean())),
    ).round(3)
    return table, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-j", "--workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    issues = set(pd.read_csv(OPEN_ISSUES, sep="\t")["crop"])
    crops = [crop for crop in gt_crops() if crop["name"] not in issues]
    print(f"{len(crops)} crops")
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [row for crop_rows in pool.map(run_crop, crops) for row in crop_rows]
    runs = pd.DataFrame(rows)
    runs.to_csv(os.path.join(args.output, "runs_count.tsv"), sep="\t", index=False)
    table, overall = summarize(runs)
    table.to_csv(os.path.join(args.output, "summary_count.tsv"), sep="\t")
    print(table.to_string())
    print(overall.to_string())


if __name__ == "__main__":
    main()
