"""Validate the region proposal (`--region A B`) on the refined IHC annotations: leave a stretch out.

For each crop, a stretch of 2 or 3 neighbouring typical row cells between two typical row cells is removed from
the annotation. The region proposal then fills the stretch from the S3 IHC model segmentations of the cochlea and
their row correction. Each stretch has its own row path, typical IHC and row correction, from the annotation
without the stretch.

The rules:
- newest: the eligible candidate of the newest model wins (the default of the tool);
- score: the eligible candidate with the best score wins, of any model;
- v11: as newest, with IHC_v11 and its row correction only;
- template: the typical IHC shape at each position, without candidates.

A proposed cell matches an annotated cell with IoU 0.5 or more. The crops of `gt_crops` (IHC post-processing) are
used without the crops with open issues. IHC_v11 trained on the train crops of its split, so the val crops are the
fair ones for IHC_v11.

Usage:
    python validate_region.py -o OUT -j 4
    python validate_region.py -o OUT --summary
"""
import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile

from flamingo_tools.annotation.crop_reference import ihc_sources, locate_crop, read_reference
from flamingo_tools.annotation.ids import compact
from flamingo_tools.annotation.issues import cell_voxels
from flamingo_tools.annotation.region import (
    ANNOTATION, CORRECTED, _Frame, propose_region, row_context, source_variants,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "..", "ihc_postprocessing"))
from common import gt_crops  # noqa: E402

OPEN_ISSUES = os.path.join(HERE, "ihc_v11_refined_open_issues.tsv")
CACHE = os.path.expanduser("~/Documents/lightsheet-cochlea/annotation_tool/detect_issues/reference_cache")
VOXEL_SIZE = 0.38
LENGTHS = (2, 3)
RULES = ("newest", "score", "v11", "template")


def stretches(labels, frame, lengths=LENGTHS):
    """One stretch of each length per crop: the inner cells and the two end cells, all typical row cells.

    The stretch is the middle one of all such stretches of the crop.
    """
    cells = []
    for label_id, voxels in cell_voxels(labels).items():
        candidate = frame.candidate(ANNOTATION, voxels)
        border = (voxels.min(axis=0) == 0).any() or (voxels.max(axis=0) == np.array(labels.shape) - 1).any()
        if candidate.distance < 15.0:
            cells.append((candidate.position, label_id, candidate.eligible and not border))
    cells.sort()
    result = []
    for length in lengths:
        windows = [cells[i:i + length + 2] for i in range(len(cells) - length - 1)]
        windows = [w for w in windows if all(typical for _, _, typical in w)]
        if windows:
            window = windows[len(windows) // 2]
            result.append(([label_id for _, label_id, _ in window[1:-1]], (window[0][1], window[-1][1])))
    return result


def match(cells, gt, inner):
    """Greedy matching of the proposed cells with the annotated inner cells, IoU 0.5 or more."""
    pairs = []
    for index, cell in enumerate(cells):
        values, counts = np.unique(gt[tuple(cell.voxels.T)], return_counts=True)
        for value, count in zip(values, counts):
            if value in inner:
                union = len(cell.voxels) + int((gt == value).sum()) - count
                pairs.append((count / union, index, int(value)))
    pairs.sort(reverse=True)
    used_cells, used_gt, matches = set(), set(), []
    for iou, index, value in pairs:
        if iou >= 0.5 and index not in used_cells and value not in used_gt:
            used_cells.add(index)
            used_gt.add(value)
            matches.append((index, value))
    return matches


def dice_of(voxels, mask):
    inside = mask[tuple(voxels.T)].sum()
    return 2 * inside / (len(voxels) + mask.sum())


def run_crop(crop):
    start = time.time()
    name = crop["name"]
    gt, _ = compact(tifffile.imread(crop["path"]))
    image = tifffile.imread(crop["path"].replace("_annotations.tif", ".tif"))
    info, types = locate_crop(name, image)
    if info is None:
        return []
    names = ihc_sources(types)
    sources = {source: compact(read_reference(info, source, CACHE))[0] for source in names}
    path, row = row_context(gt, sources, VOXEL_SIZE)
    rows = []
    for inner, ends in stretches(gt, _Frame(row, gt.shape, VOXEL_SIZE)):
        labels = np.where(np.isin(gt, inner), 0, gt).astype("uint32")
        path, row = row_context(labels, sources, VOXEL_SIZE)
        variants = source_variants(sources, path, row.reference, VOXEL_SIZE)
        for rule in RULES:
            if rule == "template":
                rule_sources = {}
            elif rule == "v11":
                rule_sources = {k: v for k, v in variants.items() if k in ("IHC_v11", "IHC_v11" + CORRECTED)}
                if not rule_sources:
                    continue
            else:
                rule_sources = variants
            try:
                proposal = propose_region(labels, labels, ends, rule_sources, VOXEL_SIZE, path_row=(path, row),
                                          correct=None, rule="score" if rule == "score" else "newest")
            except ValueError as error:
                print(f"{name} {ends}: {error}")
                continue
            matches = match(proposal.cells, gt, set(inner))
            dices = [dice_of(proposal.cells[index].voxels, gt == value) for index, value in matches]
            rows.append({
                "crop": name, "val": crop["val"], "species": "gerbil" if name.startswith("G") else "mouse",
                "sources": ",".join(names), "length": len(inner), "rule": rule, "n_proposed": len(proposal.cells),
                "tp": len(matches), "fp": len(proposal.cells) - len(matches), "fn": len(inner) - len(matches),
                "dice": float(np.mean(dices)) if dices else 0.0,
                "winners": ";".join(cell.comment.split(":")[0] for cell in proposal.cells),
            })
    print(f"{name}: {len(rows)} runs, {time.time() - start:.0f} s", flush=True)
    return rows


def summarize(out_dir):
    runs = pd.read_csv(os.path.join(out_dir, "runs_region.tsv"), sep="\t")
    common = runs.groupby(["crop", "length"])["rule"].transform(lambda r: set(RULES) <= set(r))
    runs = runs[common]

    def scores(group):
        tp, fp, fn = group["tp"].sum(), group["fp"].sum(), group["fn"].sum()
        return pd.Series({
            "stretches": len(group), "cells": int(group["length"].sum()), "f1": 2 * tp / (2 * tp + fp + fn),
            "matched_dice": (group["dice"] * group["tp"]).sum() / max(tp, 1),
            "count_error": (group["n_proposed"] - group["length"]).abs().mean(),
        })

    table = runs.groupby(["val", "rule"]).apply(scores, include_groups=False).round(3)
    table.to_csv(os.path.join(out_dir, "summary_region.tsv"), sep="\t")
    print(table.to_string())
    winners = runs[runs["rule"] == "newest"]["winners"].str.split(";").explode()
    winners = winners.str.replace(r" \+ .*", " + union", regex=True).value_counts()
    winners.to_csv(os.path.join(out_dir, "winners_region.tsv"), sep="\t")
    print("\nWinners of the rule 'newest':\n", winners.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if not args.summary:
        issues = set(pd.read_csv(OPEN_ISSUES, sep="\t")["crop"])
        crops = [crop for crop in gt_crops() if crop["name"] not in issues]
        print(f"{len(crops)} crops")
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [row for crop_rows in pool.map(run_crop, crops) for row in crop_rows]
        pd.DataFrame(rows).to_csv(os.path.join(args.output, "runs_region.tsv"), sep="\t", index=False)
    summarize(args.output)


if __name__ == "__main__":
    main()
