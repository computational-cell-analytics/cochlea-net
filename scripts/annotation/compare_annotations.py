"""Compare corrected annotations of one crop: the changes against the base and the cleanliness of the cells.

For each output, the changed cells (labels that differ from the base) are measured:
- dice_base: Dice with the same label in the base. A cell that was no item target should stay near 1.
- small_pieces, small_voxels: the pieces of a label besides its largest one, with less than 1000 voxels.
- second_piece: the size of the second largest piece; 1000 voxels or more is a second cell with the same id.
- face_pieces: the pieces of a label besides its largest one with face connectivity (a piece can touch the cell
  at an edge or corner only).
- holes_3d: enclosed background voxels. holes_2d: enclosed background voxels on the z slices; holes_2d_all: on the
  planes along each axis.
- interleaved: the voxels at the borders with other changed cells whose 3x3x3 neighbourhood holds more of the
  other label than of their own.
With `--reference`, `best_reference_dice` is the Dice with the reference label of the largest overlap.

Usage:
    python compare_annotations.py --base s3_crop.tif --reference refined_annotations.tif \\
        --output round1=round1/train/X_annotations.tif --changes round1=round1/train/X_changes.tsv \\
        --output round2=train/X_annotations.tif -o comparison.tsv
"""
import argparse
import os

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

SPLIT_SIZE = 1000


def label_metrics(labels: np.ndarray, label_id: int) -> dict:
    mask = labels == label_id
    box = ndimage.find_objects(mask.astype("uint8"))[0]
    box = tuple(slice(max(sl.start - 1, 0), sl.stop + 1) for sl in box)
    mask, region = mask[box], labels[box]
    pieces, n_pieces = ndimage.label(mask, np.ones((3, 3, 3), dtype=bool))
    sizes = np.sort(np.bincount(pieces.ravel())[1:])[::-1]
    small = sizes[1:][sizes[1:] < SPLIT_SIZE]
    holes = ndimage.binary_fill_holes(mask) & (region == 0)
    holes_2d = [sum(int((ndimage.binary_fill_holes(m) & (r == 0)).sum())
                    for m, r in zip(np.moveaxis(mask, axis, 0), np.moveaxis(region, axis, 0)) if m.any())
                for axis in range(3)]
    return {"n_voxels": int(mask.sum()), "small_pieces": len(small), "small_voxels": int(small.sum()),
            "second_piece": int(sizes[1]) if len(sizes) > 1 else 0, "face_pieces": ndimage.label(mask)[1] - 1,
            "holes_3d": int(holes.sum()), "holes_2d": holes_2d[0], "holes_2d_all": sum(holes_2d)}


def interleaved(labels: np.ndarray, ids) -> dict:
    """The minority voxels of each label at its borders with the other listed labels."""
    counts = {int(i): 0 for i in ids}
    ids = list(ids)
    for a in ids:
        mask_a = labels == a
        box = ndimage.find_objects(mask_a.astype("uint8"))[0]
        box = tuple(slice(max(sl.start - 2, 0), sl.stop + 2) for sl in box)
        region = labels[box]
        own = ndimage.uniform_filter((region == a).astype("float32"), 3)
        for b in ids:
            if b == a or not (region == b).any():
                continue
            other = ndimage.uniform_filter((region == b).astype("float32"), 3)
            counts[int(a)] += int(((region == a) & (other > own)).sum())
    return counts


def compare(name: str, labels: np.ndarray, base: np.ndarray, reference, targets) -> pd.DataFrame:
    changed = labels != base
    ids = np.union1d(np.unique(labels[changed]), np.unique(base[changed]))
    ids = [int(i) for i in ids if i > 0 and (labels == i).any()]
    mixed = interleaved(labels, ids)
    rows = []
    for label_id in ids:
        a, b = labels == label_id, base == label_id
        row = {"output": name, "label_id": label_id, "target": label_id in targets if targets is not None else None,
               "dice_base": 2 * (a & b).sum() / (a.sum() + b.sum()), **label_metrics(labels, label_id),
               "interleaved": mixed[label_id]}
        if reference is not None:
            values, counts = np.unique(reference[a], return_counts=True)
            counts[values == 0] = 0
            if counts.max() > 0:
                ref_id = values[np.argmax(counts)]
                r = reference == ref_id
                row["reference_id"], row["best_reference_dice"] = int(ref_id), 2 * (a & r).sum() / (a.sum() + r.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def summary(table: pd.DataFrame) -> pd.DataFrame:
    def summarize(group):
        untargeted = group[group["target"] == False]  # noqa: E712
        return pd.Series({
            "changed_cells": len(group),
            "untargeted_cells": len(untargeted) if group["target"].notna().any() else np.nan,
            "untargeted_dice_base": untargeted["dice_base"].mean() if len(untargeted) else np.nan,
            "small_pieces": group["small_pieces"].sum(),
            "small_voxels": group["small_voxels"].sum(),
            "duplicate_ids": int((group["second_piece"] >= SPLIT_SIZE).sum()),
            "face_pieces": group["face_pieces"].sum(),
            "holes_3d": group["holes_3d"].sum(),
            "holes_2d": group["holes_2d"].sum(),
            "holes_2d_all": group["holes_2d_all"].sum(),
            "interleaved": group["interleaved"].sum(),
        })
    return table.groupby("output", sort=False).apply(summarize, include_groups=False)


def pairs(values):
    return dict(value.split("=", 1) for value in values or [])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="The segmentation that the corrections started from.")
    parser.add_argument("--reference", help="A reference annotation, for example the refined annotation.")
    parser.add_argument("--output", action="append", required=True, help="name=path of a corrected annotation.")
    parser.add_argument("--changes", action="append",
                        help="name=path of the change table of an output; its label_id column gives the targets.")
    parser.add_argument("-o", "--table", help="The per-cell table (tsv). The summary goes next to it.")
    args = parser.parse_args()
    base = tifffile.imread(args.base)
    reference = tifffile.imread(args.reference) if args.reference else None
    changes = pairs(args.changes)
    tables = []
    for name, path in pairs(args.output).items():
        targets = None
        if name in changes:
            targets = set(pd.read_csv(changes[name], sep="\t")["label_id"].dropna().astype(int))
        tables.append(compare(name, tifffile.imread(path), base, reference, targets))
    table = pd.concat(tables, ignore_index=True)
    result = summary(table)
    print(result.round(3).to_string())
    if args.table:
        os.makedirs(os.path.dirname(os.path.abspath(args.table)), exist_ok=True)
        table.round(4).to_csv(args.table, sep="\t", index=False)
        result.round(4).to_csv(args.table.replace(".tsv", "_summary.tsv"), sep="\t")


if __name__ == "__main__":
    main()
