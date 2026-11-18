"""Run the issue detectors of the annotation correction on the refined IHC annotations.

Stages:
- indent: the indent fraction of each complete cell for several closing radii, to set the indent limit.
- detect: the items per crop, and the recall on the open issues of `ihc_v11_refined_open_issues.tsv`. The
  detectors run without the issue table, so the recall shows what they find alone. With `--reference`, the S3
  reference segmentation (the latest IHC version of each cochlea) is compared too.

Usage:
    python detect_issues.py -o OUT --stage indent
    python detect_issues.py -o OUT --stage detect [--reference]
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage
from skimage.morphology import ball

from flamingo_tools.annotation.ids import compact
from flamingo_tools.annotation.issues import MIN_CELL_SIZE, cell_voxels, detect, read_issue_table

HERE = os.path.dirname(os.path.abspath(__file__))
CROP_ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea/IHC_v11_training_data_refined")
OPEN_ISSUES = os.path.join(HERE, "ihc_v11_refined_open_issues.tsv")
VOXEL_SIZE = 0.38
RADII = (1.5, 2.5, 3.5)


def crop_paths():
    paths = sorted(glob.glob(os.path.join(CROP_ROOT, "*", "*_annotations.tif")))
    return [p for p in paths if not p.endswith("_empty_annotations.tif")]


def crop_name(path):
    return os.path.basename(path)[:-len("_annotations.tif")]


def species_of(name):
    return "gerbil" if name.startswith("G") else "mouse"


def indent_fractions(path):
    """The indent fraction (closing - cell - other labels, relative to the cell) of each complete cell."""
    name = crop_name(path)
    labels, original = compact(tifffile.imread(path))
    issues = read_issue_table(OPEN_ISSUES, name)
    flagged = {}
    for record in issues.to_dict("records"):
        if isinstance(record["label_id"], str) and record["label_id"] != "all":
            flagged[int(record["label_id"])] = record["reason"]
    rows = []
    for label_id, voxels in cell_voxels(labels).items():
        at_border = (voxels.min(axis=0) == 0).any() or (voxels.max(axis=0) == np.array(labels.shape) - 1).any()
        if at_border or len(voxels) < MIN_CELL_SIZE:
            continue
        row = {"crop": name, "species": species_of(name), "label_id": int(original[label_id]),
               "issue": flagged.get(int(original[label_id]), "")}
        for radius in RADII:
            r = round(radius / VOXEL_SIZE)
            low = np.maximum(voxels.min(axis=0) - 2 * r, 0)
            high = np.minimum(voxels.max(axis=0) + 2 * r + 1, labels.shape)
            region = labels[tuple(slice(lo, hi) for lo, hi in zip(low, high))]
            cell = np.pad(region == label_id, r)
            closed = ndimage.binary_closing(cell, ball(r).astype(bool))[(slice(r, -r),) * 3]
            row[f"indent_{radius}"] = (closed & (region == 0)).sum() / len(voxels)
        rows.append(row)
    print(f"{name}: {len(rows)} cells", flush=True)
    return rows


def reference_crop(name, shape, cache_dir):
    """The S3 reference segmentation of a crop with compact ids, or None if the crop has no position."""
    from flamingo_tools.annotation.crop_reference import load_reference

    image_path = os.path.join(CROP_ROOT, "*", f"{name}.tif")
    image = tifffile.imread(glob.glob(image_path)[0])
    reference, _, _ = load_reference(name, image, "ihc", cache_dir)
    return None if reference is None else compact(reference)[0]


def detect_crop(path, use_reference, cache_dir):
    name = crop_name(path)
    labels, original = compact(tifffile.imread(path))
    reference = reference_crop(name, labels.shape, cache_dir) if use_reference else None
    items, row = detect(labels, VOXEL_SIZE, "ihc", reference=reference, original=original,
                        species=species_of(name))
    rows = []
    for item in items:
        rows.append({"crop": name, "item": item.item_id, "kind": item.kind,
                     "label_id": int(original[item.label_id]) if item.label_id else 0,
                     "z": item.anchor[0], "y": item.anchor[1], "x": item.anchor[2],
                     "reasons": ",".join(item.reasons), "proposal": 0 if item.proposal is None else len(item.proposal),
                     "comment": item.comment})
    n_cells = int(labels.max())
    print(f"{name}: {n_cells} labels, {len(items)} items, reference {reference is not None}", flush=True)
    return rows, {"crop": name, "labels": n_cells, "items": len(items), "reference": reference is not None,
                  "row": row is not None}


def recall(items, issues, voxel_size=VOXEL_SIZE, max_distance=10.0):
    """Which open issues an item covers: the same label, or a missing item within `max_distance` um."""
    rows = []
    for record in issues.to_dict("records"):
        found = items[items["crop"] == record["crop"]]
        if isinstance(record["label_id"], str) and record["label_id"] != "all":
            hit = found[found["label_id"] == int(record["label_id"])]
        elif record["label_id"] == "all":
            hit = found[found["kind"] == "cell"]
        else:
            position = np.array([record["z"], record["y"], record["x"]], dtype=float)
            distance = np.linalg.norm(found[["z", "y", "x"]].values - position, axis=1) * voxel_size
            hit = found[(found["kind"] == "missing") & (distance <= max_distance)]
        rows.append({**record, "found": len(hit) > 0, "reasons_found": ";".join(hit["reasons"].tolist())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--stage", choices=("indent", "detect"), required=True)
    parser.add_argument("--reference", action="store_true", help="Compare with the S3 reference segmentation.")
    parser.add_argument("-j", "--workers", type=int, default=4)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    paths = crop_paths()

    if args.stage == "indent":
        with ProcessPoolExecutor(args.workers) as pool:
            cells = pd.DataFrame([row for rows in pool.map(indent_fractions, paths) for row in rows])
        cells.to_csv(os.path.join(args.output, "indent_cells.tsv"), sep="\t", index=False)
        columns = [f"indent_{radius}" for radius in RADII]
        summary = cells.assign(group=np.where(cells["issue"] == "", "no issue", cells["issue"])).groupby(
            ["species", "group"])[columns].describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(3)
        print(summary.T.to_string())
        return

    cache_dir = os.path.join(args.output, "reference_cache")
    os.makedirs(cache_dir, exist_ok=True)
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(detect_crop, paths, [args.reference] * len(paths), [cache_dir] * len(paths)))
    items = pd.DataFrame([row for rows, _ in results for row in rows])
    crops = pd.DataFrame([info for _, info in results])
    suffix = "_reference" if args.reference else ""
    items.to_csv(os.path.join(args.output, f"items{suffix}.tsv"), sep="\t", index=False)
    crops.to_csv(os.path.join(args.output, f"crops{suffix}.tsv"), sep="\t", index=False)
    reasons = items["reasons"].str.split(",").explode().str.replace(r"^row_.*", "row", regex=True)
    print("\nItems per crop:", crops["items"].describe().round(1).to_dict())
    print("Item share of the labels:", round(crops["items"].sum() / crops["labels"].sum(), 3))
    print("Reasons:", reasons.value_counts().to_dict())
    found = recall(items, pd.read_csv(OPEN_ISSUES, sep="\t", dtype={"label_id": str}))
    found.to_csv(os.path.join(args.output, f"open_issue_recall{suffix}.tsv"), sep="\t", index=False)
    print("\nRecall on the open issues:")
    print(found.groupby("reason")["found"].agg(["sum", "size"]).to_string())


if __name__ == "__main__":
    main()
