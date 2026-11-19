"""Build IHC_v11_training_data_finalized, the final version of the IHC_v11 training crops.

The script combines three folders under ROOT:
- IHC_v11_training_data: the original images and annotations;
- IHC_v11_training_data_refined: the refined annotations of 2026-11-02 (README, changes.tsv);
- IHC_v11_training_data_corrected: the crops corrected after the refinement (README, crop_mapping.tsv).

Steps for each crop of the refined folder:
1. The source: the corrected crop of crop_mapping.tsv with its own image and new name, else the corrected
   annotation, else the refined annotation. The other crops use the original image.
2. The name: the 'Standardized' name of doc/data/IHC_v11.tsv.
3. Label IDs above ID_LIMIT get the next free IDs.
4. The cleanup: particles, holes enclosed in 3D, plane holes, and holes enclosed in 3D again. The hole limits are
   above the largest holes of the IHC_v11 crops, so all holes that one cell encloses and all islands between cells
   are filled.
5. The checks.

Outputs in TARGET: the folders train and val, changes.tsv (each change from the original to the final
annotation), id_map.tsv and crops.tsv. crops.tsv also goes to doc/revised_training_data_IHC_v11.tsv. The README.md
of TARGET is written by hand and stays.

Usage: python 2026-11_finalize_ihc_training_data.py [-f]
"""
import argparse
import importlib.util
import os
import re
import shutil
from collections import Counter
from glob import glob

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from flamingo_tools.annotation.dataset_cleanup import (
    fill_enclosed_background, fill_plane_holes, remap_high_ids, remove_particles,
)
from flamingo_tools.annotation.ids import compact

ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea")
SOURCE = os.path.join(ROOT, "IHC_v11_training_data")
REFINED = os.path.join(ROOT, "IHC_v11_training_data_refined")
CORRECTED = os.path.join(ROOT, "IHC_v11_training_data_corrected")
TARGET = os.path.join(ROOT, "IHC_v11_training_data_finalized")
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
NAME_TABLE = os.path.join(REPO, "doc", "data", "IHC_v11.tsv")
CROP_TABLE = os.path.join(REPO, "doc", "revised_training_data_IHC_v11.tsv")
TRAINING_SCRIPT = os.path.join(REPO, "scripts", "training", "train_distance_unet.py")
SPLITS = {"train": 75, "val": 14}

# All final IDs fit into 16 bits. The IDs above the limit are doubled IDs of split cells, e.g. 30993099 next to 3099.
ID_LIMIT = 65535
# A label inside the crop with fewer voxels is a particle (the size limit of the instance counts in doc/data).
MIN_PARTICLE_SIZE = 1000
# A label at the crop border with fewer voxels is a particle. A larger one is a cell that the crop cuts.
MIN_BORDER_SIZE = 100
# The largest island between several cells that the cleanup fills, in voxels. The largest island of the IHC_v11
# crops has 248 voxels.
MAX_ISLAND = 300
# The largest plane hole that the cleanup fills, in pixels (28.9 um2 at 0.38 um). The largest plane hole of the
# IHC_v11 crops has 178 pixels.
MAX_PLANE_HOLE = 200

# The reason of each corrected crop, from the README of the corrected folder.
CORRECTIONS = {
    "M-AMD-N97-L_crop_0216-0315-0680_Vglut3":
        "2 missing cells between 71 and 399: region 101-399 redone with the annotation tool (round 3) and Martin's "
        "cell 459",
    "MOTOF1L_0580-1175-1000":
        "missing cell next to 42825: Martin's manual edit (new cell 1 at the z border, 42825 gives 489 voxels)",
    "M-AMD-N97-L_crop_0632-0340-0948_Vglut3":
        "cells between 140 and 160 not separable from the image: new box z 152-511; 120 and 130 are two halves of "
        "one cell",
    "M-LR-000249-R_crop_0800-0680-0277_Vglut3":
        "missing tip of 21: Martin's manual edit (tip of 21, border 20/21)",
}

# The reasons of the refinement steps that have no comment (step column of the refined changes.tsv).
REFINED_STEPS = {
    "short_id": "ID above the uint32 maximum",
    "new_id": "reused ID: the second cell gets a new ID (the largest piece keeps the ID)",
    "new_id_s3": "reused ID: the second cell gets a new ID (the piece at the S3 anchor keeps the ID)",
    "merged": "small piece of a reused ID, merged into the touching label",
    "merged_s3": "small piece of a reused ID, merged into the touching label",
    "removed": "small piece of a reused ID without contact, removed",
    "removed_s3": "small piece of a reused ID without contact, removed",
}

GENERIC = {
    "id_remap": f"ID above {ID_LIMIT}",
    "split_id": "second cell with the same ID",
    "merge": "merged into another cell",
    "remove": "removed",
    "new_cell": "new cell",
    "edit": "",
}

CHANGE_COLUMNS = ["name", "original", "split", "step", "original_id", "final_id", "n_added", "n_removed",
                  "z", "y", "x", "reason"]


def crop_list():
    """The crops of the refined folder with their sources, names and splits."""
    names = pd.read_csv(NAME_TABLE, sep="\t", dtype=str).set_index("Original")
    mapping = pd.read_csv(os.path.join(CORRECTED, "crop_mapping.tsv"), sep="\t", dtype=str).set_index("original")
    crops = []
    for split in SPLITS:
        for path in sorted(glob(os.path.join(REFINED, split, "*_annotations.tif"))):
            original = os.path.basename(path)[:-len("_annotations.tif")]
            crop = {
                "original": original, "split": split, "name": names.loc[original, "Standardized"],
                "center_original": names.loc[original, "Crop_center"],
                "original_annotation": os.path.join(SOURCE, split, f"{original}_annotations.tif"),
                "original_image": os.path.join(SOURCE, split, f"{original}.tif"),
                "image": os.path.join(SOURCE, split, f"{original}.tif"), "offset": None,
            }
            corrected = os.path.join(CORRECTED, split, f"{original}_annotations.tif")
            if original in mapping.index:
                row = mapping.loc[original]
                if row["split"] != split:
                    raise ValueError(f"{original}: split {row['split']} in crop_mapping.tsv, {split} in the folder.")
                new = row["corrected"]
                crop.update(name=new, source="corrected, new box",
                            annotation=os.path.join(CORRECTED, split, f"{new}_annotations.tif"),
                            image=os.path.join(CORRECTED, split, f"{new}.tif"),
                            offset=tuple(int(v) for v in row["offset_zyx"].split(",")))
            elif os.path.exists(corrected):
                crop.update(source="corrected", annotation=corrected)
            else:
                crop.update(source="refined", annotation=path)
            crops.append(crop)
    return crops


def parse_ids(value):
    return [int(v) for v in re.findall(r"\d+", str(value))]


def refined_reasons(crop, refined_changes, short_ids):
    """The reasons of the refinement for each original ID of a crop, as (step, text) pairs."""
    to_original = {refined: original for original, refined in short_ids.items()}
    reasons = {}
    for _, row in refined_changes[refined_changes["crop"] == crop].iterrows():
        comment = row["comment"] if isinstance(row["comment"], str) and row["comment"] else None
        text = comment or REFINED_STEPS.get(row["step"], row["step"])
        for label_id in parse_ids(row["label_id"]) + [int(row["target"])]:
            if label_id > 0:
                reasons.setdefault(to_original.get(label_id, label_id), []).append((row["step"], text))
    return reasons


def box_union(*boxes):
    boxes = [box for box in boxes if box is not None]
    return tuple(slice(min(b[d].start for b in boxes), max(b[d].stop for b in boxes)) for d in range(len(boxes[0])))


def mask_center(mask, box):
    if not mask.any():
        return (None, None, None)
    return tuple(int(round(c)) + sl.start for c, sl in zip(ndimage.center_of_mass(mask), box))


def contingency(original_labels, final_labels, block=32):
    """The voxel counts of each pair (original compact ID, final compact ID)."""
    n_o, n_f = int(original_labels.max()) + 1, int(final_labels.max()) + 1
    counts = np.zeros(n_o * n_f, dtype="int64")
    for start in range(0, original_labels.shape[0], block):
        pairs = original_labels[start:start + block].astype("int64") * n_f + final_labels[start:start + block]
        counts += np.bincount(pairs.ravel(), minlength=n_o * n_f)
    return counts.reshape(n_o, n_f)


def curation_changes(original, curated, id_map):
    """The changes from the original to the curated annotation, one row per changed cell.

    A cell is matched to the original cell with the same ID (after `id_map`) if one of them is the main partner of
    the other. Equal IDs alone do not suffice, because a correction can give an ID to another cell.
    """
    o_lab, o_ids = compact(original)
    c_lab, c_ids = compact(curated)
    counts = contingency(o_lab, c_lab)
    o_size, c_size = counts.sum(axis=1), counts.sum(axis=0)
    o_boxes, c_boxes = ndimage.find_objects(o_lab), ndimage.find_objects(c_lab)
    c_index = {int(c): j for j, c in enumerate(c_ids)}
    n_o, n_c = len(o_ids), len(c_ids)
    best_c = counts.argmax(axis=1)
    best_o = counts[1:].argmax(axis=0) + 1 if n_o > 1 else np.zeros(n_c, dtype=int)

    matched = {}
    for i in range(1, n_o):
        j = c_index.get(id_map.get(int(o_ids[i]), int(o_ids[i])))
        if j and counts[i, j] > 0 and (best_o[j] == i or best_c[i] == j):
            matched[i] = j
    taken = set(matched.values())
    split_of, new_cells = {}, []
    for j in range(1, n_c):
        if j in taken:
            continue
        i = int(best_o[j])
        if n_o > 1 and counts[i, j] > 0 and counts[i, j] >= 0.5 * c_size[j]:
            split_of[j] = i
        else:
            new_cells.append(j)
    merged_into, removed = {}, []
    for i in range(1, n_o):
        if i in matched:
            continue
        j = int(best_c[i])
        if j > 0 and counts[i, j] >= 0.5 * o_size[i]:
            merged_into[i] = j
        else:
            removed.append(i)

    def row(step, i, j, n_added, n_removed, center):
        return {"step": step, "original_id": int(o_ids[i]) if i else None, "final_id": int(c_ids[j]) if j else 0,
                "n_added": int(n_added), "n_removed": int(n_removed), "center": center}

    def cell_center(lab, boxes, index):
        box = boxes[index - 1]
        return mask_center(lab[box] == index, box)

    rows = []
    for i, j in sorted(matched.items()):
        if c_ids[j] != o_ids[i]:
            rows.append(row("id_remap", i, j, 0, 0, cell_center(c_lab, c_boxes, j)))
        splits = [s for s, source in split_of.items() if source == i]
        merges = [m for m, target in merged_into.items() if target == j]
        n_removed = o_size[i] - counts[i, j] - sum(counts[i, s] for s in splits)
        n_added = c_size[j] - counts[i, j] - sum(counts[m, j] for m in merges)
        if n_added or n_removed:
            box = box_union(o_boxes[i - 1], c_boxes[j - 1])
            o_box, c_box = o_lab[box], c_lab[box]
            changed = (((c_box == j) & (o_box != i) & ~np.isin(o_box, merges))
                       | ((o_box == i) & (c_box != j) & ~np.isin(c_box, splits)))
            rows.append(row("edit", i, j, n_added, n_removed, mask_center(changed, box)))
    for j, i in sorted(split_of.items()):
        rows.append(row("split_id", i, j, c_size[j], 0, cell_center(c_lab, c_boxes, j)))
    for j in new_cells:
        rows.append(row("new_cell", 0, j, c_size[j], 0, cell_center(c_lab, c_boxes, j)))
    for i, j in sorted(merged_into.items()):
        box = box_union(o_boxes[i - 1], c_boxes[j - 1])
        rows.append(row("merge", i, j, counts[i, j], o_size[i] - counts[i, j],
                        mask_center((o_lab[box] == i) & (c_lab[box] == j), box)))
    for i in removed:
        rows.append(row("remove", i, 0, 0, o_size[i], cell_center(o_lab, o_boxes, i)))
    return rows


def filled_by_label(before, after):
    """The voxel count and the center of the filled voxels of each label."""
    changed = after != before
    if not changed.any():
        return []
    coords = np.nonzero(changed)
    ids, inverse, n_voxels = np.unique(after[coords], return_inverse=True, return_counts=True)
    centers = [np.bincount(inverse, weights=c) / n_voxels for c in coords]
    return [(int(label_id), int(n), tuple(int(round(c[k])) for c in centers))
            for k, (label_id, n) in enumerate(zip(ids, n_voxels))]


def check_annotation(name, labels, image_shape):
    """The checks of a final annotation; raises an error if one fails."""
    if labels.dtype != np.uint32 or labels.max() > ID_LIMIT:
        raise ValueError(f"{name}: dtype {labels.dtype}, largest ID {labels.max()}.")
    if labels.shape != tuple(image_shape):
        raise ValueError(f"{name}: annotation shape {labels.shape}, image shape {image_shape}.")
    full = np.ones((3,) * labels.ndim, dtype=bool)
    for label_id, box in enumerate(ndimage.find_objects(labels), start=1):
        if box is None:
            continue
        box = tuple(slice(max(sl.start - 1, 0), sl.stop + 1) for sl in box)
        region = labels[box]
        mask = region == label_id
        if ndimage.label(mask, full)[1] != 1:
            raise ValueError(f"{name}: ID {label_id} has several 26-connected pieces.")
        pieces, n_pieces = ndimage.label(mask)
        if n_pieces > 1 and np.sort(np.bincount(pieces.ravel())[1:])[-2] < MIN_PARTICLE_SIZE:
            raise ValueError(f"{name}: ID {label_id} has a small face-connected piece.")
        enclosed, n_enclosed = ndimage.label(ndimage.binary_fill_holes(mask) & ~mask)
        if n_enclosed:
            foreground = np.bincount(enclosed.ravel(), weights=region.ravel() > 0, minlength=n_enclosed + 1)[1:]
            if (foreground == 0).any():
                raise ValueError(f"{name}: ID {label_id} encloses background.")


def check_box_image(crop, target_image):
    """The image of a crop with a new box is the original image in that box."""
    image = tifffile.imread(target_image)
    box = tuple(slice(o, o + s) for o, s in zip(crop["offset"], image.shape))
    if not np.array_equal(tifffile.imread(crop["original_image"])[box], image):
        raise ValueError(f"{crop['name']}: the image differs from the original image in the new box.")


def crop_rows(crop, original, original_box, final):
    rows = []
    if crop["name"] != crop["original"]:
        reason = "standard name (doc/data/IHC_v11.tsv)" if crop["offset"] is None else \
            "new box: the name holds the center of the new box"
        rows.append({"step": "rename", "reason": reason})
    if crop["offset"] is not None:
        full_ids, full_counts = np.unique(original, return_counts=True)
        box_ids, box_counts = np.unique(original_box, return_counts=True)
        box_count = dict(zip(box_ids.tolist(), box_counts.tolist()))
        outside = [int(i) for i in full_ids if i > 0 and int(i) not in box_count]
        cut = [int(i) for i, n in zip(full_ids, full_counts) if i > 0 and 0 < box_count.get(int(i), 0) < n]
        start = ", ".join(str(o) for o in crop["offset"])
        reason = (f"new box {final.shape} at ({start}) of the original box {original.shape}; IDs outside the new "
                  f"box: {', '.join(map(str, outside)) or 'none'}; IDs cut by the new box: "
                  f"{', '.join(map(str, cut)) or 'none'}")
        rows.append({"step": "crop_box", "reason": reason})
    if original.dtype != np.uint32:
        rows.append({"step": "dtype", "reason": f"{original.dtype} -> uint32"})
    return rows


def process(crop, refined_changes):
    original = tifffile.imread(crop["original_annotation"])
    source = tifffile.imread(crop["annotation"])
    with tifffile.TiffFile(crop["image"]) as f:
        image_shape = tuple(f.series[0].shape)
    if crop["offset"] is None:
        original_box = original
    else:
        original_box = original[tuple(slice(o, o + s) for o, s in zip(crop["offset"], source.shape))]

    refined_rows = refined_changes[(refined_changes["crop"] == crop["original"])
                                   & (refined_changes["step"] == "short_id")]
    short_ids = {int(row["label_id"]): int(row["target"]) for _, row in refined_rows.iterrows()}
    curated, remapped = remap_high_ids(source, ID_LIMIT)
    id_map = {o: remapped.get(r, r) for o, r in short_ids.items()}
    id_map.update({o: n for o, n in remapped.items() if o not in short_ids.values()})

    cleaned, particles = remove_particles(curated, MIN_PARTICLE_SIZE, MIN_BORDER_SIZE)
    holes_3d, _ = fill_enclosed_background(cleaned, MAX_ISLAND)
    holes_2d, _ = fill_plane_holes(holes_3d, MAX_PLANE_HOLE)
    final, _ = fill_enclosed_background(holes_2d, MAX_ISLAND)
    check_annotation(crop["name"], final, image_shape)

    to_original = {n: o for o, n in id_map.items()}
    reasons = refined_reasons(crop["original"], refined_changes, short_ids)
    correction = CORRECTIONS.get(crop["original"])
    rows = crop_rows(crop, original, original_box, final)
    for change in curation_changes(original_box, curated, id_map):
        final_original = to_original.get(change["final_id"], change["final_id"])
        steps = reasons.get(change["original_id"], []) + reasons.get(final_original, [])
        if change["step"] == "id_remap":
            texts = [text for step, text in steps if step == "short_id"] or [GENERIC["id_remap"]]
        else:
            texts = [text for step, text in steps if step != "short_id"]
            if correction and (not texts or change["step"] == "new_cell"):
                texts.append(correction)
        texts = list(dict.fromkeys(texts)) or [GENERIC[change["step"]]]
        center = change.pop("center")
        rows.append({**change, **dict(zip("zyx", center)), "reason": "; ".join(t for t in texts if t)})
    for record in particles:
        if record["kind"] == "piece":
            kind = f"face-connected piece with less than {MIN_PARTICLE_SIZE} voxels"
        elif record["at_border"]:
            kind = f"label at the crop border with less than {MIN_BORDER_SIZE} voxels"
        else:
            kind = f"label inside the crop with less than {MIN_PARTICLE_SIZE} voxels"
        where = "removed" if record["target"] == 0 else "to the touching label"
        rows.append({"step": "particle", "original_id": to_original.get(record["label_id"], record["label_id"]),
                     "final_id": record["target"], "n_added": 0, "n_removed": record["n_voxels"],
                     **dict(zip("zyx", record["center"])), "reason": f"particle ({kind}), {where}"})
    for step, before, after, reason in [
        ("hole_3d", cleaned, holes_3d, "background enclosed in 3D"),
        ("hole_2d", holes_3d, holes_2d, f"plane holes of at most {MAX_PLANE_HOLE} pixels"),
        ("hole_3d", holes_2d, final, "background enclosed in 3D after the plane holes"),
    ]:
        for label_id, n_voxels, center in filled_by_label(before, after):
            rows.append({"step": step, "original_id": to_original.get(label_id, label_id), "final_id": label_id,
                         "n_added": n_voxels, "n_removed": 0, **dict(zip("zyx", center)), "reason": reason})
    for row in rows:
        row.update(name=crop["name"], original=crop["original"], split=crop["split"])

    id_rows = [{"name": crop["name"], "original": crop["original"], "original_id": o, "final_id": n,
                "reason": REFINED_STEPS["short_id"] if o in short_ids else GENERIC["id_remap"]}
               for o, n in sorted(id_map.items())]
    original_ids = np.unique(original)
    final_ids = np.unique(final)
    changed = (np.vectorize(lambda v: id_map.get(int(v), int(v)), otypes=["int64"])(original_ids)
               if len(id_map) else original_ids)
    lookup = np.searchsorted(original_ids, original_box)
    n_changed = int((changed[lookup] != final).sum())
    match = re.search(r"crop_(\d{4}-\d{4}-\d{4})", crop["name"])
    table_row = {
        "Original": crop["original"], "Name": crop["name"], "Split": crop["split"], "Source": crop["source"],
        "Crop_center_original": crop["center_original"], "Crop_center": match.group(1) if match else "unknown",
        "Shape_original_zyx": str(original.shape), "Shape_zyx": str(final.shape),
        "n_IHC_original": int((original_ids > 0).sum()), "n_IHC": int((final_ids > 0).sum()),
        "n_voxels_changed": n_changed, "Changes": summary(rows),
    }
    table_row["n_IHC_change"] = table_row["n_IHC"] - table_row["n_IHC_original"]
    return final, rows, id_rows, table_row


def summary(rows):
    """A short text of the changes of one crop."""
    steps = Counter(row["step"] for row in rows)
    volume = Counter()
    for row in rows:
        if row["step"] in ("particle", "hole_3d", "hole_2d"):
            volume[row["step"]] += row["n_added"] + row["n_removed"]
    parts = []
    for step, text in [("rename", "renamed"), ("crop_box", "new box")]:
        if steps[step]:
            parts.append(text)
    for step, singular, plural in [
        ("id_remap", "ID remapped", "IDs remapped"),
        ("split_id", "reused ID split", "reused IDs split"), ("merge", "cell merged", "cells merged"),
        ("remove", "cell removed", "cells removed"), ("new_cell", "new cell", "new cells"),
        ("edit", "cell edited", "cells edited"), ("particle", "particle", "particles"),
    ]:
        if steps[step]:
            parts.append(f"{steps[step]} {singular if steps[step] == 1 else plural}")
    if volume["particle"]:
        parts[-1] += f" ({volume['particle']} vox)"
    for step, text in [("hole_3d", "3D holes"), ("hole_2d", "plane holes")]:
        if volume[step]:
            parts.append(f"{text} {volume[step]} vox")
    return "; ".join(parts) or "none"


def check_set(crops, target):
    names = [crop["name"] for crop in crops]
    if len(set(names)) != len(names) or any("annotations" in name for name in names) or "annotations" in target:
        raise ValueError("The names are not unique, or a name or the target path contains 'annotations'.")
    counts = Counter(crop["split"] for crop in crops)
    if dict(counts) != SPLITS:
        raise ValueError(f"Split sizes {dict(counts)}, expected {SPLITS}.")
    spec = importlib.util.spec_from_file_location("train_distance_unet", TRAINING_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    images, labels, _ = module.get_image_and_label_paths(target)
    if len(images) != len(crops) or len(labels) != len(crops):
        raise ValueError(f"The training script finds {len(images)} images, expected {len(crops)}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--force", action="store_true", help="Replace the split folders and tables.")
    args = parser.parse_args()

    split_dirs = [os.path.join(TARGET, split) for split in SPLITS]
    if any(os.path.exists(path) for path in split_dirs):
        if not args.force:
            raise FileExistsError(f"{TARGET} holds split folders. Use -f to replace them.")
        for path in split_dirs:
            shutil.rmtree(path, ignore_errors=True)
    for path in split_dirs:
        os.makedirs(path)

    refined_changes = pd.read_csv(os.path.join(REFINED, "changes.tsv"), sep="\t", dtype={"comment": str})
    crops = crop_list()
    changes, id_rows, table = [], [], []
    for k, crop in enumerate(crops, start=1):
        final, rows, crop_ids, table_row = process(crop, refined_changes)
        target_dir = os.path.join(TARGET, crop["split"])
        target_image = os.path.join(target_dir, f"{crop['name']}.tif")
        shutil.copyfile(crop["image"], target_image)
        if crop["offset"] is not None:
            check_box_image(crop, target_image)
        tifffile.imwrite(os.path.join(target_dir, f"{crop['name']}_annotations.tif"), final, compression="zlib")
        changes += rows
        id_rows += crop_ids
        table.append(table_row)
        print(f"{k}/{len(crops)} {crop['name']}: {table_row['n_IHC_original']} -> {table_row['n_IHC']} IHCs; "
              f"{table_row['Changes']}", flush=True)
    check_set(crops, TARGET)

    integers = ["original_id", "final_id", "n_added", "n_removed", "z", "y", "x"]
    pd.DataFrame(changes, columns=CHANGE_COLUMNS).astype(dict.fromkeys(integers, "Int64")).to_csv(
        os.path.join(TARGET, "changes.tsv"), sep="\t", index=False)
    pd.DataFrame(id_rows, columns=["name", "original", "original_id", "final_id", "reason"]).to_csv(
        os.path.join(TARGET, "id_map.tsv"), sep="\t", index=False)
    columns = ["Original", "Name", "Split", "Source", "Crop_center_original", "Crop_center", "Shape_original_zyx",
               "Shape_zyx", "n_IHC_original", "n_IHC", "n_IHC_change", "n_voxels_changed", "Changes"]
    df = pd.DataFrame(table, columns=columns).sort_values(["Split", "Name"])
    df.to_csv(os.path.join(TARGET, "crops.tsv"), sep="\t", index=False)
    df.to_csv(CROP_TABLE, sep="\t", index=False)
    print(f"{len(crops)} crops, {len(changes)} changes. Tables in {TARGET} and {CROP_TABLE}.")


if __name__ == "__main__":
    main()
