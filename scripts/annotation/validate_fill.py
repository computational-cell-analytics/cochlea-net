"""Validate the sparse-slice fill of the annotation correction on the refined IHC annotations.

A simulated user corrects one complete cell per run. Each visit takes the slices of `select_slices` for the
current cell, paints the annotation on them, and runs `fill_sparse`. The visits repeat until the painted slices
change nothing, as in the correction loop. The scenarios:
- cut2, cut4: one tip of the cell is cut by 2 or 4 um along its long axis.
- cut4_noisy: as cut4, but the user paints the cell 1 pixel too large or too small on each slice.
- removed: the cell is missing. The slices come from the cell eroded by 2 um, as from an imperfect proposal.
- boundary: the voxels of the cell within 1.5 um of a touching neighbour belong to the neighbour.
- <scenario>_brush (stage cleanup): as the scenario, but the user paints like a brush: gaps of 2 x 2 pixels in
  the cell, stray pixels next to it, and left-over pixels of the label that the cell covers.

The stage 'cleanup' compares the processing of the fill: 'off' (`fill_sparse` alone, all labels editable, as in
round 1), 'focus' (only the cell and the cells that touch it change), and 'clean_<sigma>' (focus, the cleanup of
the slices and of the result, and the border smoothing with sigma in pixels).

The crops with open issues (`ihc_v11_refined_open_issues.tsv`) are left out, because their annotation is not
a reference.

Usage:
    python validate_fill.py -o OUT --stage params
    python validate_fill.py -o OUT --stage axis
    python validate_fill.py -o OUT --stage cleanup
    python validate_fill.py -o OUT --stage summary
"""
import argparse
import glob
import itertools
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

from flamingo_tools.annotation.controller import Settings, fill_visit, focus_ids, visit_geometry
from flamingo_tools.annotation.fill import fill_sparse
from flamingo_tools.annotation.ids import compact

HERE = os.path.dirname(os.path.abspath(__file__))
CROP_ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea/IHC_v11_training_data_refined")
OPEN_ISSUES = os.path.join(HERE, "ihc_v11_refined_open_issues.tsv")
VOXEL_SIZE = 0.38
MIN_CELL_SIZE = 5000
MAX_VISITS = 6
TOLERANCE = 0.1
MIN_DIFFERENCE = 20
SCENARIOS = ("cut2", "cut4", "cut4_noisy", "removed", "boundary")
CLEANUP_SCENARIOS = ("cut4", "removed", "boundary", "cut4_brush", "removed_brush", "boundary_brush")
CLEANUP = ("off", "focus", "clean_0", "clean_1", "clean_2")


def stage_grid(stage):
    """The fill settings of a stage, as a list of dicts."""
    base = {"spacing": 2.0, "cap": 2.0, "vanish": 0.7, "max_slices": 16}
    if stage == "cleanup":
        return [{**base, "cleanup": name} for name in CLEANUP]
    if stage == "axis":
        return [{**base, "axis": "z", "n_slices": 0}] + [
            {**base, "axis": "long", "n_slices": n} for n in (8, 10, 12)]
    return [
        {"spacing": s, "cap": c, "vanish": v, "max_slices": 30}
        for s, c, v in itertools.product((2.0, 3.0), (2.0, 3.0), (0.3, 0.5, 0.7))
    ]


def crops():
    issues = set(pd.read_csv(OPEN_ISSUES, sep="\t")["crop"])
    paths = []
    for path in sorted(glob.glob(os.path.join(CROP_ROOT, "*", "*_annotations.tif"))):
        name = os.path.basename(path)[:-len("_annotations.tif")]
        if name not in issues and not name.endswith("_empty"):
            paths.append(path)
    return paths


def complete_cells(labels):
    """The cells that do not touch the crop border and have MIN_CELL_SIZE voxels or more."""
    cells = []
    for label_id, bbox in enumerate(ndimage.find_objects(labels), start=1):
        if bbox is None or any(sl.start == 0 or sl.stop == size for sl, size in zip(bbox, labels.shape)):
            continue
        if (labels[bbox] == label_id).sum() >= MIN_CELL_SIZE:
            cells.append(label_id)
    return cells


def long_axis(mask):
    voxels = np.argwhere(mask).astype(float)
    return np.linalg.eigh(np.cov(voxels.T))[1][:, -1]


def neighbour_of(labels, cell):
    """The label with the largest contact to the cell, or 0."""
    shell = ndimage.binary_dilation(labels == cell, iterations=1) & (labels != cell) & (labels > 0)
    if not shell.any():
        return 0
    ids, counts = np.unique(labels[shell], return_counts=True)
    return int(ids[np.argmax(counts)])


def apply_scenario(gt, cell, scenario, rng):
    """The labels with the error of the scenario, and the mask for the first slice selection."""
    labels = gt.copy()
    mask = gt == cell
    if scenario.startswith("cut"):
        cut = 2.0 if scenario == "cut2" else 4.0
        voxels = np.argwhere(mask)
        projection = voxels @ long_axis(mask) * VOXEL_SIZE
        if rng.random() < 0.5:
            projection = -projection
        removed = voxels[projection > projection.max() - cut]
        labels[tuple(removed.T)] = 0
        return labels, labels == cell
    if scenario == "removed":
        labels[mask] = 0
        proposal = ndimage.binary_erosion(mask, iterations=round(2.0 / VOXEL_SIZE))
        return labels, proposal if proposal.any() else mask
    neighbour = neighbour_of(gt, cell)
    if neighbour == 0:
        return None, None
    near = ndimage.distance_transform_edt(gt != neighbour) * VOXEL_SIZE <= 1.5
    labels[mask & near] = neighbour
    return labels, labels == cell


def brush_noise(painted, before, cell, rng):
    """A painted slice with the errors of a brush: gaps in the cell, stray pixels and left-over pixels."""
    out = painted.copy()
    cell_mask = out == cell
    n = max(1, int(cell_mask.sum()) // 150)

    def pick(mask, k):
        points = np.argwhere(mask)
        return points[rng.choice(len(points), min(k, len(points)), replace=False)] if len(points) else points

    for y, x in pick(ndimage.binary_erosion(cell_mask, iterations=2), n):
        gap = out[y:y + 2, x:x + 2]
        gap[gap == cell] = 0
    ring = ndimage.binary_dilation(cell_mask, iterations=3) & ~ndimage.binary_dilation(cell_mask) & (out == 0)
    for y, x in pick(ring, n):
        out[y, x] = cell
    for y, x in pick(cell_mask & (before > 0) & (before != cell), n):
        out[y, x] = before[y, x]
    return out


def user_slices(current_slices, gt_slices, cell, noisy, rng, brush=False):
    """The slices as the simulated user paints them.

    The user repaints a slice only if it differs from the annotation in more than TOLERANCE of the cell area on
    the slice (at least MIN_DIFFERENCE pixels), like an annotator who ignores small deviations.
    """
    painted = current_slices.copy()
    for index, (current, slice_) in enumerate(zip(current_slices, gt_slices)):
        cell_mask = slice_ == cell
        if (current != slice_).sum() <= max(MIN_DIFFERENCE, TOLERANCE * cell_mask.sum()):
            continue
        painted[index] = brush_noise(slice_, current, cell, rng) if brush else slice_
        if not noisy or not cell_mask.any():
            continue
        if rng.random() < 0.5:
            grown = ndimage.binary_dilation(cell_mask) & (slice_ == 0)
            painted[index][grown] = cell
        else:
            painted[index][cell_mask & ~ndimage.binary_erosion(cell_mask)] = 0
    return painted


def local_box(gt, cell, margin=15.0):
    """The bounding box of a cell with a margin in micrometer, clipped to the crop."""
    bbox = ndimage.find_objects((gt == cell).astype("uint8"))[0]
    m = round(margin / VOXEL_SIZE)
    return tuple(slice(max(0, sl.start - m), min(size, sl.stop + m)) for sl, size in zip(bbox, gt.shape))


def processing_settings(settings):
    """The controller settings of a run; the cleanup settings follow the variant of the stage 'cleanup'."""
    variant = settings.get("cleanup", "off")
    sigma = float(variant.split("_")[1]) if variant.startswith("clean_") else 0.0
    cleanup = variant.startswith("clean_")
    return Settings(
        voxel_size=VOXEL_SIZE, slice_axis=settings.get("axis", "z"), spacing=settings["spacing"],
        n_slices=settings.get("n_slices", 10), cap=settings["cap"], max_slices=settings["max_slices"],
        vanish=settings["vanish"], min_area=2.0 if cleanup else 0.0, boundary_sigma=sigma,
        split_size=1000 if cleanup else 10 ** 9,
    ), variant


def cleanliness(labels, ids):
    """Small pieces, 2D hole voxels and interleaved border voxels of the listed labels."""
    small, holes, mixed = 0, 0, 0
    structure = np.ones((3, 3, 3), dtype=bool)
    for label_id in ids:
        mask = labels == label_id
        if not mask.any():
            continue
        sizes = np.sort(np.bincount(ndimage.label(mask, structure)[0].ravel())[1:])[::-1]
        small += int((sizes[1:] < 1000).sum())
        holes += sum(int((ndimage.binary_fill_holes(m) & (r == 0)).sum()) for m, r in zip(mask, labels) if m.any())
        own = ndimage.uniform_filter(mask.astype("float32"), 3)
        for other_id in ids:
            if other_id != label_id:
                other = ndimage.uniform_filter((labels == other_id).astype("float32"), 3)
                mixed += int((mask & (other > own)).sum())
    return small, holes, mixed


def simulate(gt, cell, scenario, settings, seed):
    rng = np.random.default_rng(seed)
    labels, mask = apply_scenario(gt, cell, scenario.replace("_brush", ""), rng)
    if labels is None:
        return None
    initial = labels.copy()
    visits, changed_slices, painted_slices = 0, 0, 0
    start = time.time()
    geometry, variant = processing_settings(settings)
    next_label = [int(labels.max()) + 1]
    editable_all = set()

    def new_label():
        next_label[0] += 1
        return next_label[0] - 1

    for _ in range(MAX_VISITS):
        cell_mask = (mask | (labels == cell)) if mask.any() else labels == cell
        axis, roi, z = visit_geometry(labels, cell_mask, geometry)
        current = labels[roi]
        edit = current.copy()
        moved_current, moved_edit = np.moveaxis(current, axis, 0), np.moveaxis(edit, axis, 0)
        moved_edit[z] = user_slices(moved_current[z], np.moveaxis(gt[roi], axis, 0)[z], cell,
                                    scenario.endswith("noisy"), rng, brush=scenario.endswith("brush"))
        n_changed = sum(not np.array_equal(moved_edit[s], moved_current[s]) for s in z)
        if n_changed == 0:
            break
        visits += 1
        changed_slices += n_changed
        painted_slices += len(z)
        if variant == "off":
            result, _ = fill_sparse(moved_current, moved_edit, z, VOXEL_SIZE, vanish=settings["vanish"])
            labels[roi] = np.moveaxis(result, 0, axis)
            editable_all |= set(np.unique(current).tolist())
        else:
            editable = focus_ids(current, cell_mask[roi], cell, geometry.min_contact,
                                 round(geometry.focus_gap / VOXEL_SIZE))
            editable_all |= set(editable)
            labels[roi] = fill_visit(current, edit, axis, z, editable, geometry, new_label)[0]
        mask = labels == cell

    truth, result = gt == cell, labels == cell
    dice = 2 * (truth & result).sum() / (truth.sum() + result.sum())
    near = ndimage.binary_dilation(truth, iterations=round(5.0 / VOXEL_SIZE))
    others = near & (gt > 0) & ~truth
    neighbours = [i for i in np.unique(gt[near]) if i not in (0, cell)]
    small, holes, mixed = cleanliness(labels, [cell] + neighbours)
    allowed = list(editable_all | {0}) + list(range(int(initial.max()) + 1, next_label[0]))
    changed = labels != initial
    locked = changed & (~np.isin(initial, allowed) | ~np.isin(labels, allowed))
    return {
        "dice": dice,
        "excess": (result & ~truth).sum() / truth.sum(),
        "missing": (truth & ~result).sum() / truth.sum(),
        "neighbour_error": float(np.mean(labels[others] != gt[others])) if others.any() else 0.0,
        "small_pieces": small, "holes_2d": holes, "interleaved": mixed, "locked_changes": int(locked.sum()),
        "visits": visits,
        "changed_slices": changed_slices,
        "slices": painted_slices,
        "seconds": time.time() - start,
    }


def run_crop(path, grid, n_cells, seed, scenarios=SCENARIOS):
    name = os.path.basename(path)[:-len("_annotations.tif")]
    gt, _ = compact(tifffile.imread(path))
    cells = complete_cells(gt)
    rng = np.random.default_rng(seed)
    cells = sorted(rng.choice(cells, size=min(n_cells, len(cells)), replace=False).tolist()) if cells else []
    rows = []
    for cell in cells:
        box = local_box(gt, cell)
        local_gt = gt[box]
        orientation = "z" if abs(long_axis(local_gt == cell)[0]) > 0.5 else "xy"
        for scenario, settings in itertools.product(scenarios, grid):
            result = simulate(local_gt, cell, scenario, settings, seed + cell)
            if result is not None:
                rows.append({"crop": name, "species": "gerbil" if name.startswith("G") else "mouse", "cell": cell,
                             "orientation": orientation, "scenario": scenario, **settings, **result})
    print(f"{name}: {len(cells)} cells, {len(rows)} runs", flush=True)
    return rows


def summarize(out_dir):
    for stage in ("params", "axis", "cleanup"):
        path = os.path.join(out_dir, f"runs_{stage}.tsv")
        if not os.path.exists(path):
            continue
        runs = pd.read_csv(path, sep="\t")
        keys = ["spacing", "cap", "vanish", "max_slices"]
        if "axis" in runs:
            keys += ["axis", "n_slices"]
        if "cleanup" in runs:
            summarize_cleanup(runs, out_dir)
            continue
        table = runs.groupby(keys + ["scenario"]).agg(
            dice=("dice", "median"), dice_p10=("dice", lambda d: np.percentile(d, 10)),
            excess=("excess", "median"), neighbour_error=("neighbour_error", "mean"),
            visits=("visits", "mean"), changed_slices=("changed_slices", "mean"), n=("dice", "size"),
        ).round(3).reset_index()
        table.to_csv(os.path.join(out_dir, f"summary_{stage}.tsv"), sep="\t", index=False)
        overall = runs.groupby(keys + ["orientation"])["dice"].median().unstack("orientation").round(3)
        overall["mean_dice"] = runs.groupby(keys)["dice"].mean().round(3)
        overall["dice_p10"] = runs.groupby(keys)["dice"].quantile(0.1).round(3)
        overall["neighbour_error"] = runs.groupby(keys)["neighbour_error"].mean().round(4)
        overall["changed_slices"] = runs.groupby(keys)["changed_slices"].mean().round(1)
        overall.sort_values("mean_dice", ascending=False).to_csv(
            os.path.join(out_dir, f"overall_{stage}.tsv"), sep="\t")
        print(f"\n{stage}:\n", overall.sort_values("mean_dice", ascending=False).head(15).to_string())


def summarize_cleanup(runs, out_dir):
    runs = runs.assign(brush=runs["scenario"].str.endswith("_brush"))
    metrics = dict(dice=("dice", "mean"), dice_p10=("dice", lambda d: np.percentile(d, 10)),
                   neighbour_error=("neighbour_error", "mean"), small_pieces=("small_pieces", "mean"),
                   holes_2d=("holes_2d", "mean"), interleaved=("interleaved", "mean"),
                   locked_changes=("locked_changes", "sum"), visits=("visits", "mean"),
                   changed_slices=("changed_slices", "mean"), n=("dice", "size"))
    table = runs.groupby(["cleanup", "scenario"]).agg(**metrics).round(4).reset_index()
    table.to_csv(os.path.join(out_dir, "summary_cleanup.tsv"), sep="\t", index=False)
    overall = runs.groupby(["cleanup", "brush"]).agg(**metrics).round(4)
    overall.to_csv(os.path.join(out_dir, "overall_cleanup.tsv"), sep="\t")
    print("\ncleanup:\n", overall.to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--stage", choices=("params", "axis", "cleanup", "summary"), required=True)
    parser.add_argument("--n_cells", type=int, default=2, help="Cells per crop.")
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    if args.stage == "summary":
        summarize(args.output)
        return

    grid = stage_grid(args.stage)
    scenarios = CLEANUP_SCENARIOS if args.stage == "cleanup" else SCENARIOS
    paths = crops()
    print(f"{len(paths)} crops, {len(grid)} settings")
    with ProcessPoolExecutor(args.workers) as pool:
        results = pool.map(run_crop, paths, [grid] * len(paths), [args.n_cells] * len(paths),
                           [args.seed] * len(paths), [scenarios] * len(paths))
        rows = [row for crop_rows in results for row in crop_rows]
    pd.DataFrame(rows).to_csv(os.path.join(args.output, f"runs_{args.stage}.tsv"), sep="\t", index=False)
    summarize(args.output)


if __name__ == "__main__":
    main()
