"""Repair label IDs that cover two disjoint places in an OME-Zarr segmentation.

A segmentation stage that inherited a stale 'seeds.zarr' from an earlier run can fuse an object
with an unrelated, far-away seed, so one label ID ends up as a normal cell plus a small fragment
hundreds of micrometer away. This script keeps the largest connected component of every affected
label and removes the rest, without renumbering anything. Preserving the label IDs is the point:
the marker classification and the object measures are keyed by label ID.

The repair runs on a local rclone copy of the pyramid and writes local artifacts only. Use
'--stage fetch' first, then run the remaining stages in order. Nothing is uploaded; the final
stage prints the rclone commands.

Example:
    W=./repair_M_AMD_OTOF27_L
    python scripts/repair_split_labels.py -w $W --stage fetch
    python scripts/repair_split_labels.py -w $W --stage analyze
    python scripts/repair_split_labels.py -w $W --stage repair --allow_ambiguous
    python scripts/repair_split_labels.py -w $W --stage table
    python scripts/repair_split_labels.py -w $W --stage measures
    python scripts/repair_split_labels.py -w $W --stage verify
"""
import argparse
import glob
import os
import shutil
import subprocess

import numpy as np
import pandas as pd
import zarr
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

VOXEL_SIZE = 0.38
BUCKET = "cochlea-lightsheet"
RCLONE_REMOTE = "cochlea-lightsheet"

# Keeping the largest component is only obviously right when it dominates. Labels below these
# limits are reported for a decision instead of being repaired silently.
MIN_SIZE_RATIO = 5.0
MAX_DROP_FRACTION = 0.2


def _paths(work_dir, segmentation):
    return {
        "zarr": os.path.join(work_dir, f"{segmentation}.ome.zarr"),
        "backup": os.path.join(work_dir, f"backup_{segmentation}.ome.zarr"),
        "tables_orig": os.path.join(work_dir, "tables_orig"),
        "tables_new": os.path.join(work_dir, "tables_repaired"),
        "report": os.path.join(work_dir, "report"),
    }


def _chunk_keys(level_dir):
    """Chunk indices that exist on disk. The volume is sparse, so the full grid is not iterated."""
    keys = []
    for path in glob.glob(os.path.join(level_dir, "*", "*", "*")):
        parts = path.split(os.sep)[-3:]
        if all(part.isdigit() for part in parts):
            keys.append(tuple(int(part) for part in parts))
    return sorted(keys)


def read_foreground(array, level_dir):
    """Return (labels, coords_zyx) for every non-zero voxel, reading only the chunks that exist."""
    cz, cy, cx = array.chunks
    labs, zs, ys, xs = [], [], [], []
    for kz, ky, kx in _chunk_keys(level_dir):
        z0, y0, x0 = kz * cz, ky * cy, kx * cx
        block = array[z0:z0 + cz, y0:y0 + cy, x0:x0 + cx]
        nz = np.nonzero(block)
        if nz[0].size == 0:
            continue
        labs.append(block[nz].astype("uint32"))
        zs.append((nz[0] + z0).astype("int32"))
        ys.append((nz[1] + y0).astype("int32"))
        xs.append((nz[2] + x0).astype("int32"))
    if not labs:
        return np.zeros(0, "uint32"), np.zeros((0, 3), "int32")
    return np.concatenate(labs), np.stack([np.concatenate(zs), np.concatenate(ys), np.concatenate(xs)], axis=1)


def _encode(coords, shape):
    stride_z, stride_y = shape[1] * shape[2], shape[2]
    return (coords[:, 0].astype("int64") * stride_z + coords[:, 1].astype("int64") * stride_y +
            coords[:, 2].astype("int64"))


def label_components_6(coords, shape):
    """6-connectivity connected components over one label's voxel coordinates."""
    keys = _encode(coords, shape)
    order = np.argsort(keys)
    keys_sorted = keys[order]
    rows, cols = [], []
    for stride in (shape[1] * shape[2], shape[2], 1):
        neighbors = keys_sorted + stride
        pos = np.searchsorted(keys_sorted, neighbors)
        valid = pos < keys_sorted.size
        valid[valid] &= keys_sorted[pos[valid]] == neighbors[valid]
        if valid.any():
            rows.append(np.nonzero(valid)[0])
            cols.append(pos[valid])
    if rows:
        a, b = np.concatenate(rows), np.concatenate(cols)
        graph = coo_matrix((np.ones(a.size, "int8"), (a, b)), shape=(keys_sorted.size,) * 2)
        n_comp, comp_sorted = connected_components(graph, directed=False)
    else:
        n_comp, comp_sorted = keys_sorted.size, np.arange(keys_sorted.size)
    comp = np.empty(coords.shape[0], dtype="int64")
    comp[order] = comp_sorted
    return n_comp, comp


def base_columns(labs, coords):
    """The base 'default.tsv' columns for the given voxels.

    Reproduces the existing table exactly: 'anchor_*' is the plain centre of mass in voxel indices
    times the resolution, and 'bb_max_*' is the inclusive maximum index times the resolution.
    Verified against all 1592 unaffected labels of M_AMD_OTOF27_L / IHC_v11.
    """
    order = np.argsort(labs, kind="stable")
    uniq, starts, counts = np.unique(labs[order], return_index=True, return_counts=True)
    rows = []
    for u, start, count in zip(uniq, starts, counts):
        c = coords[order[start:start + count]].astype("int64")
        bb_min, bb_max, com = c.min(0), c.max(0), c.mean(0)
        row = {"label_id": int(u), "n_pixels": int(count)}
        for axis, name in enumerate(("z", "y", "x")):
            row[f"anchor_{name}"] = float(com[axis] * VOXEL_SIZE)
            row[f"bb_min_{name}"] = float(bb_min[axis] * VOXEL_SIZE)
            row[f"bb_max_{name}"] = float(bb_max[axis] * VOXEL_SIZE)
        rows.append(row)
    columns = (["label_id"] +
               [f"{prefix}_{ax}" for prefix in ("anchor", "bb_min", "bb_max") for ax in ("x", "y", "z")] +
               ["n_pixels"])
    return pd.DataFrame(rows)[columns]


def zero_voxels(array, coords, label_ids):
    """Set the given voxels to zero, one chunk at a time. Nothing else is modified."""
    if coords.shape[0] == 0:
        return 0
    chunks = np.asarray(array.chunks)
    chunk_idx = coords // chunks
    grid = chunk_idx.max(0) + 1
    keys = (chunk_idx[:, 0] * grid[1] + chunk_idx[:, 1]) * grid[2] + chunk_idx[:, 2]
    n_zeroed = 0
    for key in np.unique(keys):
        sel = keys == key
        local = coords[sel]
        kz, ky, kx = chunk_idx[sel][0]
        z0, y0, x0 = kz * chunks[0], ky * chunks[1], kx * chunks[2]
        block = array[z0:z0 + chunks[0], y0:y0 + chunks[1], x0:x0 + chunks[2]]
        inner = (local[:, 0] - z0, local[:, 1] - y0, local[:, 2] - x0)
        # Compare against the label this coordinate was recorded for, not the set of trimmed labels:
        # at coarse levels a voxel may hold a different label than the one whose fragment maps there.
        hit = block[inner] == label_ids[sel]
        idx = tuple(a[hit] for a in inner)
        if idx[0].size:
            block[idx] = 0
            array[z0:z0 + chunks[0], y0:y0 + chunks[1], x0:x0 + chunks[2]] = block
            n_zeroed += int(idx[0].size)
    return n_zeroed


def stage_fetch(args, paths):
    os.makedirs(args.work_dir, exist_ok=True)
    remote = f"{RCLONE_REMOTE}:{BUCKET}/{args.cochlea}"
    if not os.path.isdir(paths["zarr"]):
        subprocess.run(["rclone", "--progress", "copyto",
                        f"{remote}/images/ome-zarr/{args.segmentation}.ome.zarr", paths["zarr"]], check=True)
    if not os.path.isdir(paths["tables_orig"]):
        subprocess.run(["rclone", "--progress", "copy",
                        f"{remote}/tables/{args.segmentation}", paths["tables_orig"]], check=True)
    if not os.path.isdir(paths["backup"]):
        shutil.copytree(paths["zarr"], paths["backup"])
    print(f"Working copy : {paths['zarr']}")
    print(f"Backup       : {paths['backup']}  (never written; restore from here to roll back)")


def _analyze(paths, which="zarr"):
    """Find every label with more than one connected component at s0."""
    level_dir = os.path.join(paths[which], "s0")
    array = zarr.open(level_dir, mode="r")
    labs, coords = read_foreground(array, level_dir)
    print(f"s0 {array.shape} {array.dtype}: {labs.size} foreground voxels, {np.unique(labs).size} labels")

    order = np.argsort(labs, kind="stable")
    uniq, starts, counts = np.unique(labs[order], return_index=True, return_counts=True)
    rows, kept, dropped = [], {}, {}
    for u, start, count in zip(uniq, starts, counts):
        idx = order[start:start + count]
        c = coords[idx]
        n_comp, comp = label_components_6(c, array.shape)
        sizes = np.bincount(comp)
        best = int(np.argmax(sizes))
        row = {"label_id": int(u), "n_vox": int(count), "n_components": int(n_comp),
               "kept": int(sizes[best]), "dropped": int(count - sizes[best]),
               "second": int(np.sort(sizes)[::-1][1]) if n_comp > 1 else 0,
               "comp_sizes": np.sort(sizes)[::-1][:6].tolist()}
        rows.append(row)
        if n_comp > 1:
            kept[int(u)] = c[comp == best]
            dropped[int(u)] = c[comp != best]
    report = pd.DataFrame(rows)
    report["drop_frac"] = report.dropped / report.n_vox
    report["ratio"] = np.where(report.second > 0, report.kept / report.second, np.inf)
    return report, kept, dropped, labs, coords, array.shape


def stage_analyze(args, paths):
    report, kept, dropped, _, _, _ = _analyze(paths)
    os.makedirs(paths["report"], exist_ok=True)
    table = pd.read_csv(os.path.join(paths["tables_orig"], "default.tsv"), sep="\t")
    table["lid"] = table.label_id.astype(int)
    report = report.merge(table[["lid", "component_labels", "marker_labels"]], left_on="label_id", right_on="lid")
    report = report.drop(columns=["lid"])
    out = os.path.join(paths["report"], "split_labels.tsv")
    report.to_csv(out, sep="\t", index=False)

    split = report[report.n_components > 1]
    print(f"\nsplit labels: {len(split)} of {len(report)}")
    print(f"  components per label : {split.n_components.value_counts().sort_index().to_dict()}")
    print(f"  marker-classified    : {int((split.marker_labels != 0).sum())}")
    print(f"  voxels to remove     : {int(split.dropped.sum())} of {int(report.n_vox.sum())}"
          f" ({100 * split.dropped.sum() / report.n_vox.sum():.3f} %)")
    ambiguous = split[(split.ratio < MIN_SIZE_RATIO) | (split.drop_frac > MAX_DROP_FRACTION)]
    print(f"\nambiguous (ratio < {MIN_SIZE_RATIO} or drop_frac > {MAX_DROP_FRACTION}): {len(ambiguous)}")
    if len(ambiguous):
        print(ambiguous[["label_id", "n_vox", "comp_sizes", "drop_frac", "ratio",
                         "component_labels", "marker_labels"]].to_string(index=False))
    print(f"\nwrote {out}")
    return report, kept, dropped, ambiguous


def stage_repair(args, paths):
    report, kept, dropped, _, _, shape = _analyze(paths)
    split = report[report.n_components > 1]
    ambiguous = split[(split.ratio < MIN_SIZE_RATIO) | (split.drop_frac > MAX_DROP_FRACTION)]
    if len(ambiguous) and not args.allow_ambiguous:
        raise SystemExit(
            f"{len(ambiguous)} labels are ambiguous: {sorted(ambiguous.label_id.astype(int))}\n"
            "Review them with '--stage analyze', then pass --allow_ambiguous to keep the largest "
            "component for these as well."
        )

    n_levels = len([d for d in os.listdir(paths["zarr"]) if d.startswith("s")])
    total = 0
    for level in range(n_levels):
        array = zarr.open(os.path.join(paths["zarr"], f"s{level}"), mode="a")
        factor = 2 ** level
        coords_list, ids_list = [], []
        for label_id, drop in dropped.items():
            # A coarse voxel is cleared only when no kept voxel of this label maps into it, so the
            # main body can never be erased. Cascaded nearest-neighbour sampling means every coarse
            # voxel holding the label maps back to one of its own s0 voxels.
            drop_coarse = np.unique(drop // factor, axis=0)
            keep_coarse = np.unique(kept[label_id] // factor, axis=0)
            if keep_coarse.size:
                keep_keys = np.sort(_encode(keep_coarse, array.shape))
                drop_keys = _encode(drop_coarse, array.shape)
                pos = np.searchsorted(keep_keys, drop_keys)
                pos = np.clip(pos, 0, keep_keys.size - 1)
                drop_coarse = drop_coarse[keep_keys[pos] != drop_keys]
            if drop_coarse.size:
                coords_list.append(drop_coarse)
                ids_list.append(np.full(drop_coarse.shape[0], label_id, dtype="uint64"))
        if coords_list:
            n = zero_voxels(array, np.concatenate(coords_list), np.concatenate(ids_list))
        else:
            n = 0
        total += n
        print(f"  s{level} {array.shape}: zeroed {n} voxels")
    print(f"\nremoved {total} voxels across {n_levels} levels of {paths['zarr']}")


def stage_table(args, paths):
    os.makedirs(paths["tables_new"], exist_ok=True)
    level_dir = os.path.join(paths["zarr"], "s0")
    array = zarr.open(level_dir, mode="r")
    labs, coords = read_foreground(array, level_dir)

    orig = pd.read_csv(os.path.join(paths["tables_orig"], "default.tsv"), sep="\t")
    orig["lid"] = orig.label_id.astype(int)
    report = pd.read_csv(os.path.join(paths["report"], "split_labels.tsv"), sep="\t")
    changed = sorted(report.loc[report.n_components > 1, "label_id"].astype(int))

    # Only the repaired labels moved. Recomputing just those keeps every other row byte-identical.
    sel = np.isin(labs, np.asarray(changed, dtype="uint32"))
    fresh = base_columns(labs[sel], coords[sel]).set_index("label_id")
    table = orig.set_index("lid").copy()
    for column in fresh.columns:
        table.loc[fresh.index, column] = fresh[column].values
    table["label_id"] = table.index.astype(float)
    table = table.reset_index(drop=True).sort_values("label_id").reset_index(drop=True)

    if set(np.unique(labs).astype(int)) != set(table.label_id.astype(int)):
        raise RuntimeError("label IDs in the repaired volume differ from the table")

    interim = os.path.join(paths["tables_new"], "default_geometry.tsv")
    table.to_csv(interim, sep="\t", index=False)
    print(f"recomputed geometry for {len(changed)} labels -> {interim}")

    # Re-run the component labeling, which depends on the corrected anchors. The tonotopic columns
    # are carried over unchanged: they were already computed against an earlier component labeling,
    # and recomputing them would move the 'length_fraction' that the marker classification uses.
    from flamingo_tools.postprocessing.label_components import label_components_single
    out = os.path.join(paths["tables_new"], "default.tsv")
    label_components_single(interim, out, force_overwrite=True, cell_type="ihc",
                            component_list=args.components, max_edge_distance=args.max_edge_distance,
                            min_component_length=args.min_component_length, min_size=args.min_size)
    final = pd.read_csv(out, sep="\t").sort_values("label_id").reset_index(drop=True)
    final.to_csv(out, sep="\t", index=False)

    print("\ncomponent_labels before:", orig.component_labels.value_counts().sort_index().to_dict())
    print("component_labels after :", final.component_labels.value_counts().sort_index().to_dict())
    moved = final.label_id[final.component_labels.values != orig.sort_values("label_id")
                           .reset_index(drop=True).component_labels.values]
    print(f"labels that changed component: {len(moved)} -> {sorted(moved.astype(int))[:40]}")
    print(f"\nwrote {out}")


def stage_measures(args, paths):
    """Update the object measures for the repaired labels and check the marker classes still hold.

    The '*-bg-mask' tables hold the median over the object mask dilated by 4, minus a local
    background median (radius 75 um, subtracted; see 'object_measures_single'). Removing satellite
    voxels changes only the first term, which is recomputed here exactly from the backup and the
    repaired volume, so the correction is a plain delta on the stored value. The background term is
    left alone; labels whose anchor moved far enough to shift the background window are reported.
    """
    from scipy.ndimage import binary_dilation
    from flamingo_tools.s3_utils import get_s3_path

    dilation, extension = 4, 5
    report = pd.read_csv(os.path.join(paths["report"], "split_labels.tsv"), sep="\t")
    split = set(report.loc[report.n_components > 1, "label_id"].astype(int))
    table = pd.read_csv(os.path.join(paths["tables_new"], "default.tsv"), sep="\t")
    orig = pd.read_csv(os.path.join(paths["tables_orig"], "default.tsv"), sep="\t")
    shift = dict(zip(orig.label_id.astype(int), np.linalg.norm(
        np.stack([orig.anchor_x - table.anchor_x, orig.anchor_y - table.anchor_y,
                  orig.anchor_z - table.anchor_z], axis=1), axis=1)))
    marker = dict(zip(table.label_id.astype(int), table.marker_labels))

    backup_dir = os.path.join(paths["backup"], "s0")
    backup = zarr.open(backup_dir, mode="r")
    labs, coords = read_foreground(backup, backup_dir)
    order = np.argsort(labs, kind="stable")
    uniq, starts, counts = np.unique(labs[order], return_index=True, return_counts=True)
    index = {int(u): (s, c) for u, s, c in zip(uniq, starts, counts)}

    os.makedirs(paths["tables_new"], exist_ok=True)
    for path in sorted(glob.glob(os.path.join(paths["tables_orig"], "*object-measures*.tsv"))):
        name = os.path.basename(path)
        channel = name.split("_")[0]
        measures = pd.read_csv(path, sep="\t")
        affected = sorted(set(measures.label_id.astype(int)) & split)
        if not affected:
            shutil.copy(path, os.path.join(paths["tables_new"], name))
            continue
        store, _ = get_s3_path(f"{args.cochlea}/images/ome-zarr/{channel}.ome.zarr")
        image = zarr.open(store, mode="r")["s0"]

        def object_median(component_coords, label_id):
            lo = np.maximum(component_coords.min(0) - extension, 0)
            hi = np.minimum(component_coords.max(0) + extension + 1, np.asarray(backup.shape))
            bb = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
            mask = binary_dilation(backup[bb] == label_id, iterations=dilation)
            return image[bb][mask]

        deltas = {}
        for label_id in affected:
            start, count = index[label_id]
            c = coords[order[start:start + count]]
            n_comp, comp = label_components_6(c, backup.shape)
            sizes = np.bincount(comp)
            best = int(np.argmax(sizes))
            kept_samples = object_median(c[comp == best], label_id)
            drop_samples = [object_median(c[comp == k], label_id) for k in range(n_comp) if k != best]
            old = float(np.median(np.concatenate([kept_samples] + drop_samples)))
            deltas[label_id] = float(np.median(kept_samples)) - old

        sel = measures.label_id.astype(int).isin(affected)
        measures.loc[sel, "median"] += measures.loc[sel, "label_id"].astype(int).map(deltas)
        measures.to_csv(os.path.join(paths["tables_new"], name), sep="\t", index=False)

        changed = {k: v for k, v in deltas.items() if v != 0}
        # A positive label whose median rises, or a negative one whose median falls, moves away from
        # its threshold and cannot change class. Only the opposite direction carries any risk.
        toward = {k: v for k, v in changed.items()
                  if (marker[k] == 1 and v < 0) or (marker[k] == 2 and v > 0)}
        spread = measures["median"].std()
        print(f"\n{name}")
        print(f"  updated {len(affected)} of {len(measures)} rows; "
              f"{len(changed)} medians changed, max |delta| = "
              f"{max((abs(v) for v in changed.values()), default=0):g}")
        print(f"  moves toward a threshold: {len(toward)}, largest "
              f"{max((abs(v) for v in toward.values()), default=0):g} "
              f"against a spread of {spread:.1f} across all objects")
        far = sorted(k for k in affected if shift.get(k, 0) > args.max_anchor_shift)
        if far:
            print(f"  anchor moved more than {args.max_anchor_shift} um for {len(far)} labels, so their"
                  f" background term is not revalidated: {far}")
    print("\nRe-run 'flamingo_tools.object_measures' with"
          f" reproducibility/object_measures/{args.cochlea}_IHC.json for exact values.")


def stage_verify(args, paths):
    before = pd.read_csv(os.path.join(paths["report"], "split_labels.tsv"), sep="\t")
    n_before = int((before.n_components > 1).sum())
    report, _, _, _, _, _ = _analyze(paths)
    n_split = int((report.n_components > 1).sum())
    print(f"1. labels with more than one connected component in the repaired s0: {n_split}"
          f"  [was {n_before}]")

    table = pd.read_csv(os.path.join(paths["tables_new"], "default.tsv"), sep="\t")
    orig = pd.read_csv(os.path.join(paths["tables_orig"], "default.tsv"), sep="\t")
    ext = np.stack([table.bb_max_x - table.bb_min_x, table.bb_max_y - table.bb_min_y,
                    table.bb_max_z - table.bb_min_z], axis=1)
    ratio = np.linalg.norm(ext, axis=1) / (6 * table.n_pixels * VOXEL_SIZE ** 3 / np.pi) ** (1 / 3)
    print(f"2. table detector (bb diagonal / equivalent diameter > 10): {int((ratio > 10).sum())}")

    same_ids = set(table.label_id.astype(int)) == set(orig.label_id.astype(int))
    print(f"3. label IDs unchanged: {same_ids} ({len(table)} rows)")
    print(f"   marker_labels: {table.marker_labels.value_counts().sort_index().to_dict()}"
          f" (was {orig.marker_labels.value_counts().sort_index().to_dict()})")

    merged = table[["label_id", "n_pixels"]].merge(orig[["label_id", "n_pixels"]], on="label_id",
                                                   suffixes=("_new", "_old"))
    grew = merged[merged.n_pixels_new > merged.n_pixels_old]
    shrank = merged[merged.n_pixels_new < merged.n_pixels_old]
    print(f"4. n_pixels: {len(shrank)} labels shrank, {len(grew)} grew (must be 0)")

    print("\nUpload with, after checking the report:")
    remote = f"{RCLONE_REMOTE}:{BUCKET}/{args.cochlea}"
    print(f"  rclone --progress copyto {paths['zarr']} "
          f"{remote}/images/ome-zarr/{args.segmentation}.ome.zarr")
    for name in sorted(os.listdir(paths["tables_new"])):
        if name.endswith(".tsv") and not name.startswith("default_geometry"):
            print(f"  rclone --progress copyto {os.path.join(paths['tables_new'], name)} "
                  f"{remote}/tables/{args.segmentation}/{name}")
    print("\nDo not re-run mobie.add_segmentation for this source: it rebuilds default.tsv from"
          " scratch and would drop component_labels, the tonotopic columns and marker_labels.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", required=True,
                        choices=["fetch", "analyze", "repair", "table", "measures", "verify"])
    parser.add_argument("-c", "--cochlea", default="M_AMD_OTOF27_L")
    parser.add_argument("-s", "--segmentation", default="IHC_v11")
    parser.add_argument("-w", "--work_dir", required=True,
                        help="Directory for the local pyramid copy and the repair artifacts. "
                             "Nothing is written outside of it.")
    parser.add_argument("--allow_ambiguous", action="store_true",
                        help="Repair labels whose largest component does not clearly dominate.")
    parser.add_argument("--components", type=int, nargs="+", default=[1])
    parser.add_argument("--max_edge_distance", type=float, default=50)
    parser.add_argument("--min_component_length", type=int, default=50)
    parser.add_argument("--min_size", type=int, default=1000)
    parser.add_argument("--max_anchor_shift", type=float, default=2.0,
                        help="Anchor shift in um above which the background term is flagged as not revalidated.")
    args = parser.parse_args()

    paths = _paths(args.work_dir, args.segmentation)
    {"fetch": stage_fetch, "analyze": stage_analyze, "repair": stage_repair,
     "table": stage_table, "measures": stage_measures, "verify": stage_verify}[args.stage](args, paths)


if __name__ == "__main__":
    main()
