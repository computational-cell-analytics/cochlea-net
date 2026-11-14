"""Audit existing segmentations for objects that end on the block grid (tables only).

An object can end exactly on a block face for three reasons:

* the neighbour block was skipped by the input mask and predicted 0 (mask cut);
* the block-wise seeded watershed (halo (2, 8, 8)) did not see the seed of the object from the
  neighbour block, so the part beyond the face was lost or given to another label (halo cut);
* a seam between two predicted blocks split the object, so a partner object starts on the other
  side of the same plane (seam split).

A neighbour block that holds the anchor of any other object was predicted, so a face hit without
a partner and with an occupied neighbour block is not a mask cut. A face hit without a partner
and with an empty neighbour block is a mask-cut candidate.

The reference runs used Spark-fused n5 inputs with mask blocks of (128, 256, 256) or 256^3
voxels, while the prediction blocks are 128^3. A face at residue 128 modulo 256 in y or x is
therefore inside a mask block, and only a seam can cause it.
"""
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest

from mask_common import BLOCK, VOXEL_SIZE_S0, load_datasets, output_path, read_s3_table, table_bboxes_s0

AXES = "zyx"


def baseline_rate(coords, period):
    """Return the chance rate of one residue value, estimated away from the grid planes."""
    residues = coords % period
    away = (residues >= 3) & (residues <= period - 3)
    return np.mean(away) / (period - 5)


def find_partners(bb_min, bb_max, idx, axis, side, plane):
    """Return the indices of objects across the plane whose in-plane extent overlaps object idx."""
    others = [a for a in range(3) if a != axis]
    if side == "min":
        candidates = np.where(bb_max[:, axis] == plane)[0]
    else:
        candidates = np.where(bb_min[:, axis] == plane)[0]
    overlap = np.ones(len(candidates), dtype=bool)
    for a in others:
        overlap &= (bb_min[candidates, a] < bb_max[idx, a]) & (bb_max[candidates, a] > bb_min[idx, a])
    return candidates[overlap]


def occupied_blocks(table):
    anchors = np.stack([table[f"anchor_{ax}"].values for ax in AXES], axis=1) / VOXEL_SIZE_S0
    return set(map(tuple, (anchors // BLOCK).astype("int64")))


def neighbour_occupied(occupied, bb_min, bb_max, idx, axis, side, plane):
    """Check if any block across the face, within the in-plane extent of the object, holds an anchor."""
    across = plane // BLOCK - 1 if side == "min" else plane // BLOCK
    ranges = []
    for a in range(3):
        if a == axis:
            ranges.append([across])
        else:
            ranges.append(range(bb_min[idx, a] // BLOCK, (bb_max[idx, a] - 1) // BLOCK + 1))
    return any((z, y, x) in occupied for z in ranges[0] for y in ranges[1] for x in ranges[2])


def audit_dataset(table, components):
    bb_min, bb_max = table_bboxes_s0(table)
    occupied = occupied_blocks(table)
    relevant = table.component_labels.isin(components).values
    label_ids = table.label_id.values
    hits, stats = [], defaultdict(float)
    for axis, ax_name in enumerate(AXES):
        for side, coords in (("min", bb_min[:, axis]), ("max", bb_max[:, axis])):
            rel_coords = coords[relevant]
            rate = baseline_rate(rel_coords, 2 * BLOCK)
            for residue in (0, BLOCK):
                on_plane = relevant & (coords % (2 * BLOCK) == residue) & (coords > 0)
                n_hit = int(on_plane.sum())
                stats[f"{ax_name}{side}_r{residue}_hits"] = n_hit
                stats[f"{ax_name}{side}_r{residue}_expected"] = rate * len(rel_coords)
                stats[f"{ax_name}{side}_r{residue}_p"] = binomtest(n_hit, len(rel_coords), rate,
                                                                   alternative="greater").pvalue
                for idx in np.where(on_plane)[0]:
                    partners = find_partners(bb_min, bb_max, idx, axis, side, coords[idx])
                    hits.append({
                        "label_id": int(label_ids[idx]), "axis": ax_name, "side": side, "plane": int(coords[idx]),
                        "residue256": residue, "partner_ids": " ".join(str(int(label_ids[p])) for p in partners),
                        "partner_relevant": bool(relevant[partners].any()) if len(partners) else False,
                        "neighbour_occupied": neighbour_occupied(occupied, bb_min, bb_max, idx, axis, side,
                                                                 coords[idx]),
                    })
    hits = pd.DataFrame(hits)
    n_relevant = int(relevant.sum())
    stats["n_relevant"] = n_relevant
    if len(hits):
        hits["kind"] = np.where(hits.partner_ids != "", "seam_split",
                                np.where(hits.neighbour_occupied, "halo_cut", "mask_cut_candidate"))
        # One object can have several face hits; report the most mask-like kind per object.
        order = {"mask_cut_candidate": 0, "halo_cut": 1, "seam_split": 2}
        per_object = hits.groupby("label_id").kind.apply(lambda k: min(k, key=order.get))
        stats["objects_on_grid"] = len(per_object)
        for kind in order:
            stats[f"objects_{kind}"] = int((per_object == kind).sum())
    else:
        stats["objects_on_grid"] = 0
        for kind in ("mask_cut_candidate", "halo_cut", "seam_split"):
            stats[f"objects_{kind}"] = 0
    expected = sum(v for k, v in stats.items() if k.endswith("_expected"))
    stats["expected_face_hits"] = expected
    stats["face_hits"] = len(hits)
    return dict(stats), hits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keys", nargs="*", help="Dataset keys, e.g. M_AMD_OTOF27_R_IHC. Default: all.")
    args = parser.parse_args()

    from flamingo_tools.s3_utils import create_s3_target
    fs = create_s3_target()
    datasets = load_datasets()["datasets"]
    if args.keys:
        datasets = [ds for ds in datasets if ds["key"] in args.keys]

    rows, all_hits = [], []
    for ds in datasets:
        table = read_s3_table(fs, ds["cochlea"], ds["table_folder"])
        stats, hits = audit_dataset(table, ds["components"])
        stats.update({key: ds[key] for key in ("key", "class", "stain", "segmentation", "stratum", "split")})
        stats["reference_rule"] = "{rule_kind} p{rule_percentile} T{rule_threshold}".format(
            **{k: ds["reference_mask"].get(k) for k in ("rule_kind", "rule_percentile", "rule_threshold")})
        stats["reference_input"] = ds["reference_mask"].get("input_format")
        rows.append(stats)
        if len(hits):
            hits.insert(0, "key", ds["key"])
            all_hits.append(hits)
        print(f"{ds['key']:28s} relevant {stats['n_relevant']:6d}  face hits {stats['face_hits']:5d} "
              f"(chance {stats['expected_face_hits']:7.1f})  on grid {stats['objects_on_grid']:5d}  "
              f"seam {stats['objects_seam_split']:5d}  halo {stats['objects_halo_cut']:5d}  "
              f"mask? {stats['objects_mask_cut_candidate']:5d}")

    summary = pd.DataFrame(rows)
    summary.to_csv(output_path("audit", "summary.csv"), index=False)
    if all_hits:
        pd.concat(all_hits).to_csv(output_path("audit", "face_hits.csv"), index=False)
    print(f"-> {output_path('audit', 'summary.csv')}")


if __name__ == "__main__":
    main()
