"""Replay a correction session with the current fill and cleanup, and write the corrected annotation.

A session stores the raw edit of each visit with its slices and its focus cells, so a later version of the fill
can replay it. The replay starts from the base of the session (the cached S3 segmentation, or `--base`):
- a region patch: the stored cells of the region proposal, cleaned with `cleanup.clean_labels` as in
  `region.apply_region`. The proposal itself is not computed again, because the visits after it edited its cells;
- a visit: `controller.fill_visit` with the stored edit, slices and focus;
- a deletion: the label is removed.
The painted slices of the user stay as they were; only the voxels between them and the cleanup can change.

Usage:
    python replay_session.py --session out/X_session -o X_annotations_replayed.tif [--base X_annotations.tif] \\
        [--table metrics.tsv]
"""
import argparse
import json
import os
from dataclasses import fields

import numpy as np
import pandas as pd
import tifffile

from flamingo_tools.annotation.cleanup import clean_labels
from flamingo_tools.annotation.controller import Settings, fill_visit
from flamingo_tools.annotation.ids import compact, restore
from flamingo_tools.annotation.session import Session, array_sha1

from compare_annotations import interleaved, label_metrics


def session_base(folder: str, state: dict) -> np.ndarray:
    """The base labels of a session with an S3 base, from the cached reference crop."""
    if not state["base"].startswith("s3:"):
        raise ValueError("The base of the session is the annotation. Give it with --base.")
    return tifffile.imread(os.path.join(folder, f"{state['crop']}_{state['base'][3:]}.tif"))


def replay(base: np.ndarray, session: Session):
    """The labels of the session replayed with the current code, the stored labels, and the original ids."""
    state = session.state
    stored = {f.name: state["settings"][f.name] for f in fields(Settings) if f.name in state.get("settings", {})}
    settings = Settings(**stored)
    labels, original = compact(base)
    stored_labels = session.replay(labels.copy())
    next_label = [int(state.get("next_label", labels.max() + 1))]

    def new_label():
        next_label[0] += 1
        return next_label[0] - 1

    for entry in state["patches"]:
        roi = tuple(slice(s, s + n) for s, n in zip(entry["start"], entry["shape"]))
        before = labels[roi].copy()
        with np.load(os.path.join(session.folder, "patches", f"{entry['patch']:05d}.npz")) as data:
            arrays = {key: data[key] for key in data.files}
        if entry["action"] == "region":
            cells = [int(label_id) for label_id, delta in entry["delta"].items() if delta > 0]
            after, _ = clean_labels(arrays["after"], before, cells, settings.voxel_size, settings.boundary_sigma,
                                    settings.boundary_band)
        elif "edit" in arrays:
            local = np.asarray(entry["slices"]) - entry["start"][entry["axis"]]
            after, splits = fill_visit(before, arrays["edit"], entry["axis"], local, entry["focus"], settings,
                                       new_label)
            for split_id, label_id in splits.items():
                print(f"Patch {entry['patch']}: a separate piece of {original_id(label_id, original)} became "
                      f"{original_id(split_id, original)}.")
        elif entry["action"] == "delete":
            after = before.copy()
            after[after == entry["label_id"]] = 0
        else:
            raise ValueError(f"The patch {entry['patch']} ({entry['action']}) has no raw edit to replay.")
        labels[roi] = after
    return labels, stored_labels, original


def original_id(label_id: int, original: np.ndarray) -> int:
    return int(restore(np.array([label_id], dtype="uint32"), original)[0])


def metrics(labels: np.ndarray, ids, name: str) -> pd.DataFrame:
    present = [i for i in ids if (labels == i).any()]
    mixed = interleaved(labels, present)
    return pd.DataFrame([{"labels": name, "label_id": i, **label_metrics(labels, i), "interleaved": mixed[i]}
                         for i in present])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", required=True, help="The session folder <crop>_session.")
    parser.add_argument("--base", help="The base labels, if the base of the session is the annotation.")
    parser.add_argument("-o", "--output", required=True, help="The replayed annotation tif.")
    parser.add_argument("--table", help="The metrics of the changed cells, stored and replayed (tsv).")
    args = parser.parse_args()
    with open(os.path.join(args.session, "state.json")) as f:
        state = json.load(f)
    base = tifffile.imread(args.base) if args.base else session_base(args.session, state)
    session = Session.open(args.session, array_sha1(base))
    labels, stored, original = replay(base, session)
    tifffile.imwrite(args.output, restore(labels, original), compression="zlib")

    base_compact = compact(base)[0]
    changed = (labels != base_compact) | (stored != base_compact)
    ids = sorted(int(i) for i in np.union1d(np.unique(labels[changed]), np.unique(stored[changed])) if i > 0)
    table = pd.concat([metrics(stored, ids, "stored"), metrics(labels, ids, "replayed")], ignore_index=True)
    table["label_id"] = [original_id(i, original) for i in table["label_id"]]
    columns = ["n_voxels", "face_pieces", "small_pieces", "holes_3d", "holes_2d_all", "interleaved"]
    print(table.pivot(index="label_id", columns="labels", values=columns).to_string())
    print(f"{int((labels != stored).sum())} voxels differ between the stored and the replayed labels.")
    if args.table:
        table.to_csv(args.table, sep="\t", index=False)


if __name__ == "__main__":
    main()
