"""The stages of the correction of one crop, stored in a session folder.

The folder `<output>/<crop>_session/` holds:
- `state.json`: the state of the correction loop (items, queue, settings) and the list of patches;
- `patches/NNNNN.npz`: one patch per commit, with the labels of the region before and after, and the raw edit
  of the user;
- `proposals.npz`: the proposal voxels of the items;
- the cached S3 reference crop.

A resumed session replays the patches on the input labels. All files are written to a temporary file first and
then moved into place, so an interrupted write leaves the previous version.
"""
import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import tifffile

from .ids import restore

STATE_FILE = "state.json"
PROPOSAL_FILE = "proposals.npz"


def array_sha1(array: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(array).tobytes()).hexdigest()


def _replace_atomic(path: str, write) -> None:
    """Write a file with `write(temporary_path)`, then move it to `path`."""
    folder = os.path.dirname(os.path.abspath(path))
    suffix = os.path.splitext(path)[1]
    handle, temporary = tempfile.mkstemp(dir=folder, suffix=suffix)
    os.close(handle)
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def write_tif_atomic(path: str, data: np.ndarray) -> None:
    _replace_atomic(path, lambda temporary: tifffile.imwrite(temporary, data, compression="zlib"))


class Session:
    """The session folder of one crop. Use `Session.create` or `Session.open`."""

    def __init__(self, folder: str, state: Dict[str, Any]):
        self.folder = folder
        self.state = state

    @staticmethod
    def exists(folder: str) -> bool:
        return os.path.exists(os.path.join(folder, STATE_FILE))

    @classmethod
    def create(cls, folder: str, input_sha1: str, state: Optional[Dict[str, Any]] = None) -> "Session":
        """Start a new session. An existing session in the folder is removed."""
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(os.path.join(folder, "patches"))
        session = cls(folder, {**(state or {}), "input_sha1": input_sha1, "patches": [], "n_patches": 0})
        session.save()
        return session

    @classmethod
    def open(cls, folder: str, input_sha1: str) -> "Session":
        """Open an existing session. The input labels must be the ones of the session."""
        with open(os.path.join(folder, STATE_FILE)) as f:
            state = json.load(f)
        if state["input_sha1"] != input_sha1:
            raise ValueError(
                f"The session in {folder} belongs to other input labels. Restart the session to discard it."
            )
        return cls(folder, state)

    def save(self) -> None:
        def write(temporary):
            with open(temporary, "w") as f:
                json.dump(self.state, f, indent=1)
        _replace_atomic(os.path.join(self.folder, STATE_FILE), write)

    def save_arrays(self, arrays: Dict[str, np.ndarray], name: str = PROPOSAL_FILE) -> None:
        _replace_atomic(os.path.join(self.folder, name), lambda temporary: np.savez_compressed(temporary, **arrays))

    def load_arrays(self, name: str = PROPOSAL_FILE) -> Dict[str, np.ndarray]:
        path = os.path.join(self.folder, name)
        if not os.path.exists(path):
            return {}
        with np.load(path) as data:
            return {key: data[key] for key in data.files}

    def _patch_path(self, index: int) -> str:
        return os.path.join(self.folder, "patches", f"{index:05d}.npz")

    def commit(
        self, start: Tuple[int, ...], before: np.ndarray, after: np.ndarray, meta: Optional[Dict[str, Any]] = None,
        edit: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, Any]]:
        """Store a change of the labels in a region. Returns the patch entry, or None if nothing changed.

        The entry holds the label changes (compact ids) and the napari center of the changed voxels in the crop.
        `edit` is the raw edit of the user (the edit layer, compact ids), stored with the patch so that a later
        version of the fill can replay it.
        """
        changed = before != after
        if not changed.any():
            return None
        index = self.state["n_patches"]
        ids = np.union1d(np.unique(before[changed]), np.unique(after[changed]))
        delta = {int(i): int((after == i).sum() - (before == i).sum()) for i in ids if i != 0}
        center = (np.argwhere(changed).mean(axis=0) + np.array(start)).round().astype(int).tolist()
        entry = {**(meta or {}), "patch": index, "start": [int(s) for s in start], "shape": list(before.shape),
                 "n_changed": int(changed.sum()), "delta": delta, "center": center}
        arrays = {"before": before, "after": after}
        if edit is not None:
            arrays["edit"] = edit
        _replace_atomic(self._patch_path(index), lambda temporary: np.savez_compressed(temporary, **arrays))
        self.state["patches"].append(entry)
        self.state["n_patches"] = index + 1
        self.save()
        return entry

    def replay(self, labels: np.ndarray) -> np.ndarray:
        """Apply all patches to the input labels in place."""
        for entry in self.state["patches"]:
            with np.load(self._patch_path(entry["patch"])) as data:
                labels[_region(entry)] = data["after"]
        return labels

    def undo_last(self, labels: np.ndarray) -> Optional[Tuple[Dict[str, Any], Optional[np.ndarray]]]:
        """Revert the last patch in place. Returns its entry and its raw edit (or None), or None without a patch."""
        if not self.state["patches"]:
            return None
        entry = self.state["patches"].pop()
        path = self._patch_path(entry["patch"])
        with np.load(path) as data:
            labels[_region(entry)] = data["before"]
            edit = data["edit"] if "edit" in data.files else None
        self.save()
        os.remove(path)
        return entry, edit

    def changes(self, original: np.ndarray) -> pd.DataFrame:
        """One row per patch, with the original label ids, see `ids.compact`."""
        def original_id(label_id):
            return int(restore(np.array([label_id], dtype="uint32"), original)[0])

        rows = []
        for entry in self.state["patches"]:
            delta = {original_id(int(i)): d for i, d in entry["delta"].items()}
            rows.append({
                "patch": entry["patch"], "item": entry.get("item"), "action": entry.get("action"),
                "label_id": original_id(entry["label_id"]) if entry.get("label_id") else None,
                "reasons": entry.get("reasons"), "n_changed": entry["n_changed"],
                "delta": ",".join(f"{i}:{d:+d}" for i, d in delta.items()),
                "z": entry["center"][0], "y": entry["center"][1], "x": entry["center"][2],
            })
        return pd.DataFrame(rows)

    def export(
        self, labels: np.ndarray, original: np.ndarray, output_path: str, table_path: str, keep: bool = True,
    ) -> None:
        """Write the corrected labels with the original ids and the change table.

        Args:
            labels: The corrected labels with compact ids.
            original: The original id of each compact id, see `ids.compact`.
            output_path: The path of the annotation tif.
            table_path: The path of the change table.
            keep: Keep the session folder. It holds the record of the edits.
        """
        write_tif_atomic(output_path, restore(labels, original))
        changes = self.changes(original)
        _replace_atomic(table_path, lambda temporary: changes.to_csv(temporary, sep="\t", index=False))
        if not keep:
            shutil.rmtree(self.folder)


def _region(entry: Dict[str, Any]) -> Tuple[slice, ...]:
    return tuple(slice(s, s + n) for s, n in zip(entry["start"], entry["shape"]))
