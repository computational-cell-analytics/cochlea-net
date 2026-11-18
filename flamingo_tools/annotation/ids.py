"""Compact label ids for the annotation correction, and restore the original ids for the export.

Annotation crops can carry the ids of the cochlea segmentation (up to about 3e9). `scipy.ndimage.find_objects`
allocates memory in proportion to the largest id, so the correction works with consecutive ids.
"""
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def compact(labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Map the label ids of a segmentation to 1..n.

    Args:
        labels: The segmentation.

    Returns:
        The segmentation with the ids 1..n, as uint32.
        The original id of each compact id. Index 0 is the background.
    """
    original = np.union1d([0], pd.unique(labels.ravel())).astype("uint64")
    compact_labels = np.empty(labels.shape, dtype="uint32")
    # In blocks along the first axis, so the int64 output of searchsorted stays small.
    for start in range(0, labels.shape[0], 32):
        compact_labels[start:start + 32] = np.searchsorted(original, labels[start:start + 32])
    return compact_labels, original


def restore(labels: np.ndarray, original: np.ndarray) -> np.ndarray:
    """Map compact ids back to the original ids, see `compact`.

    A compact id above the known ids is a new cell. It gets the next free id after the largest original id.

    Args:
        labels: The segmentation with compact ids.
        original: The original id of each compact id, as returned by `compact`.

    Returns:
        The segmentation with the original ids, as uint32.
    """
    n_known = len(original)
    n_ids = max(int(labels.max()) + 1, n_known)
    table = np.zeros(n_ids, dtype="uint64")
    table[:n_known] = original
    table[n_known:] = int(original.max()) + np.arange(1, n_ids - n_known + 1)
    if table.max() >= 2 ** 32:
        raise ValueError("The restored label ids do not fit into uint32.")
    return table[labels].astype("uint32")


class IdMap:
    """The file ids of compact ids, for the display: the ids of the input tif, the issue table and the output.

    A compact id above the known ids is a new cell with the file id that `restore` gives it.
    """

    def __init__(self, original: np.ndarray):
        self.original = np.asarray(original, dtype="uint64")
        self.n_known = len(self.original)
        self.max_id = int(self.original.max())

    def file_id(self, label_id: int) -> int:
        label_id = int(label_id)
        return int(self.original[label_id]) if label_id < self.n_known else self.max_id + label_id - self.n_known + 1

    def compact_id(self, file_id: int, next_label: int) -> Optional[int]:
        """The compact id of a file id, or None if the id is unknown. Ids below `next_label` exist."""
        file_id = int(file_id)
        index = int(np.searchsorted(self.original, file_id))
        if index < self.n_known and int(self.original[index]) == file_id:
            return index
        label_id = self.n_known + file_id - self.max_id - 1
        return label_id if self.n_known <= label_id < next_label else None

    def to_file(self, labels: np.ndarray) -> np.ndarray:
        return restore(labels, self.original)
