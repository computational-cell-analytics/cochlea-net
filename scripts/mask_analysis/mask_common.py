"""Shared constants and helpers for the input mask analysis in scripts/mask_analysis/.

Coordinates follow the repository convention: tables store micrometers in (x, y, z) order,
arrays use (z, y, x) order.
"""
import importlib.util
import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTPUT_ROOT = os.path.join(REPO_ROOT, "mask_threshold_analysis")
DATASETS_JSON = os.path.join(OUTPUT_ROOT, "mask_datasets.json")
MASK_JOBS_CSV = os.path.join(OUTPUT_ROOT, "mask_jobs.csv")

# Voxel size of the s0 level of the flamingo light-sheet data, in micrometers.
VOXEL_SIZE_S0 = 0.38
# The production prediction block. find_mask decides per block of 2 x input chunks, which is
# 128^3 for the OME-Zarr inputs (64^3 chunks) and 256^3 for the Spark-fused n5 inputs.
BLOCK = 128
CLASSES = ("SGN", "IHC")


def output_path(*parts: str) -> str:
    path = os.path.join(OUTPUT_ROOT, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def load_datasets(path: str = DATASETS_JSON) -> Dict:
    with open(path) as f:
        return json.load(f)


def dataset_key(cochlea: str, cls: str) -> str:
    return f"{cochlea}_{cls}"


def load_figure_util():
    """Import scripts/figures/util.py by path, because the module name 'util' is ambiguous."""
    path = os.path.join(REPO_ROOT, "scripts", "figures", "util.py")
    spec = importlib.util.spec_from_file_location("figure_util", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_s3_json(fs, path: str) -> Dict:
    from flamingo_tools.s3_utils import BUCKET_NAME
    with fs.open(os.path.join(BUCKET_NAME, path), "r") as f:
        return json.load(f)


def read_s3_table(fs, cochlea: str, table_folder: str, name: str = "default.tsv") -> pd.DataFrame:
    from flamingo_tools.s3_utils import BUCKET_NAME
    with fs.open(os.path.join(BUCKET_NAME, cochlea, table_folder, name), "r") as f:
        return pd.read_csv(f, sep="\t")


def open_image(cochlea: str, source: str, level: str, local_root: Optional[str] = None):
    """Open one pyramid level of an image source, from S3 or from a local MoBIE copy."""
    import zarr
    rel = os.path.join(cochlea, "images", "ome-zarr", f"{source}.ome.zarr")
    if local_root is None:
        from flamingo_tools.s3_utils import get_s3_path
        store, _ = get_s3_path(rel)
        return zarr.open(store, mode="r")[level]
    return zarr.open(os.path.join(local_root, rel), mode="r")[level]


def table_bboxes_s0(table: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return the object bounding boxes in s0 voxels, (z, y, x) order, with an exclusive end.

    The MoBIE tables on S3 store an inclusive maximum: a face on the block grid shows as
    bb_min % 128 == 0 or bb_max % 128 == 127. Adding one voxel gives an exclusive end.
    """
    bb_min = np.stack([table[f"bb_min_{ax}"].values for ax in "zyx"], axis=1) / VOXEL_SIZE_S0
    bb_max = np.stack([table[f"bb_max_{ax}"].values for ax in "zyx"], axis=1) / VOXEL_SIZE_S0
    bb_min = np.round(bb_min).astype("int64")
    bb_max = np.round(bb_max).astype("int64") + 1
    return bb_min, bb_max


def object_block_ranges(bb_min: np.ndarray, bb_max: np.ndarray, block: int = BLOCK,
                        grid_shape: Optional[Tuple[int, int, int]] = None):
    """Return the first and the last (inclusive) block index that each bounding box touches."""
    first = bb_min // block
    last = (bb_max - 1) // block
    if grid_shape is not None:
        upper = np.array(grid_shape) - 1
        first = np.clip(first, 0, upper)
        last = np.clip(last, 0, upper)
    return first, last


def object_coverage(keep: np.ndarray, first: np.ndarray, last: np.ndarray) -> np.ndarray:
    """Classify objects against a block mask: 2 = complete, 1 = cut, 0 = lost.

    `keep` is the boolean block grid. An object is complete if every block that its bounding
    box touches is kept, cut if only some of them are kept, and lost if none are.
    """
    keep = keep.astype("int64")
    # A 3D summed-area table gives the number of kept blocks in every box in O(1).
    sat = np.zeros(tuple(s + 1 for s in keep.shape), dtype="int64")
    sat[1:, 1:, 1:] = keep.cumsum(0).cumsum(1).cumsum(2)
    z0, y0, x0 = first.T
    z1, y1, x1 = (last + 1).T
    n_kept = (
        sat[z1, y1, x1] - sat[z0, y1, x1] - sat[z1, y0, x1] - sat[z1, y1, x0]
        + sat[z0, y0, x1] + sat[z0, y1, x0] + sat[z1, y0, x0] - sat[z0, y0, x0]
    )
    n_total = np.prod(last - first + 1, axis=1)
    status = np.where(n_kept == n_total, 2, np.where(n_kept > 0, 1, 0))
    return status
