"""Prepare masks for incrementally extending an existing block-wise IHC prediction.

The source mask is dilated on its native chunk grid. Three independent masks are written:

* ``mask_original.zarr``: an exact block-level copy of the source mask;
* ``mask_delta.zarr``: only blocks introduced by the dilation, used for GPU inference;
* ``mask_union.zarr``: original plus new blocks, retained as the final run mask.

``mask.zarr`` is created as a relative symlink to ``mask_delta.zarr``. This lets the regular
prediction pipeline write only newly exposed prediction blocks into a copy of the previous
prediction. The downstream watershed job switches the symlink to ``mask_union.zarr``.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import z5py
from scipy.ndimage import binary_dilation


MASK_KEY = "mask"
PREDICTION_BLOCK_SHAPE = (128, 128, 128)


def _stored_mask_blocks(mask_path: str, grid_shape) -> np.ndarray:
    """Return coarse block occupancy from a sparse Zarr-v2 mask."""
    chunk_folder = Path(mask_path) / MASK_KEY
    occupancy = np.zeros(grid_shape, dtype=bool)
    for entry in chunk_folder.iterdir():
        if entry.name.startswith("."):
            continue
        try:
            block = tuple(int(coord) for coord in entry.name.split("."))
        except ValueError:
            continue
        if len(block) != 3 or any(coord < 0 or coord >= grid_shape[axis] for axis, coord in enumerate(block)):
            raise ValueError(f"Unexpected mask chunk key {entry.name!r} in {chunk_folder}.")
        occupancy[block] = True
    return occupancy


def _write_mask(path: str, shape, chunks, occupancy: np.ndarray) -> None:
    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to replace existing mask path: {path}")
    mask_file = z5py.File(path, "a")
    mask = mask_file.create_dataset(
        MASK_KEY, shape=shape, chunks=chunks, compression="gzip", dtype="uint8"
    )
    for block in np.argwhere(occupancy):
        begin = block * np.asarray(chunks)
        end = np.minimum(begin + chunks, shape)
        bounding_box = tuple(slice(int(start), int(stop)) for start, stop in zip(begin, end))
        mask[bounding_box] = 1


def _prediction_blocks(occupancy: np.ndarray, shape, mask_chunks) -> set:
    """Resolve prediction-block indices touched by a coarse mask occupancy grid."""
    prediction_blocks = set()
    pred_shape = np.ceil(np.asarray(shape) / PREDICTION_BLOCK_SHAPE).astype(int)
    for block in np.argwhere(occupancy):
        begin = block * np.asarray(mask_chunks)
        end = np.minimum(begin + mask_chunks, shape)
        first = begin // PREDICTION_BLOCK_SHAPE
        last = np.ceil(end / PREDICTION_BLOCK_SHAPE).astype(int)
        for z in range(int(first[0]), min(int(last[0]), int(pred_shape[0]))):
            for y in range(int(first[1]), min(int(last[1]), int(pred_shape[1]))):
                for x in range(int(first[2]), min(int(last[2]), int(pred_shape[2]))):
                    prediction_blocks.add((z, y, x))
    return prediction_blocks


def prepare(source_mask: str, output_folder: str, iterations: int = 1) -> dict:
    if iterations < 1:
        raise ValueError("Dilation iterations must be positive.")
    source = z5py.File(source_mask, "r")[MASK_KEY]
    shape = tuple(source.shape)
    chunks = tuple(source.chunks)
    grid_shape = tuple(np.ceil(np.asarray(shape) / chunks).astype(int))
    original = _stored_mask_blocks(source_mask, grid_shape)
    if not original.any():
        raise ValueError(f"The source mask has no stored foreground blocks: {source_mask}")

    union = binary_dilation(
        original, structure=np.ones((3, 3, 3), dtype=bool), iterations=iterations
    )
    delta = np.logical_and(union, np.logical_not(original))

    os.makedirs(output_folder, exist_ok=True)
    paths = {
        "original": os.path.join(output_folder, "mask_original.zarr"),
        "delta": os.path.join(output_folder, "mask_delta.zarr"),
        "union": os.path.join(output_folder, "mask_union.zarr"),
    }
    _write_mask(paths["original"], shape, chunks, original)
    _write_mask(paths["delta"], shape, chunks, delta)
    _write_mask(paths["union"], shape, chunks, union)

    active_mask = os.path.join(output_folder, "mask.zarr")
    if os.path.lexists(active_mask):
        raise FileExistsError(f"Refusing to replace existing active mask: {active_mask}")
    os.symlink("mask_delta.zarr", active_mask)

    original_prediction_blocks = _prediction_blocks(original, shape, chunks)
    delta_prediction_blocks = _prediction_blocks(delta, shape, chunks)
    union_prediction_blocks = _prediction_blocks(union, shape, chunks)
    if original_prediction_blocks & delta_prediction_blocks:
        raise RuntimeError("Original and delta masks unexpectedly touch the same prediction blocks.")
    if original_prediction_blocks | delta_prediction_blocks != union_prediction_blocks:
        raise RuntimeError("Original and delta prediction blocks do not exactly reconstruct the union.")

    manifest = {
        "source_mask": os.path.realpath(source_mask),
        "shape": shape,
        "mask_chunks": chunks,
        "prediction_block_shape": PREDICTION_BLOCK_SHAPE,
        "dilation_iterations": iterations,
        "original_mask_blocks": int(original.sum()),
        "delta_mask_blocks": int(delta.sum()),
        "union_mask_blocks": int(union.sum()),
        "original_prediction_blocks": len(original_prediction_blocks),
        "delta_prediction_blocks": len(delta_prediction_blocks),
        "union_prediction_blocks": len(union_prediction_blocks),
    }
    manifest_path = os.path.join(output_folder, "dilated_mask_manifest.json")
    with open(manifest_path, "w") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mask", required=True)
    parser.add_argument("--output-folder", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    args = parser.parse_args()
    manifest = prepare(args.source_mask, args.output_folder, args.iterations)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
