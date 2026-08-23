"""Sharded, workspace-local prediction for the SGN v2 seed-variance experiment.

The four SGN v2 seed variants are applied to the five F1-validation cochleae to measure the spread
in accuracy caused by the training seed alone. Compared to the standard workflow in
`flamingo_tools.segmentation.unet_prediction` only the storage of the prediction changes:
`predictions.zarr` is a zarr v3 array whose shards aggregate 4 x 4 x 4 prediction blocks and all
three output channels into one file. That cuts the file count of the full experiment from ~740k to
at most ~12k, which matters because the workspace shares a project inode quota.

A zarr shard is a single file, and a partial shard write is a load-merge-rewrite with
last-writer-wins, so a shard spanning several prediction blocks cannot be filled by concurrent
writers. Each slurm array task therefore owns whole shards: it predicts the blocks of one shard
into memory and writes the shard exactly once, which is a complete-shard write and skips the
read-modify-write path entirely. `predict_with_halo_pipelined` reads every block's halo from the
global input regardless of `roi`, and the shard grid is aligned with the block grid, so the block
bounding boxes -- and therefore the predictions -- are identical to those of a whole-volume run.

Subcommands:
    manifest  Count the in-mask blocks per shard and assign the non-empty shards to the tasks.
    init      Create the empty sharded prediction array (single process, see `cmd_init`).
    predict   Predict the shards assigned to this array task.
    convert   Copy an existing unsharded prediction into the sharded layout.
    verify    Check a prediction against its manifest, or against a reference array.
"""

import argparse
import itertools
import json
import math
import multiprocessing as mp
import os
from concurrent import futures
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import z5py
import zarr
from zarr.codecs import ZstdCodec

# The block geometry is fixed here instead of being taken from
# `unet_prediction._get_device_and_tiling`, whose default depends on `torch.cuda.is_available()`.
# The manifest is built on a CPU node and the prediction runs on a GPU node, and both must derive
# the same block grid, so the value cannot depend on the node type. (128, 128, 128) is the GPU
# default of that function and the geometry the existing predictions were computed with.
BLOCK_SHAPE = (128, 128, 128)
HALO = (16, 32, 32)

# Spatial extent of one shard; the shard always spans all output channels, because one block write
# covers the full channel range and can therefore never conflict on that axis.
SHARD_SHAPE = (512, 512, 512)
OUTPUT_CHANNELS = 3

CHUNKS = (1,) + BLOCK_SHAPE
SHARDS = (OUTPUT_CHANNELS,) + SHARD_SHAPE

# zstd level 0 is what `prediction_impl` writes today, via elf's "zstd" -> ZstdCodec() mapping.
ZSTD_LEVEL = 0

# Sigma of the gaussian applied to the boundary distance channel, from `prediction_impl`.
DISTANCE_SMOOTHING_SIGMA = 2.0

MANIFEST_NAME = "shard_manifest.json"
PREDICTION_NAME = "predictions.zarr"
PREDICTION_KEY = "prediction"
MASK_NAME = "mask.zarr"
MASK_KEY = "mask"
MEAN_STD_NAME = "mean_std.json"


#
# --- geometry ---
#


def block_shape_for(shape: Sequence[int]) -> Tuple[int, ...]:
    """Clip the block shape to the volume, exactly as `_get_device_and_tiling` does."""
    return tuple(min(bs, sh) for bs, sh in zip(BLOCK_SHAPE, shape))


def halo_for(shape: Sequence[int]) -> Tuple[int, ...]:
    """Clip the halo to the block shape, exactly as `_get_device_and_tiling` does."""
    return tuple(min(ha, bs // 2) for ha, bs in zip(HALO, block_shape_for(shape)))


def _grid(shape: Sequence[int], tile: Sequence[int]) -> Tuple[int, ...]:
    return tuple(math.ceil(sh / ti) for sh, ti in zip(shape, tile))


def _unravel(index: int, grid: Sequence[int]) -> Tuple[int, ...]:
    return tuple(int(i) for i in np.unravel_index(index, grid))


def _ravel(position: Sequence[int], grid: Sequence[int]) -> int:
    return int(np.ravel_multi_index(position, grid))


def block_bounding_box(block_id: int, shape: Sequence[int]) -> Tuple[slice, ...]:
    """The bounding box of a prediction block, matching the grid of a whole-volume run."""
    block = block_shape_for(shape)
    grid = _grid(shape, block)
    position = _unravel(block_id, grid)
    return tuple(
        slice(pos * bs, min((pos + 1) * bs, sh)) for pos, bs, sh in zip(position, block, shape)
    )


def shard_bounding_box(shard_id: int, shape: Sequence[int]) -> Tuple[slice, ...]:
    """The spatial bounding box of a shard, clipped to the volume."""
    grid = _grid(shape, SHARD_SHAPE)
    position = _unravel(shard_id, grid)
    return tuple(
        slice(pos * ss, min((pos + 1) * ss, sh)) for pos, ss, sh in zip(position, SHARD_SHAPE, shape)
    )


def shard_of_block(block_id: int, shape: Sequence[int]) -> int:
    """The id of the shard that contains a block. The grids are aligned, so this is well defined."""
    bb = block_bounding_box(block_id, shape)
    shard_grid = _grid(shape, SHARD_SHAPE)
    position = tuple(bb_ax.start // ss for bb_ax, ss in zip(bb, SHARD_SHAPE))
    return _ravel(position, shard_grid)


def blocks_of_shard(shard_id: int, shape: Sequence[int]) -> List[int]:
    """Global ids of the prediction blocks inside one shard.

    The ids index the C-order block grid of a whole-volume `Blocking`, which is what
    `predict_with_halo_pipelined` builds when no `roi` is passed.
    """
    block = block_shape_for(shape)
    block_grid = _grid(shape, block)
    shard_bb = shard_bounding_box(shard_id, shape)
    ranges = [
        range(bb.start // bs, math.ceil(bb.stop / bs)) for bb, bs in zip(shard_bb, block)
    ]
    return [_ravel(position, block_grid) for position in itertools.product(*ranges)]


def shard_key_path(prediction_path: str, shard_id: int, shape: Sequence[int]) -> str:
    """Path of the file backing one shard.

    The zarr v3 'default' chunk key encoding with the '/' separator names the *outer* grid cell,
    and the shard spans the whole channel axis, so the leading index is always 0.
    """
    position = _unravel(shard_id, _grid(shape, SHARD_SHAPE))
    return os.path.join(prediction_path, PREDICTION_KEY, "c", "0", *(str(p) for p in position))


#
# --- manifest ---
#


def _n_threads(n_threads: Optional[int]) -> int:
    return min(16, mp.cpu_count()) if n_threads is None else int(n_threads)


def in_mask_blocks(mask_path: str, n_threads: Optional[int] = None) -> Tuple[List[int], Tuple[int, ...]]:
    """Ids of the prediction blocks that hold at least one mask voxel.

    These are the blocks the prediction actually writes: `_prepare_block_input` skips a block whose
    mask is empty. The mask chunks are 2x the raw chunks, i.e. 128^3, so one block is one chunk.
    """
    with z5py.File(mask_path, "r") as f:
        mask = f[MASK_KEY]
        shape = tuple(int(s) for s in mask.shape)
        block = block_shape_for(shape)
        n_blocks = int(np.prod(_grid(shape, block)))

        def check(block_id):
            bb = block_bounding_box(block_id, shape)
            return block_id if mask[bb].any() else None

        with futures.ThreadPoolExecutor(_n_threads(n_threads)) as tp:
            found = list(tp.map(check, range(n_blocks)))

    return sorted(b for b in found if b is not None), shape


def assign_shards(
    shard_counts: Dict[int, int], prediction_instances: int
) -> Tuple[List[List[int]], List[int]]:
    """Distribute whole shards over the array tasks, balancing the number of blocks.

    Greedy longest-processing-time: the shards are dealt out largest first to the currently
    least-loaded task. Sorting by (-count, id) and breaking ties on the lowest task index makes the
    result a pure function of the manifest, so every task derives the same partition.
    """
    order = sorted(shard_counts, key=lambda shard_id: (-shard_counts[shard_id], shard_id))
    assignment = [[] for _ in range(prediction_instances)]
    loads = [0] * prediction_instances
    for shard_id in order:
        target = min(range(prediction_instances), key=lambda task: (loads[task], task))
        assignment[target].append(shard_id)
        loads[target] += shard_counts[shard_id]
    return [sorted(shards) for shards in assignment], loads


def cmd_manifest(args) -> None:
    """Write the shard manifest for one cochlea."""
    mask_path = os.path.join(args.folder, MASK_NAME)
    blocks, shape = in_mask_blocks(mask_path, args.n_threads)

    shard_counts: Dict[int, int] = {}
    for block_id in blocks:
        shard_id = shard_of_block(block_id, shape)
        shard_counts[shard_id] = shard_counts.get(shard_id, 0) + 1

    assignment, loads = assign_shards(shard_counts, args.prediction_instances)
    shard_grid = _grid(shape, SHARD_SHAPE)
    manifest = {
        "shape": list(shape),
        "block_shape": list(block_shape_for(shape)),
        "halo": list(halo_for(shape)),
        "shard_shape": list(SHARD_SHAPE),
        "shard_grid": list(shard_grid),
        "n_shards": int(np.prod(shard_grid)),
        "n_blocks_in_mask": len(blocks),
        "n_shards_in_mask": len(shard_counts),
        "prediction_instances": args.prediction_instances,
        # JSON keys must be strings; parsed back with int() in `load_manifest`.
        "shard_block_counts": {str(k): v for k, v in sorted(shard_counts.items())},
        "assignment": assignment,
        "loads": loads,
    }

    if args.expect_blocks is not None and len(blocks) != args.expect_blocks:
        raise ValueError(
            f"Expected {args.expect_blocks} blocks in the mask, found {len(blocks)}. The mask does "
            "not match the one the reference predictions were computed with."
        )

    out_path = os.path.join(args.folder, MANIFEST_NAME)
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent="\t")

    print(f"Shape {shape}, block shape {manifest['block_shape']}, shard grid {shard_grid}")
    print(f"{len(blocks)} of {int(np.prod(_grid(shape, block_shape_for(shape))))} blocks in mask")
    print(f"{len(shard_counts)} of {manifest['n_shards']} shards in mask "
          f"(= the expected number of shard files)")
    print(f"Blocks per task: min {min(loads)}, max {max(loads)}")
    print(f"Wrote {out_path}")


def load_manifest(folder: str) -> dict:
    with open(os.path.join(folder, MANIFEST_NAME)) as f:
        manifest = json.load(f)
    manifest["shape"] = tuple(manifest["shape"])
    manifest["shard_block_counts"] = {int(k): v for k, v in manifest["shard_block_counts"].items()}
    return manifest


#
# --- the sharded array ---
#


def create_prediction(output_folder: str, shape: Sequence[int], overwrite: bool = False) -> zarr.Array:
    """Create the empty sharded prediction array."""
    path = os.path.join(output_folder, PREDICTION_NAME)
    os.makedirs(output_folder, exist_ok=True)
    group = zarr.open_group(path, mode="a", zarr_format=3)
    if PREDICTION_KEY in group and not overwrite:
        array = group[PREDICTION_KEY]
        _check_geometry(array, shape)
        return array
    return group.create_array(
        PREDICTION_KEY,
        shape=(OUTPUT_CHANNELS,) + tuple(shape),
        dtype="float32",
        chunks=CHUNKS,
        shards=SHARDS,
        compressors=ZstdCodec(level=ZSTD_LEVEL),
        fill_value=0,
        overwrite=overwrite,
    )


def _check_geometry(array: zarr.Array, shape: Sequence[int]) -> None:
    expected = (OUTPUT_CHANNELS,) + tuple(shape)
    if tuple(array.shape) != expected:
        raise ValueError(f"Prediction has shape {tuple(array.shape)}, expected {expected}.")
    if tuple(array.chunks) != CHUNKS:
        raise ValueError(f"Prediction has chunks {tuple(array.chunks)}, expected {CHUNKS}.")
    if array.shards is None or tuple(array.shards) != SHARDS:
        raise ValueError(f"Prediction has shards {array.shards}, expected {SHARDS}.")


def open_prediction(output_folder: str, shape: Sequence[int], mode: str = "r+") -> zarr.Array:
    """Open an existing sharded prediction array and validate its geometry."""
    path = os.path.join(output_folder, PREDICTION_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} does not exist. Run the 'init' subcommand before predicting: elf's "
            "require_dataset ignores 'shards' for an existing array, so the array must be created "
            "once, by a single process."
        )
    array = zarr.open_array(os.path.join(path, PREDICTION_KEY), mode=mode)
    _check_geometry(array, shape)
    return array


def cmd_init(args) -> None:
    """Create the empty prediction arrays for all versions of one cochlea.

    This has to happen in a single process. elf's `require_dataset` shim validates only dtype and
    shape for an existing array and silently ignores `chunks` and `shards`, and the array tasks
    would otherwise race to create it.
    """
    manifest = load_manifest(args.folder)
    for version in args.versions:
        output_folder = os.path.join(args.folder, f"SGN_v2-{version}")
        array = create_prediction(output_folder, manifest["shape"], overwrite=args.force)
        print(f"SGN_v2-{version}: shape {tuple(array.shape)}, chunks {tuple(array.chunks)}, "
              f"shards {tuple(array.shards)}")


#
# --- prediction ---
#


class ShardBuffer:
    """In-memory output for the blocks of one shard, so the shard is written in one go.

    `torch_em.util.prediction._write_prediction` writes a block as
    ``output[(slice(None),) + spatial_bb] = prediction`` with *global* coordinates and inspects
    ``output.ndim``. This translates those writes into a buffer covering only the shard.
    """

    def __init__(self, shard_bb: Tuple[slice, ...], n_channels: int = OUTPUT_CHANNELS):
        self._offset = tuple(bb.start for bb in shard_bb)
        extent = tuple(bb.stop - bb.start for bb in shard_bb)
        self.buffer = np.zeros((n_channels,) + extent, dtype="float32")
        self.shape = self.buffer.shape
        self.ndim = self.buffer.ndim

    def __setitem__(self, key, value) -> None:
        channels, spatial = key[0], key[1:]
        local = tuple(
            slice(bb.start - off, bb.stop - off) for bb, off in zip(spatial, self._offset)
        )
        self.buffer[(channels,) + local] = value


def _prefetch_workers(num_prefetch_workers: Optional[int]) -> int:
    """Resolve the number of prefetch threads, matching `prediction_impl`'s default."""
    if num_prefetch_workers is not None:
        return int(num_prefetch_workers)
    n_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    n_cpus = int(n_cpus) if n_cpus else mp.cpu_count()
    return max(1, min(8, n_cpus - 1))


def _load_mean_std(path: str) -> Tuple[float, float]:
    with open(path) as f:
        data = json.load(f)
    # Python floats, not numpy scalars: a numpy scalar would run the normalization in float64 and
    # round differently than a whole-volume run. `prediction_impl` coerces for the same reason.
    return float(data["mean"]), float(data["std"])


def prediction_setup(input_path: str, input_key: str, cochlea_folder: str, model_path: str, shape):
    """Everything the prediction needs, set up exactly as `prediction_impl` would.

    Returns the raw input, the mask, the model, and the preprocess / postprocess callbacks.
    """
    import torch
    from bioimage_cpp.filters import gaussian_smoothing

    from flamingo_tools.file_utils import read_image_data

    from distance_unet_checkpoint import load_distance_unet

    input_ = read_image_data(input_path, input_key)
    if tuple(input_.shape) != tuple(shape):
        raise ValueError(f"Input has shape {tuple(input_.shape)}, manifest says {tuple(shape)}.")

    mask = z5py.File(os.path.join(cochlea_folder, MASK_NAME), "r")[MASK_KEY]
    if tuple(mask.shape) != tuple(shape):
        # A mismatch would silently engage the ResizedVolume path in `prediction_impl`; all masks
        # of this experiment are at full resolution, so treat it as an error instead.
        raise ValueError(f"Mask has shape {tuple(mask.shape)}, expected {tuple(shape)}.")

    mean, std = _load_mean_std(os.path.join(cochlea_folder, MEAN_STD_NAME))

    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available. The prediction is only worth running on a GPU node.")
    model = load_distance_unet(model_path).to("cuda")
    model.eval()

    def preprocess(raw):
        raw = raw.astype("float32")
        raw -= mean
        raw /= std
        return raw

    def postprocess(prediction):
        prediction[1] = gaussian_smoothing(prediction[1], sigma=DISTANCE_SMOOTHING_SIGMA)
        return prediction

    print(f"Mean {mean}, std {std}")
    return input_, mask, model, preprocess, postprocess


def cmd_predict(args) -> None:
    """Predict the shards assigned to this array task."""
    from torch_em.util.prediction import predict_with_halo_pipelined

    manifest = load_manifest(args.cochlea_folder)
    shape = manifest["shape"]
    n_instances = manifest["prediction_instances"]
    if not 0 <= args.task_id < n_instances:
        raise ValueError(
            f"Task id {args.task_id} is out of range: the manifest assigns the shards to "
            f"{n_instances} instances, so the slurm array must be 0-{n_instances - 1}."
        )
    shard_ids = manifest["assignment"][args.task_id]
    if not shard_ids:
        print(f"Task {args.task_id} has no shards assigned, nothing to do.")
        return

    block_shape, halo = block_shape_for(shape), halo_for(shape)
    output = open_prediction(args.output_folder, shape, mode="r+")
    input_, mask, model, preprocess, postprocess = prediction_setup(
        args.input, args.input_key, args.cochlea_folder, args.model, shape
    )

    print(f"Task {args.task_id}: {len(shard_ids)} shards, "
          f"{sum(manifest['shard_block_counts'][s] for s in shard_ids)} blocks")
    print(f"Block shape {block_shape}, halo {halo}")

    for n, shard_id in enumerate(shard_ids, start=1):
        shard_bb = shard_bounding_box(shard_id, shape)
        key_path = shard_key_path(
            os.path.join(args.output_folder, PREDICTION_NAME), shard_id, shape
        )
        if args.skip_existing and os.path.exists(key_path):
            print(f"[{n}/{len(shard_ids)}] shard {shard_id} exists, skipping.")
            continue

        buffer = ShardBuffer(shard_bb)
        predict_with_halo_pipelined(
            input_, model,
            gpu_ids=[0], block_shape=block_shape, halo=halo,
            output=buffer, preprocess=preprocess, postprocess=postprocess, mask=mask,
            roi=shard_bb,
            batch_size=args.batch_size,
            num_prefetch_workers=_prefetch_workers(args.num_prefetch_workers),
            # The buffer is in memory and the shard is written once, after all its blocks are in.
            num_write_workers=1,
            disable_tqdm=True,
        )
        output[(slice(None),) + shard_bb] = buffer.buffer
        print(f"[{n}/{len(shard_ids)}] shard {shard_id} "
              f"({manifest['shard_block_counts'][shard_id]} blocks) written.", flush=True)

    print(f"Task {args.task_id} done.")


def cmd_selftest(args) -> None:
    """Check that predicting one shard at a time reproduces a whole-volume run exactly.

    This is the claim the sharded layout rests on: the per-shard driver passes `roi=` so that the
    block grid is built over the shard instead of the volume, and it is only safe because
    `_prepare_block_input` reads each block's halo from the global input regardless of `roi`, and
    because the shard grid is aligned with the block grid. The test predicts one shard both ways --
    once with `roi=`, once over the global block grid restricted to the same block ids -- and
    compares the results.
    """
    from torch_em.util.prediction import predict_with_halo_pipelined

    manifest = load_manifest(args.cochlea_folder)
    shape = manifest["shape"]
    counts = manifest["shard_block_counts"]
    # The fullest shard by default, so the comparison covers a shard with no empty blocks.
    shard_id = args.shard_id
    if shard_id is None:
        shard_id = max(counts, key=lambda s: (counts[s], -s))
    if shard_id not in counts:
        raise ValueError(f"Shard {shard_id} holds no in-mask blocks.")

    shard_bb = shard_bounding_box(shard_id, shape)
    block_ids = blocks_of_shard(shard_id, shape)
    block_shape, halo = block_shape_for(shape), halo_for(shape)
    print(f"Shard {shard_id}: bounding box {shard_bb}, {len(block_ids)} blocks, "
          f"{counts[shard_id]} of them in the mask")

    input_, mask, model, preprocess, postprocess = prediction_setup(
        args.input, args.input_key, args.cochlea_folder, args.model, shape
    )
    common = dict(
        gpu_ids=[0], block_shape=block_shape, halo=halo, preprocess=preprocess,
        postprocess=postprocess, mask=mask, num_write_workers=1, disable_tqdm=True,
    )

    # What the driver does: the blocking is built over the shard.
    per_shard = ShardBuffer(shard_bb)
    predict_with_halo_pipelined(input_, model, output=per_shard, roi=shard_bb, **common)

    # The reference: the blocking is the whole-volume grid, restricted to this shard's blocks.
    whole_volume = ShardBuffer(shard_bb)
    predict_with_halo_pipelined(input_, model, output=whole_volume, iter_list=block_ids, **common)

    equal = np.array_equal(per_shard.buffer, whole_volume.buffer)
    max_diff = float(np.abs(per_shard.buffer - whole_volume.buffer).max())
    n_nonzero = int((per_shard.buffer != 0).sum())
    print(f"Non-zero voxels: {n_nonzero} (a shard is {per_shard.buffer.size})")
    print(f"Identical: {equal}, maximum absolute difference: {max_diff}")
    if n_nonzero == 0:
        raise SystemExit("The shard is empty, the comparison proves nothing. Pick another one.")
    if not equal:
        raise SystemExit("Per-shard prediction does NOT reproduce the whole-volume run.")
    print("Self-test passed.")


#
# --- conversion of an existing unsharded prediction ---
#


def cmd_convert(args) -> None:
    """Copy an existing unsharded prediction into the sharded layout."""
    from bioimage_py import copy

    source = zarr.open_array(os.path.join(args.input, PREDICTION_KEY), mode="r")
    shape = tuple(int(s) for s in source.shape[1:])
    if source.shape[0] != OUTPUT_CHANNELS:
        raise ValueError(f"Source has {source.shape[0]} channels, expected {OUTPUT_CHANNELS}.")

    target = open_prediction(args.output_folder, shape, mode="r+")
    print(f"Converting {args.input} -> {os.path.join(args.output_folder, PREDICTION_NAME)}")
    print(f"Source chunks {tuple(source.chunks)}, shards {source.shards}")
    print(f"Target chunks {tuple(target.chunks)}, shards {tuple(target.shards)}")

    # block_shape == shards makes every write a complete shard write, so there is no
    # read-modify-write of a partially filled shard. bioimage_py additionally routes the blocks of
    # a shard to a single worker, so this is safe regardless.
    copy(source, target, block_shape=SHARDS, num_workers=args.num_workers)
    print("Conversion done.")


#
# --- verification ---
#


def count_shard_files(output_folder: str) -> int:
    """Number of shard files on disk. One file per shard, so this is the number of written shards."""
    root = os.path.join(output_folder, PREDICTION_NAME, PREDICTION_KEY, "c")
    return sum(len(files) for _, _, files in os.walk(root))


def missing_shards(output_folder: str, shard_ids: Sequence[int], shape: Sequence[int]) -> List[int]:
    """Shards that should have been written but have no file.

    Each shard is written in a single complete write, and zarr writes a temporary file and renames
    it, so a shard file that exists is a finished shard. Checking for its presence is therefore a
    sufficient completeness check.
    """
    prediction_path = os.path.join(output_folder, PREDICTION_NAME)
    return [
        shard_id for shard_id in shard_ids
        if not os.path.exists(shard_key_path(prediction_path, shard_id, shape))
    ]


def cmd_verify(args) -> None:
    """Check the written shards against the manifest and, optionally, the values against a reference."""
    manifest = load_manifest(args.cochlea_folder)
    shape = manifest["shape"]
    expected_shards = sorted(manifest["shard_block_counts"])
    missing = missing_shards(args.output_folder, expected_shards, shape)
    found = count_shard_files(args.output_folder)
    print(f"Shard files: {found}, expected {len(expected_shards)}")
    ok = not missing and found == len(expected_shards)

    if missing:
        print(f"Missing {len(missing)} shards: {missing[:10]}"
              + (" ..." if len(missing) > 10 else ""))
        print("Re-run the apply job; --skip_existing makes it redo only the missing shards.")
        print("(A shard whose every in-mask voxel predicts exactly 0 in all three channels is not "
              "written by zarr. That is possible in principle and would show up here as a spurious "
              "missing shard; check the block count of the shard in the manifest if you suspect it.)")
    elif found > len(expected_shards):
        print(f"{found - len(expected_shards)} unexpected extra shard files. This looks like a "
              "prediction left over from a different mask; delete predictions.zarr and re-run.")

    if args.reference is not None:
        rng = np.random.default_rng(args.seed)
        target = open_prediction(args.output_folder, shape, mode="r")
        source = zarr.open_array(os.path.join(args.reference, PREDICTION_KEY), mode="r")
        # Sample from the in-mask shards, where the prediction is actually non-trivial.
        in_mask_shards = sorted(manifest["shard_block_counts"])
        sampled = rng.choice(
            in_mask_shards, size=min(args.n_samples, len(in_mask_shards)), replace=False
        )
        n_equal = 0
        for shard_id in sampled:
            bb = (slice(None),) + shard_bounding_box(int(shard_id), shape)
            if np.array_equal(source[bb], target[bb]):
                n_equal += 1
            else:
                print(f"MISMATCH in shard {shard_id}")
        print(f"Exactly equal shards: {n_equal}/{len(sampled)}")
        ok = ok and n_equal == len(sampled)

    if not ok:
        raise SystemExit("Verification FAILED.")
    print("Verification passed.")


#
# --- CLI ---
#


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("manifest", help="Count in-mask blocks per shard and assign them.")
    p.add_argument("-f", "--folder", required=True, help="The cochlea folder, holding mask.zarr.")
    p.add_argument("--prediction_instances", type=int, default=10)
    p.add_argument("--n_threads", type=int, default=None)
    p.add_argument("--expect_blocks", type=int, default=None,
                   help="Fail if the mask does not hold exactly this many in-mask blocks.")
    p.set_defaults(func=cmd_manifest)

    p = subparsers.add_parser("init", help="Create the empty sharded prediction arrays.")
    p.add_argument("-f", "--folder", required=True, help="The cochlea folder.")
    p.add_argument("--versions", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--force", action="store_true", help="Overwrite an existing array.")
    p.set_defaults(func=cmd_init)

    p = subparsers.add_parser("predict", help="Predict the shards of one array task.")
    p.add_argument("-i", "--input", required=True, help="Path to the raw data.")
    p.add_argument("--input_key", default="s0")
    p.add_argument("-c", "--cochlea_folder", required=True,
                   help="Folder holding mask.zarr, mean_std.json and the manifest.")
    p.add_argument("-o", "--output_folder", required=True, help="Folder holding predictions.zarr.")
    p.add_argument("-m", "--model", required=True, help="Path to the model checkpoint folder.")
    p.add_argument("-t", "--task_id", type=int, required=True, help="The slurm array task id.")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_prefetch_workers", type=int, default=None)
    p.add_argument("--skip_existing", action="store_true", help="Skip shards already written.")
    p.set_defaults(func=cmd_predict)

    p = subparsers.add_parser(
        "selftest", help="Check that per-shard prediction reproduces a whole-volume run."
    )
    p.add_argument("-i", "--input", required=True, help="Path to the raw data.")
    p.add_argument("--input_key", default="s0")
    p.add_argument("-c", "--cochlea_folder", required=True)
    p.add_argument("-m", "--model", required=True)
    p.add_argument("--shard_id", type=int, default=None,
                   help="Shard to test. Defaults to the one with the most in-mask blocks.")
    p.set_defaults(func=cmd_selftest)

    p = subparsers.add_parser("convert", help="Convert an unsharded prediction to the new layout.")
    p.add_argument("-i", "--input", required=True, help="The existing predictions.zarr.")
    p.add_argument("-o", "--output_folder", required=True, help="Folder for the new predictions.zarr.")
    p.add_argument("--num_workers", type=int, default=8)
    p.set_defaults(func=cmd_convert)

    p = subparsers.add_parser("verify", help="Check a prediction for completeness and equality.")
    p.add_argument("-c", "--cochlea_folder", required=True)
    p.add_argument("-o", "--output_folder", required=True)
    p.add_argument("--reference", default=None,
                   help="Optional unsharded predictions.zarr to compare the values against.")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_verify)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
