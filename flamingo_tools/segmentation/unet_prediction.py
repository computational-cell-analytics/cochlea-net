"""Prediction using distance U-Net.
Parallelization using multiple GPUs is currently only possible by calling functions directly.
Functions for the parallelization end with '_slurm'
and divide the process into preprocessing, prediction, and segmentation.
"""
import json
import multiprocessing as mp
import os
import re
from concurrent import futures
from functools import partial
from typing import Dict, Optional, Tuple, Union

import elf.parallel as parallel
import numpy as np
import torch
import z5py

from scipy import ndimage
from bioimage_cpp.filters import gaussian_smoothing
from bioimage_cpp.utils import Blocking

from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper, SimpleTransformationWrapperWithHalo
from elf.wrapper.base import MultiTransformationWrapper
from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from elf.util import normalize_index, squeeze_singletons
from skimage.filters import gaussian
from torch_em.util.prediction import predict_with_halo_pipelined
from tqdm import tqdm

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.model_utils import load_model_from_path


class SelectChannel(SimpleTransformationWrapper):
    """Wrapper to select a chanel from an array-like dataset object.

    Args:
        volume: The array-like input dataset.
        channel: The channel that will be selected.
    """
    def __init__(self, volume: np.typing.ArrayLike, channel: int):
        self.channel = channel
        super().__init__(volume, lambda x: x[self.channel], with_channels=True)

    def __getitem__(self, key):
        # Index the channel in the store instead of reading all channels and discarding
        # all but one. The predictions are chunked per channel, so the base-class
        # implementation decompresses one chunk per channel for every read.
        index, to_squeeze = normalize_index(key, self.shape)
        return squeeze_singletons(self._volume[(self.channel,) + index], to_squeeze)

    @property
    def shape(self):
        return self._volume.shape[1:]

    @property
    def chunks(self):
        return self._volume.chunks[1:]

    @property
    def ndim(self):
        return self._volume.ndim - 1


def _available_cpus():
    """Return the number of cores this process may use.

    `multiprocessing.cpu_count` reports the cores of the whole node, which oversubscribes
    a job on a shared slurm partition. Prefer the slurm allocation and the CPU affinity mask.
    """
    n_slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if n_slurm is not None:
        return int(n_slurm)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return mp.cpu_count()


# Slurm sets these when the job holds a GPU allocation.
_SLURM_GPU_ENV_VARS = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS_ON_NODE")


def _require_gpu_if_allocated():
    """Fail if the job holds a GPU allocation that torch cannot see.

    CPU inference is ~50x slower than an A100 and does not finish inside a realistic time
    limit, so a silent fallback wastes the whole allocation. Set FLAMINGO_ALLOW_CPU=1 to run
    on the CPU inside a GPU allocation on purpose.
    """
    if torch.cuda.is_available() or os.environ.get("FLAMINGO_ALLOW_CPU"):
        return
    allocation = {var: os.environ[var] for var in _SLURM_GPU_ENV_VARS if os.environ.get(var)}
    if not allocation:
        return
    details = ", ".join(f"{var}={value!r}" for var, value in allocation.items())
    raise RuntimeError(
        f"The job on {os.environ.get('SLURMD_NODENAME')!r} holds a GPU allocation ({details}) "
        f"but torch cannot see a device: CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}, device_count="
        f"{torch.cuda.device_count()}, torch={torch.__version__}, "
        f"cuda_build={torch.version.cuda}. CPU inference is ~50x slower and will not finish "
        "inside the time limit. Requeue on another node, or set FLAMINGO_ALLOW_CPU=1 to run "
        "on the CPU anyway."
    )


def _get_device_and_tiling(block_shape, halo, input_):
    have_cuda = torch.cuda.is_available()
    if block_shape is None:
        block_shape = (128, 128, 128) if have_cuda else getattr(input_, "chunks", (64, 64, 64))
    if halo is None:
        # A z halo of 16 leaves a seam in the center distances at the z faces of the blocks, which
        # splits cells there (doc/input_mask.md).
        halo = (32, 32, 32)
    # Clip to the volume. A block larger than the input makes the halo dominate the forward
    # pass: a (64, 128, 128) volume with a (64, 256, 256) block is padded to 96 x 384 x 384.
    # The clip only depends on the input shape, so all slurm array tasks get the same grid.
    block_shape = tuple(min(bs, sh) for bs, sh in zip(block_shape, input_.shape))
    halo = tuple(min(ha, bs // 2) for ha, bs in zip(halo, block_shape))
    # The U-Nets (depth 4) need every input axis divisible by 16. The clip can break this:
    # 61 slices + 2 * 16 halo = 93. Grow the block; torch_em pads the excess.
    block_shape = tuple(bs + (-(bs + 2 * ha)) % 16 for bs, ha in zip(block_shape, halo))
    gpu_ids = [0] if have_cuda else ["cpu"]
    return gpu_ids, block_shape, halo


def prediction_impl(
    input_path,
    input_key,
    output_folder,
    model_path,
    scale,
    block_shape,
    halo,
    output_channels=3,
    apply_postprocessing=True,
    prediction_instances=1,
    slurm_task_id=0,
    mean=None,
    std=None,
    mask=None,
    batch_size=1,
    num_prefetch_workers=None,
):
    """@private
    """
    _require_gpu_if_allocated()
    # The prefetch workers hide the block loading behind the GPU forward pass, so they only
    # help up to the number of cores the job actually has.
    if num_prefetch_workers is None:
        num_prefetch_workers = max(1, min(8, _available_cpus() - 1))
    else:
        num_prefetch_workers = int(num_prefetch_workers)
    batch_size = int(batch_size)
    model = load_model_from_path(model_path)

    input_ = read_image_data(input_path, input_key)

    if output_folder is None:
        image_mask = mask
    else:
        mask_path = os.path.join(output_folder, "mask.zarr")
        if os.path.exists(mask_path):
            image_mask = z5py.File(mask_path, "r")["mask"]
            # resize mask
            image_shape = input_.shape
            mask_shape = image_mask.shape
            if image_shape != mask_shape:
                image_mask = ResizedVolume(image_mask, image_shape, order=0)
        else:
            image_mask = mask

    if scale is None or np.isclose(scale, 1):
        original_shape = None
    else:
        original_shape = input_.shape
        new_shape = tuple(
            int(round(sh / scale)) for sh in original_shape
        )
        print("The input is processed downsampled by a factor of scale", scale)
        print("Corresponding to shape", new_shape, "instead of", original_shape)
        input_ = ResizedVolume(input_, shape=new_shape, order=3)
        image_mask = ResizedVolume(image_mask, new_shape, order=0)

    # The first layer of the U-Nets is an InstanceNorm3d, which standardizes every block by its
    # own statistics. A global mean and std cancel out, so the default passes the raw values.
    # A given mean and std are a legacy option, to reproduce older predictions bit for bit.
    if mean is None or std is None:
        def preprocess(raw):
            return raw.astype("float32")
    else:
        # Coerce to Python floats: a numpy scalar would make the normalization below run in
        # float64 and round differently than the values read back from 'mean_std.json'.
        mean, std = float(mean), float(std)
        print(f"Normalize with the mean {mean} and the standard deviation {std}.")

        def preprocess(raw):
            raw = raw.astype("float32")
            raw -= mean
            raw /= std
            return raw

    if apply_postprocessing:
        # Smooth the distance prediction channel.
        def postprocess(x):
            x[1] = gaussian_smoothing(x[1], sigma=2.0)
            return x
    elif output_channels > 1:
        postprocess = None
    else:
        def postprocess(x):
            return x[0] if x.ndim == 4 else x.squeeze()

    gpu_ids, block_shape, halo = _get_device_and_tiling(block_shape, halo, input_)
    shape = input_.shape
    ndim = len(shape)
    if output_channels > 1:
        output_shape = (output_channels,) + input_.shape
        output_chunks = (1,) + block_shape
    else:
        output_shape = input_.shape
        output_chunks = block_shape

    blocking = Blocking([0] * ndim, shape, block_shape)
    n_blocks = blocking.number_of_blocks
    device = "CPU" if gpu_ids == ["cpu"] else "GPU"
    print(f"Predict with {device}: shape {tuple(shape)}, block_shape {block_shape}, "
          f"halo {halo}, {n_blocks} blocks")
    if prediction_instances != 1:
        # shuffle indexes with fixed seed to balance out segmentation blocks for slurm workers
        rng = np.random.default_rng(seed=1234)
        iteration_ids = [x.tolist() for x in np.array_split(list(rng.permutation(n_blocks)), prediction_instances)]
        slurm_iteration = iteration_ids[slurm_task_id]
    else:
        slurm_iteration = list(range(n_blocks))

    if output_folder is None:
        output = np.zeros(output_shape, dtype=np.float32)
        predict_with_halo_pipelined(
            input_, model,
            gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
            output=output, preprocess=preprocess, postprocess=postprocess,
            mask=image_mask,
            iter_list=slurm_iteration,
            batch_size=batch_size, num_prefetch_workers=num_prefetch_workers,
        )

    else:
        output_path = os.path.join(output_folder, "predictions.zarr")
        with open_file(output_path, "a") as f:
            output = f.require_dataset(
                "prediction",
                shape=output_shape,
                # zstd decompresses much faster than gzip, which dominates the cost of the
                # block-wise reads in the detection and segmentation steps that follow.
                compression="zstd",
                chunks=output_chunks,
                dtype="float32",
            )

            predict_with_halo_pipelined(
                input_, model,
                gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
                output=output, preprocess=preprocess, postprocess=postprocess,
                mask=image_mask,
                iter_list=slurm_iteration,
                batch_size=batch_size, num_prefetch_workers=num_prefetch_workers,
                # The chunks are aligned with block_shape, so each chunk has exactly one writer.
                num_write_workers=2,
            )

    if output_folder is None:
        return original_shape, output
    else:
        return original_shape, None


# The mask grid of the default rule is the GPU prediction block, so one mask block is one U-Net
# block. The rule reads the pyramid level LOWRES_LEVELS steps of 2x below the input, where one mask
# block is (128 / 8)^3 = 16^3 voxels.
MASK_BLOCK = (128, 128, 128)
LOWRES_LEVELS = 3
LOWRES_FACTOR = 2 ** LOWRES_LEVELS
# Per class: the block percentile at the low-res level, the factor on the background level b, and
# whether a kept block adds its 26 neighbours. Fitted on 100 and validated on 89 other SGN and IHC
# segmentations of whole cochleae from the S3 bucket, split by animal: every validation segmentation
# keeps >= 99.5% of its relevant objects complete (doc/input_mask.md).
MASK_RULES = {
    "sgn": {"percentile": 95, "background_factor": 1.08, "dilation": False},
    "ihc": {"percentile": 99, "background_factor": 1.75, "dilation": True},
}
# The pyramid level must be a mean (or other averaging) downsampling of the input. In a background
# block, its mean must agree with 8^3 mean pooling of the input within this fraction of b, and its
# standard deviation must stay below this multiple of the pooled one (+ 1). A subsampled pyramid
# keeps the full noise of the input, about 4x the pooled standard deviation in the S3 data.
_POOLING_MEAN_TOLERANCE = 0.05
_POOLING_STD_FACTOR = 2.0


def _lowres_key(input_key: Optional[str]) -> Optional[str]:
    """Return the key of the pyramid level LOWRES_LEVELS below `input_key`, e.g. 's0' -> 's3'."""
    if input_key is None:
        return None
    head, _, last = input_key.rstrip("/").rpartition("/")
    match = re.fullmatch(r"s(\d+)", last)
    if match is None:
        return None
    level = f"s{int(match.group(1)) + LOWRES_LEVELS}"
    return f"{head}/{level}" if head else level


def _read_pyramid_level(input_path, input_key, shape):
    """Return the low-res pyramid level of the input as an array, or None if it does not fit."""
    key = _lowres_key(input_key)
    if key is None:
        return None, None
    try:
        level = read_image_data(input_path, key)
    except Exception as e:  # A missing level is a normal case, e.g. for a dataset without pyramid.
        print(f"No pyramid level {key}: {e}")
        return None, None
    fits = all(ls in (sh // LOWRES_FACTOR, -(-sh // LOWRES_FACTOR)) for ls, sh in zip(level.shape, shape))
    if not fits:
        print(f"The shape {level.shape} of {key} is not the input shape {shape} / {LOWRES_FACTOR}.")
        return None, None
    return level[:], key


def _mean_pool(raw, n_threads):
    """Downsample the input by LOWRES_FACTOR with the mean, like the 2x mean pyramids."""
    factor = LOWRES_FACTOR
    shape = tuple(-(-sh // factor) for sh in raw.shape)
    integer = np.issubdtype(raw.dtype, np.integer)
    lowres = np.zeros(shape, dtype=raw.dtype if integer else "float32")
    grid = tuple(-(-sh // bs) for sh, bs in zip(raw.shape, MASK_BLOCK))

    def pool(block_index):
        index = np.unravel_index(block_index, grid)
        bb = tuple(slice(i * bs, min((i + 1) * bs, sh)) for i, bs, sh in zip(index, MASK_BLOCK, raw.shape))
        data = np.asarray(raw[bb], dtype="float64")
        padded = np.pad(data, [(0, -sh % factor) for sh in data.shape])
        out_shape = tuple(sh // factor for sh in padded.shape)
        sums = padded.reshape(out_shape[0], factor, out_shape[1], factor, out_shape[2], factor).sum(axis=(1, 3, 5))
        # The number of voxels of each output voxel. It is below factor^3 only at the volume border.
        counts = [np.minimum(factor, sh - factor * np.arange(n)) for sh, n in zip(data.shape, out_shape)]
        means = sums / (counts[0][:, None, None] * counts[1][None, :, None] * counts[2][None, None, :])
        out_bb = tuple(slice(b.start // factor, b.start // factor + n) for b, n in zip(bb, out_shape))
        lowres[out_bb] = np.rint(means) if integer else means

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tp.map(pool, range(int(np.prod(grid)))))
    return lowres


def _lowres_statistics(lowres, percentile):
    """Return the background level b and the statistics of every mask block of the low-res volume.

    b is the median of the non-zero voxels, interpolated inside the histogram bin for integer
    data. Fusion padding is exactly 0, so it does not shift b. The block score is the percentile
    of the block voxels. Blocks at the volume border are padded with zeros.
    """
    block = tuple(bs // LOWRES_FACTOR for bs in MASK_BLOCK)
    grid = tuple(-(-sh // bs) for sh, bs in zip(lowres.shape, block))
    blocks = {name: np.zeros(grid, dtype="float32") for name in ("score", "mean", "std", "nonzero")}
    integer = np.issubdtype(lowres.dtype, np.integer)
    hist, values = np.zeros(1, dtype="int64"), []
    # Process one slab of blocks at a time to bound the memory of the percentile computation.
    for gz in range(grid[0]):
        slab = np.asarray(lowres[gz * block[0]:(gz + 1) * block[0]])
        nonzero = slab[slab > 0]
        if integer:
            counts = np.bincount(nonzero.ravel())
            if len(counts) > len(hist):
                hist = np.pad(hist, (0, len(counts) - len(hist)))
            hist[:len(counts)] += counts
        else:
            values.append(nonzero.ravel())
        padded = np.zeros((block[0], grid[1] * block[1], grid[2] * block[2]), dtype=slab.dtype)
        padded[:slab.shape[0], :slab.shape[1], :slab.shape[2]] = slab
        view = padded.reshape(block[0], grid[1], block[1], grid[2], block[2]).transpose(1, 3, 0, 2, 4)
        view = view.reshape(grid[1] * grid[2], -1)
        blocks["score"][gz] = np.percentile(view, percentile, axis=1).reshape(grid[1:])
        blocks["mean"][gz] = view.mean(axis=1).reshape(grid[1:])
        blocks["std"][gz] = view.std(axis=1).reshape(grid[1:])
        blocks["nonzero"][gz] = (view > 0).mean(axis=1).reshape(grid[1:])

    if not integer:
        values = np.concatenate(values)
        background = float(np.median(values)) if values.size else 0.0
    elif hist.sum() == 0:
        background = 0.0
    else:
        cdf = np.cumsum(hist).astype("float64")
        target = 0.5 * cdf[-1]
        v = int(np.searchsorted(cdf, target, side="left"))
        below = cdf[v - 1] if v > 0 else 0.0
        background = v - 0.5 + (target - below) / max(hist[v], 1)
    return float(background), blocks


def _matches_mean_pooling(raw, lowres, background, blocks):
    """Check the pyramid level against 8^3 mean pooling of the input, in one background block.

    The block is the one with the smallest standard deviation without zero voxels: there, the
    pooling of the noise shows most clearly.
    """
    candidates = blocks["nonzero"] == 1
    if not candidates.any():
        candidates = blocks["mean"] > 0
    if not candidates.any():
        return True
    index = np.unravel_index(int(np.argmin(np.where(candidates, blocks["std"], np.inf))), candidates.shape)
    block = tuple(bs // LOWRES_FACTOR for bs in MASK_BLOCK)
    lo = [i * bs for i, bs in zip(index, block)]
    hi = [min(a + bs, ls, sh // LOWRES_FACTOR) for a, bs, ls, sh in zip(lo, block, lowres.shape, raw.shape)]
    if any(b <= a for a, b in zip(lo, hi)):
        return False
    level = np.asarray(lowres[tuple(slice(a, b) for a, b in zip(lo, hi))], dtype="float64")
    data = np.asarray(raw[tuple(slice(a * LOWRES_FACTOR, b * LOWRES_FACTOR) for a, b in zip(lo, hi))], dtype="float64")
    n, f = level.shape, LOWRES_FACTOR
    pooled = data.reshape(n[0], f, n[1], f, n[2], f).mean(axis=(1, 3, 5))
    mean_ok = abs(level.mean() - pooled.mean()) <= max(_POOLING_MEAN_TOLERANCE * background, 1.0)
    std_ok = level.std() <= _POOLING_STD_FACTOR * pooled.std() + 1.0
    return bool(mean_ok and std_ok)


def _fit_grid(grid_values, grid_shape):
    """Crop or pad (with the edge values) a block grid to `grid_shape`.

    A pyramid level with floor(shape / 8) voxels misses a last slab of fewer than 8 planes. Its
    blocks take the values of their neighbours.
    """
    grid_values = grid_values[tuple(slice(0, g) for g in grid_shape)]
    pad = [(0, g - s) for g, s in zip(grid_shape, grid_values.shape)]
    return np.pad(grid_values, pad, mode="edge") if any(p[1] for p in pad) else grid_values


def compute_mask_blocks(
    input_path: str,
    input_key: Optional[str],
    seg_class: str,
    background_factor: Optional[float] = None,
) -> Tuple[np.ndarray, Dict]:
    """Decide which blocks of MASK_BLOCK the U-Net predicts, with the rule of `seg_class`.

    The rule reads the input at 8x lower resolution: the pyramid level 3 steps below `input_key`
    ('s0' -> 's3'), if it exists and averages the input, else the 8^3 mean of the input. With b the
    median of the non-zero voxels (the background level), a block is kept if the percentile of
    MASK_RULES[seg_class] of its voxels is above `background_factor * b`. For 'ihc', the 26
    neighbours of a kept block are also kept, so that cells that cross a block face stay complete.

    Args:
        input_path: The file path to the image data, or a zarr store.
        input_key: The key / internal path of the image data.
        seg_class: 'sgn' or 'ihc'.
        background_factor: The threshold as a multiple of the background level b.
            By default, the value of MASK_RULES[seg_class].

    Returns:
        The boolean grid of kept blocks, with ceil(shape / MASK_BLOCK) blocks per axis.
        The statistics of the decision, as written to 'mask_statistics.json'.
    """
    if seg_class not in MASK_RULES:
        raise ValueError(f"The mask rule needs seg_class 'sgn' or 'ihc', got {seg_class}.")
    rule = MASK_RULES[seg_class]
    factor = rule["background_factor"] if background_factor is None else float(background_factor)
    n_threads = min(16, mp.cpu_count())

    raw = read_image_data(input_path, input_key)
    lowres, source = _read_pyramid_level(input_path, input_key, raw.shape)
    if lowres is not None:
        background, blocks = _lowres_statistics(lowres, rule["percentile"])
        if not _matches_mean_pooling(raw, lowres, background, blocks):
            print(f"The pyramid level {source} is not an averaging downsampling of the input.")
            lowres = None
    if lowres is None:
        print(f"Downsample the input {LOWRES_FACTOR}x with the mean.")
        lowres, source = _mean_pool(raw, n_threads), "mean pooling of the input"
        background, blocks = _lowres_statistics(lowres, rule["percentile"])

    grid_shape = tuple(-(-sh // bs) for sh, bs in zip(raw.shape, MASK_BLOCK))
    scores = _fit_grid(blocks["score"], grid_shape)
    threshold = factor * background
    keep = scores > threshold
    if rule["dilation"]:
        keep = ndimage.binary_dilation(keep, structure=np.ones((3, 3, 3), dtype=bool))

    stats = {
        "rule": "background", "seg_class": seg_class, "percentile": rule["percentile"],
        "background_factor": factor, "dilation": "26 neighbours" if rule["dilation"] else "none",
        "background": background, "threshold": threshold, "lowres_source": source,
        "block_shape": list(MASK_BLOCK), "n_blocks": int(keep.size), "kept_blocks": int(keep.sum()),
        "kept_fraction": float(keep.mean()),
    }
    print(f"Mask rule for {seg_class}: block p{rule['percentile']} at {LOWRES_FACTOR}x lower resolution "
          f"> {factor} x b = {threshold:.1f} (b = {background:.1f}, from {source}), dilation {stats['dilation']}.")
    return keep, stats


def _legacy_mask_blocks(raw, seg_class, relative_threshold, absolute_threshold):
    """Decide the mask blocks with the percentile rule of flamingo-tools before 2026-11.

    The blocks are 2 x the input chunks. A block is kept if its upper percentile at full
    resolution exceeds `min_intensity`: `absolute_threshold` if given, else `relative_threshold`
    times the median p99.9 of 16 random blocks, clipped to the class range.
    """
    if seg_class in ("ihc", "ihc_low"):
        upper_percentile, absolute_min, absolute_max = 99, 250, 400
    else:
        upper_percentile, absolute_min, absolute_max = 95, 100, 200
    print(f"Calculating the legacy mask for segmentation class {seg_class}.")

    chunks = getattr(raw, "chunks", (64, 64, 64))
    block_shape = tuple(2 * ch for ch in chunks)
    blocking = Blocking([0, 0, 0], raw.shape, block_shape)
    n_blocks = blocking.number_of_blocks

    if absolute_threshold is None:
        # Estimate the global signal level from a random sample of blocks using the
        # median of per-block p99.9 values. The median is robust to outlier spikes
        # from fluorescence residues that would inflate a simple maximum.
        rng = np.random.default_rng(42)
        sample_ids = rng.choice(n_blocks, size=min(n_blocks, 16), replace=False).tolist()
        sample_highs = []
        for bid in sample_ids:
            block = blocking.get_block(bid)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            sample_highs.append(float(np.percentile(raw[bb], 99.9)))
        global_high = float(np.median(sample_highs))
        min_intensity = max(absolute_min, min(relative_threshold * global_high, absolute_max))
        print(f"Adaptive min_intensity: {min_intensity:.1f} (global_high={global_high:.1f}, cap={absolute_max})")
    else:
        min_intensity = float(absolute_threshold)
        print(f"Using absolute min_intensity: {min_intensity:.1f}")

    grid_shape = tuple(-(-sh // bs) for sh, bs in zip(raw.shape, block_shape))
    keep = np.zeros(grid_shape, dtype=bool)

    def find_mask_block(block_id):
        block = blocking.get_block(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        if np.percentile(raw[bb], upper_percentile) > min_intensity:
            keep[tuple(beg // bs for beg, bs in zip(block.begin, block_shape))] = True

    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(find_mask_block, range(n_blocks)), total=n_blocks))

    stats = {
        "rule": "legacy", "seg_class": seg_class, "percentile": upper_percentile,
        "threshold": min_intensity, "threshold_mode": "adaptive" if absolute_threshold is None else "absolute",
        "block_shape": list(block_shape), "n_blocks": int(keep.size), "kept_blocks": int(keep.sum()),
        "kept_fraction": float(keep.mean()),
    }
    return keep, block_shape, stats


def find_mask(
    input_path: str,
    input_key: Optional[str],
    output_folder: Optional[str],
    seg_class: Optional[str] = "sgn",
    relative_threshold: float = 0.7,
    absolute_threshold: Optional[float] = None,
    legacy_mask: bool = False,
    background_factor: Optional[float] = None,
) -> Optional[np.ndarray]:
    """Determine the mask of the blocks that the U-Net predicts.

    For 'sgn' and 'ihc', `compute_mask_blocks` decides the blocks relative to the background
    level of the volume. The legacy rule of flamingo-tools before 2026-11 applies if
    `legacy_mask` is set, or for every other `seg_class` (None, 'sgn_low', 'ihc_low'): a block of
    2 x the input chunks is kept if its p95 ('ihc', 'ihc_low': p99) at full resolution exceeds
    `absolute_threshold`, or an adaptive threshold derived from `relative_threshold`.

    The default rule assumes that most non-zero voxels are background, as in a whole cochlea. It
    was validated on whole cochleae only. In a small crop that cells fill for the most part, the
    background level is too high; use `use_mask=False` (`--disable_masking`) for such a crop.

    The mask is stored in 'mask.zarr' (key 'mask', uint8, the shape of the input, one chunk per
    mask block, only the kept blocks written) and the decision in 'mask_statistics.json'. An
    existing mask is not recomputed.

    Args:
        input_path: The file path to the image data.
        input_key: The key / internal path of the image data.
        output_folder: The output folder for storing the mask data. If None, the mask is returned.
        seg_class: The segmentation class: 'sgn', 'ihc', 'sgn_low' or 'ihc_low'.
        relative_threshold: Legacy rule only. The fraction of the global signal level used as the
            adaptive threshold.
        absolute_threshold: Legacy rule only. The threshold of the block percentile.
        legacy_mask: Use the legacy rule for 'sgn' and 'ihc'.
        background_factor: Default rule only. The threshold as a multiple of the background level.

    Returns:
        The mask as a uint8 array if `output_folder` is None, else None.
    """
    legacy = legacy_mask or seg_class not in MASK_RULES
    if absolute_threshold is not None and not legacy:
        raise ValueError(
            f"absolute_threshold={absolute_threshold} applies only to the legacy mask rule. "
            "Use --legacy_mask (legacy_mask=True) to keep the absolute threshold."
        )
    if background_factor is not None and legacy:
        raise ValueError("background_factor applies only to the default mask rule, not to the legacy rule.")

    if output_folder is not None:
        mask_path = os.path.join(output_folder, "mask.zarr")
        if os.path.isdir(mask_path) and "mask" in z5py.File(mask_path, "r"):
            return None

    raw = read_image_data(input_path, input_key)
    if legacy:
        keep, block_shape, stats = _legacy_mask_blocks(raw, seg_class, relative_threshold, absolute_threshold)
    else:
        keep, stats = compute_mask_blocks(input_path, input_key, seg_class, background_factor)
        block_shape = MASK_BLOCK
    n_blocks, n_kept = keep.size, int(keep.sum())
    print(f"{n_kept}/{n_blocks} ({round(n_kept / n_blocks * 100, 2)} %) used for segmentation.")

    if output_folder is None:
        mask = np.zeros(raw.shape, dtype="uint8")
    else:
        mask = z5py.File(mask_path, "a").create_dataset(
            "mask", shape=raw.shape, compression="gzip", dtype="uint8", chunks=block_shape
        )

    def write_block(index):
        mask[tuple(slice(i * bs, min((i + 1) * bs, sh)) for i, bs, sh in zip(index, block_shape, raw.shape))] = 1

    with futures.ThreadPoolExecutor(min(16, mp.cpu_count())) as tp:
        list(tp.map(write_block, map(tuple, np.argwhere(keep))))

    if output_folder is None:
        return mask
    with open(os.path.join(output_folder, "mask_statistics.json"), "w") as f:
        json.dump(stats, f, indent=2)
    return None


def distance_watershed_implementation(
    input_path: str,
    output_folder: Optional[str] = None,
    min_size: int = 1000,
    center_distance_threshold: Optional[float] = 0.4,
    boundary_distance_threshold: Optional[float] = None,
    fg_threshold: float = 0.5,
    distance_smoothing: float = 0.0,
    original_shape: Optional[Tuple[int, int, int]] = None,
    halo: Tuple[int, int, int] = (16, 32, 32),
) -> None:
    """Parallel implementation of the distance-prediction based watershed.

    The seeds and the segmentation are rewritten from scratch on every call, so calling this
    function twice for the same output folder replaces the previous segmentation. The seeds must
    not be inherited from an earlier run: `elf.parallel.label` skips blocks that hold no mask
    voxels without zeroing them, and it merges labels across block faces based on the content of
    the seed volume. Stale seeds therefore get fused into the current labels, which produces
    single label IDs that cover two places far apart in the volume.

    Note that `predictions.zarr` and `mask.zarr` in the output folder are reused instead. The
    prediction accumulates over the slurm array tasks that each write a subset of the blocks, and
    the mask is reused through explicit checks in the callers. Predicting with a different model
    into an existing output folder therefore keeps stale predictions in the blocks that the mask
    excludes.

    Args:
        input_path: The path to the zarr file with the network predictions.
        output_folder: The folder for storing the segmentation and intermediate results.
        min_size: The minimal size of objects in the segmentation.
        center_distance_threshold: The threshold applied to the distance center predictions to derive seeds.
        boundary_distance_threshold: The threshold applied to the boundary predictions to derive seeds.
            By default this is set to 'None', in which case the boundary distances are not used for the seeds.
        fg_threshold: The threshold applied to the foreground prediction for deriving the watershed mask.
        distance_smoothing: The sigma value for smoothing the distance predictions with a gaussian kernel.
            This may help to reduce border artifacts. If set to 0 (the default) smoothing is not applied.
        original_shape: The original shape to resize the segmentation to.
        halo: The halo of the block-wise seeded watershed. A block must see the seed of a cell that
            crosses its face; the former (2, 8, 8) split large cells, such as IHCs, at block faces.
    """
    if isinstance(input_path, str):
        input_ = open_file(input_path, "r")["prediction"]
    else:
        input_ = input_path

    print(f"Using center distance threshold: {center_distance_threshold}")
    print(f"Using boundary distance threshold: {boundary_distance_threshold}")
    print(f"Using distance smoothing: {distance_smoothing}")

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())

    # Get the foreground mask.
    mask = ThresholdWrapper(SelectChannel(input_, 0), threshold=fg_threshold)

    # Get the the center and boundary distances.
    center_distances = SelectChannel(input_, 1)
    boundary_distances = SelectChannel(input_, 2)

    # Apply (lazy) smoothing to both channels if distance smoothing was set.
    if distance_smoothing > 0:
        smooth = partial(gaussian, sigma=distance_smoothing)
        # We assume that the gaussian is truncated at 5.3 sigma (tolerance of 1e-6)
        smoothing_halo = 3 * (int(np.ceil(5.3 * distance_smoothing)),)
        center_distances = SimpleTransformationWrapperWithHalo(
            center_distances, transformation=smooth, halo=smoothing_halo
        )
        boundary_distances = SimpleTransformationWrapperWithHalo(
            boundary_distances, transformation=smooth, halo=smoothing_halo
        )

    # Allocate the (zarr) array for the seeds.
    if output_folder is None:
        block_shape = (20, 128, 128)
        seeds = np.zeros(center_distances.shape, dtype=np.uint64)
    else:
        block_shape = center_distances.chunks
        seed_path = os.path.join(output_folder, "seeds.zarr")
        seed_file = open_file(seed_path, "a")
        seeds = seed_file.create_dataset(
            "seeds", shape=center_distances.shape, chunks=block_shape, compression="gzip", dtype="uint64",
            overwrite=True,
        )

    # Compute the seed inputs:
    if boundary_distance_threshold is None and center_distance_threshold is None:
        raise ValueError("Either boundary_distance_threshold, center_distance_threshold, or both have to be specified.")
    elif boundary_distance_threshold is None:
        seed_inputs = ThresholdWrapper(center_distances, threshold=center_distance_threshold, operator=np.less)
    elif center_distance_threshold is None:
        seed_inputs = ThresholdWrapper(boundary_distances, threshold=boundary_distance_threshold, operator=np.less)
    else:
        seed_inputs1 = ThresholdWrapper(center_distances, threshold=center_distance_threshold, operator=np.less)
        seed_inputs2 = ThresholdWrapper(boundary_distances, threshold=boundary_distance_threshold, operator=np.less)
        seed_inputs = MultiTransformationWrapper(np.logical_and, seed_inputs1, seed_inputs2)

    # Compute the seeds via connected components on the seed inputs.
    parallel.label(
        data=seed_inputs, out=seeds, block_shape=block_shape, mask=mask, verbose=True, n_threads=n_threads
    )

    # Allocate the (zarr) array for the segmentation.
    if output_folder is None:
        seg = np.zeros(seeds.shape, dtype=np.uint64)
    else:
        seg_path = os.path.join(output_folder, "segmentation.zarr" if original_shape is None else "seg_downscaled.zarr")
        seg_file = open_file(seg_path, "a")
        seg = seg_file.create_dataset(
            "segmentation", shape=seeds.shape, chunks=block_shape, compression="gzip", dtype="uint64",
            overwrite=True,
        )

    # Compute the segmentation with a seeded watershed
    parallel.seeded_watershed(
        boundary_distances, seeds, out=seg, block_shape=block_shape, halo=halo, mask=mask, verbose=True,
        n_threads=n_threads,
    )

    # Apply size filter.
    if min_size > 0:
        parallel.size_filter(
            seg, seg, min_size=min_size, block_shape=block_shape, mask=mask,
            verbose=True, n_threads=n_threads, relabel=True,
        )

    # Reshape to original shape if given.
    if original_shape is not None:
        out_path = os.path.join(output_folder, "segmentation.zarr")

        output_seg = ResizedVolume(seg, shape=original_shape, order=0)
        with open_file(out_path, "a") as f:
            out_seg_volume = f.create_dataset(
                "segmentation", shape=original_shape, compression="gzip", dtype="uint64", chunks=block_shape,
                overwrite=True,
            )
            blocking = Blocking([0] * len(original_shape), output_seg.shape, block_shape)

            def write_block(block_id):
                block = blocking.get_block(block_id)
                bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                out_seg_volume[bb] = output_seg[bb]

            with futures.ThreadPoolExecutor(n_threads) as tp:
                tp.map(write_block, range(blocking.number_of_blocks))

    if output_folder is None:
        return seg
    else:
        return None


def calc_mean_and_std(input_path: str, input_key: str, output_folder: str) -> None:
    """Calculate mean and standard deviation of the input volume.

    The parameters are saved in 'mean_std.json' in the output folder. This is a legacy option:
    the prediction no longer needs them, because the U-Nets standardize every block in their
    first layer. The prediction stages use an existing 'mean_std.json', which reproduces an
    older prediction bit for bit. Call this after the mask is created, because the statistics
    are computed inside 'mask.zarr'.

    Args:
        input_path: The file path to the image data.
        input_key: The key / internal path of the image data.
        output_folder: The output folder for storing the segmentation related data.
    """
    json_file = os.path.join(output_folder, "mean_std.json")
    mask_path = os.path.join(output_folder, "mask.zarr")

    input_ = read_image_data(input_path, input_key)
    chunks = getattr(input_, "chunks", (64, 64, 64))

    # The mask is optional: prediction without masking has no mask.zarr in the output folder.
    if os.path.exists(mask_path):
        image_mask = z5py.File(mask_path, "r")["mask"]
        if image_mask.shape != input_.shape:
            image_mask = ResizedVolume(image_mask, input_.shape, order=0)
    else:
        image_mask = None

    # Compute the global mean and standard deviation.
    n_threads = min(16, mp.cpu_count())
    mean, std = parallel.mean_and_std(
        input_, block_shape=tuple([2 * i for i in chunks]), n_threads=n_threads, verbose=True, mask=image_mask
    )
    ddict = {"mean": float(mean), "std": float(std)}
    with open(json_file, "w") as f:
        json.dump(ddict, f)


def _load_mean_std(output_folder: str) -> Tuple[Optional[float], Optional[float]]:
    """Read the legacy normalization values, see `calc_mean_and_std`.

    Without 'mean_std.json' the prediction uses the raw values.
    """
    mean_std_file = os.path.join(output_folder, "mean_std.json")
    if not os.path.isfile(mean_std_file):
        return None, None
    with open(mean_std_file) as f:
        values = json.load(f)
    return float(values["mean"]), float(values["std"])


def run_unet_prediction(
    input_path: str,
    input_key: Optional[str],
    output_folder: Optional[str],
    model_path: str,
    min_size: int,
    scale: Optional[float] = None,
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    use_mask: bool = True,
    center_distance_threshold: Optional[float] = 0.4,
    boundary_distance_threshold: Optional[float] = None,
    fg_threshold: float = 0.5,
    distance_smoothing: float = 0.0,
    seg_class: Optional[str] = None,
    relative_threshold: float = 0.6,
    batch_size: int = 1,
    num_prefetch_workers: Optional[int] = None,
    legacy_mask: bool = False,
    absolute_threshold: Optional[float] = None,
    background_factor: Optional[float] = None,
) -> None:
    """Run prediction and segmentation with a distance U-Net.

    Args:
        input_path: The path to the input data.
        input_key: The key / internal path of the image data.
        output_folder: The output folder for storing the segmentation related data.
        model_path: The path to the model to use for segmentation.
        min_size: The minimal size of segmented objects in the output.
        scale: A factor to rescale the data before prediction.
            By default the data will not be rescaled.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        use_mask: Whether to use the masking heuristics to not run inference on empty blocks.
        center_distance_threshold: The threshold applied to the distance center predictions to derive seeds.
        boundary_distance_threshold: The threshold applied to the boundary predictions to derive seeds.
            By default this is set to 'None', in which case the boundary distances are not used for the seeds.
        fg_threshold: The threshold applied to the foreground prediction for deriving the watershed mask.
        distance_smoothing: The sigma value for smoothing the distance predictions with a gaussian kernel.
            This may help to reduce border artifacts. If set to 0 (the default) smoothing is not applied.
        seg_class: The segmentation class of the mask, see `find_mask`. None uses the legacy rule.
        relative_threshold: Passed to find_mask. The adaptive threshold factor of the legacy rule.
        batch_size: The number of blocks stacked into a single forward pass.
        num_prefetch_workers: The number of threads that load blocks while the GPU predicts.
            By default it is derived from the number of cores available to the job.
        legacy_mask: Passed to find_mask. Use the legacy mask rule for 'sgn' and 'ihc'.
        absolute_threshold: Passed to find_mask. The threshold of the legacy mask rule.
            By default, it is derived from the data.
        background_factor: Passed to find_mask. The threshold of the default mask rule as a multiple
            of the background level. By default, the value of MASK_RULES[seg_class].
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    if use_mask:
        mask = find_mask(
            input_path, input_key, output_folder=output_folder,
            seg_class=seg_class, relative_threshold=relative_threshold, absolute_threshold=absolute_threshold,
            legacy_mask=legacy_mask, background_factor=background_factor,
        )
    else:
        mask = None

    original_shape, prediction = prediction_impl(
        input_path=input_path, input_key=input_key, output_folder=output_folder, model_path=model_path, scale=scale,
        block_shape=block_shape, halo=halo, mask=mask,
        batch_size=batch_size, num_prefetch_workers=num_prefetch_workers,
    )

    if output_folder is None:
        pmap_out = prediction
    else:
        pmap_out = os.path.join(output_folder, "predictions.zarr")

    segmentation = distance_watershed_implementation(
        pmap_out, output_folder, min_size=min_size, original_shape=original_shape,
        center_distance_threshold=center_distance_threshold,
        boundary_distance_threshold=boundary_distance_threshold,
        fg_threshold=fg_threshold,
        distance_smoothing=distance_smoothing,
    )

    return segmentation


#
# ---Workflow for parallel prediction using slurm---
#


def run_unet_prediction_preprocess_slurm(
    input_path: str,
    output_folder: str,
    input_key: Optional[str] = None,
    s3: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    s3_credentials: Optional[str] = None,
    seg_class: Optional[str] = None,
    absolute_threshold: Optional[float] = None,
    legacy_mask: Union[bool, str] = False,
    background_factor: Optional[Union[float, str]] = None,
) -> None:
    """Pre-processing for the parallel prediction with U-Net models.
    Masks are stored in mask.zarr in the output folder.

    Args:
        input_path: The path to the input data.
        output_folder: The output folder for storing the segmentation related data.
        input_key: The key / internal path of the image data.
        s3: Flag for accessing data stored on S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
        seg_class: The segmentation class of the mask, see `find_mask`.
        absolute_threshold: The threshold of the legacy mask rule. 'None' derives it from the data.
        legacy_mask: Use the legacy mask rule. The slurm templates pass strings, so the strings
            "", "0", "false" and "no" are off.
        background_factor: The threshold of the default mask rule as a multiple of the background
            level. 'None' uses the class default.
    """
    input_path = s3_utils.resolve_path(
        input_path, s3, bucket_name=s3_bucket_name,
        service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
    )

    if isinstance(absolute_threshold, str):
        try:
            absolute_threshold = float(absolute_threshold)
        except ValueError:
            absolute_threshold = None
        print(f"Using absolute threshold {absolute_threshold}")
    if isinstance(legacy_mask, str):
        legacy_mask = legacy_mask.strip().lower() not in ("", "0", "false", "no")
    if isinstance(background_factor, str):
        background_factor = None if background_factor.strip() in ("", "None") else float(background_factor)

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(
            input_path, input_key, output_folder, seg_class=seg_class, absolute_threshold=absolute_threshold,
            legacy_mask=legacy_mask, background_factor=background_factor,
        )


def run_unet_prediction_slurm(
    input_path: str,
    output_folder: str,
    model_path: str,
    input_key: Optional[str] = None,
    scale: Optional[float] = None,
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    prediction_instances: Optional[int] = 1,
    s3: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    s3_credentials: Optional[str] = None,
    batch_size: int = 1,
    num_prefetch_workers: Optional[int] = None,
    seg_class: Optional[str] = None,
) -> None:
    """Run prediction of distance U-Net for data stored locally or on an S3 bucket.

    Args:
        input_path: The path to the input data.
        output_folder: The output folder for storing the segmentation related data.
        model_path: The path to the model to use for segmentation.
        input_key: The key / internal path of the image data.
        scale: A factor to rescale the data before prediction.
            By default the data will not be rescaled.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        prediction_instances: Number of instances for parallel prediction.
        s3: Flag for accessing data stored on S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
        batch_size: The number of blocks stacked into a single forward pass.
        num_prefetch_workers: The number of threads that load blocks while the GPU predicts.
            By default it is derived from the number of cores available to the job.
        seg_class: The segmentation class of the mask, if the output folder has no 'mask.zarr' yet.
            See `find_mask`.
    """
    os.makedirs(output_folder, exist_ok=True)
    prediction_instances = int(prediction_instances)
    if isinstance(scale, str):
        scale = float(scale)
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    input_path = s3_utils.resolve_path(
        input_path, s3, bucket_name=s3_bucket_name,
        service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
    )

    if slurm_task_id is not None:
        slurm_task_id = int(slurm_task_id)
    else:
        raise ValueError("The SLURM_ARRAY_TASK_ID is not set. Ensure that you are using the '-a' option with SBATCH.")

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(input_path, input_key, output_folder, seg_class=seg_class)

    mean, std = _load_mean_std(output_folder)

    prediction_impl(
        input_path, input_key, output_folder, model_path, scale, block_shape, halo,
        prediction_instances=prediction_instances, slurm_task_id=slurm_task_id,
        mean=mean, std=std,
        batch_size=batch_size, num_prefetch_workers=num_prefetch_workers,
    )


# does NOT need GPU, FIXME: only run on CPU
def run_unet_segmentation_slurm(
    output_folder: str,
    min_size: int,
    center_distance_threshold: Optional[float] = 0.4,
    boundary_distance_threshold: Optional[float] = 0.5,
    fg_threshold: float = 0.5,
    distance_smoothing: float = 0.0,
    original_shape: Optional[Tuple[int, int, int]] = None,
    watershed_params: Optional[str] = None,
) -> None:
    """Create segmentation from prediction.

    Args:
        output_folder: The output folder for storing the segmentation related data.
        min_size: The minimal size of segmented objects in the output.
        center_distance_threshold: The threshold applied to the distance center predictions to derive seeds.
        boundary_distance_threshold: The threshold applied to the boundary predictions to derive seeds.
            'None' does not use the boundary distances for the seeds.
        fg_threshold: The threshold applied to the foreground prediction for deriving the watershed mask.
        distance_smoothing: The sigma value for smoothing the distance predictions with a gaussian kernel.
            This may help to reduce border artifacts. If set to 0 (the default) smoothing is not applied.
        original_shape: The original shape of the output, in case the prediction was resized.
    """
    if watershed_params is not None and os.path.exists(watershed_params):
        with open(watershed_params) as fh:
            data = json.load(fh)
        print(f"Loaded cached best params from {watershed_params}")
        center_distance_threshold = data["params"]["center_distance_threshold"]
        boundary_distance_threshold = data["params"]["boundary_distance_threshold"]
        distance_smoothing = data["params"]["distance_smoothing"]

    min_size = int(min_size)
    center_distance_threshold = None if center_distance_threshold is None else float(center_distance_threshold)
    boundary_distance_threshold = None if boundary_distance_threshold is None else float(boundary_distance_threshold)
    distance_smoothing = float(distance_smoothing)
    pmap_out = os.path.join(output_folder, "predictions.zarr")
    distance_watershed_implementation(pmap_out, output_folder, center_distance_threshold=center_distance_threshold,
                                      boundary_distance_threshold=boundary_distance_threshold,
                                      fg_threshold=fg_threshold,
                                      distance_smoothing=distance_smoothing,
                                      min_size=min_size,
                                      original_shape=original_shape)
