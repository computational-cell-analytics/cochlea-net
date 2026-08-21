"""Prediction using distance U-Net.
Parallelization using multiple GPUs is currently only possible by calling functions directly.
Functions for the parallelization end with '_slurm'
and divide the process into preprocessing, prediction, and segmentation.
"""
import importlib
import json
import multiprocessing as mp
import os
import warnings
from concurrent import futures
from functools import partial
from typing import Dict, List, Optional, Tuple

import elf.parallel as parallel
import imageio.v3 as imageio
import numpy as np
import torch
import z5py

from bioimage_cpp.filters import gaussian_smoothing
from bioimage_cpp.utils import Blocking

from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper, SimpleTransformationWrapperWithHalo
from elf.wrapper.base import MultiTransformationWrapper
from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from elf.util import normalize_index, squeeze_singletons
from skimage.filters import gaussian
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo_pipelined
from tqdm import tqdm

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.file_utils import read_image_data


def _model_from_checkpoint(ckpt):
    """Reconstruct a model from a torch_em-style trainer checkpoint dict."""
    model_class_path = ckpt["init"]["model_class"]
    model_kwargs = ckpt["init"]["model_kwargs"]
    module_path, class_name = model_class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class(**model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    return model


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
        halo = (16, 32, 32)
    # Clip to the volume. A block larger than the input makes the halo dominate the forward
    # pass: a (64, 128, 128) volume with a (64, 256, 256) block is padded to 96 x 384 x 384.
    # The clip only depends on the input shape, so all slurm array tasks get the same grid.
    block_shape = tuple(min(bs, sh) for bs, sh in zip(block_shape, input_.shape))
    halo = tuple(min(ha, bs // 2) for ha, bs in zip(halo, block_shape))
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.path.isdir(model_path):
            model = load_model(model_path)
        else:
            obj = torch.load(model_path, weights_only=False)
            model = _model_from_checkpoint(obj) if isinstance(obj, dict) and "model_state" in obj else obj

    input_ = read_image_data(input_path, input_key)
    chunks = getattr(input_, "chunks", (64, 64, 64))

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

    if mean is None or std is None:
        # Compute the global mean and standard deviation.
        n_threads = min(16, mp.cpu_count())
        mean, std = parallel.mean_and_std(
            input_, block_shape=tuple([2 * i for i in chunks]), n_threads=n_threads, verbose=True,
            mask=image_mask
        )
    # Coerce to Python floats: a numpy scalar would make the normalization below run in
    # float64 and round differently than the values read back from 'mean_std.json'. The
    # single-job and the slurm-array workflow must produce the same prediction.
    mean, std = float(mean), float(std)
    print("Mean and standard deviation computed for the full volume:")
    print(mean, std)

    # Preprocess with fixed mean and standard deviation.
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


def sweep_mask_thresholds(
    input_path: str,
    output_path: str,
    input_key: Optional[str] = None,
    min_intensities: List[float] = [100, 150, 200, 300, 400, 600, 1000],
    seg_class: Optional[str] = "sgn",
    threshold_map_path: Optional[str] = None,
    percentile_map_path: Optional[str] = None,
) -> Dict[float, float]:
    """Compute the fraction of blocks that would be included in the mask for each
    candidate min_intensity value, using a single parallel read pass over the data.

    Args:
        input_path: The file path to the image data.
        input_key: The key / internal path of the image data.
        min_intensities: Intensity thresholds to evaluate.
        seg_class: Determines the upper percentile used per block (same as find_mask).
        threshold_map_path: Optional output path for a downscaled TIF with one voxel
            per masking block, where each voxel holds the largest value in
            min_intensities that the block's percentile still exceeds (0 if none).
        percentile_map_path: Optional output path for a downscaled TIF with one voxel
            per masking block, where each voxel holds the block's raw percentile value.

    Returns:
        Dict mapping each min_intensity to the fraction of blocks that would be masked.
    """
    if seg_class == "ihc":
        upper_percentile = 99
    else:
        upper_percentile = 95

    raw = read_image_data(input_path, input_key)
    chunks = getattr(raw, "chunks", (64, 64, 64))
    block_shape = tuple(2 * ch for ch in chunks)
    blocking = Blocking([0, 0, 0], raw.shape, block_shape)
    n_blocks = blocking.number_of_blocks

    def percentile_for_block(block_id):
        block = blocking.get_block(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        return float(np.percentile(raw[bb], upper_percentile))

    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        block_percentiles = list(tqdm(tp.map(percentile_for_block, range(n_blocks)), total=n_blocks,
                                      desc="Computing block percentiles"))

    block_percentiles = np.array(block_percentiles)

    if threshold_map_path is not None or percentile_map_path is not None:
        grid_shape = tuple(blocking.blocks_per_axis)

        if percentile_map_path is not None:
            out_dir = os.path.dirname(percentile_map_path)
            os.makedirs(out_dir, exist_ok=True)
            percentile_map = block_percentiles.reshape(grid_shape).astype("float32")
            imageio.imwrite(percentile_map_path, percentile_map, compression="zlib")

        if threshold_map_path is not None:
            out_dir = os.path.dirname(threshold_map_path)
            os.makedirs(out_dir, exist_ok=True)
            sorted_thresholds = np.array(sorted(min_intensities), dtype=np.float32)
            exceed_counts = (block_percentiles[:, None] > sorted_thresholds[None, :]).sum(axis=1)
            threshold_values = np.concatenate([[0.0], sorted_thresholds])[exceed_counts]
            threshold_map = threshold_values.reshape(grid_shape).astype("float32")
            imageio.imwrite(threshold_map_path, threshold_map, compression="zlib")

    print(f"\n{'min_intensity':>15}  {'masked blocks':>15}  {'fraction (%)':>12}")
    print("-" * 46)
    result = {}
    for threshold in sorted(min_intensities):
        n_masked = int(np.sum(block_percentiles > threshold))
        fraction = n_masked / n_blocks
        print(f"{threshold:>15.1f}  {n_masked:>9}/{n_blocks:<5}  {fraction * 100:>11.2f}")
        result[threshold] = fraction

    out_dir = os.path.dirname(output_path)
    os.makedirs(out_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent='\t', separators=(',', ': '))

    return result


def find_mask(
    input_path: str,
    input_key: Optional[str],
    output_folder: Optional[str],
    seg_class: Optional[str] = "sgn",
    relative_threshold: float = 0.7,
    absolute_threshold: Optional[float] = None,
) -> None:
    """Determine the mask for running prediction.

    The mask marks blocks that contain actual signal (not just noise/background).
    A block is included when its upper-percentile intensity exceeds `min_intensity`.

    `min_intensity` is computed adaptively: it equals `relative_threshold` times a
    robust estimate of the global signal level (median of per-block p99.9 values drawn
    from a random sample of blocks), then capped at the per-class absolute maximum so
    that spike-inflated estimates (fluorescence residues) never push the threshold above
    the proven fixed default and dim stainings get a proportionally lower threshold.

    Args:
        input_path: The file path to the image data.
        input_key: The key / internal path of the image data.
        output_folder: The output folder for storing the mask data.
        seg_class: Specifier for exclusion criterias for mask generation.
        relative_threshold: Fraction of the global signal level used as the lower
            inclusion threshold. Capped per class so the threshold never exceeds the
            class default (200 for sgn, 150 for ihc).
    """
    if seg_class == "sgn":
        upper_percentile = 95
        absolute_max = 200
        absolute_min = 100
        print(f"Calculating mask for segmentation class {seg_class}.")
    elif seg_class == "ihc":
        upper_percentile = 99
        absolute_max = 400
        absolute_min = 250
        print(f"Calculating mask for segmentation class {seg_class}.")
    else:
        upper_percentile = 95
        absolute_max = 200
        absolute_min = 100
        print("Calculating mask with default values.")

    raw = read_image_data(input_path, input_key)
    chunks = getattr(raw, "chunks", (64, 64, 64))

    block_shape = tuple(2 * ch for ch in chunks)
    blocking = Blocking([0, 0, 0], raw.shape, block_shape)
    n_blocks = blocking.number_of_blocks

    if output_folder is None:
        ds_mask = np.zeros(raw.shape, dtype=np.uint64)

    else:
        mask_path = os.path.join(output_folder, "mask.zarr")
        f = z5py.File(mask_path, "a")
        mask_key = "mask"
        if mask_key in f:
            return

        ds_mask = f.create_dataset(mask_key, shape=raw.shape, compression="gzip", dtype="uint8", chunks=block_shape)

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
        # Cap at absolute_max: for spike-inflated or normally bright images the
        # relative term exceeds the cap and we fall back to the fixed class default;
        # for dim stainings the relative term is below the cap and adapts downward.
        min_intensity = max(absolute_min, min(relative_threshold * global_high, absolute_max))
        print(f"Adaptive min_intensity: {min_intensity:.1f} (global_high={global_high:.1f}, cap={absolute_max})")
    else:
        min_intensity = float(absolute_threshold)
        print(f"Using absolute min_intensity: {min_intensity:.1f}")

    def find_mask_block(block_id):
        block = blocking.get_block(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        threshold = np.percentile(raw[bb], upper_percentile)
        if threshold > min_intensity:
            ds_mask[bb] = 1
            return True
        return False

    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        results = list(tqdm(tp.map(find_mask_block, range(n_blocks)), total=n_blocks))

    seg_mask_blocks = sum(results)
    relative_blocks = round(seg_mask_blocks / n_blocks * 100, 2)
    print(f"{seg_mask_blocks}/{n_blocks} ({relative_blocks} %) used for segmentation.")

    if output_folder is None:
        return ds_mask
    else:
        return None


def distance_watershed_implementation(
    input_path: str,
    output_folder: Optional[str] = None,
    min_size: int = 1000,
    center_distance_threshold: Optional[float] = 0.4,
    boundary_distance_threshold: Optional[float] = None,
    fg_threshold: float = 0.5,
    distance_smoothing: float = 0.0,
    original_shape: Optional[Tuple[int, int, int]] = None
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
        halo = int(np.ceil(5.3 * distance_smoothing))
        halo = 3 * (halo,)
        center_distances = SimpleTransformationWrapperWithHalo(center_distances, transformation=smooth, halo=halo)
        boundary_distances = SimpleTransformationWrapperWithHalo(boundary_distances, transformation=smooth, halo=halo)

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
        raise ValueError("Either boundary_distance_threshold, center_distance_threshold, or both have to be specifie.")
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
    halo = (2, 8, 8)
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

    The parameters are saved in 'mean_std.json' in the output folder.

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
    center_distance_threshold: float = 0.4,
    boundary_distance_threshold: Optional[float] = None,
    fg_threshold: float = 0.5,
    distance_smoothing: float = 0.0,
    seg_class: Optional[str] = None,
    relative_threshold: float = 0.6,
    batch_size: int = 1,
    num_prefetch_workers: Optional[int] = None,
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
        seg_class: Specifier for exclusion criterias for mask generation.
        relative_threshold: Passed to find_mask. Fraction of the estimated global signal level
            used as the block inclusion threshold (capped at the per-class absolute maximum).
        batch_size: The number of blocks stacked into a single forward pass.
        num_prefetch_workers: The number of threads that load blocks while the GPU predicts.
            By default it is derived from the number of cores available to the job.
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    if use_mask:
        mask = find_mask(
            input_path, input_key, output_folder=output_folder,
            seg_class=seg_class, relative_threshold=relative_threshold,
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
) -> None:
    """Pre-processing for the parallel prediction with U-Net models.
    Masks are stored in mask.zarr in the output folder.
    The mean and standard deviation are precomputed for later usage during prediction
    and stored in a JSON file within the output folder as mean_std.json.

    Args:
        input_path: The path to the input data.
        output_folder: The output folder for storing the segmentation related data.
        input_key: The key / internal path of the image data.
        s3: Flag for accessing data stored on S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
        seg_class: Specifier for exclusion criterias for mask generation.
    """
    if s3 is not None:
        input_path, fs = s3_utils.get_s3_path(
            input_path, bucket_name=s3_bucket_name,
            service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
        )

    if isinstance(absolute_threshold, str):
        try:
            absolute_threshold = float(absolute_threshold)
        except ValueError:
            absolute_threshold = None
        print(f"Using absolute threshold {absolute_threshold}")

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(input_path, input_key, output_folder, seg_class=seg_class, absolute_threshold=absolute_threshold)

    if not os.path.isfile(os.path.join(output_folder, "mean_std.json")):
        calc_mean_and_std(input_path, input_key, output_folder)


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
    """
    os.makedirs(output_folder, exist_ok=True)
    prediction_instances = int(prediction_instances)
    if isinstance(scale, str):
        scale = float(scale)
    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if s3 is not None:
        input_path, fs = s3_utils.get_s3_path(
            input_path, bucket_name=s3_bucket_name,
            service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
        )

    if slurm_task_id is not None:
        slurm_task_id = int(slurm_task_id)
    else:
        raise ValueError("The SLURM_ARRAY_TASK_ID is not set. Ensure that you are using the '-a' option with SBATCH.")

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(input_path, input_key, output_folder)

    # get pre-computed mean and standard deviation of full volume from JSON file
    if os.path.isfile(os.path.join(output_folder, "mean_std.json")):
        with open(os.path.join(output_folder, "mean_std.json")) as f:
            d = json.load(f)
            mean = float(d["mean"])
            std = float(d["std"])
    else:
        mean = None
        std = None

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
    center_distance_threshold: float = 0.4,
    boundary_distance_threshold: float = 0.5,
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
            By default this is set to 'None', in which case the boundary distances are not used for the seeds.
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
    boundary_distance_threshold = float(boundary_distance_threshold)
    distance_smoothing = float(distance_smoothing)
    pmap_out = os.path.join(output_folder, "predictions.zarr")
    distance_watershed_implementation(pmap_out, output_folder, center_distance_threshold=center_distance_threshold,
                                      boundary_distance_threshold=boundary_distance_threshold,
                                      fg_threshold=fg_threshold,
                                      distance_smoothing=distance_smoothing,
                                      min_size=min_size,
                                      original_shape=original_shape)
