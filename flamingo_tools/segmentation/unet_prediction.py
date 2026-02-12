"""Prediction using distance U-Net.
Parallelization using multiple GPUs is currently only possible by calling functions directly.
Functions for the parallelization end with '_slurm'
and divide the process into preprocessing, prediction, and segmentation.
"""
import json
import multiprocessing as mp
import os
import warnings
from concurrent import futures
from functools import partial
from typing import Optional, Tuple

import elf.parallel as parallel
import numpy as np
import nifty.tools as nt
import vigra
import torch
import z5py

from elf.wrapper import ThresholdWrapper, SimpleTransformationWrapper, SimpleTransformationWrapperWithHalo
from elf.wrapper.base import MultiTransformationWrapper
from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from skimage.filters import gaussian
from torch_em.util import load_model
from torch_em.util.prediction import predict_with_halo
from tqdm import tqdm

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.file_utils import read_image_data


class SelectChannel(SimpleTransformationWrapper):
    """Wrapper to select a chanel from an array-like dataset object.

    Args:
        volume: The array-like input dataset.
        channel: The channel that will be selected.
    """
    def __init__(self, volume: np.typing.ArrayLike, channel: int):
        self.channel = channel
        super().__init__(volume, lambda x: x[self.channel], with_channels=True)

    @property
    def shape(self):
        return self._volume.shape[1:]

    @property
    def chunks(self):
        return self._volume.chunks[1:]

    @property
    def ndim(self):
        return self._volume.ndim - 1


def _get_device_and_tiling(block_shape, halo, input_):
    have_cuda = torch.cuda.is_available()
    if block_shape is None:
        block_shape = (128, 128, 128) if have_cuda else getattr(input_, "chunks", (64, 64, 64))
    if halo is None:
        halo = (16, 32, 32)
    if have_cuda:
        print("Predict with GPU")
        gpu_ids = [0]
    else:
        print("Predict with CPU")
        gpu_ids = ["cpu"]
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
    mask=None
):
    """@private
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if os.path.isdir(model_path):
            model = load_model(model_path)
        else:
            model = torch.load(model_path, weights_only=False)

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
            x[1] = vigra.filters.gaussianSmoothing(x[1], sigma=2.0)
            return x
    else:
        postprocess = None if output_channels > 1 else lambda x: x.squeeze()

    gpu_ids, block_shape, halo = _get_device_and_tiling(block_shape, halo, input_)
    shape = input_.shape
    ndim = len(shape)
    if output_channels > 1:
        output_shape = (output_channels,) + input_.shape
        output_chunks = (1,) + block_shape
    else:
        output_shape = input_.shape
        output_chunks = block_shape

    blocking = nt.blocking([0] * ndim, shape, block_shape)
    n_blocks = blocking.numberOfBlocks
    if prediction_instances != 1:
        # shuffle indexes with fixed seed to balance out segmentation blocks for slurm workers
        rng = np.random.default_rng(seed=1234)
        iteration_ids = [x.tolist() for x in np.array_split(list(rng.permutation(n_blocks)), prediction_instances)]
        slurm_iteration = iteration_ids[slurm_task_id]
    else:
        slurm_iteration = list(range(n_blocks))

    if output_folder is None:
        output = np.zeros(output_shape, dtype=np.float32)
        predict_with_halo(
            input_, model,
            gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
            output=output, preprocess=preprocess, postprocess=postprocess,
            mask=image_mask,
            iter_list=slurm_iteration,
        )

    else:
        output_path = os.path.join(output_folder, "predictions.zarr")
        with open_file(output_path, "a") as f:
            output = f.require_dataset(
                "prediction",
                shape=output_shape,
                chunks=output_chunks,
                compression="gzip",
                dtype="float32",
            )

            predict_with_halo(
                input_, model,
                gpu_ids=gpu_ids, block_shape=block_shape, halo=halo,
                output=output, preprocess=preprocess, postprocess=postprocess,
                mask=image_mask,
                iter_list=slurm_iteration,
            )

    if output_folder is None:
        return original_shape, output
    else:
        return original_shape, None


def find_mask(
    input_path: str,
    input_key: Optional[str],
    output_folder: Optional[str],
    seg_class: Optional[str] = "sgn"
) -> None:
    """Determine the mask for running prediction.

    The mask corresponds to data that contains actual signal and not just noise.
    This is determined by checking if the 95th percentile of the intensity
    of a local block has a value larger than 200. It may be necesary to choose a
    different criterion if the data acquisition changes.

    Args:
        input_path: The file path to the image data.
        input_key: The key / internal path of the image data.
        output_folder: The output folder for storing the mask data.
        seg_class: Specifier for exclusion criterias for mask generation.
    """
    # set parameters for the exclusion of chunks within mask generation
    if seg_class == "sgn":
        upper_percentile = 95
        min_intensity = 200
        print(f"Calculating mask for segmentation class {seg_class}.")
    elif seg_class == "ihc":
        upper_percentile = 99
        min_intensity = 150
        print(f"Calculating mask for segmentation class {seg_class}.")
    elif seg_class == "ihc_low":
        upper_percentile = 99.5
        min_intensity = 100
        print(f"Calculating mask for segmentation class {seg_class}.")
    else:
        upper_percentile = 95
        min_intensity = 200
        print("Calculating mask with default values.")

    raw = read_image_data(input_path, input_key)
    chunks = getattr(raw, "chunks", (64, 64, 64))

    block_shape = tuple(2 * ch for ch in chunks)
    blocking = nt.blocking([0, 0, 0], raw.shape, block_shape)
    n_blocks = blocking.numberOfBlocks

    if output_folder is None:
        ds_mask = np.zeros(raw.shape, dtype=np.uint64)

    else:
        mask_path = os.path.join(output_folder, "mask.zarr")
        f = z5py.File(mask_path, "a")
        mask_key = "mask"
        if mask_key in f:
            return

        ds_mask = f.create_dataset(mask_key, shape=raw.shape, compression="gzip", dtype="uint8", chunks=block_shape)

    # TODO more sophisticated criterion?!
    def find_mask_block(block_id):
        block = blocking.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
        data = raw[bb]
        max_ = np.percentile(data, upper_percentile)
        if max_ > min_intensity:
            ds_mask[bb] = 1

    n_threads = min(16, mp.cpu_count())
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(find_mask_block, range(n_blocks)), total=n_blocks))

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
        seed_file = open_file(os.path.join(seed_path), "a")
        seeds = seed_file.require_dataset(
            "seeds", shape=center_distances.shape, chunks=block_shape, compression="gzip", dtype="uint64"
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
            "segmentation", shape=seeds.shape, chunks=block_shape, compression="gzip", dtype="uint64"
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
            )
            blocking = parallel.common.get_blocking(output_seg, block_shape, roi=None, n_threads=n_threads)

            def write_block(block_id):
                block = blocking.getBlock(block_id)
                bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
                out_seg_volume[bb] = output_seg[bb]

            with futures.ThreadPoolExecutor(n_threads) as tp:
                tp.map(write_block, range(blocking.numberOfBlocks))

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
    image_mask = z5py.File(mask_path, "r")["mask"]

    input_ = read_image_data(input_path, input_key)
    chunks = getattr(input_, "chunks", (64, 64, 64))

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
    """
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)

    if use_mask:
        mask = find_mask(input_path, input_key, output_folder=output_folder, seg_class=seg_class)
    else:
        mask = None

    original_shape, prediction = prediction_impl(
        input_path=input_path, input_key=input_key, output_folder=output_folder, model_path=model_path, scale=scale,
        block_shape=block_shape, halo=halo, mask=mask
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

    if not os.path.isdir(os.path.join(output_folder, "mask.zarr")):
        find_mask(input_path, input_key, output_folder, seg_class=seg_class)

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
