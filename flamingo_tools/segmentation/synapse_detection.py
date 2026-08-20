"""Detection of ribbon synapses from a CtBP2 stain.

Parallelization over multiple slurm tasks is only possible by calling functions directly.
Functions for the parallelization end with '_slurm' and divide the process into
preprocessing, prediction, and detection.
"""
import json
import os
import warnings
from concurrent import futures
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import zarr
from scipy.ndimage import binary_dilation

from elf.parallel.local_maxima import find_local_maxima
from elf.parallel.distance_transform import map_points_to_objects
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.segmentation.unet_prediction import (
    _available_cpus, calc_mean_and_std, prediction_impl, SelectChannel,
)
import flamingo_tools.s3_utils as s3_utils

# Must match the sigma used in CsvHeatmapFlowTransform during training.
_HEATMAP_FLOW_SIGMA = 1

# Peak detection reads whole chunks. A block that equals one chunk makes the halo
# dominate the read volume, so grow the block until it reaches this voxel budget.
_DETECTION_BLOCK_VOXELS = 32_000_000

# The prediction block grid determines how blocks are split across slurm array tasks.
# Both the single-job and the parallel entry point must use the same values.
_PREDICTION_BLOCK_SHAPE = (64, 256, 256)
_PREDICTION_HALO = (16, 64, 64)


def _normalize_voxel_size(voxel_size):
    """Return the voxel size as an (x, y, z) tuple of floats.

    Accepts a scalar, a sequence of one or three values, or a string, so that the slurm
    entry points can take their arguments straight from the environment.
    """
    if isinstance(voxel_size, str):
        voxel_size = [float(v) for v in voxel_size.replace(",", " ").split()]
    elif isinstance(voxel_size, (int, float)):
        voxel_size = [float(voxel_size)]
    else:
        voxel_size = [float(v) for v in voxel_size]

    if len(voxel_size) == 1:
        voxel_size = voxel_size * 3
    if len(voxel_size) != 3:
        raise ValueError(f"Expect a voxel size with one or three values, got {voxel_size}.")
    return tuple(voxel_size)


def _get_model_out_channels(model_path):
    """Return the number of output channels of a model file or trainer checkpoint."""
    try:
        import sys
        import flamingo_tools.synapse_detection.detection_dataset as _dd
        sys.modules.setdefault("detection_dataset", _dd)
    except ImportError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        obj = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model_state" in obj:
        return obj["init"]["model_kwargs"].get("out_channels", 1)
    return obj.state_dict()["out_conv.bias"].shape[0]


def _detection_block_shape(chunks):
    """Return the smallest multiple of *chunks* that fits into the block voxel budget."""
    block = list(chunks)
    while 2 * int(np.prod(block)) <= _DETECTION_BLOCK_VOXELS:
        block[int(np.argmin(block))] *= 2
    return tuple(block)


def _apply_flow_correction(pred, peak_coords, n_threads):
    """Shift peaks to sub-voxel positions with the stereographic flow channels.

    The flow values are read one chunk at a time. A per-peak read decompresses a
    full chunk for every single lookup.
    """
    chunk_shape = np.asarray(getattr(pred, "chunks", pred.shape)[-3:])
    chunk_ids = peak_coords // chunk_shape
    flat_ids = np.ravel_multi_index(chunk_ids.T, chunk_ids.max(axis=0) + 1)

    order = np.argsort(flat_ids, kind="stable")
    sorted_ids = flat_ids[order]
    group_starts = np.flatnonzero(np.r_[True, sorted_ids[1:] != sorted_ids[:-1]])
    groups = np.split(order, group_starts[1:])

    adjusted = np.empty((len(peak_coords), 3), dtype="float64")

    def correct_group(indices):
        points = peak_coords[indices]
        start = points.min(axis=0)
        bounding_box = (slice(1, 5),) + tuple(
            slice(int(beg), int(end)) for beg, end in zip(start, points.max(axis=0) + 1)
        )
        flow = np.asarray(pred[bounding_box])
        local = points - start
        values = flow[:, local[:, 0], local[:, 1], local[:, 2]].astype("float64")
        denominator = 1.0 + values[0] + 1e-8
        adjusted[indices] = points + _HEATMAP_FLOW_SIGMA * values[1:].T / denominator[:, None]

    with futures.ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(correct_group, groups))

    return adjusted


def _flow_corrected_detections(pred, min_distance, threshold_abs, block_shape, n_threads):
    """Detect peaks and refine their positions using stereographic flow channels.

    Args:
        pred: Zarr/array of shape (Z, Y, X) for single-channel models or
              (5, Z, Y, X) for heatmap+flow models.
        min_distance: Minimum distance between detected peaks in voxels.
        threshold_abs: Absolute heatmap threshold for peak detection.
        block_shape: Spatial block shape for parallel peak detection.
        n_threads: Number of threads.

    Returns:
        Tuple `(coords, raw_coords)` of (N, 3) float arrays of [z, y, x] coordinates.
        `coords` is sub-voxel if flow correction was applied, otherwise identical to
        `raw_coords`. `raw_coords` is always the un-corrected local-maxima positions.
    """
    have_flow = pred.ndim == 4 and pred.shape[0] >= 5
    # SelectChannel presents the 4-D (C, Z, Y, X) zarr as a 3-D (Z, Y, X) view
    # so find_local_maxima can work out-of-core without loading the full volume.
    heatmap = SelectChannel(pred, 0) if have_flow else pred

    peak_coords = find_local_maxima(
        heatmap, block_shape=block_shape, min_distance=min_distance,
        threshold_abs=threshold_abs, verbose=True, n_threads=n_threads,
    )
    raw_coords = peak_coords.astype(float)

    if not have_flow or len(peak_coords) == 0:
        print("Use peak detection from local maxima.")
        return raw_coords, raw_coords

    print("Adjusting peak detection using heatmap.")
    return _apply_flow_correction(pred, peak_coords, n_threads), raw_coords


def map_and_filter_detections(
    segmentation: np.ndarray,
    detections: pd.DataFrame,
    max_distance: float,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    n_threads: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Map synapse detections to segmented IHCs and filter out detections above a distance threshold to the IHCs.

    Args:
        segmentation: The IHC segmentation.
        detections: The synapse marker detections.
        max_distance: The maximal distance in micrometer for a valid match of synapse markers to IHCs.
        voxel_size: The voxel size of the data in micrometer.
        n_threads: The number of threads for parallelizing the mapping of detections to objects.
        verbose: Whether to print the progress of the mapping procedure.

    Returns:
        The filtered dataframe with the detections mapped to the segmentation.
    """
    # Get the point coordinates in pixel by scaling with resolution, rounding, and conversion to integers
    scaling_factors = {"x": 1 / voxel_size[0], "y": 1 / voxel_size[1], "z": 1 / voxel_size[2]}
    points = detections[["z", "y", "x"]].mul(scaling_factors).round().values.astype("int")

    # Set the block shape (this could also be exposed as a parameter; it should not matter much though).
    block_shape = (128, 128, 128)

    # Determine the halo. We set it to 2 pixels + the max-distance in pixels, to ensure all distances
    # that are smaller than the max distance are measured.
    # The halo and the sampling are applied to the ZYX segmentation, so both use the reversed
    # (x, y, z) voxel size.
    voxel_size_zyx = tuple(voxel_size)[::-1]
    halo = tuple(2 + int(np.ceil(max_distance / vs)) for vs in voxel_size_zyx)

    # Map the detections to the objects in the (IHC) segmentation.
    object_ids, object_distances = map_points_to_objects(
        segmentation=segmentation,
        points=points,
        block_shape=block_shape,
        halo=halo,
        sampling=voxel_size_zyx,
        n_threads=n_threads,
        verbose=verbose,
    )
    assert len(object_ids) == len(points)
    assert len(object_distances) == len(points)

    # Add matched ids and distances to the dataframe.
    detections["matched_ihc"] = object_ids
    # map_points_to_objects already returns physical distances because sampling
    # is set to the voxel size above.
    detections["distance_to_ihc"] = object_distances

    # Filter the dataframe by the max distance.
    detections = detections[detections.distance_to_ihc <= max_distance]
    return detections


def _to_mobie_format(coords: np.ndarray, voxel_size: Tuple[float, float, float]) -> pd.DataFrame:
    """Convert (N, 3) [z, y, x] pixel coordinates to a MoBIE-compatible spot table."""
    coords = np.concatenate(
        [np.arange(1, len(coords) + 1)[:, None], coords[:, ::-1]], axis=1
    )
    table = pd.DataFrame(coords, columns=["spot_id", "x", "y", "z"])
    table["x"] *= voxel_size[0]
    table["y"] *= voxel_size[1]
    table["z"] *= voxel_size[2]
    return table


def synapse_detection_from_prediction(
    prediction_path: str,
    detection_path: str,
    block_shape: Optional[Tuple[int, int, int]] = None,
    prediction_key: str = "prediction",
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    force_overwrite: bool = False,
    threshold: float = 0.5,
    n_threads: Optional[int] = None,
    save_no_flow: bool = False,
) -> pd.DataFrame:
    """Run synapse detection for prediction.

    Args:
        prediction_path: Input path to synapse prediction in ZARR format.
        detection_path: Output path for synapse detection.
        block_shape: The block-shape for peak detection. By default it is derived from the
            chunks of the prediction.
        prediction_key: Input key for prediction.
        voxel_size: The voxel size of the data in micrometer.
        force_overwrite: Forcefully overwrite output detection.
        threshold: Absolute heatmap threshold for peak detection. If None, the
            threshold is loaded from cache or determined via gridsearch on the
            validation set used during training (requires *model_path*).
        n_threads: The number of threads for peak detection and flow correction.
        save_no_flow: Whether to additionally save the un-corrected peak detections
            (before sub-voxel flow correction) to a sibling file next to
            *detection_path*, named `<name>_no-flow<ext>`. Written only when
            *detection_path* is (re)computed, not when it is loaded from cache.

    Returns:
        The detections in MoBIE compatible format, with coordinates in micrometer.
    """
    print(f"Using detection threshold: {threshold:.3f}")
    n_threads = min(16, _available_cpus()) if n_threads is None else n_threads

    if not os.path.exists(detection_path) or force_overwrite:
        pred = zarr.open(prediction_path, mode="r")[prediction_key]
        # Use the spatial chunk shape (drop the leading channel dim for multi-channel predictions).
        det_block_shape = block_shape or _detection_block_shape(pred.chunks[-3:])
        coords, no_flow_coords = _flow_corrected_detections(
            pred, min_distance=2, threshold_abs=threshold,
            block_shape=det_block_shape, n_threads=n_threads,
        )
        detections = _to_mobie_format(coords, voxel_size)
        detections.to_csv(detection_path, index=False, sep="\t")

        if save_no_flow:
            base, ext = os.path.splitext(detection_path)
            no_flow_path = f"{base}_no-flow{ext}"
            _to_mobie_format(no_flow_coords, voxel_size).to_csv(no_flow_path, index=False, sep="\t")
    else:
        print(f"Skipping peak detection. {detection_path} already exists.")
        detections = pd.read_csv(detection_path, sep="\t")

    return detections


def build_ihc_mask(
    mask_path: str,
    output_folder: str,
    mask_input_key: str = "s4",
) -> None:
    """Derive the prediction mask from an IHC segmentation.

    The segmentation is read at a low scale level, binarized and dilated. The result is
    written to 'mask.zarr' in the output folder, where `prediction_impl` picks it up and
    resizes it to the full resolution. Synapses are matched to IHCs within `max_distance`
    later on, so restricting inference to the dilated IHC region discards no detection
    that survives that filter.

    Args:
        mask_path: Path to the IHC segmentation.
        output_folder: Output folder for synapse segmentation and marker detection.
        mask_input_key: Key to the undersampled IHC segmentation.
    """
    output_file = os.path.join(output_folder, "mask.zarr")
    mask_key = "mask"
    if os.path.exists(output_file) and mask_key in zarr.open(output_file, mode="r"):
        print(f"Skipping mask creation. {output_file} already exists.")
        return

    segmentation = read_image_data(mask_path, mask_input_key)
    # binary_dilation casts its input to bool, so the label ids act as foreground directly.
    dilated = binary_dilation(segmentation, structure=np.ones((9, 9, 9))).astype("uint8")

    os.makedirs(output_folder, exist_ok=True)
    f_out = zarr.open(output_file, mode="w")
    f_out.create_array(mask_key, data=dilated, compressors=zarr.codecs.GzipCodec())


def _predict_synapses(
    input_path, input_key, output_folder, model_path, block_shape, halo,
    prediction_instances=1, slurm_task_id=0, mean=None, std=None,
):
    """Run the U-Net inference stage of synapse detection."""
    prediction_impl(
        input_path, input_key, output_folder, model_path,
        scale=None,
        block_shape=_PREDICTION_BLOCK_SHAPE if block_shape is None else block_shape,
        halo=_PREDICTION_HALO if halo is None else halo,
        apply_postprocessing=False,
        output_channels=_get_model_out_channels(model_path),
        prediction_instances=prediction_instances, slurm_task_id=slurm_task_id,
        mean=mean, std=std,
    )


def run_prediction(
    input_path: str,
    input_key: str,
    output_folder: str,
    model_path: str,
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    threshold: float = 0.5,
    n_threads: Optional[int] = None,
):
    """Run prediction for synapse detection.

    Args:
        input_path: Input path to image channel for synapse detection.
        input_key: Input key for resolution of image channel and mask channel.
        output_folder: Output folder for synapse segmentation and marker detection.
        model_path: Path to model for synapse detection.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        voxel_size: The voxel size of the data in micrometer.
        threshold: Threshold for peak detection.
        n_threads: The number of threads for peak detection and flow correction.
    """

    # Skip existing prediction, which is saved in output_folder/predictions.zarr.
    # The check only tests that the dataset exists, not that every block was written, so it
    # is valid for this single-job path alone. See run_synapse_prediction_slurm.
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    skip_prediction = os.path.exists(output_path) and prediction_key in zarr.open(output_path, mode="r")

    if not skip_prediction:
        _predict_synapses(input_path, input_key, output_folder, model_path, block_shape, halo)

    detection_path = os.path.join(output_folder, "synapse_detection.tsv")
    synapse_detection_from_prediction(
        output_path, detection_path,
        prediction_key=prediction_key,
        voxel_size=voxel_size,
        threshold=threshold,
        n_threads=n_threads,
    )


def marker_detection(
    input_path: str,
    input_key: str,
    mask_path: Optional[str],
    output_folder: str,
    model_path: str,
    mask_input_key: Optional[str] = "s4",
    max_distance: float = 3,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
):
    """Streamlined workflow for marker detection, mapping, and filtering.

    Args:
        input_path: Input path to image channel for synapse detection.
        input_key: Input key for resolution of image channel and mask channel.
        mask_path: Path to IHC segmentation used to mask input.
        output_folder: Output folder for synapse segmentation and marker detection.
        model_path: Path to model for synapse detection.
        mask_input_key: Key to undersampled IHC segmentation for masking input for synapse detection.
        max_distance: The maximal distance in micrometer for a valid match of synapse markers to IHCs.
        voxel_size: The voxel size of the data in micrometer.
    """
    voxel_size = _normalize_voxel_size(voxel_size)

    # 1.) Determine mask for inference based on the IHC segmentation.
    if mask_path is not None:
        build_ihc_mask(mask_path, output_folder, mask_input_key=mask_input_key)

    # 2.) Run inference and detection of maxima.

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, mode="r"):
        skip_prediction = True

    # skip prediction if post-processed output exists
    detection_path = os.path.join(output_folder, "synapse_detection.tsv")
    if os.path.exists(detection_path):
        skip_prediction = True

    if not skip_prediction:
        out_channels = _get_model_out_channels(model_path)
        prediction_impl(
            input_path, input_key, output_folder, model_path,
            scale=None, apply_postprocessing=False, output_channels=out_channels,
            block_shape=None, halo=None,
        )

    detections = synapse_detection_from_prediction(
        output_path, detection_path, prediction_key=prediction_key, voxel_size=voxel_size
    )

    # 3.) Map the detections to IHC and filter them based on a distance criterion.
    # Use the function 'map_and_filter_detections' from above.
    if mask_path is not None:
        input_ = read_image_data(mask_path, input_key)
        detections_filtered = map_and_filter_detections(
            segmentation=input_,
            detections=detections,
            max_distance=max_distance,
            voxel_size=voxel_size,
        )

        # Save the result in MoBIE compatible format.
        detection_path = os.path.join(output_folder, "synapse_detection_filtered.tsv")
        detections_filtered.to_csv(detection_path, index=False, sep="\t")


#
# ---Workflow for parallel synapse detection using slurm---
#


def run_synapse_prediction_preprocess_slurm(
    input_path: str,
    output_folder: str,
    input_key: Optional[str] = None,
    mask_path: Optional[str] = None,
    mask_input_key: str = "s4",
    s3: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    s3_credentials: Optional[str] = None,
) -> None:
    """Pre-processing for the parallel synapse prediction.

    This is the first of three steps. It runs as a single job before the prediction array.
    The optional mask is stored in 'mask.zarr' in the output folder. The mean and standard
    deviation are stored in 'mean_std.json'. Every array task must use the same values, and
    recomputing them per task would read the full volume once per task.

    Args:
        input_path: Input path to image channel for synapse detection.
        output_folder: Output folder for synapse segmentation and marker detection.
        input_key: Input key for resolution of the image channel.
        mask_path: Path to an IHC segmentation used to restrict the prediction.
            By default the prediction runs on the full volume.
        mask_input_key: Key to the undersampled IHC segmentation.
        s3: Flag for accessing data stored on S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
    """
    os.makedirs(output_folder, exist_ok=True)

    if s3 is not None:
        input_path, _ = s3_utils.get_s3_path(
            input_path, bucket_name=s3_bucket_name,
            service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
        )

    if mask_path is not None:
        build_ihc_mask(mask_path, output_folder, mask_input_key=mask_input_key)

    if not os.path.isfile(os.path.join(output_folder, "mean_std.json")):
        calc_mean_and_std(input_path, input_key, output_folder)


def run_synapse_prediction_slurm(
    input_path: str,
    output_folder: str,
    model_path: str,
    input_key: Optional[str] = None,
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    prediction_instances: int = 1,
    s3: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    s3_credentials: Optional[str] = None,
) -> None:
    """Run one task of the parallel synapse prediction.

    This is the second of three steps. Submit it as a slurm array with as many tasks as
    `prediction_instances`. Each task predicts its own subset of the blocks and writes them
    into the shared 'predictions.zarr' in the output folder.

    Args:
        input_path: Input path to image channel for synapse detection.
        output_folder: Output folder for synapse segmentation and marker detection.
        model_path: Path to model for synapse detection.
        input_key: Input key for resolution of the image channel.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        prediction_instances: Number of instances for parallel prediction.
            This must match the size of the slurm array.
        s3: Flag for accessing data stored on S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
    """
    os.makedirs(output_folder, exist_ok=True)
    prediction_instances = int(prediction_instances)

    slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_task_id is None:
        raise ValueError("The SLURM_ARRAY_TASK_ID is not set. Ensure that you are using the '-a' option with SBATCH.")
    slurm_task_id = int(slurm_task_id)
    if slurm_task_id >= prediction_instances:
        raise ValueError(
            f"The SLURM_ARRAY_TASK_ID {slurm_task_id} exceeds the {prediction_instances} prediction instances. "
            "The size of the slurm array and 'prediction_instances' must match."
        )

    if s3 is not None:
        input_path, _ = s3_utils.get_s3_path(
            input_path, bucket_name=s3_bucket_name,
            service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
        )

    # Get the pre-computed mean and standard deviation of the full volume from the JSON file.
    mean_std_file = os.path.join(output_folder, "mean_std.json")
    if os.path.isfile(mean_std_file):
        with open(mean_std_file) as f:
            values = json.load(f)
        mean, std = float(values["mean"]), float(values["std"])
    else:
        raise ValueError(
            f"{mean_std_file} does not exist. Run 'run_synapse_prediction_preprocess_slurm' first, so that all "
            "array tasks normalize the input identically."
        )

    # No skip check on the existing prediction here: the dataset is created by whichever task
    # starts first, so skipping on its existence would leave the other tasks' blocks empty.
    _predict_synapses(
        input_path, input_key, output_folder, model_path, block_shape, halo,
        prediction_instances=prediction_instances, slurm_task_id=slurm_task_id,
        mean=mean, std=std,
    )


def run_synapse_detection_slurm(
    output_folder: str,
    voxel_size: Union[float, Tuple[float, float, float]] = (0.38, 0.38, 0.38),
    threshold: float = 0.5,
    n_threads: Optional[int] = None,
    mask_path: Optional[str] = None,
    mask_input_key: Optional[str] = None,
    max_distance: float = 3.0,
) -> None:
    """Detect the synapse markers in a finished prediction.

    This is the third of three steps. Run it as a single job after the prediction array
    finished. It needs no GPU. The peak detection is thread-parallel, so give the job cores.

    Args:
        output_folder: Output folder for synapse segmentation and marker detection.
            It must contain the 'predictions.zarr' written by the prediction array.
        voxel_size: The voxel size of the data in micrometer.
        threshold: Threshold for peak detection.
        n_threads: The number of threads for peak detection and flow correction.
            By default it is derived from the number of cores available to the job.
        mask_path: Path to an IHC segmentation. If given, the detections are matched to the
            IHCs and filtered by 'max_distance'.
        mask_input_key: Key to the IHC segmentation at full resolution.
        max_distance: The maximal distance in micrometer for a valid match of synapse markers to IHCs.
    """
    voxel_size = _normalize_voxel_size(voxel_size)
    threshold = float(threshold)
    n_threads = None if n_threads is None else int(n_threads)

    prediction_path = os.path.join(output_folder, "predictions.zarr")
    detection_path = os.path.join(output_folder, "synapse_detection.tsv")
    detections = synapse_detection_from_prediction(
        prediction_path, detection_path, prediction_key="prediction",
        voxel_size=voxel_size, threshold=threshold, n_threads=n_threads,
    )

    if mask_path is not None:
        segmentation = read_image_data(mask_path, mask_input_key)
        detections_filtered = map_and_filter_detections(
            segmentation=segmentation, detections=detections,
            max_distance=float(max_distance), voxel_size=voxel_size,
        )
        detections_filtered.to_csv(
            os.path.join(output_folder, "synapse_detection_filtered.tsv"), index=False, sep="\t"
        )
