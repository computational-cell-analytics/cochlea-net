import multiprocessing
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
from flamingo_tools.segmentation.unet_prediction import prediction_impl, SelectChannel

# Must match the sigma used in CsvHeatmapFlowTransform during training.
_HEATMAP_FLOW_SIGMA = 1

# Peak detection reads whole chunks. A block that equals one chunk makes the halo
# dominate the read volume, so grow the block until it reaches this voxel budget.
_DETECTION_BLOCK_VOXELS = 32_000_000


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
    halo = (
        2 + int(np.ceil(max_distance / voxel_size[0])),
        2 + int(np.ceil(max_distance / voxel_size[1])),
        2 + int(np.ceil(max_distance / voxel_size[2])),
    )

    # Map the detections to the objects in the (IHC) segmentation.
    object_ids, object_distances = map_points_to_objects(
        segmentation=segmentation,
        points=points,
        block_shape=block_shape,
        halo=halo,
        sampling=voxel_size,
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
    save_no_flow: bool = True,
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
    n_threads = min(16, multiprocessing.cpu_count()) if n_threads is None else n_threads

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

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, mode="r"):
        skip_prediction = True

    if not skip_prediction:
        out_channels = _get_model_out_channels(model_path)
        prediction_impl(
            input_path, input_key, output_folder, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=out_channels,
        )

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
    if not isinstance(voxel_size, float):
        if len(voxel_size) == 1:
            voxel_size = voxel_size * 3
        assert len(voxel_size) == 3
    else:
        voxel_size = (voxel_size,) * 3

    # 1.) Determine mask for inference based on the IHC segmentation.
    # Best approach: load IHC segmentation at a low scale level, binarize it,
    # dilate it and use this as mask. It can be mapped back to the full resolution
    # with `elf.wrapper.ResizedVolume`.
    skip_masking = False

    mask_preprocess_key = "mask"
    output_file = os.path.join(output_folder, "mask.zarr")

    if mask_path is None or (os.path.exists(output_file) and mask_preprocess_key in zarr.open(output_file, mode="r")):
        skip_masking = True

    if not skip_masking:
        mask_ = read_image_data(mask_path, mask_input_key)
        new_mask = np.zeros(mask_.shape)
        new_mask[mask_ != 0] = 1
        arr_bin = binary_dilation(mask_, structure=np.ones((9, 9, 9))).astype(int)

        f_out = zarr.open(output_file, mode="w")
        f_out.create_array(mask_preprocess_key, data=arr_bin, compressors=zarr.codecs.GzipCodec())

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
