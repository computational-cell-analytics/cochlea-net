import multiprocessing
import os
import threading
from concurrent import futures
from threadpoolctl import threadpool_limits
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import zarr

from elf.io import open_file
from elf.parallel.local_maxima import find_local_maxima
from flamingo_tools.segmentation.unet_prediction import prediction_impl
from tqdm import tqdm

from elf.parallel.common import get_blocking


def distance_based_marker_extension(
    markers: np.ndarray,
    output: ArrayLike,
    extension_distance: float,
    sampling: Union[float, Tuple[float, ...]],
    block_shape: Tuple[int, ...],
    n_threads: Optional[int] = None,
    verbose: bool = False,
    roi: Optional[Tuple[slice, ...]] = None,
):
    """
    Extend SGN detection to emulate shape of SGNs for better visualization.

    Args:
        markers: Array of coordinates for seeding watershed.
        output: Output for watershed.
        extension_distance: Distance in micrometer for extension.
        sampling: Resolution in micrometer.
        block_shape:
        n_threads:
        verbose:
        roi:
    """
    n_threads = multiprocessing.cpu_count() if n_threads is None else n_threads
    blocking = get_blocking(output, block_shape, roi, n_threads)

    lock = threading.Lock()

    # determine the correct halo in pixels based on the sampling and the extension distance.
    halo = [round(extension_distance / s) + 2 for s in sampling]

    @threadpool_limits.wrap(limits=1)  # restrict the numpy threadpool to 1 to avoid oversubscription
    def extend_block(block_id):
        block = blocking.getBlockWithHalo(block_id, halo)
        outer_block = block.outerBlock
        inner_block = block.innerBlock

        # get the indices and coordinates of the markers in the INNER block
        mask = (
            (inner_block.begin[0] <= markers[:, 0]) & (markers[:, 0] <= inner_block.end[0]) &
            (inner_block.begin[1] <= markers[:, 1]) & (markers[:, 1] <= inner_block.end[1]) &
            (inner_block.begin[2] <= markers[:, 2]) & (markers[:, 2] <= inner_block.end[2])
        )
        markers_in_block_ids = np.where(mask)[0]
        markers_in_block_coords = markers[markers_in_block_ids]

        # proceed if detections fall within inner block
        if len(markers_in_block_coords) > 0:
            markers_in_block_coords = [coord - outer_block.begin for coord in markers_in_block_coords]
            markers_in_block_coords = [[round(c) for c in coord] for coord in markers_in_block_coords]

            markers_in_block_coords = np.array(markers_in_block_coords, dtype=int)
            z, y, x = markers_in_block_coords.T

            # Shift index by one so that zero is reserved for background id
            markers_in_block_ids += 1

            # Create the seed volume.
            outer_block_shape = tuple(end - begin for begin, end in zip(outer_block.begin, outer_block.end))
            seeds = np.zeros(outer_block_shape, dtype="uint32")
            seeds[z, y, x] = markers_in_block_ids

            # Compute the distance map.
            distance = distance_transform_edt(seeds == 0, sampling=sampling)

            # And extend the seeds
            mask = distance < extension_distance
            segmentation = watershed(distance.max() - distance, markers=seeds, mask=mask)

            # Write the segmentation. Note: we need to lock here because we write outside of our inner block
            bb = tuple(slice(begin, end) for begin, end in zip(outer_block.begin, outer_block.end))
            with lock:
                this_output = output[bb]
                this_output[mask] = segmentation[mask]
                output[bb] = this_output

    n_blocks = blocking.numberOfBlocks
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(extend_block, range(n_blocks)), total=n_blocks, desc="Marker extension", disable=not verbose
        ))


def sgn_detection(
    input_path: str,
    input_key: str,
    output_folder: str,
    model_path: str,
    extension_distance: float,
    sampling: Union[float, Tuple[float, ...]],
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    n_threads: Optional[int] = None,
    threshold_abs: float = 0.5,
    min_distance: int = 4,
):
    """Run prediction for SGN detection.

    Args:
        input_path: Input path to image channel for SGN detection.
        input_key: Input key for resolution of image channel and mask channel.
        output_folder: Output folder for SGN segmentation.
        model_path: Path to model for SGN detection.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        spot_radius: Radius in pixel to convert spot detection of SGNs into a volume.
    """
    if block_shape is None:
        block_shape = (12, 128, 128)
    if halo is None:
        halo = (10, 64, 64)

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, mode="r"):
        skip_prediction = True

    if not skip_prediction:
        prediction_impl(
            input_path, input_key, output_folder, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=1,
        )

    detection_path = os.path.join(output_folder, "SGN_detection.tsv")
    input_ = zarr.open(output_path, mode="r")[prediction_key]
    if not os.path.exists(detection_path):
        block_shape = (12, 128, 128)  # bigger block to avoid edge effects
        detections_maxima = find_local_maxima(
            input_, block_shape=block_shape, min_distance=min_distance, threshold_abs=threshold_abs,
            verbose=True, n_threads=16,
        )

        # Save the result in mobie compatible format.
        detections = np.concatenate(
            [np.arange(1, len(detections_maxima) + 1)[:, None], detections_maxima[:, ::-1]], axis=1
        )
        detections = pd.DataFrame(detections, columns=["spot_id", "x", "y", "z"])
        detections.to_csv(detection_path, index=False, sep="\t")
    else:
        detections_maxima = None

    segmentation_path = os.path.join(output_folder, "segmentation.zarr")
    # extend detection
    if not os.path.exists(segmentation_path):
        shape = input_.shape
        chunks = (128, 128, 128)

        if detections_maxima is None:
            detections_maxima = pd.read_csv(detection_path, sep="\t")
            detections_maxima = detections_maxima[["z", "y", "x"]].values

        output = open_file(segmentation_path, mode="a")
        segmentation_key = "segmentation"
        output_dataset = output.create_dataset(
            segmentation_key, shape=shape, dtype=np.uint64,
            chunks=chunks, compression="gzip"
        )

        distance_based_marker_extension(
            markers=detections_maxima,
            output=output_dataset,
            extension_distance=extension_distance,
            sampling=sampling,
            block_shape=(128, 128, 128),
            n_threads=n_threads,
            verbose=True,
        )
