import os
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from scipy.ndimage import binary_dilation

from elf.parallel.local_maxima import find_local_maxima
from elf.parallel.distance_transform import map_points_to_objects
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.segmentation.unet_prediction import prediction_impl


def map_and_filter_detections(
    segmentation: np.ndarray,
    detections: pd.DataFrame,
    max_distance: float,
    resolution: float = 0.38,
    n_threads: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Map synapse detections to segmented IHCs and filter out detections above a distance threshold to the IHCs.

    Args:
        segmentation: The IHC segmentation.
        detections: The synapse marker detections.
        max_distance: The maximal distance in micrometer for a valid match of synapse markers to IHCs.
        resolution: The resolution / voxel size of the data in micrometer.
        n_threads: The number of threads for parallelizing the mapping of detections to objects.
        verbose: Whether to print the progress of the mapping procedure.

    Returns:
        The filtered dataframe with the detections mapped to the segmentation.
    """
    # Get the point coordinates in pixel by scaling with resolution, rounding, and conversion to integers
    points = detections[["z", "y", "x"]].mul(1 / resolution).round().values.astype("int")

    # Set the block shape (this could also be exposed as a parameter; it should not matter much though).
    block_shape = (128, 128, 128)

    # Determine the halo. We set it to 2 pixels + the max-distance in pixels, to ensure all distances
    # that are smaller than the max distance are measured.
    halo = (2 + int(np.ceil(max_distance / resolution)),) * 3

    # Map the detections to the objects in the (IHC) segmentation.
    object_ids, object_distances = map_points_to_objects(
        segmentation=segmentation,
        points=points,
        block_shape=block_shape,
        halo=halo,
        sampling=resolution,
        n_threads=n_threads,
        verbose=verbose,
    )
    assert len(object_ids) == len(points)
    assert len(object_distances) == len(points)

    # Add matched ids and distances to the dataframe.
    detections["matched_ihc"] = object_ids
    detections["distance_to_ihc"] = object_distances * resolution

    # Filter the dataframe by the max distance.
    detections = detections[detections.distance_to_ihc <= max_distance]
    return detections


def run_prediction(
    input_path: str,
    input_key: str,
    output_folder: str,
    model_path: str,
    block_shape: Optional[Tuple[int, int, int]] = None,
    halo: Optional[Tuple[int, int, int]] = None,
    resolution: float = 0.38,
):
    """Run prediction for synapse detection.

    Args:
        input_path: Input path to image channel for synapse detection.
        input_key: Input key for resolution of image channel and mask channel.
        output_folder: Output folder for synapse segmentation and marker detection.
        model_path: Path to model for synapse detection.
        block_shape: The block-shape for running the prediction.
        halo: The halo (= block overlap) to use for prediction.
        resolution: The resolution / voxel size of the data in micrometer.
    """

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, "r"):
        skip_prediction = True

    if not skip_prediction:
        prediction_impl(
            input_path, input_key, output_folder, model_path,
            scale=None, block_shape=block_shape, halo=halo,
            apply_postprocessing=False, output_channels=1,
        )

    detection_path = os.path.join(output_folder, "synapse_detection.tsv")
    if not os.path.exists(detection_path):
        input_ = zarr.open(output_path, "r")[prediction_key]
        detections = find_local_maxima(
            input_, block_shape=block_shape, min_distance=2, threshold_abs=0.5, verbose=True, n_threads=16,
        )
        # Save the result in MoBIE compatible format.
        detections = np.concatenate(
            [np.arange(1, len(detections) + 1)[:, None], detections[:, ::-1]], axis=1
        )
        detections = pd.DataFrame(detections, columns=["spot_id", "x", "y", "z"])

        # scale coordinates
        detections["x"] *= resolution
        detections["y"] *= resolution
        detections["z"] *= resolution

        detections.to_csv(detection_path, index=False, sep="\t")


def marker_detection(
    input_path: str,
    input_key: str,
    mask_path: Optional[str],
    output_folder: str,
    model_path: str,
    mask_input_key: Optional[str] = "s4",
    max_distance: float = 3,
    resolution: float = 0.38,
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
        resolution: The resolution / voxel size of the data in micrometer.
    """

    # 1.) Determine mask for inference based on the IHC segmentation.
    # Best approach: load IHC segmentation at a low scale level, binarize it,
    # dilate it and use this as mask. It can be mapped back to the full resolution
    # with `elf.wrapper.ResizedVolume`.
    skip_masking = False

    mask_preprocess_key = "mask"
    output_file = os.path.join(output_folder, "mask.zarr")

    if mask_path is None or (os.path.exists(output_file) and mask_preprocess_key in zarr.open(output_file, "r")):
        skip_masking = True

    if not skip_masking:
        mask_ = read_image_data(mask_path, mask_input_key)
        new_mask = np.zeros(mask_.shape)
        new_mask[mask_ != 0] = 1
        arr_bin = binary_dilation(mask_, structure=np.ones((9, 9, 9))).astype(int)

        with zarr.open(output_file, mode="w") as f_out:
            f_out.create_dataset(mask_preprocess_key, data=arr_bin, compression="gzip")

    # 2.) Run inference and detection of maxima.

    # Skip existing prediction, which is saved in output_folder/predictions.zarr
    skip_prediction = False
    output_path = os.path.join(output_folder, "predictions.zarr")
    prediction_key = "prediction"
    if os.path.exists(output_path) and prediction_key in zarr.open(output_path, "r"):
        skip_prediction = True

    # skip prediction if post-processed output exists
    detection_path = os.path.join(output_folder, "synapse_detection.tsv")
    if os.path.exists(detection_path):
        skip_prediction = True

    if not skip_prediction:
        prediction_impl(
            input_path, input_key, output_folder, model_path,
            scale=None, apply_postprocessing=False, output_channels=1,
            block_shape=None, halo=None,
        )

    if not os.path.exists(detection_path):
        input_ = zarr.open(output_path, "r")[prediction_key]
        block_shape = tuple(input_.chunks)
        detections = find_local_maxima(
            input_, block_shape=block_shape, min_distance=2, threshold_abs=0.5, verbose=True, n_threads=16,
        )
        # Save the result in MoBIE compatible format.
        detections = np.concatenate(
            [np.arange(1, len(detections) + 1)[:, None], detections[:, ::-1]], axis=1
        )
        detections = pd.DataFrame(detections, columns=["spot_id", "x", "y", "z"])

        # scale coordinates
        detections["x"] *= resolution
        detections["y"] *= resolution
        detections["z"] *= resolution

        detections.to_csv(detection_path, index=False, sep="\t")

    else:
        with open(detection_path, "r") as f:
            detections = pd.read_csv(f, sep="\t")

    # 3.) Map the detections to IHC and filter them based on a distance criterion.
    # Use the function 'map_and_filter_detections' from above.
    if mask_path is not None:
        input_ = read_image_data(mask_path, input_key)
        detections_filtered = map_and_filter_detections(
            segmentation=input_,
            detections=detections,
            max_distance=max_distance,
            resolution=resolution,
        )

        # Save the result in MoBIE compatible format.
        detection_path = os.path.join(output_folder, "synapse_detection_filtered.tsv")
        detections_filtered.to_csv(detection_path, index=False, sep="\t")
