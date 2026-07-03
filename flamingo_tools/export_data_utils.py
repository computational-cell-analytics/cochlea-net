"""@private
"""

from typing import List, Tuple

import numpy as np


def compute_crop_bb(
    crop_center: List[float],
    roi_halo: List[int],
    voxel_size: float,
    scale: int,
    shape: Tuple[int, ...],
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box start/stop for a crop in ZYX pixel coordinates.

    Args:
        crop_center: Crop center position as [x, y, z] in µm.
        roi_halo: Halo around the center as [halo_x, halo_y, halo_z] in pixels at the target scale.
        voxel_size: Isotropic voxel size in µm at full resolution (scale 0).
        scale: Scale level (0 = full resolution, each step doubles the effective voxel size).
        shape: Array shape in ZYX order at the target scale.

    Returns:
        start: ZYX start pixel coordinates, clamped to zero.
        stop: ZYX stop pixel coordinates, clamped to shape.
    """
    center_zyx = np.round(np.array(crop_center[::-1]) / (voxel_size * 2 ** scale)).astype(int)
    halo_zyx = np.array(roi_halo[::-1])
    start = np.maximum(0, center_zyx - halo_zyx)
    stop = np.minimum(np.array(shape), center_zyx + halo_zyx)
    return start, stop


def crop_filter_volume(
    filter_volume: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    us_factor: int,
) -> np.ndarray:
    """Extract and upscale the sub-region of a downsampled cochlea filter volume for a crop.

    Instead of upscaling the entire filter volume to full resolution and then slicing, this
    function only upscales the small portion covering the requested crop, which is far more
    memory-efficient when exporting high-resolution crops.

    Args:
        filter_volume: Downsampled boolean cochlea mask in ZYX order.
        start: Crop start in scale-s pixel coordinates, ZYX.
        stop: Crop stop in scale-s pixel coordinates, ZYX.
        us_factor: Upscale factor to go from filter_volume resolution to scale-s resolution
            (i.e. ds_factor // 2**scale).

    Returns:
        Boolean mask aligned to the crop region, shape == stop - start.
    """
    # Sub-region of filter_volume covering the crop (1-cell margin for safety).
    start_fv = np.maximum(0, start // us_factor - 1)
    stop_fv = np.minimum(np.array(filter_volume.shape), (stop - 1) // us_factor + 2)
    sub = filter_volume[start_fv[0]:stop_fv[0], start_fv[1]:stop_fv[1], start_fv[2]:stop_fv[2]]

    # Upscale only the sub-region.
    big = np.repeat(np.repeat(np.repeat(sub, us_factor, 0), us_factor, 1), us_factor, 2)

    # Align to exact crop coordinates.
    offset = start - start_fv * us_factor
    crop_size = stop - start
    return big[offset[0]:offset[0] + crop_size[0],
               offset[1]:offset[1] + crop_size[1],
               offset[2]:offset[2] + crop_size[2]]
