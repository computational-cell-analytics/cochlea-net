"""@private
"""

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

# Physical size of one cell of the down-sampled cochlea filter volume, in µm.
# 48 * 0.38 reproduces the down-sampling factor that was hard-coded for isotropic 0.38 µm data.
FILTER_CELL_SIZE = 48 * 0.38


def normalize_voxel_size(voxel_size: Union[float, Sequence[float]]) -> Tuple[float, float, float]:
    """Normalize a voxel size to a (x, y, z) tuple of three values.

    Args:
        voxel_size: Voxel size in µm, either a single value for isotropic data, a sequence with a
            single value, or a sequence of three values in (x, y, z) order.

    Returns:
        Voxel size as a tuple of three values in (x, y, z) order.
    """
    if isinstance(voxel_size, (int, float)):
        return (float(voxel_size),) * 3
    values = tuple(float(v) for v in voxel_size)
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"voxel_size must have 1 or 3 values, got {len(values)}.")
    return values


def filter_volume_downscale_factors(
    voxel_size: Union[float, Sequence[float]],
    cell_size: float = FILTER_CELL_SIZE,
) -> Tuple[int, int, int]:
    """Compute the per-axis down-sampling factor for the cochlea filter volume.

    The factors are chosen so that one cell of the down-sampled volume covers `cell_size` µm on
    every axis. This keeps the physical extent of a dilation step independent of the voxel size.

    Args:
        voxel_size: Voxel size in µm, scalar or (x, y, z).
        cell_size: Target physical size of one cell of the down-sampled volume, in µm.

    Returns:
        Down-sampling factor in pixels per axis, in (x, y, z) order.
    """
    return tuple(max(1, int(round(cell_size / v))) for v in normalize_voxel_size(voxel_size))


def compute_crop_bb(
    crop_center: List[float],
    roi_halo: Optional[List[int]],
    voxel_size: Union[float, Sequence[float]],
    scale: int,
    shape: Tuple[int, ...],
    axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box start/stop for a crop in ZYX pixel coordinates.

    Args:
        crop_center: Crop center position as [x, y, z] in µm.
        roi_halo: Halo around the center as [halo_x, halo_y, halo_z] in pixels at the target
            scale. Optional when `axis` is given: the two axes not collapsed by `axis` then
            span the full array extent (`shape`), i.e. the whole cross-sectional plane is
            cropped. Required otherwise.
        voxel_size: Voxel size in µm at full resolution (scale 0), scalar for isotropic data or
            three values in (x, y, z) order.
        scale: Scale level (0 = full resolution, each step doubles the effective voxel size).
        shape: Array shape in ZYX order at the target scale.
        axis: Optional axis index into (x, y, z), i.e. 0, 1, or 2. When given, that axis is
            collapsed to a single-pixel slice starting at the crop center, producing a 2D
            crop. Default: None (3D crop; requires roi_halo).

    Returns:
        start: ZYX start pixel coordinates, clamped to zero.
        stop: ZYX stop pixel coordinates, clamped to shape.
    """
    if roi_halo is None and axis is None:
        raise ValueError("roi_halo is required unless axis is also given.")

    voxel_size_zyx = np.array(normalize_voxel_size(voxel_size)[::-1])
    center_zyx = np.round(np.array(crop_center[::-1]) / (voxel_size_zyx * 2 ** scale)).astype(int)

    if roi_halo is None:
        start = np.zeros(3, dtype=int)
        stop = np.array(shape, dtype=int)
    else:
        halo_zyx = np.array(roi_halo[::-1])
        start = np.maximum(0, center_zyx - halo_zyx)
        stop = np.minimum(np.array(shape), center_zyx + halo_zyx)

    if axis is not None:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        # crop_center/roi_halo are given in (x, y, z) order; the pixel arrays above are ZYX.
        axis_zyx = 2 - axis
        start[axis_zyx] = center_zyx[axis_zyx]
        stop[axis_zyx] = center_zyx[axis_zyx] + 1

    return start, stop


def crop_suffix(crop_center: List[float], axis: Optional[int] = None, suffix: Optional[str] = None) -> str:
    """Build the output filename suffix for a crop.

    Args:
        crop_center: Crop center position as [x, y, z] in µm.
        axis: Optional axis index into (x, y, z) used to collapse the crop to a 2D slice
            (see `compute_crop_bb`). Appended as "_axis-<axis>" when given.
        suffix: Optional extra label appended last, e.g. a tonotopic position name
            ("apex", "mid", "base").

    Returns:
        Suffix of the form "_crop_<x>-<y>-<z>[_axis-<axis>][_<suffix>]".
    """
    coord_string = "-".join(str(int(round(c))).zfill(4) for c in crop_center)
    parts = [f"_crop_{coord_string}"]
    if axis is not None:
        parts.append(f"_axis-{axis}")
    if suffix is not None:
        parts.append(f"_{suffix}")
    return "".join(parts)


def crop_filter_volume(
    filter_volume: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    us_factor: Union[float, Sequence[float]],
) -> np.ndarray:
    """Extract the sub-region of a down-sampled cochlea filter volume that covers a crop.

    Instead of upscaling the entire filter volume to full resolution and then slicing, this
    function maps each target pixel to its filter volume cell, which is far more memory-efficient
    when exporting high-resolution crops.

    Args:
        filter_volume: Down-sampled boolean cochlea mask in ZYX order.
        start: Crop start in scale-s pixel coordinates, ZYX.
        stop: Crop stop in scale-s pixel coordinates, ZYX.
        us_factor: Size of one filter_volume cell in scale-s pixels, either a single value or one
            value per axis in ZYX order (i.e. ds_factor / 2**scale). May be non-integer.

    Returns:
        Boolean mask aligned to the crop region, shape == stop - start. Regions of the crop
        that fall outside `filter_volume`'s covered extent (it only spans the segmentation
        table's anchor points plus a fixed padding, see `filter_cochlea_volume_single`/
        `filter_cochlea_volume`, and can be smaller than the full image -- e.g. for a
        whole-plane 2D crop) are zero-padded, i.e. treated as "not part of the cochlea".
    """
    factors = np.broadcast_to(np.asarray(us_factor, dtype=float), (3,))
    if np.any(factors <= 0):
        raise ValueError(f"us_factor must be positive, got {us_factor}.")

    indices, valid = [], []
    for ax in range(3):
        index = np.floor(np.arange(start[ax], stop[ax]) / factors[ax]).astype(int)
        valid.append((index >= 0) & (index < filter_volume.shape[ax]))
        indices.append(np.clip(index, 0, filter_volume.shape[ax] - 1))

    cropped = filter_volume[np.ix_(*indices)].astype(bool)
    cropped &= valid[0][:, None, None]
    cropped &= valid[1][None, :, None]
    cropped &= valid[2][None, None, :]
    return cropped
