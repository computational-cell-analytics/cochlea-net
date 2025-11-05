import multiprocessing as mp
import os
import warnings
from concurrent import futures
from functools import partial
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import trimesh
from elf.io import open_file
from elf.wrapper.resized_volume import ResizedVolume
from elf.wrapper.base import WrapperBase
from elf.util import normalize_index, squeeze_singletons
from nifty.tools import blocking
from skimage.measure import marching_cubes, regionprops_table
from skimage.transform import downscale_local_mean
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from .file_utils import read_image_data
from .segmentation.postprocessing import compute_table_on_the_fly
import flamingo_tools.s3_utils as s3_utils


def _measure_volume_and_surface(mask, resolution):
    # Use marching_cubes for 3D data
    verts, faces, normals, _ = marching_cubes(mask, spacing=(resolution,) * 3)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    surface = mesh.area
    if mesh.is_watertight:
        volume = np.abs(mesh.volume)
    else:
        volume = np.nan

    return volume, surface


def _get_bounding_box_and_center(table, seg_id, resolution, shape, dilation):
    row = table[table.label_id == seg_id]

    if dilation is not None and dilation > 0:
        bb_extension = dilation + 1
    else:
        bb_extension = 1

    bb_min = np.array([
        row.bb_min_z.item(), row.bb_min_y.item(), row.bb_min_x.item()
    ]).astype("float32") / resolution
    bb_min = np.round(bb_min, 0).astype("int32")

    bb_max = np.array([
        row.bb_max_z.item(), row.bb_max_y.item(), row.bb_max_x.item()
    ]).astype("float32") / resolution
    bb_max = np.round(bb_max, 0).astype("int32")

    bb = tuple(
        slice(max(bmin - bb_extension, 0), min(bmax + bb_extension, sh))
        for bmin, bmax, sh in zip(bb_min, bb_max, shape)
    )

    if isinstance(resolution, float):
        resolution = (resolution,) * 3

    center = (
        int(row.anchor_z.item() / resolution[0]),
        int(row.anchor_y.item() / resolution[1]),
        int(row.anchor_x.item() / resolution[2]),
    )

    return bb, center


def _spherical_mask(shape, radius, center=None):
    if center is None:
        center = tuple(s // 2 for s in shape)
    if len(shape) != len(center):
        raise ValueError("`shape` and `center` must have same length")

    # Build a 1-D open grid for every axis
    grids = np.ogrid[tuple(slice(0, s) for s in shape)]
    dist2 = sum((g - c) ** 2 for g, c in zip(grids, center))
    return (dist2 <= radius ** 2).astype(bool)


def _normalize_background(measures, image, mask, center, radius, norm, median_only):
    # Compute the bounding box and get the local image data.
    bb = tuple(
        slice(max(0, int(ce - radius)), min(int(ce + radius), sh)) for ce, sh in zip(center, image.shape)
    )
    local_image = image[bb]

    # Create a mask with radius around the center.
    radius_mask = _spherical_mask(local_image.shape, radius)

    # Intersect the radius mask with the foreground mask (if given).
    if mask is not None:
        assert mask.shape == image.shape, f"{mask.shape}, {image.shape}"
        local_mask = mask[bb]
        radius_mask = np.logical_and(radius_mask, local_mask)

        # For debugging.
        # import napari
        # v = napari.Viewer()
        # v.add_image(local_image)
        # v.add_labels(local_mask)
        # v.add_labels(radius_mask)
        # napari.run()

    # Compute the features over the mask.
    masked_intensity = local_image[radius_mask]

    # Standardize the measures.
    bg_measures = {"median": np.median(masked_intensity)}
    if not median_only:
        bg_measures = {
            "mean": np.mean(masked_intensity),
            "stdev": np.std(masked_intensity),
            "min": np.min(masked_intensity),
            "max": np.max(masked_intensity),
        }
        for percentile in (5, 10, 25, 75, 90, 95):
            bg_measures[f"percentile-{percentile}"] = np.percentile(masked_intensity, percentile)

    for measure, val in bg_measures.items():
        measures[measure] = norm(measures[measure], val)

    return measures


def _default_object_features(
    seg_id, table, image, segmentation, resolution,
    background_mask=None, background_radius=None, norm=np.divide, median_only=False, dilation=None
):
    bb, center = _get_bounding_box_and_center(table, seg_id, resolution, image.shape, dilation)

    local_image = image[bb]
    mask = segmentation[bb] == seg_id
    assert mask.sum() > 0, f"Segmentation ID {seg_id} is empty."
    if dilation is not None and dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    masked_intensity = local_image[mask]

    # Do the base intensity measurements.
    measures = {"label_id": seg_id, "median": np.median(masked_intensity)}
    if not median_only:
        measures.update({
            "mean": np.mean(masked_intensity),
            "stdev": np.std(masked_intensity),
            "min": np.min(masked_intensity),
            "max": np.max(masked_intensity),
        })
        for percentile in (5, 10, 25, 75, 90, 95):
            measures[f"percentile-{percentile}"] = np.percentile(masked_intensity, percentile)

    if background_radius is not None:
        # The radius passed is given in micrometer.
        # The resolution is given in micrometer per pixel.
        # So we have to divide by the resolution to obtain the radius in pixel.
        radius_in_pixel = background_radius / resolution
        measures = _normalize_background(measures, image, background_mask, center, radius_in_pixel, norm, median_only)

    # Do the volume and surface measurement.
    if not median_only:
        volume, surface = _measure_volume_and_surface(mask, resolution)
        measures["volume"] = volume
        measures["surface"] = surface
    return measures


def _regionprops_features(seg_id, table, image, segmentation, resolution, background_mask=None, dilation=None):
    bb, _ = _get_bounding_box_and_center(table, seg_id, resolution, image.shape, dilation)

    local_image = image[bb]
    local_segmentation = segmentation[bb]
    mask = local_segmentation == seg_id
    assert mask.sum() > 0, f"Segmentation ID {seg_id} is empty."
    if dilation is not None and dilation > 0:
        mask = binary_dilation(mask, iterations=dilation)
    local_segmentation[~mask] = 0

    features = regionprops_table(
        local_segmentation, local_image, properties=[
            "label", "area", "axis_major_length", "axis_minor_length",
            "equivalent_diameter_area", "euler_number", "extent",
            "feret_diameter_max", "inertia_tensor_eigvals",
            "intensity_max", "intensity_mean", "intensity_min",
            "intensity_std", "moments_central",
            "moments_weighted", "solidity",
        ]
    )

    features["label_id"] = features.pop("label")
    return features


def get_object_measures_from_table(arr_seg, table):
    """Return object measurements for label IDs wthin array.
    """
    # iterate through segmentation ids in reference mask
    ref_ids = list(np.unique(arr_seg)[1:])
    measure_ids = list(table["label_id"])
    object_ids = [id for id in ref_ids if id in measure_ids]
    if len(object_ids) < len(ref_ids):
        warnings.warn(f"Not all IDs were found in measurement table. Using {len(object_ids)}/{len(ref_ids)}.")

    median_values = [table.at[table.index[table["label_id"] == label_id][0], "median"] for label_id in object_ids]

    measures = pd.DataFrame({
        "label_id": object_ids,
        "median": median_values,
    })
    return measures


# Maybe also support:
# - spherical harmonics
# - line profiles
FEATURE_FUNCTIONS = {
    "default": _default_object_features,
    "skimage": _regionprops_features,
    "default_background_norm": partial(_default_object_features, background_radius=75, norm=np.divide),
    "default_background_subtract": partial(_default_object_features, background_radius=75, norm=np.subtract),
}
"""The different feature functions that are supported in `compute_object_measures` and
that can be selected via the feature_set argument. Currently this supports:
- 'default': The default features which compute standard intensity statistics and volume + surface.
- 'skimage': The scikit image regionprops features.
- 'default_background_norm': The default features with background normalization.
- 'default_background_subtract': The default features with background subtraction.

For the background normalized measures, we compute the background intensity in a sphere with radius of 75 micrometer
around each object.
"""


def compute_object_measures_impl(
    image: np.typing.ArrayLike,
    segmentation: np.typing.ArrayLike,
    n_threads: Optional[int] = None,
    resolution: float = 0.38,
    table: Optional[pd.DataFrame] = None,
    feature_set: str = "default",
    background_mask: Optional[np.typing.ArrayLike] = None,
    median_only: bool = False,
    dilation: Optional[int] = None,
) -> pd.DataFrame:
    """Compute simple intensity and morphology measures for each segmented cell in a segmentation.

    See `compute_object_measures` for details.

    Args:
        image: The image data.
        segmentation: The segmentation.
        n_threads: The number of threads to use for computation.
        resolution: The resolution / voxel size of the data.
        table: The segmentation table. Will be computed on the fly if it is not given.
        feature_set: The features to compute for each object. Refer to `FEATURE_FUNCTIONS` for details.
        background_mask: An optional mask indicating the area to use for computing background correction values.
        median_only: Whether to only compute the median intensity.
        dilation: Value for dilating the segmentation before computing measurements.
            By default no dilation is applied.

    Returns:
        The table with per object measurements.
    """
    if table is None:
        table = compute_table_on_the_fly(segmentation, resolution=resolution)

    if feature_set not in FEATURE_FUNCTIONS:
        raise ValueError
    measure_function = partial(
        FEATURE_FUNCTIONS[feature_set],
        table=table,
        image=image,
        segmentation=segmentation,
        resolution=resolution,
        background_mask=background_mask,
        median_only=median_only,
        dilation=dilation,
    )

    seg_ids = table.label_id.values
    assert len(seg_ids) > 0, "The segmentation table is empty."
    measure_function(seg_ids[0])
    n_threads = mp.cpu_count() if n_threads is None else n_threads

    # For debugging.
    # measure_function(seg_ids[0])
    # breakpoint()

    with futures.ThreadPoolExecutor(n_threads) as pool:
        measures = list(tqdm(
            pool.map(measure_function, seg_ids), total=len(seg_ids), desc="Compute intensity measures"
        ))

    # Create the result table and save it.
    keys = measures[0].keys()
    measures = pd.DataFrame({k: [measure[k] for measure in measures] for k in keys})
    return measures


# Could also support s3 directly?
def compute_object_measures(
    image_path: str,
    segmentation_path: str,
    segmentation_table_path: Optional[str],
    output_table_path: str,
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    n_threads: Optional[int] = None,
    resolution: Union[float, Tuple[float, ...]] = 0.38,
    force: bool = False,
    feature_set: str = "default",
    s3_flag: bool = False,
    component_list: List[int] = [],
    dilation: Optional[int] = None,
    median_only: bool = False,
    background_mask: Optional[np.typing.ArrayLike] = None,
) -> None:
    """Compute simple intensity and morphology measures for each segmented cell in a segmentation.

    By default, this computes the mean, standard deviation, minimum, maximum, median and
    5th, 10th, 25th, 75th, 90th and 95th percentile of the intensity image
    per cell, as well as the volume and surface.
    Other measurements can be computed by changing the feature_set argument.

    Args:
        image_path: The filepath to the image data. Either a tif or hdf5/zarr/n5 file.
        segmentation_path: The filepath to the segmentation data. Either a tif or hdf5/zarr/n5 file.
        segmentation_table_path: The path to the segmentation table in MoBIE format.
        output_table_path: The path for saving the segmentation with intensity measures.
        image_key: The key (= internal path) for the image data. Not needed fir tif.
        segmentation_key: The key (= internal path) for the segmentation data. Not needed for tif.
        n_threads: The number of threads to use for computation.
        resolution: The resolution / voxel size of the data.
        force: Whether to overwrite an existing output table.
        feature_set: The features to compute for each object. Refer to `FEATURE_FUNCTIONS` for details.
        s3_flag:
        component_list:
        median_only: Whether to only compute the median intensity.
        dilation: Value for dilating the segmentation before computing measurements.
            By default no dilation is applied.
        background_mask: An optional mask indicating the area to use for computing background correction values.
    """
    if os.path.exists(output_table_path) and not force:
        return

    # First, we load the pre-computed segmentation table from MoBIE.
    if segmentation_table_path is None:
        table = None
    elif s3_flag:
        seg_table, fs = s3_utils.get_s3_path(segmentation_table_path)
        with fs.open(seg_table, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table = pd.read_csv(segmentation_table_path, sep="\t")

    # filter table with largest component
    if len(component_list) != 0 and "component_labels" in table.columns:
        table = table[table["component_labels"].isin(component_list)]

    # Then, open the volumes.
    image = read_image_data(image_path, image_key, from_s3=s3_flag)
    segmentation = read_image_data(segmentation_path, segmentation_key, from_s3=s3_flag)

    measures = compute_object_measures_impl(
        image, segmentation, n_threads, resolution, table=table, feature_set=feature_set,
        median_only=median_only, dilation=dilation, background_mask=background_mask,
    )
    measures.to_csv(output_table_path, sep="\t", index=False)


# Refactor to elf?
class ResizedVolumeLocalMean(WrapperBase):
    def __init__(self, volume, factors):
        super().__init__(volume)
        self._scale = factors
        self._shape = tuple(int(np.ceil(s / f)) for s, f in zip(volume.shape, self._scale))

    @property
    def shape(self):
        return self._shape

    @property
    def scale(self):
        return self._scale

    def __getitem__(self, key):
        index, to_squeeze = normalize_index(key, self.shape)
        index = tuple(slice(s.start * f, s.stop * f) for s, f in zip(index, self._scale))
        out = self.volume[index]
        out = downscale_local_mean(out, self._scale)
        return squeeze_singletons(out, to_squeeze)


def compute_sgn_background_mask(
    image_path: str,
    segmentation_path: str,
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    threshold_percentile: int = 35,
    scale_factor: Tuple[int, int, int] = (16, 16, 16),
    n_threads: Optional[int] = None,
    cache_path: Optional[str] = None,
) -> np.typing.ArrayLike:
    """Compute the background mask for intensity measurements in the SGN segmentation.

    This function computes a mask for determining the background signal in the rosenthal canal.
    It is computed by downsampling the image (PV) and segmentation (SGNs) internally,
    by thresholding the downsampled image, and by then intersecting this mask with the segmentation.
    This results in a mask that is positive for the background signal within the rosenthal canal.

    Args:
        image_path: The path to the image data with the PV channel.
        segmentation_path: The path to the SGN segmentation.
        image_key: Internal path for the image data, for zarr or similar file formats.
        segmentation_key: Internal path for the segmentation data, for zarr or similar file formats.
        threshold_percentile: The percentile threshold for separating foreground and background in the PV signal.
        scale_factor: The scale factor for internally downsampling the mask.
        n_threads: The number of threads for parallelizing the computation.
        cache_path: Optional path to save the downscaled background mask to zarr.

    Returns:
        The mask for determining the background values.
    """
    image = read_image_data(image_path, image_key)
    segmentation = read_image_data(segmentation_path, segmentation_key)
    assert image.shape == segmentation.shape

    if cache_path is not None and os.path.exists(cache_path):
        with open_file(cache_path, "r") as f:
            if "mask" in f:
                low_res_mask = f["mask"][:]
                mask = ResizedVolume(low_res_mask, shape=image.shape, order=0)
                return mask

    original_shape = image.shape
    downsampled_shape = tuple(int(np.ceil(sh / sf)) for sh, sf in zip(original_shape, scale_factor))

    low_res_mask = np.zeros(downsampled_shape, dtype="bool")

    # This corresponds to a block shape of 128 x 512 x 512 in the original resolution,
    # which roughly corresponds to the size of the blocks we use for the GFP annotation.
    chunk_shape = (8, 32, 32)

    blocks = blocking((0, 0, 0), downsampled_shape, chunk_shape)
    n_blocks = blocks.numberOfBlocks

    img_resized = ResizedVolumeLocalMean(image, scale_factor)
    seg_resized = ResizedVolume(segmentation, downsampled_shape, order=0)

    def _compute_block(block_id):
        block = blocks.getBlock(block_id)
        bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))

        img = img_resized[bb]
        threshold = np.percentile(img, threshold_percentile)

        this_mask = img > threshold
        this_seg = seg_resized[bb] != 0
        this_seg = binary_dilation(this_seg)
        this_mask[this_seg] = 0

        low_res_mask[bb] = this_mask

    n_threads = mp.cpu_count() if n_threads is None else n_threads
    randomized_blocks = np.arange(0, n_blocks)
    np.random.shuffle(randomized_blocks)
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(
            tp.map(_compute_block, randomized_blocks), total=n_blocks, desc="Compute background mask"
        ))

    if cache_path is not None:
        with open_file(cache_path, "a") as f:
            f.create_dataset("mask", data=low_res_mask, chunks=(64, 64, 64))

    mask = ResizedVolume(low_res_mask, shape=original_shape, order=0)
    return mask
