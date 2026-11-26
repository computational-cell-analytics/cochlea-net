"""Export image data or a segmentation at a lower resolution, optionally cropped and masked to the cochlea.
"""
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from elf.parallel import isin

from flamingo_tools.cpu_utils import available_cpus
from flamingo_tools.export_data_utils import (
    cochlea_table, crop_filter_volume, export_bounding_box, export_name, export_output_path,
    filter_volume_downscale_factors, normalize_scales, normalize_voxel_size, write_volume,
)
from flamingo_tools.file_utils import is_label_dtype, open_scale_level
from flamingo_tools.postprocessing.label_components import filter_cochlea_volume, filter_cochlea_volume_single
from flamingo_tools.s3_utils import MOBIE_FOLDER, default_image_path

DEFAULT_CHANNELS = ("PV", "Vglut3", "CTBP2")

# The cochlea mode clips the bright outliers of the PV and CTBP2 stains, so that an overview image shows the
# structure. The PV stain of the LaVision cochleae has no such outliers.
CLIP_INTENSITY = 1400


def default_max_intensity(cochlea: str, channel: str) -> Optional[float]:
    """Get the intensity clipping of the cochlea mode for a channel.

    Args:
        cochlea: Name of the cochlea.
        channel: Name of the channel.

    Returns:
        `CLIP_INTENSITY` for PV (except for LaVision cochleae) and CTBP2, None for all other channels.
    """
    if "CTBP2" in channel or ("PV" in channel and "LaVision" not in cochlea):
        return CLIP_INTENSITY
    return None


def filter_label_ids(segmentation: np.ndarray, table: pd.DataFrame, components: Sequence[int]) -> np.ndarray:
    """Set every label ID that is not in one of the components to 0.

    Args:
        segmentation: The segmentation, modified in place. Use the integer data type of the segmentation:
            float32 cannot represent a label ID above 2^24.
        table: Segmentation table with the columns "label_id" and "component_labels".
        components: The component labels to keep.

    Returns:
        The filtered segmentation.
    """
    keep_ids = table.label_id.values[np.isin(table.component_labels.values, components)]
    keep_ids = keep_ids.astype(segmentation.dtype)
    outside = np.zeros(segmentation.shape, dtype="bool")
    outside = ~isin(
        segmentation, keep_ids, out=outside, verbose=True, block_shape=(128, 128, 128), n_threads=available_cpus()
    )
    segmentation[outside] = 0
    return segmentation


def cochlea_mask(
    voxel_size: Union[float, Sequence[float]],
    sgn_table: Optional[pd.DataFrame] = None,
    ihc_table: Optional[pd.DataFrame] = None,
    sgn_components: Optional[List[int]] = [1],
    ihc_components: Optional[List[int]] = [1],
    dilation_iterations: int = 8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a down-sampled mask of the cochlea from the SGN and / or the IHC segmentation table.

    One table gives a dilated mask around its objects. Both tables give a mask that also covers the region
    between the SGNs and the IHCs.

    Args:
        voxel_size: Voxel size of the data in µm, scalar or (x, y, z).
        sgn_table: SGN segmentation table.
        ihc_table: IHC segmentation table.
        sgn_components: Component labels of the SGN table that the mask covers.
        ihc_components: Component labels of the IHC table that the mask covers.
        dilation_iterations: Iterations for the dilation of the mask.

    Returns:
        mask: The down-sampled binary mask, ZYX.
        cell_size: The size of one mask cell in voxels at full resolution, ZYX.
    """
    if sgn_table is None and ihc_table is None:
        raise ValueError("The cochlea mask requires an SGN or an IHC segmentation table.")

    voxel_size = normalize_voxel_size(voxel_size)
    ds_factor = filter_volume_downscale_factors(voxel_size)
    kwargs = dict(scale_factor=ds_factor, voxel_size=voxel_size, dilation_iterations=dilation_iterations)
    if sgn_table is None:
        mask = filter_cochlea_volume_single(ihc_table, components=ihc_components, **kwargs)
    elif ihc_table is None:
        mask = filter_cochlea_volume_single(sgn_table, components=sgn_components, **kwargs)
    else:
        mask = filter_cochlea_volume(
            sgn_table, ihc_table, sgn_components=sgn_components, ihc_components=ihc_components, **kwargs
        )
    return np.transpose(mask, (2, 1, 0)), np.array(ds_factor[::-1], dtype="float64")


def export_lower_resolution(
    input_path: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    input_key: Optional[str] = "s0",
    name: Optional[str] = None,
    table: Optional[pd.DataFrame] = None,
    filter_by_components: Optional[List[int]] = None,
    sgn_table: Optional[pd.DataFrame] = None,
    ihc_table: Optional[pd.DataFrame] = None,
    filter_sgn_components: Optional[List[int]] = [1],
    filter_ihc_components: Optional[List[int]] = [1],
    filter_dilation_iterations: int = 8,
    max_intensity: Optional[float] = None,
    labels: Optional[bool] = None,
    binarize: bool = False,
    ome_zarr: bool = False,
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    voxel_size: Union[float, Sequence[float]] = (0.38, 0.38, 0.38),
    force: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> None:
    """Export image data or a segmentation at one or more lower resolutions.

    The output is <output_folder>/scale<s>/<name><crop suffix>.tif for each scale s, or .ome.zarr with `ome_zarr`.
    With a cochlea mask, the folder is scale<s>_dilation<filter_dilation_iterations>. The output data type is
    float32, which holds label IDs up to 2^24 exactly.

    Args:
        input_path: Path to the data: a tif file, or a zarr/n5 container. Relative to the bucket if s3 is set.
        output_folder: Output folder.
        scale: Scale level(s). Scale s has 2^s times the voxel size of the input, see
            `flamingo_tools.file_utils.open_scale_level`.
        input_key: Key of the input in the container, e.g. "s0" or "setup0/timepoint0/s0". Ignored for a tif file.
        name: Name of the export. Default: the file name of the input without its extension.
        table: Segmentation table of the input, for `filter_by_components`.
        filter_by_components: Keep only the objects of these components of `table`. For a segmentation.
        sgn_table: SGN segmentation table, for the cochlea mask, see `cochlea_mask`.
        ihc_table: IHC segmentation table, for the cochlea mask, see `cochlea_mask`.
        filter_sgn_components: Component labels of `sgn_table` that the cochlea mask covers.
        filter_ihc_components: Component labels of `ihc_table` that the cochlea mask covers.
        filter_dilation_iterations: Iterations for the dilation of the cochlea mask.
        max_intensity: Clip the intensities of image data at this value. A segmentation is not clipped.
        labels: Whether the input is a segmentation. A segmentation is downsampled by nearest neighbor
            sampling instead of the mean. Default: True with `filter_by_components`, otherwise True for
            32 or 64 bit integer data.
        binarize: Write 1 for every non-zero voxel.
        ome_zarr: Write the data into a zarr container instead of a TIF file.
        crop_center: Optional crop center as [x, y, z] in µm. Requires roi_halo, unless axis is given.
        roi_halo: Halo around the crop center as [halo_x, halo_y, halo_z] in pixels at the target scale.
            Optional when axis is given: the whole plane is cropped in that case.
        axis: Optional axis index into (x, y, z) to crop as a single-pixel 2D slice at the crop center.
        suffix: Optional label appended to the name of a crop, e.g. a position name such as "apex".
        voxel_size: Voxel size of the input in µm, scalar or (x, y, z).
        force: Replace an existing output. Otherwise, an existing output is kept.
        s3: Read the input from the S3 bucket.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    if filter_by_components is not None and table is None:
        raise ValueError("filter_by_components requires the segmentation table.")
    voxel_size = normalize_voxel_size(voxel_size)
    name = export_name(input_path) if name is None else name
    if labels is None and filter_by_components is not None:
        labels = True

    mask = None
    if sgn_table is not None or ihc_table is not None:
        mask, mask_cell_size = cochlea_mask(
            voxel_size, sgn_table=sgn_table, ihc_table=ihc_table, sgn_components=filter_sgn_components,
            ihc_components=filter_ihc_components, dilation_iterations=filter_dilation_iterations,
        )

    for s in normalize_scales(scale):
        folder_name = f"scale{s}" if mask is None else f"scale{s}_dilation{filter_dilation_iterations}"
        out_folder = os.path.join(output_folder, folder_name)
        out_path = export_output_path(out_folder, name, ome_zarr, crop_center, axis, suffix)
        if os.path.exists(out_path) and not force:
            print(f"Skipping {out_path}. File already exists.")
            continue

        print(f"Exporting {name} at scale {s}.")
        level = open_scale_level(
            input_path, input_key, s, from_s3=s3, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint,
            credential_file=s3_credentials,
        )
        start, stop = export_bounding_box(level.shape, crop_center, roi_halo, voxel_size, s, axis)
        is_labels = is_label_dtype(level.dtype) if labels is None else labels
        data = level.read(start, stop, labels=is_labels)
        print("Data shape", data.shape)

        if filter_by_components is not None:
            print(f"Filtering {name} by components {filter_by_components}.")
            data = filter_label_ids(data, table, filter_by_components)
        if mask is not None:
            data[crop_filter_volume(mask, start, stop, mask_cell_size / 2 ** s) == 0] = 0
        if max_intensity is not None and not is_labels:
            data[data > max_intensity] = max_intensity
        if binarize:
            data = data > 0

        os.makedirs(out_folder, exist_ok=True)
        write_volume(out_path, data.astype("float32", copy=False), voxel_size, s, ome_zarr=ome_zarr)


def export_lower_resolution_cochlea(
    cochlea: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    channels: Sequence[str] = DEFAULT_CHANNELS,
    filter_by_components: Optional[List[int]] = None,
    filter_cochlea_channels: Optional[List[str]] = None,
    max_intensity: Optional[float] = None,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **kwargs,
) -> None:
    """Export channels of a cochlea in a MoBIE project at one or more lower resolutions.

    The data is read from <cochlea>/images/ome-zarr/<channel>.ome.zarr and the tables from
    <cochlea>/tables/<source>/default.tsv. The output is written to <output_folder>/<cochlea>, see
    `export_lower_resolution`.

    Args:
        cochlea: Name of the cochlea.
        output_folder: Output folder.
        scale: Scale level(s).
        channels: Names of the image or segmentation channels.
        filter_by_components: Keep only the objects of these components of each channel's own table.
        filter_cochlea_channels: Segmentation(s) for the cochlea mask. A name with "SGN" gives the SGN table,
            a name with "IHC" the IHC table.
        max_intensity: Clip the intensities of image data at this value. Default: `default_max_intensity`.
        mobie_dir: Local MoBIE project directory. Ignored when s3 is set.
        s3: Read the data from the S3 bucket instead of the local MoBIE project.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        kwargs: Further arguments of `export_lower_resolution`, e.g. crop_center, roi_halo, axis, suffix,
            voxel_size, filter_sgn_components, filter_ihc_components, filter_dilation_iterations, binarize,
            ome_zarr or force.
    """
    s3_kwargs = dict(
        s3=s3, s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name, s3_service_endpoint=s3_service_endpoint,
    )

    sgn_table, ihc_table = None, None
    if filter_cochlea_channels is not None:
        sgn_channels = [channel for channel in filter_cochlea_channels if "SGN" in channel]
        ihc_channels = [channel for channel in filter_cochlea_channels if "IHC" in channel]
        if not sgn_channels and not ihc_channels:
            raise ValueError(
                f"The channels for the cochlea mask, {filter_cochlea_channels}, contain no IHC or SGN segmentation."
            )
        if sgn_channels:
            sgn_table = cochlea_table(cochlea, sgn_channels[0], mobie_dir=mobie_dir, **s3_kwargs)
        if ihc_channels:
            ihc_table = cochlea_table(cochlea, ihc_channels[0], mobie_dir=mobie_dir, **s3_kwargs)

    for channel in channels:
        table = None
        if filter_by_components is not None:
            table = cochlea_table(cochlea, channel, mobie_dir=mobie_dir, **s3_kwargs)
        export_lower_resolution(
            default_image_path(cochlea, channel, s3=s3, mobie_dir=mobie_dir),
            os.path.join(output_folder, cochlea),
            scale,
            input_key="s0",
            name=channel,
            table=table,
            filter_by_components=filter_by_components,
            sgn_table=sgn_table,
            ihc_table=ihc_table,
            max_intensity=default_max_intensity(cochlea, channel) if max_intensity is None else max_intensity,
            **s3_kwargs,
            **kwargs,
        )
