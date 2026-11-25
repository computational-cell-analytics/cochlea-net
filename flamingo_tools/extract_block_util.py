"""@private
"""

import json
import os
from typing import Dict, Optional, List, Sequence, Union, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import zarr
from skimage.transform import rescale

from flamingo_tools.file_utils import read_image_data
from flamingo_tools.s3_utils import MOBIE_FOLDER, default_image_path, default_table_path, read_table
from flamingo_tools.analysis.density_utils import hull_to_mask


def _crop_bounds(
    coords: List[float],
    roi_halo: List[int],
    voxel_size: Union[float, Tuple[float, float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Get the unclipped ZYX begin and end voxels of the crop around a center in micrometer, (x, y, z)."""
    voxel_size = np.broadcast_to(np.asarray(voxel_size, dtype="float64"), (3,))
    center = np.round(np.asarray(coords, dtype="float64")[::-1] / voxel_size[::-1]).astype(int)
    halo = np.asarray(roi_halo, dtype=int)[::-1]
    return center - halo, center + halo


def _crop_roi(
    coords: List[float],
    roi_halo: List[int],
    voxel_size: Union[float, Tuple[float, float, float]],
    shape: Tuple[int, int, int],
) -> Tuple[slice, slice, slice]:
    """Get the ZYX crop slices around a center, clipped to the volume.

    Args:
        coords: The crop center in micrometer, (x, y, z).
        roi_halo: The halo around the center in voxels, (x, y, z).
        voxel_size: The voxel size in micrometer, scalar or (x, y, z).
        shape: The shape of the volume, ZYX.
    """
    full_begin, full_end = _crop_bounds(coords, roi_halo, voxel_size)
    begin = np.maximum(full_begin, 0)
    end = np.minimum(full_end, np.asarray(shape))
    if np.any(begin >= end):
        raise ValueError(f"The crop around {coords} lies outside of the volume of shape {tuple(shape)}.")
    if np.any(begin != full_begin) or np.any(end != full_end):
        clipped_shape = tuple(int(e - b) for b, e in zip(begin, end))
        print(f"Clipped the crop around {coords} to the volume of shape {tuple(shape)}: {clipped_shape}.")
    return tuple(slice(int(b), int(e)) for b, e in zip(begin, end))


def _crop_file_name(coords: List[float], dataset_name: Optional[str], suffix: Optional[str], extension: str) -> str:
    coord_string = "-".join([str(int(round(c))).zfill(4) for c in coords])
    prefix = "" if dataset_name is None else f"{dataset_name.replace('_', '-')}_"
    suffix = "" if suffix is None else f"_{suffix.replace('_', '-')}"
    return f"{prefix}crop_{coord_string}{suffix}{extension}"


def extract_block_single(
    input_path: str,
    coords: List[float],
    output_path: str,
    dataset_name: Optional[str] = None,
    channel_name: Optional[str] = None,
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
    roi_halo: List[int] = [128, 128, 64],
    s3: Optional[bool] = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    scale_factor: Optional[Tuple[float, float, float]] = None,
    force_overwrite: bool = False,
    label_ids: Optional[List[int]] = None,
    **_,
) -> None:
    """Extract block around coordinate from input data according to a given halo.
    Either from a local file or from an S3 bucket.
    A block at the border of the volume is clipped to the volume. An existing output is kept
    unless `force_overwrite` is set.

    Args:
        input_path: Input folder in n5 / ome-zarr format.
        coords: The block center in micrometer, (x, y, z).
        output_path: Output directory or file. In a directory, the crop is named after the dataset, the center and
            the channel.
        input_key: Input key for data in input file.
        output_key: Output key for data in n5 format. If None is supplied, output is TIF file.
        voxel_size: The voxel size of the data in micrometer.
        roi_halo: The halo around the block center in voxels, (x, y, z).
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        scale_factor: Optional factor for rescaling the extracted data.
        force_overwrite: Flag for forcefully overwriting output files.
    """
    if os.path.isdir(output_path):
        extension = ".tif" if output_key is None else ".n5"
        output_path = os.path.join(output_path, _crop_file_name(coords, dataset_name, channel_name, extension))

    if os.path.exists(output_path) and not force_overwrite:
        print(f"Skipping block extraction because {output_path} already exists.")
        return

    data_ = read_image_data(input_path, input_key, from_s3=s3,
                            bucket_name=s3_bucket_name,
                            service_endpoint=s3_service_endpoint,
                            credential_file=s3_credentials)
    roi = _crop_roi(coords, roi_halo, voxel_size, data_.shape)

    data_roi = data_[roi]
    if scale_factor is not None:
        kwargs = {"preserve_range": True}
        # Check if this is a segmentation.
        if data_roi.dtype in (np.dtype("int32"), np.dtype("uint32"), np.dtype("int64"), np.dtype("uint64")):
            kwargs.update({"order": 0, "anti_aliasing": False})
        data_roi = rescale(data_roi, scale_factor, **kwargs).astype(data_roi.dtype)

    if output_key is None:
        imageio.imwrite(output_path, data_roi, compression="zlib")
    else:
        f_out = zarr.open(output_path, mode="w")
        f_out.create_array(output_key, data=data_roi, compressors=zarr.codecs.GzipCodec())

    # filter out segmentation not corresponding to label IDs
    if label_ids is not None:
        # check for segmentation
        if data_roi.dtype in (np.dtype("int32"), np.dtype("uint32"), np.dtype("int64"), np.dtype("uint64")):
            print("Filtering segmentation with label IDs.")
            filter_mask = ~np.isin(data_roi, label_ids)
            if np.count_nonzero(data_roi[filter_mask]) != 0:
                data_roi[filter_mask] = 0
                output_path = os.path.splitext(output_path)[0] + "_filtered" + os.path.splitext(output_path)[1]
                imageio.imwrite(output_path, data_roi, compression="zlib")


def extract_density_crops(
    block_list: List[dict],
    output_path: str,
    seg_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    img_paths: Optional[List[str]] = None,
    input_key: str = "s0",
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    force_overwrite: bool = False,
) -> None:
    """Extract image crops for each computed SGN density position.

    Crops the segmentation (via `seg_path`) and any additional volumes (`img_paths`) at the crop
    center and ROI halo of each entry in `block_list`, as produced by
    `flamingo_tools.analysis.density_utils._build_block_extraction_dict`. Output file names follow
    the `extract_block_single` naming scheme.

    Args:
        block_list: List of per-position block-extraction dicts with 'crop_centers', 'roi_halo',
                    and optionally 'dataset_name' / 'segmentation_channel'.
        output_path: Output directory for the extracted crops.
        seg_path: Path to the segmentation volume to crop. None skips the segmentation crop.
        dataset_name: Dataset name used as filename prefix for entries that have none.
        img_paths: Additional volume path(s) to crop alongside the segmentation.
        input_key: Internal key for N5/Zarr/OME-ZARR data.
        voxel_size: Voxel size in µm.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        force_overwrite: Flag for forcefully overwriting output files.
    """
    os.makedirs(output_path, exist_ok=True)
    img_paths = img_paths or []

    for entry in block_list:
        center = list(entry["crop_centers"][0])
        roi_halo = list(entry["roi_halo"])
        entry_dataset_name = entry.get("dataset_name", dataset_name)

        sources = []
        if seg_path is not None:
            seg_channel = entry.get("segmentation_channel")
            channel_name = (
                seg_channel if seg_channel is not None
                else os.path.basename(seg_path.rstrip("/")).split(".")[0]
            )
            sources.append((seg_path, channel_name))
        for img_path in img_paths:
            channel_name = os.path.basename(img_path.rstrip("/")).split(".")[0]
            sources.append((img_path, channel_name))

        for input_path, channel_name in sources:
            extract_block_single(
                input_path=input_path,
                coords=center,
                output_path=output_path,
                dataset_name=entry_dataset_name,
                channel_name=channel_name,
                input_key=input_key,
                voxel_size=voxel_size,
                roi_halo=roi_halo,
                s3=s3,
                s3_credentials=s3_credentials,
                s3_bucket_name=s3_bucket_name,
                s3_service_endpoint=s3_service_endpoint,
                force_overwrite=force_overwrite,
            )


def save_mask_from_hull_vertices(
    coords: List[int],
    hull_vertices: List[List[float]],
    output_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
    roi_halo: List[int] = [128, 128, 64],
    shape: Optional[Tuple[int, int, int]] = None,
    **_,
) -> None:
    """Save the density hull as a mask TIF that is aligned with the crops of `extract_block_single`.

    Args:
        coords: The crop center in micrometer, (x, y, z).
        hull_vertices: The hull vertices in micrometer. The first entry is converted with `hull_to_mask`.
        output_path: The output directory.
        dataset_name: The dataset name, used as the file name prefix.
        voxel_size: The voxel size in micrometer, (x, y, z).
        roi_halo: The halo around the crop center in voxels, (x, y, z).
        shape: The ZYX shape of the volume. If given, the mask is clipped like a crop at the border.
    """
    density_mask = hull_to_mask(
        hull_vertices[0],
        bb_center=coords,
        roi_halo=roi_halo,
        voxel_size=voxel_size,
    )
    if shape is not None:
        full_begin, _ = _crop_bounds(coords, roi_halo, voxel_size)
        roi = _crop_roi(coords, roi_halo, voxel_size, shape)
        density_mask = density_mask[tuple(slice(r.start - b, r.stop - b) for r, b in zip(roi, full_begin))]
    output_path = os.path.join(output_path, _crop_file_name(coords, dataset_name, "hull", ".tif"))
    imageio.imwrite(output_path, density_mask, compression="zlib")


def extract_points_single(
    points: pd.DataFrame,
    coords: List[float],
    output_path: str,
    shape: Tuple[int, int, int],
    name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    voxel_size: Union[float, Tuple[float, float, float]] = 0.38,
    roi_halo: List[int] = [128, 128, 64],
    force_overwrite: bool = False,
    **_,
) -> str:
    """Write the points inside a crop as a napari points file.

    The CSV file has the columns 'index', 'axis-0', 'axis-1', 'axis-2' with the ZYX position in voxels of the
    crop, as written by `extract_block_single`, and the other columns of the point table as point properties.

    Args:
        points: The point table, with the positions in micrometer in the columns 'x', 'y', 'z'.
        coords: The crop center in micrometer, (x, y, z).
        output_path: The output directory.
        shape: The ZYX shape of the volume. A crop at the border is clipped as in `extract_block_single`.
        name: The name of the point table, used as the file name suffix.
        dataset_name: The dataset name, used as the file name prefix.
        voxel_size: The voxel size in micrometer, scalar or (x, y, z).
        roi_halo: The halo around the crop center in voxels, (x, y, z).
        force_overwrite: Overwrite an existing file.

    Returns:
        The path of the CSV file.
    """
    output_file = os.path.join(output_path, _crop_file_name(coords, dataset_name, name, ".csv"))
    if os.path.exists(output_file) and not force_overwrite:
        print(f"Skipping point extraction because {output_file} already exists.")
        return output_file

    roi = _crop_roi(coords, roi_halo, voxel_size, shape)
    begin = np.array([r.start for r in roi])
    end = np.array([r.stop for r in roi])
    voxel_size = np.broadcast_to(np.asarray(voxel_size, dtype="float64"), (3,))
    positions = points[["z", "y", "x"]].to_numpy(dtype="float64") / voxel_size[::-1]
    voxels = np.round(positions)
    inside = np.all((voxels >= begin) & (voxels < end), axis=1)
    local = positions[inside] - begin

    table = pd.DataFrame({
        "index": np.arange(len(local)), "axis-0": local[:, 0], "axis-1": local[:, 1], "axis-2": local[:, 2],
    })
    properties = points[inside].drop(columns=["x", "y", "z"]).reset_index(drop=True)
    table = pd.concat([table, properties.drop(columns=["index"], errors="ignore")], axis=1)
    table.to_csv(output_file, index=False)
    return output_file


def _volume_shape(input_path: str, s3: bool, input_key: Optional[str] = None, s3_credentials: Optional[str] = None,
                  s3_bucket_name: Optional[str] = None, s3_service_endpoint: Optional[str] = None,
                  **_) -> Tuple[int, ...]:
    data = read_image_data(input_path, input_key, from_s3=s3, bucket_name=s3_bucket_name,
                           service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
    return tuple(data.shape)


def _read_point_sources(
    param_dicts: List[dict],
    point_sources: Sequence[str],
    s3: bool,
    mobie_dir: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **_,
) -> Dict[Tuple[str, str], pd.DataFrame]:
    """Read the point table of each source name for each cochlea of the entries, keyed by (cochlea, name)."""
    tables = {}
    for params in param_dicts:
        cochlea = params["dataset_name"]
        for name in point_sources:
            if (cochlea, name) not in tables:
                tables[(cochlea, name)] = read_table(
                    default_table_path(cochlea, name, s3=s3, mobie_dir=mobie_dir), s3, bucket_name=s3_bucket_name,
                    service_endpoint=s3_service_endpoint, credential_file=s3_credentials,
                )
    return tables


def _check_point_columns(tables: Dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        missing = [axis for axis in ("x", "y", "z") if axis not in table.columns]
        if missing:
            raise ValueError(f"The point table {name} has no columns {missing}. "
                             "A point table needs 'x', 'y', 'z' in micrometer, e.g. a synapse table.")


def extract_block_json_wrapper(
    output_path: str,
    input_path: Optional[str] = None,
    json_file: Optional[str] = None,
    coords: Optional[List[float]] = None,
    mobie_dir: str = MOBIE_FOLDER,
    force: bool = False,
    s3: Optional[bool] = False,
    point_tables: Optional[Dict[str, pd.DataFrame]] = None,
    point_sources: Optional[Sequence[str]] = None,
    **kwargs,
):
    """Extract a block for each crop center and image channel of a JSON file, or one block around `coords`.

    Args:
        output_path: Output directory, or output file for one block.
        input_path: Image data for one block. Used without json_file.
        json_file: JSON file with one block parameter dictionary, or a list of them. Each entry needs
            'crop_centers'. The image data path comes from its 'dataset_name' and 'image_channel'.
        coords: The block center in micrometer, (x, y, z). Used without json_file.
        mobie_dir: Local MoBIE directory used for creating data paths when a JSON dict is provided.
        force: Flag for forcefully overwriting output files.
        s3: Flag for accessing data stored on S3 bucket.
        point_tables: Point tables with 'x', 'y', 'z' in micrometer, keyed by name. The points of each crop are
            saved as a napari points file, see `extract_points_single`.
        point_sources: Names of point tables in the MoBIE project, e.g. 'synapse_v11_ihc_v11b'. Each entry uses
            the table <cochlea>/tables/<name>/default.tsv of its cochlea, from the S3 bucket or from mobie_dir.
            The points are saved like the points of point_tables. Requires json_file.
        kwargs: Further arguments for `extract_block_single`. An entry of the JSON file overrides them.
    """
    point_tables = point_tables or {}
    point_sources = list(point_sources or [])
    shared_names = set(point_tables) & set(point_sources)
    if shared_names:
        raise ValueError(f"The names {sorted(shared_names)} are in point_tables and in point_sources.")
    _check_point_columns(point_tables)

    if json_file is not None:
        with open(json_file, "r") as f:
            param_dicts = json.loads(f.read())
        if not isinstance(param_dicts, list):
            param_dicts = [param_dicts]
        for index, params in enumerate(param_dicts):
            if "crop_centers" not in params:
                raise ValueError(
                    f"Entry {index} ({params.get('dataset_name')}) of {json_file} has no 'crop_centers'. "
                    "Find them with flamingo_tools.extract_blocks --find_centers."
                )
        if point_tables and len({params["dataset_name"] for params in param_dicts}) > 1:
            raise ValueError(
                "Local point tables belong to one cochlea, but the JSON file has entries for several cochleae. "
                "Give the name of a MoBIE table instead. Each entry reads it for its own cochlea."
            )
        source_tables = _read_point_sources(param_dicts, point_sources, s3, mobie_dir, **kwargs)
        _check_point_columns({f"{name} of {cochlea}": table for (cochlea, name), table in source_tables.items()})
        os.makedirs(output_path, exist_ok=True)
        for params in param_dicts:
            cochlea = params["dataset_name"]
            if isinstance(params["image_channel"], list):
                image_channels = params["image_channel"]
            else:
                image_channels = [params["image_channel"]]
            # The keys of one entry, e.g. 'roi_halo', must not carry over to the next entry.
            entry_kwargs = {**kwargs, **params}

            for image_channel in image_channels:
                input_path = default_image_path(cochlea, image_channel, s3=s3, mobie_dir=mobie_dir)

                for coords in params["crop_centers"]:
                    extract_block_single(
                        input_path=input_path,
                        coords=coords,
                        output_path=output_path,
                        channel_name=image_channel,
                        force_overwrite=force,
                        s3=s3,
                        **entry_kwargs,
                    )

            entry_points = {**point_tables, **{name: source_tables[(cochlea, name)] for name in point_sources}}
            if "hull_vertices" in params or entry_points:
                shape = _volume_shape(input_path, s3, **entry_kwargs)

            # crop out mask from hull vertices
            if "hull_vertices" in params:
                for coords in params["crop_centers"]:
                    save_mask_from_hull_vertices(coords=coords, output_path=output_path, shape=shape, **entry_kwargs)

            if entry_points:
                for coords in params["crop_centers"]:
                    for name, points in entry_points.items():
                        extract_points_single(points, coords, output_path, shape, name=name,
                                              force_overwrite=force, **entry_kwargs)

    else:
        if input_path is None or coords is None:
            raise ValueError("An input path to image data and coords are required, if no JSON file is supplied.")
        if point_sources:
            raise ValueError("point_sources requires a JSON file, which names the cochlea of each entry.")
        if point_tables and not os.path.isdir(output_path):
            raise ValueError("Point tables require an output directory.")
        extract_block_single(
            input_path=input_path,
            output_path=output_path,
            channel_name=kwargs.get("image_channel"),
            force_overwrite=force,
            coords=coords,
            s3=s3,
            **kwargs,
        )
        if point_tables:
            shape = _volume_shape(input_path, s3, **kwargs)
            for name, points in point_tables.items():
                extract_points_single(points, coords, output_path, shape, name=name, force_overwrite=force, **kwargs)
