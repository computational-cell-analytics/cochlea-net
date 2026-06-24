"""Utilities for reading image data.
"""

import os
import warnings
from typing import Dict, List, Optional, Union

import imageio.v3 as imageio
import numpy as np
import pooch
import tifffile
import zarr
from elf.io import open_file
from pybdv.util import relative_to_absolute_scale_factors

from .s3_utils import get_s3_path

try:
    from zarr.abc.store import Store
except ImportError:
    from zarr._storage.store import BaseStore as Store

AXES_TYPE_DICT = {
    "x": "space",
    "y": "space",
    "z": "space",
    "t": "time",
    "c": "channel"
}


def get_cache_dir() -> str:
    """Get the cache directory of CochleaNet.

    The default cache directory is "$HOME/cochlea-net"

    Returns:
        The cache directory.
    """
    cache_dir = os.path.expanduser(pooch.os_cache("cochlea-net"))
    return cache_dir


def _parse_shape(metadata_file):
    depth, height, width = None, None, None

    with open(metadata_file, "r") as f:
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            if line.startswith("AOI width"):
                width = int(line.split(" ")[-1])
            if line.startswith("AOI height"):
                height = int(line.split(" ")[-1])
            if line.startswith("Number of planes saved"):
                depth = int(line.split(" ")[-1])

    assert depth is not None
    assert height is not None
    assert width is not None
    return (depth, height, width)


def read_raw(file_path: str, metadata_file: str) -> np.memmap:
    """Read a raw file written by the flamingo microscope.

    Args:
        file_path: The file path to the raw file.
        metadata_file: The file path to the metadata describing the raw file.
            The metadata will be used to determine the shape of the data.

    Returns:
        The memory-mapped data.
    """
    shape = _parse_shape(metadata_file)
    return np.memmap(file_path, mode="r", dtype="uint16", shape=shape)


def read_tif(file_path: str) -> Union[np.ndarray, np.memmap]:
    """Read a tif file.

    Tries to memory map the file. If not possible will load the complete file into memory
    and raise a warning.

    Args:
        file_path: The file path to the tif file.

    Returns:
        The memory-mapped data. If not possible to memmap, the data in memory.
    """
    try:
        x = tifffile.memmap(file_path)
    except ValueError:
        warnings.warn(f"Cannot memmap the tif file at {file_path}. Fall back to loading it into memory.")
        x = imageio.imread(file_path)
    return x


def read_image_data(
    input_path: Union[str, Store],
    input_key: Optional[str],
    from_s3: bool = False,
    bucket_name: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    credential_file: Optional[str] = None,
) -> np.typing.ArrayLike:
    """Read flamingo image data, stored in various formats.

    Args:
        input_path: The file path to the data, or a zarr S3 store for data remotely accessed on S3.
            The data can be stored as a tif file, or a zarr/n5 container.
            Access via S3 is only supported for a zarr container.
        input_key: The key (= internal path) for a zarr or n5 container.
            Set it to None if the data is stored in a tif file.
        from_s3: Whether to read the data from S3.

    Returns:
        The data, loaded either as a numpy mem-map, a numpy array, or a zarr / n5 array.
    """
    if from_s3:
        assert input_key is not None
        s3_store, fs = get_s3_path(input_path, bucket_name, service_endpoint, credential_file)
        return zarr.open(s3_store, mode="r")[input_key]

    if input_key is None:
        input_ = read_tif(input_path)
    elif isinstance(input_path, str):
        input_ = open_file(input_path, "r")[input_key]
    else:
        f = zarr.open(input_path, mode="r")
        input_ = f[input_key]
    return input_


def _create_ngff_metadata(g, name, axes_names, scales=None, units=None):
    # axes metadata
    axes = [
        {"name": name, "type": AXES_TYPE_DICT[name]} for name in axes_names
    ]
    if units is not None:
        assert len(units) == len(axes_names)
        for ax, unit in zip(axes, units):
            if unit is not None:
                ax["unit"] = unit

    # dataset metadata including transformations
    n_scales = len(g)
    if scales is None:
        scales = [[1.0] * len(axes_names)] * n_scales
    assert len(scales) == n_scales
    assert all(len(scale) == len(axes_names) for scale in scales)

    # NOTE we might need a half pixel offset for proper scale alignment here (via a translation)
    transforms = [[{"type": "scale", "scale": scale}] for scale in scales]
    datasets = [
        {"path": f"s{level}", "coordinateTransformations": trafo} for level, trafo in enumerate(transforms)
    ]
    assert all(ds["path"] in g for ds in datasets)

    ms_entry = {
        "axes": axes,
        "datasets": datasets,
        "name": name,
        "version": "0.4"
    }

    metadata = g.attrs.get("multiscales", [])
    metadata.append(ms_entry)
    g.attrs["multiscales"] = metadata


def write_ome_zarr_metadata(
    path: str,
    metadata_dict: Dict,
    scale_factors: List[List[int]],
    prefix: str = "",
) -> None:
    """Write the multicales metadata for ome.zarr v0.4.

    Args:
        path: The file path to the zarr container.
        metadata_dict: Dictionary with additional metadata.
        scale_factors: The scale factors used for downsampling the multi-resolution image pyramid.
        prefix: The prefix for the location of the multiscales group containing the image data.
            By default we assume that the image is in the root group.
    """
    setup_name = metadata_dict.get("setup_name", None)
    setup_name = "data" if setup_name is None else setup_name
    unit = metadata_dict.get("unit", "pixel")
    scale_factors = [[1, 1, 1]] + list(scale_factors)
    scale_factors = relative_to_absolute_scale_factors(scale_factors)

    with open_file(path, mode="a") as f:
        g = f if prefix == "" else f[prefix]
        ndim = g["s0"].ndim
        axes_names = ["y", "x"] if ndim == 2 else ["z", "y", "x"]
        resolution = metadata_dict.get("resolution", [1.] * ndim)
        scales = [[sc * res for sc, res in zip(scale, resolution)] for scale in scale_factors]
        units = ndim * [unit]
        _create_ngff_metadata(g, setup_name, axes_names, units=units, scales=scales)
