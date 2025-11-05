import warnings
from typing import Optional, Union

import imageio.v3 as imageio
import numpy as np
import tifffile
import zarr
from elf.io import open_file

from .s3_utils import get_s3_path

try:
    from zarr.abc.store import Store
except ImportError:
    from zarr._storage.store import BaseStore as Store


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
    input_path: Union[str, Store], input_key: Optional[str], from_s3: bool = False
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
        s3_store, fs = get_s3_path(input_path)
        return zarr.open(s3_store, mode="r")[input_key]

    if input_key is None:
        input_ = read_tif(input_path)
    elif isinstance(input_path, str):
        input_ = open_file(input_path, "r")[input_key]
    else:
        f = zarr.open(input_path, mode="r")
        input_ = f[input_key]
    return input_
