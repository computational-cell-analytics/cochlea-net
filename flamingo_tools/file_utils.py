"""Utilities for reading image data.
"""

import itertools
import math
import os
import re
import warnings
from typing import Optional, Sequence, Tuple, Union

import imageio.v3 as imageio
import numpy as np
import pooch
import tifffile
import zarr
from elf.io import open_file

from .s3_utils import get_s3_path

try:
    from zarr.abc.store import Store
except ImportError:
    from zarr._storage.store import BaseStore as Store


def get_cache_dir() -> str:
    """Get the cache directory of CochleaNet.

    The environment variable COCHLEA_NET_CACHE_DIR sets the cache directory.
    Without it, the cache directory is the "cochlea-net" folder in the cache location of the OS,
    for example "~/.cache/cochlea-net" on Linux.

    Returns:
        The cache directory.
    """
    cache_dir = os.environ.get("COCHLEA_NET_CACHE_DIR") or pooch.os_cache("cochlea-net")
    return os.path.expanduser(cache_dir)


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


def _check_array(node, input_path, input_key):
    """Fail with the cause instead of an AttributeError deeper in the pipeline.

    A dataset directory without valid metadata ('attributes.json' for n5) reads as a group.
    This is the signature of an incomplete copy of the container.
    """
    if hasattr(node, "shape"):
        return node
    children = sorted(node.keys()) if hasattr(node, "keys") else []
    raise ValueError(
        f"The key '{input_key}' in {input_path} is a group, not an array. It contains {children}. "
        "An n5 dataset without a valid 'attributes.json' reads as a group. "
        "Check the container with scripts/check_n5_container.py."
    )


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
        return _check_array(zarr.open(s3_store, mode="r")[input_key], input_path, input_key)

    if input_key is None:
        input_ = read_tif(input_path)
    elif isinstance(input_path, str):
        input_ = _check_array(open_file(input_path, "r")[input_key], input_path, input_key)
    else:
        f = zarr.open(input_path, mode="r")
        input_ = _check_array(f[input_key], input_path, input_key)
    return input_


def pyramid_key(input_key: Optional[str], levels: int) -> Optional[str]:
    """Return the key of the pyramid level `levels` below `input_key`.

    For example, ("s0", 3) gives "s3" and ("setup0/timepoint0/s1", 2) gives "setup0/timepoint0/s3".

    Args:
        input_key: The key of a pyramid level. The last part of the key must be "s<N>".
        levels: The number of levels below `input_key`.

    Returns:
        The key, or None if `input_key` is None or is not the key of a pyramid level.
    """
    if input_key is None:
        return None
    head, _, last = input_key.rstrip("/").rpartition("/")
    match = re.fullmatch(r"s(\d+)", last)
    if match is None:
        return None
    level = f"s{int(match.group(1)) + levels}"
    return f"{head}/{level}" if head else level


def is_label_dtype(dtype: np.typing.DTypeLike) -> bool:
    """Check whether a data type is the data type of a segmentation, i.e. a 32 or 64 bit integer.

    Image data is stored as 8 or 16 bit integers or as floating point numbers.
    """
    dtype = np.dtype(dtype)
    return dtype.kind in "iu" and dtype.itemsize >= 4


def _is_tif(input_path) -> bool:
    return isinstance(input_path, str) and input_path.lower().endswith((".tif", ".tiff"))


# The minimal edge length of a block that is read for the downsampling on the fly, in voxels at full resolution.
_MIN_BLOCK_LENGTH = 256


def _block_mean(block: np.ndarray, factor: int) -> np.ndarray:
    """Average the blocks of `factor`^3 voxels. An edge block is averaged over the voxels inside the volume."""
    out_shape = tuple(-(-length // factor) for length in block.shape)
    padding = [(0, out * factor - length) for out, length in zip(out_shape, block.shape)]
    padded = np.pad(block.astype("float64"), padding)
    sums = padded.reshape(out_shape[0], factor, out_shape[1], factor, out_shape[2], factor).sum(axis=(1, 3, 5))

    counts = np.ones((1, 1, 1))
    for axis, (out, length) in enumerate(zip(out_shape, block.shape)):
        axis_counts = np.full(out, factor)
        axis_counts[-1] = length - (out - 1) * factor
        counts = counts * axis_counts.reshape([out if ax == axis else 1 for ax in range(3)])
    return (sums / counts).astype("float32")


class ScaleLevel:
    """One scale level of a volume.

    The level is either a level of the pyramid in the container, or it is downsampled on the fly from the
    input. Use `open_scale_level` to create it.

    Args:
        array: The pyramid level, or the input that is downsampled.
        factor: The downsampling factor per axis. 1 for a pyramid level.
    """
    def __init__(self, array: np.typing.ArrayLike, factor: int = 1):
        self.array = array
        self.factor = factor

    @property
    def shape(self) -> Tuple[int, int, int]:
        """The shape of the level, ZYX."""
        return tuple(-(-length // self.factor) for length in self.array.shape)

    @property
    def dtype(self) -> np.dtype:
        """The data type of the input. The mean downsampling returns float32 instead."""
        return np.dtype(self.array.dtype)

    def read(
        self,
        start: Optional[Sequence[int]] = None,
        stop: Optional[Sequence[int]] = None,
        labels: Optional[bool] = None,
    ) -> np.ndarray:
        """Read a bounding box of the level into memory.

        Args:
            start: The start of the bounding box, ZYX, in voxels of this level. Default: the volume start.
            stop: The stop of the bounding box, ZYX, in voxels of this level. Default: the volume shape.
            labels: Whether the data is a segmentation. A segmentation is downsampled by nearest neighbor
                sampling, image data by the mean. Default: `is_label_dtype` of the data type.

        Returns:
            The data. It is writeable and does not share memory with the input.
        """
        start = np.zeros(3, dtype=int) if start is None else np.asarray(start, dtype=int)
        stop = np.array(self.shape, dtype=int) if stop is None else np.asarray(stop, dtype=int)

        if self.factor == 1:
            data = self.array[tuple(slice(sta, sto) for sta, sto in zip(start, stop))]
            # A slice of a numpy array is a view. For a tif file, it is a view of the file, mapped with write access.
            return np.array(data) if isinstance(self.array, np.ndarray) else np.asarray(data)

        labels = is_label_dtype(self.dtype) if labels is None else labels
        factor = self.factor
        out = np.zeros(tuple(int(sto - sta) for sta, sto in zip(start, stop)),
                       dtype=self.dtype if labels else "float32")

        # Read blocks that cover whole chunks of the input: zarr decodes a complete chunk for every read.
        chunks = getattr(self.array, "chunks", None)
        if chunks is None or len(chunks) != 3:
            chunks = (1, 1, 1)
        block_shape = []
        for chunk in chunks:
            length = math.lcm(factor, int(chunk))
            block_shape.append(length * -(-_MIN_BLOCK_LENGTH // length))
        out_block = [length // factor for length in block_shape]

        ranges = [range(sta, sto, block) for sta, sto, block in zip(start, stop, out_block)]
        for block_start in itertools.product(*ranges):
            block_stop = [min(sta + block, sto) for sta, block, sto in zip(block_start, out_block, stop)]
            bb = tuple(
                slice(sta * factor, min(sto * factor, length))
                for sta, sto, length in zip(block_start, block_stop, self.array.shape)
            )
            block = np.asarray(self.array[bb])
            block = block[::factor, ::factor, ::factor] if labels else _block_mean(block, factor)
            out_bb = tuple(
                slice(sta - offset, sto - offset) for sta, sto, offset in zip(block_start, block_stop, start)
            )
            out[out_bb] = block
        return out


def open_scale_level(
    input_path: Union[str, Store],
    input_key: Optional[str],
    scale: int,
    from_s3: bool = False,
    bucket_name: Optional[str] = None,
    service_endpoint: Optional[str] = None,
    credential_file: Optional[str] = None,
) -> ScaleLevel:
    """Open a volume at a lower resolution.

    The scale level `scale` has 2^scale times the voxel size of the level `input_key`, on every axis. This
    function uses the pyramid level `scale` levels below `input_key`, e.g. "s3" for the key "s0" and the scale 3,
    if the container holds it with the expected shape. Otherwise, the level is downsampled from `input_key` when
    it is read.

    Args:
        input_path: The file path to the data, a tif file or a zarr/n5 container, see `read_image_data`.
        input_key: The key of the input in the container, e.g. "s0" or "setup0/timepoint0/s0".
            The key is ignored for a tif file.
        scale: The scale level, 0 for the input itself.
        from_s3: Whether to read the data from S3.
        bucket_name: S3 bucket name.
        service_endpoint: S3 service endpoint.
        credential_file: Credential file containing access key and secret key.

    Returns:
        The scale level.
    """
    if scale < 0:
        raise ValueError(f"The scale must be non-negative, got {scale}.")
    if _is_tif(input_path):
        if from_s3:
            raise ValueError("Reading a tif file from S3 is not supported.")
        input_key = None
    s3_kwargs = dict(
        from_s3=from_s3, bucket_name=bucket_name, service_endpoint=service_endpoint, credential_file=credential_file,
    )

    data = read_image_data(input_path, input_key, **s3_kwargs)
    if scale == 0:
        return ScaleLevel(data)

    factor = 2 ** scale
    key = pyramid_key(input_key, scale)
    if key is not None:
        try:
            level = read_image_data(input_path, key, **s3_kwargs)
        except KeyError:  # A container without this pyramid level.
            level = None
        if level is not None:
            if all(ls in (sh // factor, -(-sh // factor)) for ls, sh in zip(level.shape, data.shape)):
                return ScaleLevel(level)
            print(f"The shape {level.shape} of the pyramid level {key} is not the shape {data.shape} / {factor}.")

    print(f"Downsampling {input_key or input_path} by {factor} on the fly.")
    return ScaleLevel(data, factor)
