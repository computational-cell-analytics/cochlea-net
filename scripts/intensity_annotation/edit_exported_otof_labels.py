"""Edit OTOF assignments with locally exported IHC and OTOF volumes."""

import argparse
import os
import re
from contextlib import ExitStack, contextmanager
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile
import zarr

from otof_label_editor import (
    BASE_VOXEL_SIZE_XYZ,
    EditorData,
    prepare_editor_data,
    run_editor,
)


DEFAULT_TARGET_SCALE = 3
DEFAULT_BLOCK_SHAPE = (16, 128, 128)
_SCALE_FOLDER_PATTERN = re.compile(r"^scale(?P<scale>\d+)(?:_|$)")


class _TiffVolume:
    """Expose a 3D TIFF series through block-wise array indexing."""

    def __init__(self, tiff: tifffile.TiffFile):
        self._tiff = tiff
        self._series = tiff.series[0]
        self.shape = self._series.shape
        self.dtype = self._series.dtype
        self.ndim = len(self.shape)
        self._cached_array = None
        self._pages_are_z = (
            self.ndim == 3
            and len(self._series.pages) == self.shape[0]
            and self._series.pages[0].shape == self.shape[1:]
        )

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        item = item + (slice(None),) * (self.ndim - len(item))
        if not self._pages_are_z:
            if self._cached_array is None:
                self._cached_array = self._series.asarray()
            return self._cached_array[item]

        z_item = item[0]
        data = self._tiff.asarray(key=z_item, series=self._series)
        if data.ndim == 2:
            data = data[np.newaxis]
        return data[(slice(None),) + item[1:]]

    def read_z_plane(self, index: int) -> np.ndarray:
        if not self._pages_are_z:
            raise ValueError("The TIFF series does not store one page per z plane.")
        return self._tiff.asarray(key=index, series=self._series)


def _select_zarr_array(root, path: str, key: Optional[str], input_scale: int):
    if hasattr(root, "shape"):
        if key is not None:
            raise ValueError(f"Volume '{path}' is already an array and does not accept key '{key}'.")
        return root

    candidates = [key] if key is not None else ["image", f"s{input_scale}"]
    for candidate in candidates:
        if candidate in root and hasattr(root[candidate], "shape"):
            return root[candidate]

    array_keys = sorted(root.array_keys())
    if key is None and len(array_keys) == 1:
        return root[array_keys[0]]
    if key is None:
        raise ValueError(
            f"Could not select an array in '{path}'. Available arrays: {array_keys}. "
            "Use the matching --ihc-key or --otof-key option."
        )
    raise ValueError(f"Volume '{path}' has no array '{key}'. Available arrays: {array_keys}.")


@contextmanager
def open_volume(path: str, key: Optional[str], input_scale: int):
    """Open a TIFF, NumPy, or Zarr volume without loading it in full."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volume does not exist: {path}")

    lower_path = path.lower()
    if lower_path.endswith((".tif", ".tiff")):
        with tifffile.TiffFile(path) as tiff:
            yield _TiffVolume(tiff)
        return
    if lower_path.endswith(".npy"):
        if key is not None:
            raise ValueError(f"NumPy volume '{path}' does not accept an array key.")
        yield np.load(path, mmap_mode="r")
        return
    if lower_path.endswith((".zarr", ".ome.zarr")):
        root = zarr.open(path, mode="r")
        yield _select_zarr_array(root, path, key, input_scale)
        return
    raise ValueError(f"Unsupported volume format for '{path}'. Use TIFF, NPY, Zarr, or OME-Zarr.")


def _output_shape(shape: Sequence[int], factor: int) -> Tuple[int, int, int]:
    return tuple(max(1, int(size) // factor) for size in shape)


def coerce_segmentation(segmentation: np.ndarray) -> np.ndarray:
    """Convert an integer-valued exported segmentation to uint32."""
    if not np.issubdtype(segmentation.dtype, np.number):
        raise ValueError(f"The IHC segmentation must contain numbers, not {segmentation.dtype}.")
    if not np.isfinite(segmentation).all():
        raise ValueError("The IHC segmentation must contain finite values.")
    if not np.equal(segmentation, np.floor(segmentation)).all():
        raise ValueError("The IHC segmentation must contain integer-valued labels.")

    minimum = float(segmentation.min()) if segmentation.size else 0.0
    maximum = float(segmentation.max()) if segmentation.size else 0.0
    if minimum < 0:
        raise ValueError("The IHC segmentation must not contain negative label IDs.")
    if maximum > np.iinfo("uint32").max:
        raise ValueError(f"Label ID {maximum:g} exceeds the supported uint32 range.")
    return segmentation.astype("uint32", copy=False)


def _downsample_tiff_pages(
    volume: _TiffVolume,
    factor: int,
    mode: str,
    block_shape: Sequence[int],
) -> np.ndarray:
    shape = tuple(int(size) for size in volume.shape)
    out_shape = _output_shape(shape, factor)
    if mode == "nearest":
        output = np.empty(out_shape, dtype=volume.dtype)
        offsets = tuple(min(size - 1, factor // 2) for size in shape)
        for output_z in range(out_shape[0]):
            plane = volume.read_z_plane(output_z * factor + offsets[0])
            for y0 in range(0, out_shape[1], block_shape[1]):
                y1 = min(y0 + block_shape[1], out_shape[1])
                for x0 in range(0, out_shape[2], block_shape[2]):
                    x1 = min(x0 + block_shape[2], out_shape[2])
                    output[output_z, y0:y1, x0:x1] = plane[
                        y0 * factor + offsets[1]:y1 * factor + offsets[1]:factor,
                        x0 * factor + offsets[2]:x1 * factor + offsets[2]:factor,
                    ]
        return output

    axis_factors = tuple(min(factor, size) for size in shape)
    output = np.zeros(out_shape, dtype="float32")
    for output_z in range(out_shape[0]):
        for input_z in range(output_z * axis_factors[0], (output_z + 1) * axis_factors[0]):
            plane = volume.read_z_plane(input_z)
            for y0 in range(0, out_shape[1], block_shape[1]):
                y1 = min(y0 + block_shape[1], out_shape[1])
                for x0 in range(0, out_shape[2], block_shape[2]):
                    x1 = min(x0 + block_shape[2], out_shape[2])
                    input_block = plane[
                        y0 * axis_factors[1]:y1 * axis_factors[1],
                        x0 * axis_factors[2]:x1 * axis_factors[2],
                    ]
                    block_output_shape = (y1 - y0, x1 - x0)
                    input_block = input_block.reshape(
                        block_output_shape[0],
                        axis_factors[1],
                        block_output_shape[1],
                        axis_factors[2],
                    )
                    output[output_z, y0:y1, x0:x1] += input_block.mean(
                        axis=(1, 3), dtype="float32"
                    )
        output[output_z] /= axis_factors[0]
    return output


def downsample_volume(
    volume,
    factor: int,
    mode: str,
    block_shape: Sequence[int] = DEFAULT_BLOCK_SHAPE,
) -> np.ndarray:
    """Downsample a 3D volume in bounded blocks."""
    if getattr(volume, "ndim", None) != 3:
        raise ValueError(f"Expected a 3D volume, but received shape {getattr(volume, 'shape', None)}.")
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError(f"The downsampling factor must be a positive integer, not {factor}.")
    if mode not in {"nearest", "mean"}:
        raise ValueError(f"Unsupported downsampling mode '{mode}'. Use 'nearest' or 'mean'.")

    block_shape = tuple(int(size) for size in block_shape)
    if len(block_shape) != 3 or any(size < 1 for size in block_shape):
        raise ValueError(f"The block shape must contain three positive integers, not {block_shape}.")

    if isinstance(volume, _TiffVolume) and volume._pages_are_z:
        return _downsample_tiff_pages(volume, int(factor), mode, block_shape)

    shape = tuple(int(size) for size in volume.shape)
    out_shape = _output_shape(shape, int(factor))
    if mode == "nearest":
        output = np.empty(out_shape, dtype=volume.dtype)
        axis_factors = (int(factor),) * 3
        offsets = tuple(min(size - 1, axis_factor // 2) for size, axis_factor in zip(shape, axis_factors))
    else:
        output = np.empty(out_shape, dtype="float32")
        axis_factors = tuple(min(int(factor), size) for size in shape)
        offsets = (0, 0, 0)

    for z0 in range(0, out_shape[0], block_shape[0]):
        z1 = min(z0 + block_shape[0], out_shape[0])
        for y0 in range(0, out_shape[1], block_shape[1]):
            y1 = min(y0 + block_shape[1], out_shape[1])
            for x0 in range(0, out_shape[2], block_shape[2]):
                x1 = min(x0 + block_shape[2], out_shape[2])
                output_slices = (slice(z0, z1), slice(y0, y1), slice(x0, x1))
                output_bounds = ((z0, z1), (y0, y1), (x0, x1))

                if mode == "nearest":
                    input_slices = tuple(
                        slice(begin * axis_factor + offset, end * axis_factor + offset, axis_factor)
                        for (begin, end), axis_factor, offset in zip(output_bounds, axis_factors, offsets)
                    )
                    output[output_slices] = volume[input_slices]
                    continue

                input_slices = tuple(
                    slice(begin * axis_factor, end * axis_factor)
                    for (begin, end), axis_factor in zip(output_bounds, axis_factors)
                )
                block = np.asarray(volume[input_slices])
                block_output_shape = tuple(end - begin for begin, end in output_bounds)
                reshape = tuple(
                    value
                    for output_size, axis_factor in zip(block_output_shape, axis_factors)
                    for value in (output_size, axis_factor)
                )
                block = block.reshape(reshape)
                output[output_slices] = block.mean(axis=(1, 3, 5), dtype="float32")
    return output


def infer_input_scale(*paths: str) -> int:
    """Infer the pyramid scale from an exported scaleN folder."""
    scales = set()
    for path in paths:
        directory = os.path.dirname(os.path.abspath(path))
        for part in reversed(directory.split(os.sep)):
            match = _SCALE_FOLDER_PATTERN.match(part)
            if match is not None:
                scales.add(int(match.group("scale")))
                break
    if len(scales) > 1:
        raise ValueError(f"The input paths refer to different pyramid scales: {sorted(scales)}.")
    return scales.pop() if scales else 0


def _validate_voxel_size(voxel_size_xyz: Sequence[float]) -> Tuple[float, float, float]:
    values = tuple(float(value) for value in voxel_size_xyz)
    if len(values) != 3 or any(value <= 0 for value in values):
        raise ValueError(f"The voxel size must contain three positive values, not {voxel_size_xyz}.")
    return values


def load_editor_data(
    ihc_path: str,
    otof_path: str,
    table_path: str,
    target_scale: int = DEFAULT_TARGET_SCALE,
    input_scale: Optional[int] = None,
    voxel_size_xyz: Sequence[float] = BASE_VOXEL_SIZE_XYZ,
    ihc_key: Optional[str] = None,
    otof_key: Optional[str] = None,
    masking_radius: Optional[float] = None,
) -> EditorData:
    """Load local exports and downsample them to the requested pyramid scale."""
    if input_scale is None:
        input_scale = infer_input_scale(ihc_path, otof_path)
    if input_scale < 0 or target_scale < 0:
        raise ValueError("The input and target scales must be zero or greater.")
    if target_scale < input_scale:
        raise ValueError(
            f"Target scale {target_scale} is finer than input scale {input_scale}; upsampling is not supported."
        )
    if not os.path.exists(table_path):
        raise FileNotFoundError(f"IHC table does not exist: {os.path.abspath(table_path)}")

    voxel_size_xyz = _validate_voxel_size(voxel_size_xyz)
    factor = 2 ** (target_scale - input_scale)
    with ExitStack() as stack:
        ihc_source = stack.enter_context(open_volume(ihc_path, ihc_key, input_scale))
        otof_source = stack.enter_context(open_volume(otof_path, otof_key, input_scale))
        if ihc_source.ndim != 3 or otof_source.ndim != 3:
            raise ValueError(
                f"Expected two 3D volumes, but received shapes {ihc_source.shape} and {otof_source.shape}."
            )
        if ihc_source.shape != otof_source.shape:
            raise ValueError(
                f"The IHC and Otof volumes have different shapes: {ihc_source.shape} and {otof_source.shape}."
            )
        if not np.issubdtype(ihc_source.dtype, np.number):
            raise ValueError(f"The IHC segmentation must contain numbers, not {ihc_source.dtype}.")
        if not np.issubdtype(otof_source.dtype, np.number):
            raise ValueError(f"The Otof volume must contain numbers, not {otof_source.dtype}.")

        if factor == 1:
            print(f"Loading local volumes at scale{target_scale} without downsampling ...")
        else:
            print(
                f"Downsampling local volumes from scale{input_scale} to scale{target_scale} "
                f"with factor {factor} ..."
            )
        segmentation = coerce_segmentation(downsample_volume(ihc_source, factor, mode="nearest"))
        otof = downsample_volume(otof_source, factor, mode="mean")

    print(f"Loading {os.path.abspath(table_path)} ...")
    table = pd.read_csv(table_path, sep="\t")
    scale_zyx = tuple(value * (2 ** target_scale) for value in voxel_size_xyz[::-1])
    return prepare_editor_data(
        table,
        segmentation,
        otof,
        scale_zyx,
        f"scale{target_scale}",
        masking_radius=masking_radius,
    )


def default_output_path(table_path: str) -> str:
    stem, extension = os.path.splitext(os.path.abspath(table_path))
    if extension.lower() != ".tsv":
        stem = os.path.abspath(table_path)
    return f"{stem}_edited.tsv"


def default_dataset_name(ihc_path: str) -> str:
    directory = os.path.dirname(os.path.abspath(ihc_path))
    if _SCALE_FOLDER_PATTERN.match(os.path.basename(directory)):
        directory = os.path.dirname(directory)
    return os.path.basename(directory) or os.path.basename(ihc_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit positive and negative OTOF assignments in locally exported IHC data."
    )
    parser.add_argument("--ihc", "--ihc-path", dest="ihc_path", required=True, help="Exported IHC volume.")
    parser.add_argument("--otof", "--otof-path", dest="otof_path", required=True, help="Exported OTOF volume.")
    parser.add_argument("--table", "--table-path", dest="table_path", required=True, help="Source IHC TSV table.")
    parser.add_argument(
        "--input-scale",
        type=int,
        default=None,
        help="Pyramid scale of the exported volumes. Default: infer from scaleN folder, otherwise 0.",
    )
    parser.add_argument(
        "--scale",
        "-s",
        dest="target_scale",
        type=int,
        default=DEFAULT_TARGET_SCALE,
        help=f"Target pyramid scale for editing. Default: {DEFAULT_TARGET_SCALE}",
    )
    parser.add_argument(
        "--voxel-size",
        nargs=3,
        type=float,
        default=BASE_VOXEL_SIZE_XYZ,
        metavar=("X", "Y", "Z"),
        help="Scale-0 voxel size in micrometers. Default: 0.38 0.38 0.38",
    )
    parser.add_argument("--ihc-key", default=None, help="Array key for an IHC Zarr group.")
    parser.add_argument("--otof-key", default=None, help="Array key for an OTOF Zarr group.")
    parser.add_argument(
        "--masking",
        dest="masking_radius",
        type=float,
        default=None,
        metavar="RADIUS_UM",
        help="Keep OTOF signal within this radius of assigned IHCs, in micrometers. Default: no masking.",
    )
    parser.add_argument("--name", default=None, help="Dataset name shown in the napari window title.")
    parser.add_argument(
        "--output-table",
        "-o",
        default=None,
        help="Local TSV path for the edited table. Default: <input-table>_edited.tsv",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_table = args.output_table or default_output_path(args.table_path)
    dataset_name = args.name or default_dataset_name(args.ihc_path)
    data = load_editor_data(
        args.ihc_path,
        args.otof_path,
        args.table_path,
        target_scale=args.target_scale,
        input_scale=args.input_scale,
        voxel_size_xyz=args.voxel_size,
        ihc_key=args.ihc_key,
        otof_key=args.otof_key,
        masking_radius=args.masking_radius,
    )
    run_editor(dataset_name, output_table, data)


if __name__ == "__main__":
    main()
