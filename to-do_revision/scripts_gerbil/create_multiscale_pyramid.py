"""Downsample a segmentation into a multi-scale pyramid, so that it can be displayed for component selection.

The pyramid is written next to the segmentation as 'segmentation.ome.zarr', as an OME-Zarr (v0.5) in
zarr v3 format with the levels 's0', 's1', ... , so that it can be displayed without any further
conversion. The levels are sharded, so that the sparse segmentation does not waste inodes: the small
chunks that a viewer reads are packed into much larger shard files.
Each level is downsampled from the previous one, i.e. the given scale factors are relative, and the
default factors of 2 result in the levels 's1' - 's5' being downsampled by 2, 4, 8, 16 and 32
compared to the full resolution level 's0'.
"""

import argparse
import math
import multiprocessing as mp
import os
from typing import Optional, Sequence, Tuple, Union

import zarr
from zarr.codecs import ZstdCodec

from bioimage_py import copy, downsample, open_source
from bioimage_py.util import downscale_shape

PYRAMID_NAME = "segmentation.ome.zarr"
"""The name of the multi-scale pyramid, matching the MoBIE naming scheme."""

DEFAULT_SCALE_FACTORS = (2, 2, 2, 2, 2)
"""The relative scale factors, resulting in downsampling by 2, 4, 8, 16 and 32."""

DEFAULT_CHUNKS = (64, 64, 64)
"""The chunks of the pyramid levels, i.e. the granularity at which a viewer reads the data."""

DEFAULT_SHARDS = (256, 256, 256)
"""The shards of the pyramid levels, which hold 4 x 4 x 4 chunks each."""


def _write_multiscale_metadata(group, level_names, cumulative_factors, voxel_size, name):
    """Write the OME-Zarr (v0.5) multiscale metadata for the pyramid levels."""
    # The metadata is given in ZYX order, so the (x, y, z) voxel size has to be reversed.
    voxel_size_zyx = voxel_size[::-1]
    datasets = [
        {
            "coordinateTransformations": [{"scale": [size * factor for size in voxel_size_zyx], "type": "scale"}],
            "path": level_name,
        }
        for level_name, factor in zip(level_names, cumulative_factors)
    ]
    group.attrs["ome"] = {
        "version": "0.5",
        "multiscales": [{
            "axes": [{"name": axis, "type": "space", "unit": "micrometer"} for axis in "zyx"],
            "datasets": datasets,
            "name": name,
        }],
    }


def create_multiscale_pyramid(
    segmentation_folder: str,
    segmentation_key: str = "segmentation",
    scale_factors: Sequence[int] = DEFAULT_SCALE_FACTORS,
    voxel_size: Union[float, Sequence[float]] = 0.38,
    chunks: Tuple[int, int, int] = DEFAULT_CHUNKS,
    shards: Tuple[int, int, int] = DEFAULT_SHARDS,
    name: Optional[str] = None,
    n_threads: Optional[int] = None,
    force: bool = False,
) -> None:
    """Downsample a segmentation into a multi-scale OME-Zarr pyramid next to the segmentation.

    Args:
        segmentation_folder: The folder with the segmentation, i.e. the output folder of the U-Net pipeline.
        segmentation_key: The key of the segmentation in 'segmentation.zarr'.
        scale_factors: The relative scale factors, i.e. each level is downsampled from the previous one.
        voxel_size: The physical voxel spacing of the data in (x, y, z) order.
        chunks: The chunks of the pyramid levels.
        shards: The shards of the pyramid levels. Must be a multiple of the chunks.
        name: The name of the segmentation in the OME-Zarr metadata. By default the folder name is used.
        n_threads: The number of threads. By default all available cores, capped at 16, are used.
        force: Recompute levels that already exist.
    """
    if isinstance(voxel_size, float):
        voxel_size = 3 * (voxel_size,)
    voxel_size = tuple(voxel_size)
    if len(voxel_size) == 1:
        voxel_size = 3 * voxel_size
    assert len(voxel_size) == 3

    if name is None:
        name = os.path.basename(os.path.normpath(segmentation_folder))
    if n_threads is None:
        n_threads = min(16, mp.cpu_count())

    segmentation = open_source(os.path.join(segmentation_folder, "segmentation.zarr"), segmentation_key)
    pyramid_path = os.path.join(segmentation_folder, PYRAMID_NAME)
    group = zarr.open_group(pyramid_path, mode="a", zarr_format=3)

    level_kwargs = dict(
        chunks=chunks, shards=shards, dtype=segmentation.dtype, compressors=ZstdCodec(level=5), fill_value=0,
    )

    # The full resolution level and the relative factor and cumulative factor of each level.
    level_names, cumulative_factors = ["s0"], [1]
    for level, factor in enumerate(scale_factors, start=1):
        level_names.append(f"s{level}")
        cumulative_factors.append(cumulative_factors[-1] * factor)

    print(f"Creating {len(level_names)} pyramid levels in {pyramid_path} with up to {n_threads} threads.")
    previous = None
    for level_name, factor, cumulative_factor in zip(level_names, [1] + list(scale_factors), cumulative_factors):
        shape = segmentation.shape if previous is None else downscale_shape(previous.shape, factor)

        if level_name in group and not force:
            print(f"Level {level_name} already exists and is used as is. Pass --force to recompute it.")
            previous = group[level_name]
            assert tuple(previous.shape) == tuple(shape), f"{previous.shape} != {shape}"
            continue

        level = group.create_array(level_name, shape=shape, overwrite=True, **level_kwargs)
        # Each block covers exactly one shard. Writing a part of a shard rewrites the whole shard
        # file, so shard-sized blocks write each shard once instead of once per chunk it contains.
        # (bioimage_py routes the blocks of a shard to a single worker, so this is about efficiency,
        # not correctness.) With one block per shard, more workers than shards stay idle, which only
        # happens for the coarsest levels, where the level fits into a handful of shards.
        n_blocks = math.prod([math.ceil(sh / sd) for sh, sd in zip(shape, shards)])
        num_workers = min(n_threads, n_blocks)
        print(f"Writing level {level_name} with shape {tuple(shape)}, downsampled by "
              f"{cumulative_factor}, in {n_blocks} shards with {num_workers} threads.")
        if previous is None:
            # The full resolution level is copied, so that the pyramid is a self-contained OME-Zarr.
            copy(segmentation, level, num_workers=num_workers, block_shape=shards)
        else:
            # Each level is downsampled from the previous one, which is much cheaper than
            # downsampling the full resolution level again. 'order=0' keeps the label values intact.
            downsample(previous, factor, level, order=0, num_workers=num_workers, block_shape=shards)
        previous = level

    _write_multiscale_metadata(group, level_names, cumulative_factors, voxel_size, name)
    print(f"Wrote the OME-Zarr metadata for '{name}' with the levels {level_names}.")


def main():
    parser = argparse.ArgumentParser(
        description="Downsample a segmentation into a multi-scale OME-Zarr pyramid for display."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="The folder with the segmentation, i.e. the output folder of the U-Net pipeline. "
                        f"The pyramid is written into it as '{PYRAMID_NAME}'.")
    parser.add_argument("--segmentation_key", default="segmentation",
                        help="The key of the segmentation in 'segmentation.zarr'.")
    parser.add_argument("--scale_factors", type=int, nargs="+", default=list(DEFAULT_SCALE_FACTORS),
                        help="The relative scale factors, i.e. each level is downsampled from the previous one.")
    parser.add_argument("--voxel_size", type=float, nargs="+", default=[0.38],
                        help="The voxel size of the segmentation in micrometer, in (x, y, z) order.")
    parser.add_argument("--chunks", type=int, nargs=3, default=list(DEFAULT_CHUNKS),
                        help="The chunks of the pyramid levels.")
    parser.add_argument("--shards", type=int, nargs=3, default=list(DEFAULT_SHARDS),
                        help="The shards of the pyramid levels. Must be a multiple of the chunks.")
    parser.add_argument("--name", default=None,
                        help="The name of the segmentation in the OME-Zarr metadata. Default: the folder name.")
    parser.add_argument("--n_threads", type=int, default=None, help="The number of threads.")
    parser.add_argument("-f", "--force", action="store_true", help="Recompute levels that already exist.")

    args = parser.parse_args()
    create_multiscale_pyramid(
        segmentation_folder=args.input,
        segmentation_key=args.segmentation_key,
        scale_factors=args.scale_factors,
        voxel_size=args.voxel_size,
        chunks=tuple(args.chunks),
        shards=tuple(args.shards),
        name=args.name,
        n_threads=args.n_threads,
        force=args.force,
    )


if __name__ == "__main__":
    main()
