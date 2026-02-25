import argparse
import os

import numpy as np
import pandas as pd
import zarr
from elf.io import open_file
import scipy.ndimage as ndimage

from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.sgn_detection import distance_based_marker_extension
from flamingo_tools.file_utils import read_image_data


def main():
    parser = argparse.ArgumentParser(
        description="Script for the extension of an SGN detection. "
        "Either locally or on an S3 bucket.")

    parser.add_argument("-c", "--cochlea", required=True, help="Cochlea in MoBIE.")
    parser.add_argument("-s", "--seg_channel", required=True, help="Segmentation channel.")
    parser.add_argument("-o", "--output", required=True, help="Output directory for segmentation.")
    parser.add_argument("--input", default=None, help="Input tif.")

    parser.add_argument("--component_labels", type=int, nargs="+", default=[1],
                        help="Component labels of SGN_detect.")
    parser.add_argument("-d", "--extension_distance", type=float, default=12, help="Extension distance.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[3.0, 1.887779, 1.887779],
                        help="Voxel size of input in micrometer. Default: [3.0, 1.887779, 1.887779]")

    args = parser.parse_args()

    block_shape = (128, 128, 128)
    chunks = (128, 128, 128)

    if len(args.voxel_size) == 1:
        voxel_size = tuple(args.voxel_size, args.voxel_size, args.voxel_size)
    else:
        voxel_size = tuple(args.voxel_size)

    if args.input is not None:
        data = read_image_data(args.input, None)
        shape = data.shape
        # Compute centers of mass for each label (excluding background = 0)
        markers = ndimage.center_of_mass(np.ones_like(data), data, index=np.unique(data[data > 0]))
        markers = np.array(markers)

    else:

        s3_path = os.path.join(f"{args.cochlea}", "tables", f"{args.seg_channel}", "default.tsv")
        tsv_path, fs = get_s3_path(s3_path)
        with fs.open(tsv_path, 'r') as f:
            table = pd.read_csv(f, sep="\t")

        table = table.loc[table["component_labels"].isin(args.component_labels)]
        markers = list(zip(table["anchor_x"] / voxel_size[0],
                           table["anchor_y"] / voxel_size[1],
                           table["anchor_z"] / voxel_size[2]))
        markers = np.array(markers)

        s3_path = os.path.join(f"{args.cochlea}", "images", "ome-zarr", f"{args.seg_channel}.ome.zarr")
        input_key = "s0"
        s3_store, fs = get_s3_path(s3_path)
        with zarr.open(s3_store, mode="r") as f:
            data = f[input_key][:].astype("float32")

        shape = data.shape

    output_key = "extended_segmentation"
    output_path = os.path.join(args.output, f"{args.cochlea}-{args.seg_channel}.zarr")

    output = open_file(output_path, mode="a")
    output_dataset = output.create_dataset(
        output_key, shape=shape, dtype=np.dtype("uint32"),
        chunks=chunks, compression="gzip"
    )

    distance_based_marker_extension(
        markers=markers,
        output=output_dataset,
        extension_distance=args.extension_distance,
        sampling=voxel_size,
        block_shape=block_shape,
        n_threads=16,
    )


if __name__ == "__main__":
    main()
