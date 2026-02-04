import argparse
import os

import multiprocessing as mp
from concurrent import futures

import nifty.tools as nt
from tqdm import tqdm

from elf.wrapper.resized_volume import ResizedVolume
from elf.io import open_file
from flamingo_tools.file_utils import read_tif


def main(
        input_path: str,
        output_dir: str = None,
        scale: float = 0.38,
        input_key: str = "setup0/timepoint0/s0",
        interpolation_order: int = 3
):
    """Resize wrongly scaled cochleas.

    Args:
        input_path: Input path to tif file or n5 folder.
        output_dir: Output folder for rescaled data in n5 format.
        scale: Scale of output data.
        input_key: The key / internal path of the image data.
        interpolation_order: Interpolation order for resizing function.
    """
    if input_path.endswith(".tif"):
        input_ = read_tif(input_path)
        input_chunks = (128,) * 3
    else:
        input_ = open_file(input_path, "r")[input_key]
        input_chunks = input_.chunks

    abs_path = os.path.abspath(input_path)
    basename = ".".join(os.path.basename(abs_path).split(".")[:-1])
    input_dir = os.path.abspath(input_path).split(basename)[0]

    if output_dir is None:
        output_path = os.path.join(input_dir, f"{basename}_resized.n5")
    else:
        output_path = os.path.abspath(output_dir)

    shape = input_.shape
    ndim = len(shape)

    # Limit the number of cores for parallelization.
    n_threads = min(16, mp.cpu_count())

    shape = input_.shape
    new_shape = tuple(
        int(round(sh / scale)) for sh in shape
    )

    resized_volume = ResizedVolume(input_, new_shape, order=interpolation_order)

    output = open_file(output_path, mode="a")
    output_dataset = output.create_dataset(
        input_key, shape=new_shape, dtype=input_.dtype,
        chunks=input_chunks, compression="gzip"
    )
    blocking = nt.blocking([0] * ndim, new_shape, input_chunks)

    def copy_chunk(block_index):
        block = blocking.getBlock(block_index)
        volume_index = tuple(slice(begin, end) for (begin, end) in zip(block.begin, block.end))
        data = resized_volume[volume_index]
        output_dataset[volume_index] = data

    with futures.ThreadPoolExecutor(n_threads) as resize_pool:
        list(tqdm(
            resize_pool.map(copy_chunk, range(blocking.numberOfBlocks)),
            total=blocking.numberOfBlocks,
            desc=f"Resizing volume from shape {shape} to {new_shape}"
        ))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for resizing microscoopy data in n5 format.")

    parser.add_argument("-i", "--input", required=True, type=str, help="Input tif file or n5 folder.")
    parser.add_argument("-o", "--output", type=str,
                        help="Output n5 folder. Default resized output is '<basename>_resized.n5'.")

    parser.add_argument("-s", "--scale", type=float, default=0.38, help="Scale of input. Re-scaled to 1.")
    parser.add_argument("-k", "--input_key", type=str, default="setup0/timepoint0/s0", help="Input key for n5 file.")
    parser.add_argument("--interpolation_order", type=float, default=3, help="Interpolation order.")

    args = parser.parse_args()

    main(args.input, args.output, args.scale, args.input_key, args.interpolation_order)
