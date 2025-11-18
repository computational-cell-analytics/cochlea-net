"""Prediction using distance U-Net.
Parallelization using multiple GPUs is currently only possible
by calling functions located in segmentation/unet_prediction.py directly.
Functions for the parallelization end with '_slurm' and divide the process into preprocessing,
prediction, and segmentation.
"""
import argparse
import json
import time
import os

import imageio.v3 as imageio
import torch
import z5py


def main():
    from flamingo_tools.segmentation import run_unet_prediction

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-k", "--input_key", default=None)
    parser.add_argument("-s", "--scale", default=None, type=float, help="Downscale the image by the given factor.")
    parser.add_argument("-b", "--block_shape", default=None, type=str)
    parser.add_argument("--halo", default=None, type=str)
    parser.add_argument("--memory", action="store_true", help="Perform prediction in memory and save output as tif.")
    parser.add_argument("--time", action="store_true", help="Time prediction process.")
    parser.add_argument("--no_masking", action="store_true", help="Do not mask input.")
    parser.add_argument("--seg_class", default=None, type=str,
                        help="Segmentation class to load parameters for masking input.")
    parser.add_argument("--center_distance_threshold", default=0.4, type=float,
                        help="The threshold applied to the distance center predictions to derive seeds.")
    parser.add_argument("--boundary_distance_threshold", default=None, type=float,
                        help="The threshold applied to the boundary predictions to derive seeds. \
                        By default this is set to 'None', \
                        in which case the boundary distances are not used for the seeds.")
    parser.add_argument("--fg_threshold", default=0.5, type=float,
                        help="The threshold applied to the foreground prediction for deriving the watershed mask.")
    parser.add_argument("--distance_smoothing", default=0, type=float,
                        help="The sigma value for smoothing the distance predictions with a gaussian kernel.")

    args = parser.parse_args()

    scale = args.scale
    if scale is None:
        min_size = 1000
    elif scale > 1:
        min_size = 250
    elif scale < 1:
        min_size = 1000

    have_cuda = torch.cuda.is_available()
    if args.input_key is None:
        if args.block_shape is None:
            block_shape = (64, 256, 256) if have_cuda else (64, 64, 64)
        else:
            block_shape = tuple(json.loads(args.block_shape))

    else:
        if args.block_shape is None:
            chunks = z5py.File(args.input, "r")[args.input_key].chunks
            block_shape = tuple([2 * ch for ch in chunks]) if have_cuda else tuple(chunks)
        else:
            block_shape = json.loads(args.block_shape)

    if args.halo is None:
        halo = (16, 64, 64) if have_cuda else (8, 32, 32)
    else:
        halo = tuple(json.loads(args.halo))

    use_mask = ~args.no_masking

    if args.time:
        start = time.perf_counter()

    if args.memory:
        segmentation = run_unet_prediction(
            args.input, args.input_key, output_folder=None, model_path=args.model,
            scale=scale, min_size=min_size,
            block_shape=block_shape, halo=halo,
            use_mask=use_mask,
            seg_class=args.seg_class,
            center_distance_threshold=args.center_distance_threshold,
            boundary_distance_threshold=args.boundary_distance_threshold,
            fg_threshold=args.fg_threshold, distance_smoothing=args.distance_smoothing,
        )

        abs_path = os.path.abspath(args.input)
        basename = ".".join(os.path.basename(abs_path).split(".")[:-1])
        output_path = os.path.join(args.output_folder, basename + "_seg.tif")
        imageio.imwrite(output_path, segmentation, compression="zlib")
        timer_output = os.path.join(args.output_folder, basename + "_timer.json")

    else:
        run_unet_prediction(
            args.input, args.input_key, output_folder=args.output_folder, model_path=args.model,
            scale=scale, min_size=min_size,
            block_shape=block_shape, halo=halo,
            use_mask=use_mask,
            seg_class=args.seg_class,
            center_distance_threshold=args.center_distance_threshold,
            boundary_distance_threshold=args.boundary_distance_threshold,
            fg_threshold=args.fg_threshold, distance_smoothing=args.distance_smoothing,
        )
        timer_output = os.path.join(args.output_folder, "timer.json")

    if args.time:
        duration = time.perf_counter() - start
        time_dict = {"total_duration[s]": duration}
        with open(timer_output, "w") as f:
            json.dump(time_dict, f, indent='\t', separators=(',', ': '))


if __name__ == "__main__":
    main()
