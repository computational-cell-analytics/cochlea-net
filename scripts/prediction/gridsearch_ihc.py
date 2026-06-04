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

from flamingo_tools.segmentation.gridsearch import gridsearch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-b", "--block_shape", default=None, type=str)
    parser.add_argument("--halo", default=None, type=str)
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

    gridsearch(
        val_dir=args.input,
        model_path=args.model,
        result_dir=args.output_folder,
    )


if __name__ == "__main__":
    main()
