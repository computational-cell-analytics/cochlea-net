#!/bin/bash

export MODEL_NAME="nucleus_NIS3D_supervised_2025-07-17"

export IDIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/2025-07_NIS3D

export SCRIPT_DIR=/user/schilling40/u15000/flamingo-tools/scripts/training

python $SCRIPT_DIR/train_distance_unet.py -i $IDIR --name $MODEL_NAME

