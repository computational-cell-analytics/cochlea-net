#!/bin/bash
#SBATCH --job-name=ws-SGNv2var
#SBATCH --time=03:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --array=0-3
# The gerbil watershed took 53 min on a 676 GB prediction; these are 192-326 GB and the
# first run is on a sub-hour pace. Four tasks run concurrently and contend for lustre
# bandwidth, so this keeps ~4x headroom rather than cutting to 2 h.

# Seeded watershed for the four seed variants of one cochlea. Usage:
#   sbatch 2026-08-23_sbatch_watershed_SGN-v2-variance.sbatch <cochlea>
#
# The parameters are those of the reference SGN_v2 segmentation, so the four variants stay
# comparable to the published v2 accuracy. distance_watershed_implementation takes its foreground
# from channel 0 of the prediction and never reads mask.zarr.
#
# Memory: the gerbil watershed measured 189 GB MaxRSS at 34248 in-mask blocks, and the largest
# cochlea here has 16327, so 90-120 GB is expected. Check `sacct -o MaxRSS` after the smoke test
# and raise this to 400G (the value in reproducibility/templates_processing) if it comes close.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:?"pass the cochlea as the first argument"}
VERSION=${VERSIONS[$SLURM_ARRAY_TASK_ID]}

COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
OUTPUT_FOLDER=$(version_folder "$COCHLEA" "$VERSION")

require_paths "$COCHLEA_FOLDER/shard_manifest.json" "$OUTPUT_FOLDER/predictions.zarr/prediction"

echo "Cochlea: $COCHLEA, version: $VERSION"
echo "Output: $OUTPUT_FOLDER"

# Missing shards read back as zeros, so reject an incomplete prediction before starting. This
# replaces the per-channel file count of the unsharded workflow, which is meaningless here: one
# shard file holds 4 x 4 x 4 blocks and all three channels.
python -m sgn_variance verify \
	--cochlea_folder "$COCHLEA_FOLDER" \
	--output_folder "$OUTPUT_FOLDER"

echo "Watershed: min_size=1000, center=0.4, boundary=0.5, foreground=0.5, smoothing=0.0"

# The parameters are numeric literals on purpose: run_unet_segmentation_slurm coerces min_size and
# the three distance parameters, but passes fg_threshold through unconverted, so feeding it from an
# environment variable would hand a string to ThresholdWrapper.

export OUTPUT_FOLDER
python -c 'import os
from flamingo_tools.segmentation.unet_prediction import run_unet_segmentation_slurm

run_unet_segmentation_slurm(
    output_folder=os.environ["OUTPUT_FOLDER"],
    min_size=1000,
    center_distance_threshold=0.4,
    boundary_distance_threshold=0.5,
    fg_threshold=0.5,
    distance_smoothing=0.0,
)'
