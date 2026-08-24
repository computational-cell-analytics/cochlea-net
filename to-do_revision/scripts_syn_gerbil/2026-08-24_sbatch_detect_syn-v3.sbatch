#!/bin/bash
#SBATCH --job-name=detect-syn-v3
#SBATCH --time=06:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
# The peak detection is thread-parallel over blocks, so it scales with the cores.
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

# Step 3 of 3: detect the maxima in the finished prediction, match them to the IHCs and write
# the two tables. Usage:
#   sbatch 2026-08-24_sbatch_detect_syn-v3.sbatch <cochlea>
#
# Writes synapse_detection.tsv (all detections) and synapse_detection_filtered.tsv (the ones
# within MAX_DISTANCE of an IHC, with matched_ihc and distance_to_ihc). No GPU needed.
# The matching reads the IHC segmentation at full resolution, hence the memory request.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:?"pass the cochlea as the first argument"}
require_known_cochlea "$COCHLEA"

MASK=$(mask_path "$COCHLEA")
OUTPUT_FOLDER=$(output_folder "$COCHLEA")

require_paths "$MASK" "$OUTPUT_FOLDER/predictions.zarr"

echo "Cochlea: $COCHLEA"
echo "Output: $OUTPUT_FOLDER"
echo "Mask: $MASK ($MASK_FULL_KEY)"
echo "Threshold: $THRESHOLD, max distance: $MAX_DISTANCE um, voxel size: $VOXEL_SIZE"

export OUTPUT_FOLDER MASK MASK_FULL_KEY VOXEL_SIZE THRESHOLD MAX_DISTANCE
python -c 'import os
from flamingo_tools.segmentation.synapse_detection import run_synapse_detection_slurm

run_synapse_detection_slurm(
    output_folder=os.environ["OUTPUT_FOLDER"],
    voxel_size=os.environ["VOXEL_SIZE"],
    threshold=os.environ["THRESHOLD"],
    mask_path=os.environ["MASK"],
    mask_input_key=os.environ["MASK_FULL_KEY"],
    max_distance=os.environ["MAX_DISTANCE"],
)'
