#!/bin/bash
#SBATCH --job-name=pre-syn-v3
#SBATCH --time=01:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
# calc_mean_and_std is thread-parallel over blocks and uses min(16, cpu_count) threads, so
# give it 16 cores. It reads only the in-mask blocks, which for a synapse mask is 2.8 % of the
# volume. Measured 4.5 min for G_LR_000301_L (job 15469008), mask included; 1 h is ~13x that.
# G_LR_000302_R is a smaller volume, so it should not be slower.

# Step 1 of 3: build the prediction mask from the IHC segmentation and compute the mean and
# standard deviation of the masked volume. Usage:
#   sbatch 2026-08-24_sbatch_preprocess_syn-v3.sbatch <cochlea>
#
# Both results have to exist before the prediction array starts: every array task must
# normalize identically, and recomputing the statistics per task would read the volume ten
# times over.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
# activate_env before 'set -u': sourcing ~/.bashrc trips over an unbound variable in /etc/bashrc.
activate_env
set -euo pipefail

COCHLEA=${1:?"pass the cochlea as the first argument"}
require_known_cochlea "$COCHLEA"

INPUT=$(raw_path "$COCHLEA")
INPUT_KEY=$(raw_key "$COCHLEA")
MASK=$(mask_path "$COCHLEA")
OUTPUT_FOLDER=$(output_folder "$COCHLEA")

require_paths "$INPUT" "$MASK"
mkdir -p "$OUTPUT_FOLDER"

echo "Cochlea: $COCHLEA"
echo "Input: $INPUT ($INPUT_KEY)"
echo "Mask: $MASK ($MASK_INPUT_KEY)"
echo "Output: $OUTPUT_FOLDER"

export INPUT INPUT_KEY MASK MASK_INPUT_KEY OUTPUT_FOLDER
python -c 'import os
from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_preprocess_slurm

run_synapse_prediction_preprocess_slurm(
    input_path=os.environ["INPUT"],
    input_key=os.environ["INPUT_KEY"],
    output_folder=os.environ["OUTPUT_FOLDER"],
    mask_path=os.environ["MASK"],
    mask_input_key=os.environ["MASK_INPUT_KEY"],
)'
