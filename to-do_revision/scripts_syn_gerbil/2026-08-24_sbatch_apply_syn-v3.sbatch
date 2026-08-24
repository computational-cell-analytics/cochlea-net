#!/bin/bash
#SBATCH --job-name=apply-syn-v3
#SBATCH --time=04:00:00
#SBATCH --partition=grete:shared
#SBATCH --account=nim00007
#SBATCH --gpus=A100:1
# The prefetch workers load the next blocks while the GPU predicts, so give the job cores.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9
# The size of the array and PREDICTION_INSTANCES in common.sh must match: fewer tasks than
# instances leaves blocks unpredicted, more tasks than instances raises in the library.
#
# Walltime is an estimate, not a measurement: the earlier single-job run of G_LR_000302_R had
# 10 h for the whole volume and did not finish, so a tenth of the volume in 4 h is roughly
# fourfold margin over that pace. Check the first array's Elapsed before trusting it for the
# second cochlea.

# Step 2 of 3: predict the synapse maps for one array task's share of the blocks. Usage:
#   sbatch 2026-08-24_sbatch_apply_syn-v3.sbatch <cochlea>
#
# Every task writes into the same predictions.zarr, each into its own blocks. There is no
# skip-existing check inside the library for this path, by design: the dataset is created by
# whichever task starts first, so skipping on its existence would leave the other tasks'
# blocks empty. That is the failure mode that produced the truncated G_LR_000302_R result, so
# on a re-run delete predictions.zarr rather than resubmitting on top of it.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:?"pass the cochlea as the first argument"}
require_known_cochlea "$COCHLEA"

INPUT=$(raw_path "$COCHLEA")
INPUT_KEY=$(raw_key "$COCHLEA")
OUTPUT_FOLDER=$(output_folder "$COCHLEA")

require_paths "$INPUT" "$MODEL" "$OUTPUT_FOLDER/mask.zarr" "$OUTPUT_FOLDER/mean_std.json"

echo "Array task: ${SLURM_ARRAY_TASK_ID} of ${PREDICTION_INSTANCES}"
echo "Cochlea: $COCHLEA"
echo "Input: $INPUT ($INPUT_KEY)"
echo "Output: $OUTPUT_FOLDER"
echo "Model: $MODEL"

export INPUT INPUT_KEY OUTPUT_FOLDER MODEL PREDICTION_INSTANCES
python -c 'import os
from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_slurm

run_synapse_prediction_slurm(
    input_path=os.environ["INPUT"],
    input_key=os.environ["INPUT_KEY"],
    output_folder=os.environ["OUTPUT_FOLDER"],
    model_path=os.environ["MODEL"],
    prediction_instances=os.environ["PREDICTION_INSTANCES"],
)'
