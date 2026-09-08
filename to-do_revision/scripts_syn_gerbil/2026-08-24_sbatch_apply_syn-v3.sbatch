#!/bin/bash
#SBATCH --job-name=apply-syn-v3
#SBATCH --time=01:00:00
#SBATCH --partition=grete:shared
#SBATCH --account=nim00007
#SBATCH --gpus=A100:1
# The prefetch workers load the next blocks while the GPU predicts, so give the job cores.
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-4
# Requeue on preemption. This is not a convenience: a preempted task that is not requeued
# leaves its blocks empty, and the detection step writes a complete-looking table from the
# partial volume. Requeueing costs at most one task's work because a task redoes its whole
# share, which is ~250 forward passes, about 12 minutes. Append so a requeued task does not
# clobber the log of its first attempt.
#SBATCH --requeue
#SBATCH --open-mode=append
#
# The size of the array and PREDICTION_INSTANCES in common.sh must match: fewer tasks than
# instances leaves blocks unpredicted, more tasks than instances raises in the library.
#
# Walltime: measured 2.78 s per in-mask block on a 1g.20gb slice, and G_LR_000301_L has 1,238
# in-mask blocks of 44,850 (2.8 %), so 227 to 269 per task across five tasks. That is 11 to
# 12.5 minutes; walking the 8,970 assigned blocks to find them is free within the measurement.
# 1 h is ~5x that, which also covers one requeue after a preemption. See the README.

# Usage:
#   sbatch 2026-08-24_sbatch_apply_syn-v3.sbatch <cochlea>
#   sbatch --partition=grete:preemptible --gpus=3g.40gb:1 2026-08-24_sbatch_apply_syn-v3.sbatch <cochlea>
#
# Every task writes into the same predictions.zarr, each into its own blocks; the chunks are
# aligned with the block shape so each chunk has exactly one writer. There is no skip-existing
# check inside the library, by design: predictions.zarr is created by whichever task starts
# first, so skipping on its existence would leave every other task's blocks empty. Do not
# resubmit the whole array on top of a partial prediction -- run verify_prediction.py and
# resubmit only the tasks it names.

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
echo "Node: ${SLURM_NODELIST}, gpus: ${SLURM_GPUS:-unset}, restarts: ${SLURM_RESTART_COUNT:-0}"

# expandable_segments makes the prediction 1.6x faster (1.76 vs 2.78 s per block, measured
# twice on the same node) but raises peak device occupancy from 18.10 to 19.42 GiB, which is
# 100 % of a 1g.20gb slice. So it is a per-slice decision, not a global win: leave it off
# unless the slice has room. Deliberately '${VAR-default}' and not '${VAR:-default}', so that
# submit_all.sh can pass an empty SYN_ALLOC_CONF to mean "off" rather than "unset".
SYN_ALLOC_CONF=${SYN_ALLOC_CONF-}
if [[ -n "$SYN_ALLOC_CONF" ]]; then
	export PYTORCH_ALLOC_CONF="$SYN_ALLOC_CONF"
	echo "PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
fi

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

# The receipt is the only record that this task finished, so write it last and only on
# success -- 'set -e' plus the python exit status means we never get here otherwise.
# verify_prediction.py turns the set of missing receipts into the list of tasks to resubmit.
RECEIPT_DIR="$OUTPUT_FOLDER/tasks"
mkdir -p "$RECEIPT_DIR"
RECEIPT="$RECEIPT_DIR/task_${SLURM_ARRAY_TASK_ID}_of_${PREDICTION_INSTANCES}.json"
cat > "$RECEIPT" <<JSON
{
  "task_id": ${SLURM_ARRAY_TASK_ID},
  "instances": ${PREDICTION_INSTANCES},
  "job_id": "${SLURM_JOB_ID}",
  "node": "${SLURM_NODELIST}",
  "gpus": "${SLURM_GPUS:-unset}",
  "restarts": ${SLURM_RESTART_COUNT:-0},
  "elapsed_s": ${SECONDS}
}
JSON
echo "Wrote receipt $RECEIPT after ${SECONDS} s"
