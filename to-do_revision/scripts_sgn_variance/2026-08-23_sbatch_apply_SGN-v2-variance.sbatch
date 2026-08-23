#!/bin/bash
#SBATCH --job-name=apply-SGNv2var
#SBATCH --time=02:00:00
#SBATCH --partition=grete:shared
#SBATCH --account=nim00007
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=0-9
# Measured on M_LR_000226_L: 0.62 s per block on a 1g.20gb MIG slice and 0.72 s on a
# 1g.10gb one, so a task owning 963 to 1231 blocks takes 10 to 15 minutes. 2 h is roughly
# 8x that. The default here is a whole A100 on grete:shared; pass --preemptible to
# submit_all.sh to use a MIG slice instead, which is usually schedulable immediately.

# Apply one SGN v2 seed variant to one cochlea. Usage:
#   sbatch 2026-08-23_sbatch_apply_SGN-v2-variance.sbatch <cochlea> <version>
#
# Each array task owns whole 512^3 shards, listed in the shard manifest, and writes each of them
# exactly once. That is what makes the sharded output safe: a shard is one file, and a partial
# shard write is a load-merge-rewrite with last-writer-wins, so two tasks filling different blocks
# of the same shard would silently discard each other's work.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:?"pass the cochlea as the first argument"}
VERSION=${2:?"pass the version (1-4) as the second argument"}

COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
OUTPUT_FOLDER=$(version_folder "$COCHLEA" "$VERSION")
MODEL=$(model_path "$VERSION")
INPUT=$(raw_path "$COCHLEA")

require_paths "$INPUT" "$MODEL" "$CHECKPOINT_HELPER_DIR/distance_unet_checkpoint.py" \
	"$COCHLEA_FOLDER/mask.zarr" "$COCHLEA_FOLDER/mean_std.json" \
	"$COCHLEA_FOLDER/shard_manifest.json" "$OUTPUT_FOLDER/predictions.zarr"

echo "Array task: ${SLURM_ARRAY_TASK_ID} of ${PREDICTION_INSTANCES}"
echo "Cochlea: $COCHLEA, version: $VERSION"
echo "Input: $INPUT"
echo "Output: $OUTPUT_FOLDER"
echo "Model: $MODEL"

# --skip_existing makes a re-run after a timeout or a node failure cheap: a shard is written in one
# complete write, so an existing shard file is always a finished shard.
python -m sgn_variance predict \
	--input "$INPUT" --input_key s0 \
	--cochlea_folder "$COCHLEA_FOLDER" \
	--output_folder "$OUTPUT_FOLDER" \
	--model "$MODEL" \
	--task_id "${SLURM_ARRAY_TASK_ID}" \
	--skip_existing
