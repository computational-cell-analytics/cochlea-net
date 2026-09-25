#!/bin/bash
#SBATCH --job-name=train-synapse
#SBATCH -t 48:00:00

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
# The data loader workers must match num_workers in train_synapse_detection.py.
#SBATCH -c 8
#SBATCH --mem 128G

source ~/.bashrc
micromamba activate micro-sam_gpu

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
export PYTHONPATH="$SCRIPT_REPO:$PYTHONPATH"

# Directory for the checkpoints. train_synapse_detection.py writes to
# ./checkpoints/synapse_detection_<MODEL_SUFFIX>, so the job must run inside it.
WORK_DIR=$1
# Version of the training data below the training_data/synapses/training_data folder, e.g. v5
VERSION=$2
# Suffix of the model name, e.g. v6. Also seeds the train / validation split.
MODEL_SUFFIX=$3
# Optional random state for the split. Pass 42 to reuse the split of a fixed version.
RANDOM_STATE=$4
# Optional half width in voxels of the loss mask around each annotation, e.g. 16. Without it the
# loss covers the full patch, which also supervises unannotated CTBP2 spots outside the IHCs.
# This cube mask failed for v8, see to-do_revision/synapses.md. Prefer IGNORE_PERCENTILE.
MASK_RADIUS=$5
# Optional percentile, e.g. 10, that excludes the unannotated CTBP2 spots from the loss and keeps
# every other voxel supervised (--ignore_percentile). Pass "" as MASK_RADIUS to use it alone.
IGNORE_PERCENTILE=$6

cd "$WORK_DIR" || exit

echo "Training data version: ${VERSION}"
echo "Model: synapse_detection_${MODEL_SUFFIX}"
echo "Checkpoint directory: ${WORK_DIR}/checkpoints"

split_args=()
if [ -n "$RANDOM_STATE" ]; then
	split_args+=(-r "$RANDOM_STATE")
fi
if [ -n "$MASK_RADIUS" ]; then
	split_args+=(--mask_radius "$MASK_RADIUS")
fi
if [ -n "$IGNORE_PERCENTILE" ]; then
	split_args+=(--ignore_percentile "$IGNORE_PERCENTILE")
fi

# Trains the heatmap only, the output layout of synapse_detection_v3. Add --use_flow for the
# 5 channel heatmap and flow training of synapse_detection_v5, and --sampler to override the
# default sampler of the selected mode. To retrain v3 or v5 themselves, add --legacy_recipe:
# the runs before 2026-09-21 had no raw normalization and a different validation metric.
python "$SCRIPT_REPO"/scripts/synapse_marker_detection/train_synapse_detection.py \
	-v "$VERSION" -m "$MODEL_SUFFIX" "${split_args[@]}"
