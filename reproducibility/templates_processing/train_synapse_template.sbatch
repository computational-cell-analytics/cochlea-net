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
# $WORK_DIR/checkpoints/synapse_detection_<MODEL_SUFFIX>.
WORK_DIR=$1
# Version of the training data below the training_data/synapses/training_data folder, e.g. v5
VERSION=$2
# Suffix of the model name, e.g. v6. Also seeds the train / validation split.
MODEL_SUFFIX=$3
# Optional random state for the split. Pass 42 to reuse the split of a fixed version.
RANDOM_STATE=$4

cd "$WORK_DIR" || exit

echo "Training data version: ${VERSION}"
echo "Model: synapse_detection_${MODEL_SUFFIX}"
echo "Checkpoint directory: ${WORK_DIR}/checkpoints"

split_args=()
if [ -n "$RANDOM_STATE" ]; then
	split_args+=(-r "$RANDOM_STATE")
fi

# Trains the v7 recipe: heatmap only, MinPointSampler and fixed validation patches. Add
# --use_flow for the 5 channel heatmap and flow training of synapse_detection_v5. To retrain v3
# or v5 themselves, add --legacy_recipe: the runs before 2026-09-21 had a different validation
# metric, redrawn validation patches, and no sampler without flow.
python "$SCRIPT_REPO"/scripts/synapse_marker_detection/train_synapse_detection.py \
	-v "$VERSION" -m "$MODEL_SUFFIX" -s "$WORK_DIR" "${split_args[@]}"
