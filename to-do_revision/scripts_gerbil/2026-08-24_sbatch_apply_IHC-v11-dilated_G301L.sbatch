#!/bin/bash
#SBATCH --job-name=apply-IHCv11-G301L-dil
#SBATCH --time=00:45:00
#SBATCH --partition=grete-h100:shared
#SBATCH --qos=2h
#SBATCH --account=nim00007
#SBATCH --gpus=H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-9%10

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
PIPELINE_SCRIPT_DIR=$SCRIPT_REPO/to-do_revision/scripts_synapses
INPUT_PATH=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/G301L/GLR_301L_Vglut3_fused.n5
INPUT_KEY=setup0/timepoint0/s0
OUTPUT_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/IHC_v11_dilated_mask1
MODEL_PATH=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v11_cochlea_distance_unet_IHC_supervised_2026-07-20
PREDICTION_INSTANCES=10
EXPECTED_EXISTING_BLOCKS=628

for path in "$SCRIPT_REPO" "$PIPELINE_SCRIPT_DIR/distance_unet_checkpoint.py" \
        "$INPUT_PATH" "$MODEL_PATH" "$OUTPUT_FOLDER/mask_delta.zarr" \
        "$OUTPUT_FOLDER/mean_std.json" "$OUTPUT_FOLDER/predictions.zarr/prediction/zarr.json"; do
	if [[ ! -e "$path" ]]; then
		echo "Required path does not exist: $path" >&2
		exit 1
	fi
done

if [[ "$(readlink "$OUTPUT_FOLDER/mask.zarr")" != "mask_delta.zarr" ]]; then
	echo "The active mask must point to mask_delta.zarr during incremental prediction." >&2
	exit 1
fi

# The cloned prediction must contain exactly the old blocks before the first task starts.
# Once tasks overlap in time this count grows, so only reject a lower value.
for channel in 0 1 2; do
	written_blocks=$(find "$OUTPUT_FOLDER/predictions.zarr/prediction/c/$channel" -type f | wc -l)
	if [[ "$written_blocks" -lt "$EXPECTED_EXISTING_BLOCKS" ]]; then
		echo "Prediction channel $channel has only $written_blocks existing blocks; expected at least $EXPECTED_EXISTING_BLOCKS." >&2
		exit 1
	fi
done

cd "$SCRIPT_REPO"
export PYTHONPATH="$PIPELINE_SCRIPT_DIR:$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "Array task: ${SLURM_ARRAY_TASK_ID}/${PREDICTION_INSTANCES}"
echo "Input: $INPUT_PATH ($INPUT_KEY)"
echo "Output: $OUTPUT_FOLDER"
echo "Active mask: delta blocks only"
echo "Model: $MODEL_PATH"

export INPUT_PATH INPUT_KEY OUTPUT_FOLDER MODEL_PATH PREDICTION_INSTANCES
python -c 'import os
from distance_unet_checkpoint import load_distance_unet
from flamingo_tools.segmentation import unet_prediction

unet_prediction.load_model = load_distance_unet
unet_prediction.run_unet_prediction_slurm(
    input_path=os.environ["INPUT_PATH"],
    input_key=os.environ["INPUT_KEY"],
    output_folder=os.environ["OUTPUT_FOLDER"],
    model_path=os.environ["MODEL_PATH"],
    prediction_instances=os.environ["PREDICTION_INSTANCES"],
)'

mkdir -p "$OUTPUT_FOLDER/prediction_tasks_dilated"
touch "$OUTPUT_FOLDER/prediction_tasks_dilated/task_${SLURM_ARRAY_TASK_ID}_of_${PREDICTION_INSTANCES}.complete"
