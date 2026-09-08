#!/bin/bash
#SBATCH --job-name=apply-IHCv11-G301L
#SBATCH --time=02:00:00
#SBATCH --partition=grete:shared
#SBATCH --account=nim00007
#SBATCH --gpus=A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-9

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
PIPELINE_SCRIPT_DIR=$SCRIPT_REPO/to-do_revision/scripts_synapses
INPUT_PATH=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/G301L/GLR_301L_Vglut3_fused.n5
INPUT_KEY=setup0/timepoint0/s0
OUTPUT_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/IHC_v11
MODEL_PATH=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v11_cochlea_distance_unet_IHC_supervised_2026-07-20
PREDICTION_INSTANCES=10

for path in "$SCRIPT_REPO" "$PIPELINE_SCRIPT_DIR/distance_unet_checkpoint.py" "$INPUT_PATH" "$MODEL_PATH"; do
	if [[ ! -e "$path" ]]; then
		echo "Required path does not exist: $path" >&2
		exit 1
	fi
done

for path in "$OUTPUT_FOLDER/mask.zarr" "$OUTPUT_FOLDER/mean_std.json"; do
	if [[ ! -e "$path" ]]; then
		echo "Missing preprocessing result: $path" >&2
		echo "Run 2026-08-22_sbatch_mask_IHC-v11_G301L.sbatch successfully first." >&2
		exit 1
	fi
done

cd "$SCRIPT_REPO"
export PYTHONPATH="$PIPELINE_SCRIPT_DIR:$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "Array task: ${SLURM_ARRAY_TASK_ID}/${PREDICTION_INSTANCES}"
echo "Input: $INPUT_PATH ($INPUT_KEY)"
echo "Output: $OUTPUT_FOLDER"
echo "Model: $MODEL_PATH"

export INPUT_PATH INPUT_KEY OUTPUT_FOLDER MODEL_PATH PREDICTION_INSTANCES
python -c 'import os
from distance_unet_checkpoint import load_distance_unet
from flamingo_tools.segmentation import unet_prediction

# Avoid restoring the training datasets embedded in the trainer checkpoint.
unet_prediction.load_model = load_distance_unet

unet_prediction.run_unet_prediction_slurm(
    input_path=os.environ["INPUT_PATH"],
    input_key=os.environ["INPUT_KEY"],
    output_folder=os.environ["OUTPUT_FOLDER"],
    model_path=os.environ["MODEL_PATH"],
    prediction_instances=os.environ["PREDICTION_INSTANCES"],
)'
