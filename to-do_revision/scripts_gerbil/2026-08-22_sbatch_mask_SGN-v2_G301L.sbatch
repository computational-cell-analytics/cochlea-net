#!/bin/bash
#SBATCH --job-name=mask-SGNv2-G301L
#SBATCH --time=02:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
INPUT_PATH=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/G301L/GLR_301L_PV_fused.n5
INPUT_KEY=setup0/timepoint0/s0
OUTPUT_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/SGN_v2

# Fixed value used for the previous Gerbil SGN runs. For SGNs, find_mask uses
# the 95th percentile of each masking block and includes blocks above 200.
SEG_CLASS=sgn
ABSOLUTE_THRESHOLD=200

for path in "$SCRIPT_REPO" "$INPUT_PATH"; do
	if [[ ! -e "$path" ]]; then
		echo "Required path does not exist: $path" >&2
		exit 1
	fi
done

mkdir -p "$OUTPUT_FOLDER"
cd "$SCRIPT_REPO"
export PYTHONPATH="$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "Input: $INPUT_PATH ($INPUT_KEY)"
echo "Output: $OUTPUT_FOLDER"
echo "Mask: class=$SEG_CLASS, absolute_threshold=$ABSOLUTE_THRESHOLD"

export INPUT_PATH INPUT_KEY OUTPUT_FOLDER SEG_CLASS ABSOLUTE_THRESHOLD
python -c 'import os
from flamingo_tools.segmentation.unet_prediction import run_unet_prediction_preprocess_slurm

run_unet_prediction_preprocess_slurm(
    input_path=os.environ["INPUT_PATH"],
    input_key=os.environ["INPUT_KEY"],
    output_folder=os.environ["OUTPUT_FOLDER"],
    seg_class=os.environ["SEG_CLASS"],
    absolute_threshold=os.environ["ABSOLUTE_THRESHOLD"],
)'
