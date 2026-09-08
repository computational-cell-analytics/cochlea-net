#!/bin/bash
#SBATCH --job-name=ws-IHCv11-G301L-dil
#SBATCH --time=36:00:00
#SBATCH --partition=large96s
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
OUTPUT_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/IHC_v11_dilated_mask1
EXPECTED_TASKS=10
EXPECTED_PREDICTION_BLOCKS=7140

for path in "$SCRIPT_REPO" "$OUTPUT_FOLDER/mask_delta.zarr" "$OUTPUT_FOLDER/mask_union.zarr" \
        "$OUTPUT_FOLDER/predictions.zarr/prediction/zarr.json"; do
	if [[ ! -e "$path" ]]; then
		echo "Missing watershed prerequisite: $path" >&2
		exit 1
	fi
done

receipts=$(find "$OUTPUT_FOLDER/prediction_tasks_dilated" -maxdepth 1 \
	-name 'task_*_of_10.complete' -type f | wc -l)
if [[ "$receipts" -ne "$EXPECTED_TASKS" ]]; then
	echo "Incomplete prediction array: expected $EXPECTED_TASKS receipts, found $receipts." >&2
	exit 1
fi

for channel in 0 1 2; do
	chunk_dir="$OUTPUT_FOLDER/predictions.zarr/prediction/c/$channel"
	written_blocks=$(find "$chunk_dir" -type f | wc -l)
	if [[ "$written_blocks" -ne "$EXPECTED_PREDICTION_BLOCKS" ]]; then
		echo "Incomplete prediction channel $channel: expected $EXPECTED_PREDICTION_BLOCKS blocks, found $written_blocks." >&2
		exit 1
	fi
done

# Prediction used only the delta mask. Preserve both masks and atomically switch the conventional
# mask.zarr path to the full union before downstream processing.
ln -sfn mask_union.zarr "$OUTPUT_FOLDER/mask.zarr"
if [[ "$(readlink "$OUTPUT_FOLDER/mask.zarr")" != "mask_union.zarr" ]]; then
	echo "Could not activate the union mask." >&2
	exit 1
fi

cd "$SCRIPT_REPO"
export PYTHONPATH="$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_FOLDER

echo "Output: $OUTPUT_FOLDER"
echo "Prediction blocks per channel: $EXPECTED_PREDICTION_BLOCKS"
echo "Watershed: min_size=1000, center=0.7, boundary=0.6, foreground=0.5, smoothing=1.0"

python -c 'import os
from flamingo_tools.segmentation.unet_prediction import run_unet_segmentation_slurm

run_unet_segmentation_slurm(
    output_folder=os.environ["OUTPUT_FOLDER"],
    min_size=1000,
    center_distance_threshold=0.7,
    boundary_distance_threshold=0.6,
    fg_threshold=0.5,
    distance_smoothing=1.0,
)'
