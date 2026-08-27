#!/bin/bash
#SBATCH --job-name=ws-SGNv2-G301L
#SBATCH --time=36:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
OUTPUT_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/SGN_v2

# The completed SGN mask contains 8,562 mask blocks. Each mask block covers
# four 128 x 128 x 128 prediction blocks.
EXPECTED_PREDICTION_BLOCKS=34248

for path in "$SCRIPT_REPO" "$OUTPUT_FOLDER/mask.zarr" "$OUTPUT_FOLDER/predictions.zarr/prediction/zarr.json"; do
	if [[ ! -e "$path" ]]; then
		echo "Missing watershed prerequisite: $path" >&2
		exit 1
	fi
done

# Missing prediction chunks are read as zeros by Zarr, so reject an incomplete
# prediction explicitly before starting the watershed.
for channel in 0 1 2; do
	chunk_dir="$OUTPUT_FOLDER/predictions.zarr/prediction/c/$channel"
	if [[ ! -d "$chunk_dir" ]]; then
		echo "Missing prediction channel: $chunk_dir" >&2
		exit 1
	fi
	written_blocks=$(find "$chunk_dir" -type f | wc -l)
	if [[ "$written_blocks" -ne "$EXPECTED_PREDICTION_BLOCKS" ]]; then
		echo "Incomplete prediction channel $channel: expected $EXPECTED_PREDICTION_BLOCKS blocks, found $written_blocks" >&2
		exit 1
	fi
done

cd "$SCRIPT_REPO"
export PYTHONPATH="$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OUTPUT_FOLDER

echo "Output: $OUTPUT_FOLDER"
echo "Watershed: min_size=1000, center=0.4, boundary=0.5, foreground=0.5, smoothing=0.0"

python -c 'import os
from flamingo_tools.segmentation.unet_prediction import run_unet_segmentation_slurm

run_unet_segmentation_slurm(
    output_folder=os.environ["OUTPUT_FOLDER"],
    min_size=1000,
    center_distance_threshold=0.4,
    boundary_distance_threshold=0.5,
    fg_threshold=0.5,
    distance_smoothing=0.0,
)'
