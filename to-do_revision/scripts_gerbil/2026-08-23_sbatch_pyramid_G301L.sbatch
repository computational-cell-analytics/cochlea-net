#!/bin/bash
#SBATCH --job-name=pyramid-G301L
#SBATCH --time=04:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-1

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
SCRIPT=$SCRIPT_REPO/to-do_revision/scripts_gerbil/create_multiscale_pyramid.py
PREDICTION_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L

# Array task 0 processes the SGNs, task 1 the IHCs.
case "${SLURM_ARRAY_TASK_ID}" in
	0)
		SEGMENTATION_FOLDER=$PREDICTION_FOLDER/SGN_v2
		;;
	1)
		SEGMENTATION_FOLDER=$PREDICTION_FOLDER/IHC_v11
		;;
	*)
		echo "Unexpected array task id: ${SLURM_ARRAY_TASK_ID}" >&2
		exit 1
		;;
esac

for path in "$SCRIPT" "$SEGMENTATION_FOLDER/segmentation.zarr/segmentation/zarr.json"; do
	if [[ ! -e "$path" ]]; then
		echo "Required path does not exist: $path" >&2
		echo "Run the watershed for $SEGMENTATION_FOLDER successfully first." >&2
		exit 1
	fi
done

cd "$SCRIPT_REPO"
export PYTHONPATH="$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"

echo "Array task: ${SLURM_ARRAY_TASK_ID}"
echo "Segmentation: $SEGMENTATION_FOLDER/segmentation.zarr"

# The levels are downsampled by 2, 4, 8, 16 and 32, each from the previous level.
# Levels that already exist are skipped, so pass --force to recompute them.
# The default 64^3 chunks in 256^3 shards keep the number of files low for this sparse
# segmentation. Each block covers one shard, so a worker holds a 512^3 uint64 region of the
# previous level in memory (~1 GB) while downsampling; hence the memory request above.
python "$SCRIPT" -i "$SEGMENTATION_FOLDER" \
	--scale_factors 2 2 2 2 2 --voxel_size 0.38 \
	--chunks 64 64 64 --shards 256 256 256 --n_threads "${SLURM_CPUS_PER_TASK}"
