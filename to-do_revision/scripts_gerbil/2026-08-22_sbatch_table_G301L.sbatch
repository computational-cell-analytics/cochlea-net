#!/bin/bash
#SBATCH --job-name=table-G301L
#SBATCH --time=02:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-1

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
SCRIPT=$SCRIPT_REPO/to-do_revision/scripts_gerbil/create_table_and_components.py
PREDICTION_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L

# Array task 0 processes the SGNs, task 1 the IHCs.
case "${SLURM_ARRAY_TASK_ID}" in
	0)
		SEGMENTATION_FOLDER=$PREDICTION_FOLDER/SGN_v2
		CELL_TYPE=sgn
		EXTRA_ARGS=()
		;;
	1)
		SEGMENTATION_FOLDER=$PREDICTION_FOLDER/IHC_v11
		CELL_TYPE=ihc
		# The gerbil IHC spiral is usually split into several components, so the minimal component
		# length is lowered to also list the small ones, cf.
		# reproducibility/label_components/G_LR_000301_R_IHC.json.
		EXTRA_ARGS=(--min_component_length 10)
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
echo "Cell type: $CELL_TYPE"

# The segmentation table is only computed if it does not exist yet, so re-running this after
# adapting the component parameters above only repeats the (cheap) component labeling.
python "$SCRIPT" -i "$SEGMENTATION_FOLDER" --cell_type "$CELL_TYPE" \
	--voxel_size 0.38 --n_threads "${SLURM_CPUS_PER_TASK}" \
	${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}
