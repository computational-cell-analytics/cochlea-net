#!/bin/bash
#SBATCH --job-name=table-IHCv11-G301L-dil
#SBATCH --time=01:00:00
#SBATCH --partition=standard96s:test
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

source ~/.bashrc
micromamba activate new-stack || exit 1
set -euo pipefail

SCRIPT_REPO=/user/pape41/u12086/Work/my_projects/flamingo-tools
SCRIPT=$SCRIPT_REPO/to-do_revision/scripts_gerbil/create_table_and_components.py
SEGMENTATION_FOLDER=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/IHC_v11_dilated_mask1

for path in "$SCRIPT" "$SEGMENTATION_FOLDER/segmentation.zarr/segmentation/zarr.json"; do
	if [[ ! -e "$path" ]]; then
		echo "Required path does not exist: $path" >&2
		exit 1
	fi
done

cd "$SCRIPT_REPO"
export PYTHONPATH="$SCRIPT_REPO${PYTHONPATH:+:$PYTHONPATH}"

python "$SCRIPT" -i "$SEGMENTATION_FOLDER" --cell_type ihc \
	--voxel_size 0.38 --n_threads "${SLURM_CPUS_PER_TASK}" --min_component_length 10
