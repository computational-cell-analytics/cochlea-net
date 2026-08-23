#!/bin/bash
#SBATCH --job-name=table-SGNv2var
#SBATCH --time=04:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --array=0-3

# Segmentation table and connected components for the four seed variants of one cochlea. Usage:
#   sbatch 2026-08-23_sbatch_table_SGN-v2-variance.sbatch <cochlea>
#
# This replaces adding the segmentation to MoBIE, which is the usual way of obtaining the table.
# It writes default.tsv (the MoBIE columns) and default_components.tsv (with component_labels),
# both of which the local evaluation reads directly.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

COCHLEA=${1:?"pass the cochlea as the first argument"}
VERSION=${VERSIONS[$SLURM_ARRAY_TASK_ID]}

SCRIPT=$GERBIL_DIR/create_table_and_components.py
OUTPUT_FOLDER=$(version_folder "$COCHLEA" "$VERSION")

require_paths "$SCRIPT" "$OUTPUT_FOLDER/segmentation.zarr/segmentation"

echo "Cochlea: $COCHLEA, version: $VERSION"
echo "Segmentation: $OUTPUT_FOLDER/segmentation.zarr"

# default.tsv is only computed if it is missing, so re-running this after changing the component
# parameters below only repeats the cheap component labeling. The component defaults are the SGN
# ones (max_edge_distance 30 um, min_component_length 50, min_size 1000); no per-cochlea override
# exists for SGN_v2 in reproducibility/label_components.
python "$SCRIPT" -i "$OUTPUT_FOLDER" --cell_type sgn \
	--voxel_size 0.38 --n_threads "${SLURM_CPUS_PER_TASK}" -c 1
