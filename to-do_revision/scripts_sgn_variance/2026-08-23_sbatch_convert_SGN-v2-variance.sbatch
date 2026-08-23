#!/bin/bash
#SBATCH --job-name=convert-SGNv2var
#SBATCH --time=12:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --array=0-3%2

# Convert the four existing M_LR_000169_R predictions from the unsharded layout on vast into the
# sharded arrays in the workspace, preserving the values exactly. Throttled to two tasks at a time
# because this is bound by reading 326 GB from vast and writing it back to lustre, not by the CPU.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

COCHLEA=$CONVERTED_COCHLEA
VERSION=${VERSIONS[$SLURM_ARRAY_TASK_ID]}

COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
OUTPUT_FOLDER=$(version_folder "$COCHLEA" "$VERSION")
SOURCE=$OLD_PREDICTION_ROOT/$COCHLEA/SGN_v2-$VERSION/predictions.zarr

require_paths "$SOURCE/prediction" "$COCHLEA_FOLDER/shard_manifest.json" \
	"$OUTPUT_FOLDER/predictions.zarr"

echo "Cochlea: $COCHLEA, version: $VERSION"
echo "Source: $SOURCE"
echo "Target: $OUTPUT_FOLDER/predictions.zarr"

# block_shape == shards, so every write covers a whole shard and never has to load and merge a
# partially filled one. Eight workers hold a 1.6 GB read and a 1.6 GB write block each.
python -m sgn_variance convert \
	--input "$SOURCE" \
	--output_folder "$OUTPUT_FOLDER" \
	--num_workers 8

# Compare whole shards against the original before the original may be deleted.
python -m sgn_variance verify \
	--cochlea_folder "$COCHLEA_FOLDER" \
	--output_folder "$OUTPUT_FOLDER" \
	--reference "$SOURCE" \
	--n_samples 20
