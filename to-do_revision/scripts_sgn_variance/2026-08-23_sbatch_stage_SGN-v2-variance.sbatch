#!/bin/bash
#SBATCH --job-name=stage-SGNv2var
#SBATCH --time=04:00:00
#SBATCH --partition=standard96s:shared
#SBATCH --account=nim00007
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --array=0-4

# Stage the inputs of one cochlea into the workspace and prepare the sharded prediction arrays.
#
# The mask and the mean/standard deviation were computed once per cochlea and copied into the four
# version folders on vast, so they are copied from version 1 here and kept once per cochlea. Only
# the prediction needs them; the watershed reads its foreground from the prediction itself.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
activate_env
set -euo pipefail

COCHLEA=${COCHLEAE[$SLURM_ARRAY_TASK_ID]}
SOURCE_FOLDER=$OLD_PREDICTION_ROOT/$COCHLEA/SGN_v2-1
TARGET_FOLDER=$(cochlea_folder "$COCHLEA")

require_paths "$SOURCE_FOLDER/mask.zarr" "$SOURCE_FOLDER/mean_std.json" "$(raw_path "$COCHLEA")"

echo "Cochlea: $COCHLEA"
echo "Source: $SOURCE_FOLDER"
echo "Target: $TARGET_FOLDER"

# The four version folders are supposed to hold the same normalization; fail loudly if they do not,
# because that would mean the four predictions are not comparable.
reference_md5=$(md5sum "$SOURCE_FOLDER/mean_std.json" | cut -d' ' -f1)
for version in "${VERSIONS[@]}"; do
	other=$OLD_PREDICTION_ROOT/$COCHLEA/SGN_v2-$version/mean_std.json
	require_paths "$other"
	if [[ "$(md5sum "$other" | cut -d' ' -f1)" != "$reference_md5" ]]; then
		echo "mean_std.json of SGN_v2-$version differs from SGN_v2-1" >&2
		exit 1
	fi
done

mkdir -p "$TARGET_FOLDER"

# -a preserves times so a re-run of this script is a cheap no-op for the already copied mask.
if [[ -d "$TARGET_FOLDER/mask.zarr" ]]; then
	echo "mask.zarr already staged, skipping the copy."
else
	echo "Copying mask.zarr (this is many small files, it takes a few minutes)."
	cp -a "$SOURCE_FOLDER/mask.zarr" "$TARGET_FOLDER/mask.zarr.incomplete"
	mv "$TARGET_FOLDER/mask.zarr.incomplete" "$TARGET_FOLDER/mask.zarr"
fi
cp -a "$SOURCE_FOLDER/mean_std.json" "$TARGET_FOLDER/mean_std.json"

# Counts the in-mask blocks per shard, assigns them to the array tasks, and fails if the mask does
# not hold exactly the number of in-mask blocks the existing predictions were computed with.
python -m sgn_variance manifest \
	--folder "$TARGET_FOLDER" \
	--prediction_instances "$PREDICTION_INSTANCES" \
	--n_threads "${SLURM_CPUS_PER_TASK}" \
	--expect_blocks "${EXPECT_BLOCKS[$COCHLEA]}"

# Creating the arrays here, in a single process, is not optional: elf's require_dataset validates
# only dtype and shape for an existing array and silently ignores 'shards', and the ten prediction
# tasks would otherwise race to create it.
python -m sgn_variance init --folder "$TARGET_FOLDER" --versions "${VERSIONS[@]}"

echo "Staging for $COCHLEA done."
