#!/bin/bash
# Submit the full chain for one or more cochleae, with the dependencies between the stages.
#
#   bash submit_all.sh M_LR_000226_L
#   bash submit_all.sh M_AMD_000058_L M_LR_000227_L M_LR_000227_R
#
# Run one cochlea at a time: four predictions of the largest cochlea are 1.3 TB, and
# cleanup_predictions.sh has to run before the next one starts. Staging must be done first.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
set -euo pipefail

cd "$SGN_VARIANCE_DIR"

if [[ $# -eq 0 ]]; then
	echo "Usage: bash submit_all.sh <cochlea> [<cochlea> ...]" >&2
	exit 1
fi

for COCHLEA in "$@"; do
	COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
	require_paths "$COCHLEA_FOLDER/shard_manifest.json"
	for version in "${VERSIONS[@]}"; do
		require_paths "$(version_folder "$COCHLEA" "$version")/predictions.zarr"
	done

	echo "=== $COCHLEA ==="
	prediction_jobs=()

	if [[ "$COCHLEA" == "$CONVERTED_COCHLEA" ]]; then
		# The predictions of this cochlea already exist and are converted instead of recomputed.
		job=$(sbatch --parsable -J "conv-$COCHLEA" \
			2026-08-23_sbatch_convert_SGN-v2-variance.sbatch)
		echo "  convert: $job"
		prediction_jobs+=("$job")
	else
		for version in "${VERSIONS[@]}"; do
			job=$(sbatch --parsable -J "apply-$COCHLEA-v$version" \
				2026-08-23_sbatch_apply_SGN-v2-variance.sbatch "$COCHLEA" "$version")
			echo "  apply v2-$version: $job"
			prediction_jobs+=("$job")
		done
	fi

	dependency=$(IFS=:; echo "${prediction_jobs[*]}")
	watershed=$(sbatch --parsable -J "ws-$COCHLEA" \
		--dependency="afterok:$dependency" \
		2026-08-23_sbatch_watershed_SGN-v2-variance.sbatch "$COCHLEA")
	echo "  watershed: $watershed (after $dependency)"

	# aftercorr pairs task i of the table array with task i of the watershed array, so the table of
	# a version starts as soon as that version's watershed is done. Both arrays are 0-3 with the
	# same task -> version mapping, which is what makes this correct.
	table=$(sbatch --parsable -J "tab-$COCHLEA" \
		--dependency="aftercorr:$watershed" \
		2026-08-23_sbatch_table_SGN-v2-variance.sbatch "$COCHLEA")
	echo "  table: $table (aftercorr $watershed)"
	echo "  when it is done: bash cleanup_predictions.sh $COCHLEA"
done
