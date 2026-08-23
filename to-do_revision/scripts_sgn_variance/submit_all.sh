#!/bin/bash
# Submit the full chain for one or more cochleae, with the dependencies between the stages.
#
#   bash submit_all.sh M_LR_000226_L
#   bash submit_all.sh --dry_run M_AMD_000058_L M_LR_000227_L M_LR_000227_R
#
# Run one cochlea at a time: four predictions of the largest cochlea are 1.3 TB, and
# cleanup_predictions.sh has to run before the next one starts. Staging must be done first.
#
# Versions whose prediction is already complete are skipped, so re-running this after a partial run
# does not repeat the prediction (which for the converted cochlea would be four hours of I/O).

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance/common.sh
# activate_env before 'set -u': sourcing ~/.bashrc trips over an unbound variable in /etc/bashrc.
activate_env
set -euo pipefail

cd "$SGN_VARIANCE_DIR"

DRY_RUN=0
if [[ "${1:-}" == "--dry_run" ]]; then
	DRY_RUN=1
	shift
fi

if [[ $# -eq 0 ]]; then
	echo "Usage: bash submit_all.sh [--dry_run] <cochlea> [<cochlea> ...]" >&2
	exit 1
fi

submit() {
	if [[ "$DRY_RUN" -eq 1 ]]; then
		echo "    would run: sbatch $*" >&2
		echo "DRYRUN"
	else
		sbatch --parsable "$@"
	fi
}

# True when every version of this cochlea already holds all the shards the manifest expects.
predictions_complete() {
	local cochlea=$1 version
	for version in "${VERSIONS[@]}"; do
		if ! python -m sgn_variance verify \
			--cochlea_folder "$(cochlea_folder "$cochlea")" \
			--output_folder "$(version_folder "$cochlea" "$version")" >/dev/null 2>&1; then
			return 1
		fi
	done
	return 0
}

for COCHLEA in "$@"; do
	COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
	require_paths "$COCHLEA_FOLDER/shard_manifest.json"
	for version in "${VERSIONS[@]}"; do
		require_paths "$(version_folder "$COCHLEA" "$version")/predictions.zarr"
	done

	echo "=== $COCHLEA ==="
	prediction_jobs=()

	if predictions_complete "$COCHLEA"; then
		echo "  predictions already complete for all versions, skipping."
	elif [[ "$COCHLEA" == "$CONVERTED_COCHLEA" ]]; then
		# The predictions of this cochlea already exist and are converted instead of recomputed.
		job=$(submit -J "conv-$COCHLEA" 2026-08-23_sbatch_convert_SGN-v2-variance.sbatch)
		echo "  convert: $job"
		prediction_jobs+=("$job")
	else
		for version in "${VERSIONS[@]}"; do
			job=$(submit -J "apply-$COCHLEA-v$version" \
				2026-08-23_sbatch_apply_SGN-v2-variance.sbatch "$COCHLEA" "$version")
			echo "  apply v2-$version: $job"
			prediction_jobs+=("$job")
		done
	fi

	watershed_args=(-J "ws-$COCHLEA")
	if [[ ${#prediction_jobs[@]} -gt 0 ]]; then
		dependency=$(IFS=:; echo "${prediction_jobs[*]}")
		watershed_args+=(--dependency="afterok:$dependency")
	else
		dependency="none"
	fi
	watershed=$(submit "${watershed_args[@]}" \
		2026-08-23_sbatch_watershed_SGN-v2-variance.sbatch "$COCHLEA")
	echo "  watershed: $watershed (after $dependency)"

	# aftercorr pairs task i of the table array with task i of the watershed array, so the table of
	# a version starts as soon as that version's watershed is done. Both arrays are 0-3 with the
	# same task -> version mapping, which is what makes this correct.
	table=$(submit -J "tab-$COCHLEA" \
		--dependency="aftercorr:$watershed" \
		2026-08-23_sbatch_table_SGN-v2-variance.sbatch "$COCHLEA")
	echo "  table: $table (aftercorr $watershed)"
	echo "  when it is done: bash cleanup_predictions.sh $COCHLEA"
done
