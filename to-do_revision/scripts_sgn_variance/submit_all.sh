#!/bin/bash
# Submit the full chain for one or more cochleae, with the dependencies between the stages.
#
#   bash submit_all.sh M_LR_000226_L
#   bash submit_all.sh --preemptible M_AMD_000058_L
#   bash submit_all.sh --dry_run M_AMD_000058_L M_LR_000227_L M_LR_000227_R
#
# --preemptible sends the prediction to grete:preemptible on a 1g.10gb MIG slice instead of a whole
# A100 on grete:shared. That partition is usually free while grete:shared is not, and the model is
# small enough that a seventh of an A100 costs only ~15% over a 1g.20gb slice: measured 0.72 s per
# block against 0.62 s. Preemption is cheap here because a shard is written in one atomic write and
# --skip_existing resumes at the next one, so a kill costs at most one shard, about 20 s of work.
# Only the prediction moves; the watershed and the table are CPU jobs.
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
PREEMPTIBLE=0
while [[ $# -gt 0 ]]; do
	case "$1" in
		--dry_run) DRY_RUN=1; shift ;;
		--preemptible) PREEMPTIBLE=1; shift ;;
		*) break ;;
	esac
done

# Extra sbatch arguments for the prediction only. The watershed and the table run on CPU nodes.
gpu_args=()
if [[ "$PREEMPTIBLE" -eq 1 ]]; then
	gpu_args=(--partition=grete:preemptible --gpus=1g.10gb:1)
fi

if [[ $# -eq 0 ]]; then
	echo "Usage: bash submit_all.sh [--dry_run] [--preemptible] <cochlea> [<cochlea> ...]" >&2
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

# Validate every argument before submitting anything. Checking inside the submission loop meant a
# typo in the third cochlea aborted after the first two chains were already queued, leaving a
# half-submitted state to untangle.
for COCHLEA in "$@"; do
	known=0
	for candidate in "${COCHLEAE[@]}"; do
		[[ "$COCHLEA" == "$candidate" ]] && known=1
	done
	if [[ "$known" -eq 0 ]]; then
		echo "Unknown cochlea: '$COCHLEA'. Expected one of:" >&2
		printf '  %s\n' "${COCHLEAE[@]}" >&2
		exit 1
	fi

	COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")
	if [[ ! -f "$COCHLEA_FOLDER/shard_manifest.json" ]]; then
		echo "$COCHLEA is not staged: no $COCHLEA_FOLDER/shard_manifest.json." >&2
		echo "Run the staging job for it first." >&2
		exit 1
	fi
	for version in "${VERSIONS[@]}"; do
		folder=$(version_folder "$COCHLEA" "$version")
		if [[ ! -d "$folder/predictions.zarr" ]]; then
			echo "$COCHLEA SGN_v2-$version has no prediction array at $folder." >&2
			echo "Re-run the staging job for it; it creates them." >&2
			exit 1
		fi
	done
done

for COCHLEA in "$@"; do
	COCHLEA_FOLDER=$(cochlea_folder "$COCHLEA")

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
				${gpu_args[@]+"${gpu_args[@]}"} \
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
