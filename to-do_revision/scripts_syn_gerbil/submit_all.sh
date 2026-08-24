#!/bin/bash
# Submit the three-stage synapse detection for one or more cochleae, with the dependencies
# between the stages.
#
#   bash submit_all.sh --dry_run G_LR_000301_L
#   bash submit_all.sh G_LR_000301_L
#   bash submit_all.sh --reset G_LR_000302_R
#
# The preprocessing stage is skipped when mask.zarr and mean_std.json already exist, so
# re-running this after a failed prediction does not recompute them.
#
# A pre-existing predictions.zarr is refused rather than reused. The prediction array has no
# skip-existing check by design (the dataset is created by whichever task starts first), so
# resubmitting on top of a partial array leaves the missing blocks empty and the detection
# step happily writes a table from them. That is exactly how the published G_LR_000302_R
# result ended up covering 44 % of its IHCs. Pass --reset to delete the stale prediction and
# the tables derived from it.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
# activate_env before 'set -u': sourcing ~/.bashrc trips over an unbound variable in /etc/bashrc.
activate_env
set -euo pipefail

cd "$SYN_GERBIL_DIR"

DRY_RUN=0
RESET=0
while [[ $# -gt 0 ]]; do
	case "$1" in
		--dry_run) DRY_RUN=1; shift ;;
		--reset) RESET=1; shift ;;
		-*) echo "Unknown option: $1" >&2; exit 1 ;;
		*) break ;;
	esac
done

if [[ $# -eq 0 ]]; then
	echo "Usage: bash submit_all.sh [--dry_run] [--reset] <cochlea> [<cochlea> ...]" >&2
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

# Validate every argument before submitting anything: a typo in the second cochlea should not
# abort after the first chain is already queued.
for COCHLEA in "$@"; do
	require_known_cochlea "$COCHLEA"
	require_paths "$(raw_path "$COCHLEA")" "$(mask_path "$COCHLEA")" "$MODEL"

	OUTPUT_FOLDER=$(output_folder "$COCHLEA")
	if [[ -d "$OUTPUT_FOLDER/predictions.zarr" && "$RESET" -eq 0 ]]; then
		echo "$COCHLEA already has $OUTPUT_FOLDER/predictions.zarr." >&2
		echo "Resubmitting the array on top of it would leave any missing blocks empty." >&2
		echo "Pass --reset to delete it and start the prediction over." >&2
		exit 1
	fi
done

for COCHLEA in "$@"; do
	OUTPUT_FOLDER=$(output_folder "$COCHLEA")
	echo "=== $COCHLEA ==="

	if [[ -d "$OUTPUT_FOLDER/predictions.zarr" ]]; then
		# Only reachable with --reset, the validation loop above rejected it otherwise.
		echo "  --reset: removing the stale prediction and the tables derived from it"
		if [[ "$DRY_RUN" -eq 0 ]]; then
			rm -rf "$OUTPUT_FOLDER/predictions.zarr"
			rm -f "$OUTPUT_FOLDER/synapse_detection.tsv" \
				"$OUTPUT_FOLDER/synapse_detection_filtered.tsv"
		fi
	fi

	preprocess_job=""
	if [[ -d "$OUTPUT_FOLDER/mask.zarr" && -f "$OUTPUT_FOLDER/mean_std.json" ]]; then
		echo "  preprocess: already done, skipping"
	else
		preprocess_job=$(submit -J "pre-syn-$COCHLEA" \
			2026-08-24_sbatch_preprocess_syn-v3.sbatch "$COCHLEA")
		echo "  preprocess: $preprocess_job"
	fi

	apply_args=(-J "apply-syn-$COCHLEA")
	if [[ -n "$preprocess_job" ]]; then
		apply_args+=(--dependency="afterok:$preprocess_job")
	fi
	apply_job=$(submit "${apply_args[@]}" \
		2026-08-24_sbatch_apply_syn-v3.sbatch "$COCHLEA")
	echo "  apply: $apply_job (after ${preprocess_job:-none})"

	# afterok on an array job waits for every task, which is what the detection needs.
	detect_job=$(submit -J "detect-syn-$COCHLEA" \
		--dependency="afterok:$apply_job" \
		2026-08-24_sbatch_detect_syn-v3.sbatch "$COCHLEA")
	echo "  detect: $detect_job (afterok $apply_job)"
	echo "  when it is done: python check_coverage.py $COCHLEA"
done
