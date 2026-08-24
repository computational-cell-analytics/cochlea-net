#!/bin/bash
# Submit the three-stage synapse detection for one or more cochleae, with the dependencies
# between the stages.
#
#   bash submit_all.sh --dry_run G_LR_000301_L
#   bash submit_all.sh G_LR_000301_L
#   bash submit_all.sh --preemptible G_LR_000301_L
#   bash submit_all.sh --preemptible --slice 1g.20gb G_LR_000301_L
#   bash submit_all.sh --reset G_LR_000302_R
#
# --preemptible sends the prediction to grete:preemptible on a MIG slice instead of a whole
# A100 on grete:shared. Only the prediction moves; the preprocessing and the detection are CPU
# jobs. --slice overrides PREEMPTIBLE_SLICE from common.sh; 1g.10gb is rejected because the
# forward pass does not fit in it, see the README for the measurements.
#
# Preemption is safe here, but only because of three things that have to stay true:
#   - the apply job sets --requeue, so a preempted task comes back instead of vanishing;
#   - a task redoes its whole share on requeue, which is a few minutes, because there is no
#     resume inside the library;
#   - the detection depends on 'afterok' of the whole array, so it cannot run on a partial
#     volume. If a task is killed for good the detection stays PENDING with
#     DependencyNeverSatisfied. That is the intended outcome, not a bug: run
#     verify_prediction.py, resubmit the tasks it names, then submit the detection by hand.
#
# The preprocessing stage is skipped when mask.zarr and mean_std.json already exist, so
# re-running this after a failed prediction does not recompute them.
#
# A pre-existing predictions.zarr is refused rather than reused. The prediction array has no
# skip-existing check by design (the dataset is created by whichever task starts first), so
# resubmitting the whole array on top of a partial one leaves the missing blocks empty and the
# detection step happily writes a table from them. That is how the published G_LR_000302_R
# result ended up covering a third of its helix. Pass --reset to delete the stale prediction
# and the tables derived from it, or resubmit single tasks as verify_prediction.py instructs.

# sbatch runs a copy of this script from the slurm spool directory, so the path cannot be
# derived from BASH_SOURCE.
source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh
# activate_env before 'set -u': sourcing ~/.bashrc trips over an unbound variable in /etc/bashrc.
activate_env
set -euo pipefail

cd "$SYN_GERBIL_DIR"

DRY_RUN=0
RESET=0
PREEMPTIBLE=0
SLICE="$PREEMPTIBLE_SLICE"
while [[ $# -gt 0 ]]; do
	case "$1" in
		--dry_run) DRY_RUN=1; shift ;;
		--reset) RESET=1; shift ;;
		--preemptible) PREEMPTIBLE=1; shift ;;
		--slice) SLICE=${2:?"--slice needs a value"}; PREEMPTIBLE=1; shift 2 ;;
		-*) echo "Unknown option: $1" >&2; exit 1 ;;
		*) break ;;
	esac
done

if [[ $# -eq 0 ]]; then
	echo "Usage: bash submit_all.sh [--dry_run] [--reset] [--preemptible] [--slice S] <cochlea> ..." >&2
	exit 1
fi

# Extra sbatch arguments for the prediction only.
gpu_args=()
if [[ "$PREEMPTIBLE" -eq 1 ]]; then
	# Measured: the forward pass occupies 17.6 GiB of tensors plus a context per prefetch
	# worker, so a 9.5 GiB slice dies with a CUDA OOM inside the first in-mask block. Refuse
	# it rather than let the array fail ten times over.
	if [[ "$SLICE" == 1g.10gb ]]; then
		echo "1g.10gb is too small: the forward pass needs ~17.6 GiB and OOMs there." >&2
		echo "Use 1g.20gb or 3g.40gb. See the README for the measurements." >&2
		exit 1
	fi
	gpu_args=(--partition="$PREEMPTIBLE_PARTITION" --gpus="$SLICE:1")
	echo "Prediction goes to $PREEMPTIBLE_PARTITION on a $SLICE slice."
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
		echo "Run 'python verify_prediction.py $COCHLEA' to see which tasks are missing," >&2
		echo "or pass --reset to delete it and start the prediction over." >&2
		exit 1
	fi
done

for COCHLEA in "$@"; do
	OUTPUT_FOLDER=$(output_folder "$COCHLEA")
	echo "=== $COCHLEA ==="

	if [[ -d "$OUTPUT_FOLDER/predictions.zarr" ]]; then
		# Only reachable with --reset, the validation loop above rejected it otherwise.
		echo "  --reset: removing the stale prediction, its receipts and the tables from it"
		if [[ "$DRY_RUN" -eq 0 ]]; then
			rm -rf "$OUTPUT_FOLDER/predictions.zarr" "$OUTPUT_FOLDER/tasks"
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
	apply_job=$(submit "${apply_args[@]}" ${gpu_args[@]+"${gpu_args[@]}"} \
		2026-08-24_sbatch_apply_syn-v3.sbatch "$COCHLEA")
	echo "  apply: $apply_job (after ${preprocess_job:-none})"

	# afterok on an array job waits for every task, which is what the detection needs.
	detect_job=$(submit -J "detect-syn-$COCHLEA" \
		--dependency="afterok:$apply_job" \
		2026-08-24_sbatch_detect_syn-v3.sbatch "$COCHLEA")
	echo "  detect: $detect_job (afterok $apply_job)"
	echo "  after the array : python verify_prediction.py $COCHLEA"
	echo "  after the detect: python check_coverage.py $COCHLEA"
done
