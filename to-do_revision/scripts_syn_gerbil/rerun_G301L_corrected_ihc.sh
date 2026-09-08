#!/bin/bash
# Submit a clean production-v3 rerun for G_LR_000301_L against the finalized IHC_v11
# segmentation (the dilated-mask re-prediction used for the manuscript IHC count).
#
# The output is versioned and therefore does not overwrite the earlier v3 run:
#   $OUT_ROOT/G_LR_000301_L_IHC_v11_dilated_mask1
#
# Usage:
#   bash rerun_G301L_corrected_ihc.sh --dry-run
#   bash rerun_G301L_corrected_ihc.sh

set -euo pipefail

source /user/pape41/u12086/Work/my_projects/flamingo-tools/to-do_revision/scripts_syn_gerbil/common.sh

COCHLEA=G_LR_000301_L
OUTPUT_FOLDER_OVERRIDE="$OUT_ROOT/${COCHLEA}_IHC_v11_dilated_mask1"
COMPONENT_TABLE="$WS/prediction/G301L/IHC_v11_dilated_mask1/default_components.tsv"
CPU_PARTITION=large96s:test
CPU_TIME=01:00:00
DRY_RUN=0

if [[ $# -gt 1 ]]; then
	echo "Usage: bash $0 [--dry-run]" >&2
	exit 1
fi
if [[ $# -eq 1 ]]; then
	if [[ "$1" != "--dry-run" ]]; then
		echo "Unknown option: $1" >&2
		exit 1
	fi
	DRY_RUN=1
fi

cd "$SYN_GERBIL_DIR"
require_known_cochlea "$COCHLEA"
require_paths "$(raw_path "$COCHLEA")" "$(mask_path "$COCHLEA")" \
	"$COMPONENT_TABLE" "$MODEL"

# A clean output is essential. The array cannot resume safely on top of an existing prediction,
# and reusing old mask/stat files would silently use provenance from the earlier IHC segmentation.
for artifact in mask.zarr mean_std.json predictions.zarr tasks \
		synapse_detection.tsv synapse_detection_filtered.tsv; do
	if [[ -e "$OUTPUT_FOLDER_OVERRIDE/$artifact" ]]; then
		echo "Refusing to reuse $OUTPUT_FOLDER_OVERRIDE: found $artifact" >&2
		echo "Inspect or archive that versioned result before submitting another rerun." >&2
		exit 1
	fi
done

submit() {
	if [[ "$DRY_RUN" -eq 1 ]]; then
		printf 'sbatch --parsable' >&2
		printf ' %q' "$@" >&2
		printf '\n' >&2
		echo DRYRUN
	else
		sbatch --parsable "$@"
	fi
}

COMMON_EXPORT="ALL,OUTPUT_FOLDER_OVERRIDE=$OUTPUT_FOLDER_OVERRIDE"

echo "Cochlea: $COCHLEA"
echo "IHC mask: $(mask_path "$COCHLEA")"
echo "Selected-IHC table: $COMPONENT_TABLE (components 1-13; 946 IHCs)"
echo "Output: $OUTPUT_FOLDER_OVERRIDE"
echo "CPU stages: $CPU_PARTITION ($CPU_TIME)"
echo "GPU: one five-task array on $PREEMPTIBLE_PARTITION, $PREEMPTIBLE_SLICE per task"

preprocess_job=$(submit \
	-J pre-syn-G301L-ihc11 \
	--partition="$CPU_PARTITION" \
	--time="$CPU_TIME" \
	--export="$COMMON_EXPORT" \
	2026-08-24_sbatch_preprocess_syn-v3.sbatch "$COCHLEA" | tail -n 1)
echo "preprocess: $preprocess_job"

# Five array elements are the complete production block split. --requeue is also set in the
# sbatch file, so an individual preempted element repeats its own share before detection starts.
apply_job=$(submit \
	-J apply-syn-G301L-ihc11 \
	--dependency="afterok:$preprocess_job" \
	--partition="$PREEMPTIBLE_PARTITION" \
	--gpus="$PREEMPTIBLE_SLICE:1" \
	--array="0-$((PREDICTION_INSTANCES - 1))" \
	--export="$COMMON_EXPORT,SYN_ALLOC_CONF=" \
	2026-08-24_sbatch_apply_syn-v3.sbatch "$COCHLEA" | tail -n 1)
echo "prediction array: $apply_job (five GPU tasks; afterok $preprocess_job)"

detect_job=$(submit \
	-J detect-syn-G301L-ihc11 \
	--partition="$CPU_PARTITION" \
	--time="$CPU_TIME" \
	--dependency="afterok:$apply_job" \
	--export="$COMMON_EXPORT" \
	2026-08-24_sbatch_detect_syn-v3.sbatch "$COCHLEA" | tail -n 1)
echo "detection: $detect_job (afterok all five tasks of $apply_job)"

echo "After completion, validate with:"
echo "  python check_coverage.py $COCHLEA --output_folder $OUTPUT_FOLDER_OVERRIDE"
