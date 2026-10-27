#!/bin/bash
#SBATCH --job-name=synapse-eval-v11
#SBATCH -t 04:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 64G

# Rules 1 and 2 of the v11 evaluation in "Rescoring result (2026-10-23)" of ../synapses.md, for the
# latest.pt of the three seeds v11-1..3:
# 1. select threshold and min_distance with the gridsearch on the consensus val crops (data v11,
#    filtered by the IHC segmentation as in the evaluation);
# 2. score test v11 at the selected setting of each seed and at the median setting of the three.
# The settings go to $OUT/selected_settings.txt, the scores to $OUT/synapses.json (keys
# v11-<i>-cons and v11-<i>-median). There is no (0.5, 2) arm, because the fixed target peaks near 4,
# and no old arm, because the old val crops are training data of v11.
# Every step reads the code from $REPO, so do not switch its branch while the job runs.

source ~/.bashrc
micromamba activate cochlea-net || exit 1

REPO=/user/schilling40/u15000/flamingo-tools
export PYTHONPATH="$REPO:$PYTHONPATH"
ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
DATA=$ROOT/training_data/synapses
IHC=$ROOT/trained_models/IHC/v11_cochlea_distance_unet_IHC_supervised_2026-07-20
REF=$ROOT/AnnotatedImageCrops/Synapses_2026-04/consensus_annotation
TEST=$DATA/test_data/v11/images
# IHC v11 segmentations of all six consensus crops, for the gridsearch on the consensus val crops.
IHC_SEG=$ROOT/predictions/val_synapses/production_2026-09-25/v7
# IHC v11 output of the four test crops from the rescoring, linked so that it is not predicted again.
IHC_TEST=$ROOT/predictions/val_synapses/rescore_2026-10-23/v3-fixed
OUT=$ROOT/predictions/val_synapses/v11_2026-10-26
CKPT=/mnt/lustre-grete/usr/u15000/checkpoints

# Prints "<threshold> <min_distance>" of the best F1 on the val crops of a split file.
# Further arguments go to the gridsearch.
select_setting() {
	python -m flamingo_tools.synapse_detection.gridsearch -m "$1" -s "$2/train_val_split.json" \
		-i "$2/images" -l "$2/labels" "${@:3}" |
		sed -n 's/^Best: threshold \([0-9.]*\), min_distance \([0-9]*\).*/\1 \2/p'
}

mkdir -p "$OUT"
NAMES=(v11-1 v11-2 v11-3)
THRESHOLDS=()
MIN_DISTANCES=()
for NAME in "${NAMES[@]}"; do
	read -r T MD < <(select_setting "$CKPT/synapse_detection_$NAME/latest.pt" "$DATA/training_data/v11" \
		--ihc_dir "$IHC_SEG")
	[ -n "$MD" ] || { echo "The gridsearch gave no setting for $NAME."; exit 1; }
	echo "$NAME cons $T $MD" | tee -a "$OUT/selected_settings.txt"
	THRESHOLDS+=("$T")
	MIN_DISTANCES+=("$MD")
done
# The median of three values is the second in sorted order.
T_MED=$(printf '%s\n' "${THRESHOLDS[@]}" | sort -g | sed -n 2p)
MD_MED=$(printf '%s\n' "${MIN_DISTANCES[@]}" | sort -g | sed -n 2p)
echo "median $T_MED $MD_MED" | tee -a "$OUT/selected_settings.txt"

for i in "${!NAMES[@]}"; do
	NAME=${NAMES[$i]}
	for SETTING in "cons ${THRESHOLDS[$i]} ${MIN_DISTANCES[$i]}" "median $T_MED $MD_MED"; do
		read -r TAG T MD <<< "$SETTING"
		RUN="$OUT/$NAME-$TAG"
		mkdir -p "$RUN"
		ln -sfn "$IHC_TEST"/*_ihc "$RUN"/
		python "$REPO"/scripts/validation/synapses/prediction.py -o "$RUN" -i "$TEST" -g "$REF" \
			--model_synapse "$CKPT/synapse_detection_$NAME/latest.pt" --model_ihc "$IHC" \
			--threshold "$T" --min_distance "$MD"
		python "$REPO"/scripts/validation/synapses/run_evaluation.py -p "$RUN" -c "$TEST" -r "$REF" -o "$OUT"
	done
done
