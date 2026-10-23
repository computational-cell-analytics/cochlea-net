#!/bin/bash
#SBATCH --job-name=synapse-rescore
#SBATCH -t 12:00:00
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 64G

# Step 3 of "Training robustness (2026-10-21)" in ../synapses.md. For every model:
# 1. select threshold and min_distance with the gridsearch on the consensus val crops (data v11,
#    filtered by the IHC segmentation as in the evaluation) and on the old val crops (data v7,
#    the radius filter, the former practice);
# 2. score test v11 at (0.5, 2) and at both selected settings.
# The selected settings go to $OUT/selected_settings.txt, the scores to $OUT/synapses.json
# (keys <model>-fixed, <model>-cons, <model>-old). Needs the data folders v11 and test v11 of
# ../synapses.md and the branch HEAD: --ihc_dir and the consensus-only labels of the gridsearch
# do not exist in 4da9e70, and the run of 2026-10-22 on that checkout got no cons setting.
# The exports are pickled U-Nets, which fail in torch_em 0.10.7 and later ('Decoder' object has
# no attribute 'explicit_skip_channels'). micro-sam_gpu has 0.10.8; cochlea-net must have <= 0.10.6.
# The v3-1..v3-4 exports are not readable for this account, so they are not in MODELS.

source ~/.bashrc
micromamba activate cochlea-net || exit 1

REPO=/user/schilling40/u15000/flamingo-tools
export PYTHONPATH="$REPO:$PYTHONPATH"
ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
SYN=$ROOT/trained_models/Synapses
DATA=$ROOT/training_data/synapses
IHC=$ROOT/trained_models/IHC/v11_cochlea_distance_unet_IHC_supervised_2026-07-20
REF=$ROOT/AnnotatedImageCrops/Synapses_2026-04/consensus_annotation
TEST=$DATA/test_data/v11/images
# IHC v11 segmentations of all six consensus crops, for the gridsearch on the consensus val crops.
IHC_SEG=$ROOT/predictions/val_synapses/production_2026-09-25/v7
OUT=$ROOT/predictions/val_synapses/rescore_2026-10-23
# Trainer checkpoints of the runs without an export. Check the paths before you submit.
CKPT=/mnt/lustre-grete/usr/u15000/checkpoints

# v5-fold1..4/best.pt are the exports that synapses.json scores as v5-f1..f4 (four different
# splits). The v5-rep runs are the checkpoint folders v5-f1..f4: seed replicates of v5 with the
# standard split, never scored.
MODELS=(
	"v3 $SYN/synapse_detection_model_v3.pt"
	"v5 $SYN/synapse_detection_model_v5.pt"
	"v5-latest $CKPT/synapse_detection_v5/latest.pt"
	"v5-rep1 $CKPT/synapse_detection_v5-f1/best.pt"
	"v5-rep2 $CKPT/synapse_detection_v5-f2/best.pt"
	"v5-rep3 $CKPT/synapse_detection_v5-f3/best.pt"
	"v5-rep4 $CKPT/synapse_detection_v5-f4/best.pt"
	"v5-rep1-latest $CKPT/synapse_detection_v5-f1/latest.pt"
	"v5-rep2-latest $CKPT/synapse_detection_v5-f2/latest.pt"
	"v5-rep3-latest $CKPT/synapse_detection_v5-f3/latest.pt"
	"v5-rep4-latest $CKPT/synapse_detection_v5-f4/latest.pt"
	"v5-fold1 $SYN/synapse_detection_model_v5-f1.pt"
	"v5-fold2 $SYN/synapse_detection_model_v5-f2.pt"
	"v5-fold3 $SYN/synapse_detection_model_v5-f3.pt"
	"v5-fold4 $SYN/synapse_detection_model_v5-f4.pt"
	"v7 $SYN/synapse_detection_v7.pt"
	"v7-latest $SYN/synapse_detection_v7-latest.pt"
	"v9 $SYN/synapse_detection_v9.pt"
	"v9-latest $SYN/synapse_detection_v9-latest.pt"
	"v10 $SYN/synapse_detection_v10.pt"
	"v10-latest $SYN/synapse_detection_v10-latest.pt"
)

# Prints "<threshold> <min_distance>" of the best F1 on the val crops of a split file.
# Further arguments go to the gridsearch.
select_setting() {
	python -m flamingo_tools.synapse_detection.gridsearch -m "$1" -s "$2/train_val_split.json" \
		-i "$2/images" -l "$2/labels" "${@:3}" |
		sed -n 's/^Best: threshold \([0-9.]*\), min_distance \([0-9]*\).*/\1 \2/p'
}

mkdir -p "$OUT"
IHC_DONE=""
for entry in "${MODELS[@]}"; do
	read -r NAME MODEL <<< "$entry"
	read -r T_CONS MD_CONS < <(select_setting "$MODEL" "$DATA/training_data/v11" --ihc_dir "$IHC_SEG")
	read -r T_OLD MD_OLD < <(select_setting "$MODEL" "$DATA/training_data/v7")
	echo "$NAME cons $T_CONS $MD_CONS old $T_OLD $MD_OLD" | tee -a "$OUT/selected_settings.txt"

	for SETTING in "fixed 0.5 2" "cons $T_CONS $MD_CONS" "old $T_OLD $MD_OLD"; do
		read -r TAG T MD <<< "$SETTING"
		RUN="$OUT/$NAME-$TAG"
		mkdir -p "$RUN"
		# The IHC segmentation does not depend on the synapse model, so predict it once.
		if [ -n "$IHC_DONE" ]; then
			ln -sfn "$IHC_DONE"/*_ihc "$RUN"/
		fi
		python "$REPO"/scripts/validation/synapses/prediction.py -o "$RUN" -i "$TEST" -g "$REF" \
			--model_synapse "$MODEL" --model_ihc "$IHC" --threshold "$T" --min_distance "$MD"
		python "$REPO"/scripts/validation/synapses/run_evaluation.py -p "$RUN" -c "$TEST" -r "$REF" -o "$OUT"
		# Only a run with IHC output can supply it, otherwise every later run gets a broken link.
		if [ -z "$IHC_DONE" ] && compgen -G "$RUN/*_ihc" > /dev/null; then
			IHC_DONE=$RUN
		fi
	done
done
