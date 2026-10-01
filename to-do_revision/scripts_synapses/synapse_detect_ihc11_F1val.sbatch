#!/bin/bash
#SBATCH --job-name=apply_synapse-detection_supervised
#SBATCH -t 02:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 4
#SBATCH --mem 32G

source ~/.bashrc
micromamba activate cochlea-net

# Stop at the first failure. Without this the evaluation runs on a prediction that never
# happened and reports a missing detection file instead of the real error.
set -e

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

export SCRIPT_DIR=/user/schilling40/u15000/flamingo-tools

# A new root: prediction.py skips every crop that already has a prediction, and the predictions in
# production_2026-09-24 were normalized with statistics that included the zero padding.
OUTPUT_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/val_synapses/production_2026-09-25

# All entries are scored against IHC v11. --overwrite replaces the entries of the 2026-09-24 run.
# Add v9 and v9-latest once the model is trained and exported.
for VERSION in v3-ihc11 v5-ihc11 v7 v7-latest v8 v8-latest ; do
	python "$SCRIPT_DIR"/scripts/validation/synapses/prediction.py \
		-v "$VERSION" --output_root "$OUTPUT_DIR"
	python "$SCRIPT_DIR"/scripts/validation/synapses/run_evaluation.py \
		-v "$VERSION" -o "$SCRIPT_DIR"/reproducibility/model_accuracy/ --pred_root "$OUTPUT_DIR" --overwrite
done
