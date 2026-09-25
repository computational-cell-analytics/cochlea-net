#!/bin/bash
#SBATCH --job-name=train_synapse-detection_supervised
#SBATCH -t 32:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 8
#SBATCH --mem 64G

source ~/.bashrc
micromamba activate cochlea-net || exit 1

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

export SCRIPT_DIR=/user/schilling40/u15000/flamingo-tools/scripts/synapse_marker_detection
SAVE_ROOT=/mnt/lustre-grete/usr/u15000/checkpoints

# v7 with the candidate-exclusion mask. Data, split, validation patches and sampler are those of v7,
# so the mask is the only difference. Check the printed candidate count per crop before the
# training starts: the m78l_*_cr-ctbp2 crops are expected to have far more than the others.
python $SCRIPT_DIR/train_synapse_detection.py -v v7 -m v9 --random_state 42 -s $SAVE_ROOT \
    --sampler minpoint --ignore_percentile 10
