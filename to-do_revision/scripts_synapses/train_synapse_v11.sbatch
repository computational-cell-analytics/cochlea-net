#!/bin/bash
#SBATCH --job-name=train_synapse-detection_v11
#SBATCH -t 16:00:00             # 3e4 iterations, about 10 A100 hours
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends
#SBATCH --array=1-3

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
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
DATA=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data
SPLIT=$DATA/v11/train_val_split.json
[ -f "$SPLIT" ] || { echo "Missing $SPLIT, set up the v11 data folder of ../synapses.md first."; exit 1; }

# Three seed replicates of the refined recipe: the v7 recipe with the fixed target, a cosine
# learning rate schedule over 3e4 iterations, and the two consensus val crops (the three old val
# crops go back to training). Weight init and patch order are not seeded, so the array tasks
# differ only in these. Evaluate latest.pt, see "Rescoring result (2026-10-23)" in ../synapses.md.
python $SCRIPT_DIR/train_synapse_detection.py -v v11 -m v11-$SLURM_ARRAY_TASK_ID --random_state 42 \
    -s $SAVE_ROOT --split $SPLIT --fixed_target --lr_schedule cosine -n 30000
