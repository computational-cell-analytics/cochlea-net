#!/bin/bash
#SBATCH --job-name=train_synapse-detection_v11
#SBATCH -t 08:00:00             # 3e4 iterations, about 5.7 A100 hours
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends
#SBATCH --array=1-3

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 16G               # about 7 GB used

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

# Three seed replicates of the v11 recipe, the default of train_synapse_detection.py since
# 2026-10-28: the fixed target, a cosine learning rate schedule over 3e4 iterations, and the two
# consensus val crops (the three old val crops go back to training). Weight init and patch order
# are not seeded, so the array tasks differ only in these. v11-1..3 were trained with the
# equivalent flags --fixed_target --lr_schedule cosine -n 30000. Use latest.pt.
python $SCRIPT_DIR/train_synapse_detection.py -v v11 -m v11-$SLURM_ARRAY_TASK_ID --random_state 42 \
    -s $SAVE_ROOT --split $SPLIT
