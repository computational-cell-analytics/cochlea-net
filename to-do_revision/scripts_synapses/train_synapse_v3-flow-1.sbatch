#!/bin/bash
#SBATCH --job-name=train_synapse-detection_supervised
#SBATCH -t 32:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=constantin.pape@cs.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 8
#SBATCH --mem 64G

# The v3 training data with the v5 recipe: heatmap + 4 stereographic flow channels, trained
# with the MinPointSampler that --use_flow enables by default. This is the missing cell of
#
#              15-crop v3 data      29-crop v5/v6 data
#   heatmap    v3, v3-1             v6-1
#   flow       -> this run          v5
#
# At production settings v6-1 loses 0.163 recall against v3 (data effect) while v5 gains 0.191
# against v6-1 (recipe effect), so the two contrasts nearly cancel. This run separates them
# without relying on v6-1 alone.
#
# --random_state 42 is required: without it the split seed defaults to a hash of the model
# suffix, which would give a different train/val split than v3 and v3-1 and confound the
# comparison. With it the split is identical to train_val_split_v3-1.json (12 train / 3 val).

source ~/.bashrc
micromamba activate new-stack || exit 1

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# For debugging purposes.

# Run the script:
shopt -s extglob

export SCRIPT_DIR=/user/pape41/u12086/Work/my_projects/flamingo-tools/scripts/synapse_marker_detection

python $SCRIPT_DIR/train_synapse_detection.py -v v3 -m v3-flow-1 --use_flow --random_state 42
