#!/bin/bash
#SBATCH --job-name=train_synapse-detection_supervised
#SBATCH -t 32:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=constantin.pape@cs.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 8
#SBATCH --mem 64G

# Seed replicate 4 of 4 for the v3 model: heatmap only, no sampler, same as v3 and v3-1.
# Together with v3 and v3-1 this measures the run-to-run variability of the model we ship,
# with the training data held fixed.
#
# --random_state 42 fixes the train/val split, so it is identical to v3 and v3-1
# (12 train / 3 val, see train_val_split_v3-1.json). It must be passed explicitly: the
# default is a hash of the model suffix (v3-2 -> 264, v3-3 -> 265, v3-4 -> 266), which
# would resample the split and confound seed variability with split variability.
#
# Nothing in the training script, torch_em or the czii trainer sets a global seed, so the
# weight initialization and the patch sampling order differ between these runs on their own.
# That is the variability being measured; there is no seed argument to vary here.

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

python $SCRIPT_DIR/train_synapse_detection.py -v v3 -m v3-4 --random_state 42
