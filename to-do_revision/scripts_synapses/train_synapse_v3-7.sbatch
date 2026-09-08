#!/bin/bash
#SBATCH --job-name=train_synapse-v3-7-10k
#SBATCH -t 02:00:00
#SBATCH --qos=2h
#SBATCH --mail-user=constantin.pape@cs.uni-goettingen.de
#SBATCH --mail-type=FAIL

#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 8
#SBATCH --mem 32G

# Additional short seed replicate for the v3 model: heatmap only, no sampler, 10k iterations.
# The existing full-A100 runs reached 10k in 1:47 to 1:48, leaving about 12 minutes of
# headroom in the short 2h QoS.
#
# --random_state 42 fixes the train/val split, so it is identical to v3 and v3-1 .. v3-4
# (12 train / 3 val). Weight initialization and patch sampling are intentionally not seeded,
# so this run provides another sample of the training seed noise.

source ~/.bashrc
micromamba activate new-stack || exit 1

echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

export SCRIPT_DIR=/user/pape41/u12086/Work/my_projects/flamingo-tools/scripts/synapse_marker_detection

python $SCRIPT_DIR/train_synapse_detection.py \
    -v v3 -m v3-7 --random_state 42 --n_iterations 10000
