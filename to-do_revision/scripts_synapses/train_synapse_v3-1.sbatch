#!/bin/bash
#SBATCH --job-name=train_synapse-detection_supervised
#SBATCH -t 32:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=constantin.pape@cs.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 8
#SBATCH --mem 64G

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

python $SCRIPT_DIR/train_synapse_detection.py -v v3 -m v3-1 --random_state 42

