#!/bin/bash
#SBATCH --job-name=synapse-detect
#SBATCH -t 16:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 3
#SBATCH --mem 400G
#SBATCH --constraint=inet		# for downloading models

source ~/.bashrc
micromamba activate micro-sam_gpu
#micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script
SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

# data in n5 format, e.g. GEK11L_PV_GFP_01_fused.n5
DATA=$1
# channel in n5 folder, e.g. setup0/timepoint0/s0
INPUT_KEY=$2
# output folder
OUTPUT_FOLDER=$3

export DATA
export INPUT_KEY
export OUTPUT_FOLDER
MODEL_TYPE="Synapses"

echo "Detecting synapses"
echo "$DATA"

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

flamingo_tools.run_detection  --input_path "$DATA" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_type "$MODEL_TYPE"
