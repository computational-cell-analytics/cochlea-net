#!/bin/bash
#SBATCH --job-name=process-unet-SGN
#SBATCH -t 18:00:00             # for gerbil up to ~3 hours

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 12
#SBATCH --mem 400G
#SBATCH --constraint=inet		# for downloading models

source ~/.bashrc
micromamba activate micro-sam_gpu

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script

# get path to flamingo_tools repository based on the location of this script, only works for calling bash script directly
#SCRIPT_REPO="$( cd "$( dirname  "$( dirname "$( dirname "$(readlink -f "${BASH_SOURCE[0]}")" )" )" )" >/dev/null 2>&1 && pwd )"

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

# data in n5 format, e.g. GEK11L_PV_GFP_01_fused.n5
DATA=$1
# channel in n5 folder, e.g. setup0/timepoint0/s0
INPUT_KEY=$2
# output folder
OUTPUT_FOLDER=$3

MODEL_TYPE="SGN"

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

flamingo_tools.run_segmentation  --input_path "$DATA" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_type "$MODEL_TYPE" \
    --min_size 1000