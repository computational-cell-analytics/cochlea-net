#!/bin/bash
#SBATCH --job-name=apply-unet-SGN
#SBATCH -t 00:40:00             # for gerbil up to ~3 hours

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 4
#SBATCH --mem 64G
#SBATCH -a 0-9

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

# The default v2 model
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/v2_cochlea_distance_unet_SGN_supervised_2025-05-27
export PREDICTION_INSTANCES=10

export DATA
export INPUT_KEY
export OUTPUT_FOLDER

echo "Input data: ${DATA}"
echo "Output directory: ${OUTPUT_FOLDER}"
echo "Model: ${MODEL}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_prediction_slurm(input_path=os.environ["DATA"],'
	'output_folder=os.environ["OUTPUT_FOLDER"],model_path=os.environ["MODEL"],'
	'input_key=os.environ["INPUT_KEY"],'
	'prediction_instances=os.environ["PREDICTION_INSTANCES"])')
cmd="${cmd_array[*]}"
python -c "$cmd"

