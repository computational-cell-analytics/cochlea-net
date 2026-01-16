#!/bin/bash
#SBATCH --job-name=mean-std-IHC
#SBATCH -t 01:00:00             # usually takes ~20 min

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 3
#SBATCH --mem 128G

source ~/.bashrc
micromamba activate flamingo13

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

export DATA
export INPUT_KEY
export OUTPUT_FOLDER

export SEG_CLASS="ihc"

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

echo "Input data: ${DATA}"
echo "Output directory: ${OUTPUT_FOLDER}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_prediction_preprocess_slurm(input_path=os.environ["DATA"],'
	'input_key=os.environ["INPUT_KEY"],'
	'output_folder=os.environ["OUTPUT_FOLDER"],seg_class=os.environ["SEG_CLASS"])')
cmd="${cmd_array[*]}"
python -c "$cmd"
