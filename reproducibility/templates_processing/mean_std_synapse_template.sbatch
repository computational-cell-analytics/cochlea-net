#!/bin/bash
#SBATCH --job-name=mean-std-synapse
#SBATCH -t 01:00:00

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

# CTBP2 channel, e.g. CTBP2.ome.zarr or setup2/timepoint0/s0 of a fused n5
DATA=$1
# channel in n5 folder, e.g. setup2/timepoint0/s0
INPUT_KEY=$2
# output folder
OUTPUT_FOLDER=$3
# optional: IHC segmentation used to restrict the prediction to the region around the IHCs
MASK_PATH=$4

export DATA
export INPUT_KEY
export OUTPUT_FOLDER
export MASK_PATH
export MASK_INPUT_KEY="s4"

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

echo "Input data: ${DATA}"
echo "Output directory: ${OUTPUT_FOLDER}"
echo "Mask: ${MASK_PATH:-none}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import synapse_detection;'
	'synapse_detection.run_synapse_prediction_preprocess_slurm(input_path=os.environ["DATA"],'
	'input_key=os.environ["INPUT_KEY"],'
	'output_folder=os.environ["OUTPUT_FOLDER"],'
	'mask_path=os.environ.get("MASK_PATH") or None,'
	'mask_input_key=os.environ["MASK_INPUT_KEY"])')
cmd="${cmd_array[*]}"
python -c "$cmd"
