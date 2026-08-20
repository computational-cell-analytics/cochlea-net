#!/bin/bash
#SBATCH --job-name=detect-synapse-peaks
#SBATCH -t 04:00:00

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
# The peak detection is thread-parallel over blocks, so it scales with the cores.
#SBATCH -c 16
#SBATCH --mem 200G

source ~/.bashrc
micromamba activate flamingo13

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

# directory containing predictions.zarr
OUTPUT_FOLDER=$1
# optional: IHC segmentation at full resolution, for matching and filtering the detections
MASK_PATH=$2
# key to the IHC segmentation, e.g. s0
MASK_INPUT_KEY=$3

export OUTPUT_FOLDER
export MASK_PATH
export MASK_INPUT_KEY
export VOXEL_SIZE="0.38 0.38 0.38"
export THRESHOLD=0.5
export MAX_DISTANCE=3.0

echo "Output directory: ${OUTPUT_FOLDER}"
echo "Mask: ${MASK_PATH:-none}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import synapse_detection;'
	'synapse_detection.run_synapse_detection_slurm(output_folder=os.environ["OUTPUT_FOLDER"],'
	'voxel_size=os.environ["VOXEL_SIZE"],threshold=os.environ["THRESHOLD"],'
	'mask_path=os.environ.get("MASK_PATH") or None,'
	'mask_input_key=os.environ.get("MASK_INPUT_KEY") or None,'
	'max_distance=os.environ["MAX_DISTANCE"])')
cmd="${cmd_array[*]}"
python -c "$cmd"
