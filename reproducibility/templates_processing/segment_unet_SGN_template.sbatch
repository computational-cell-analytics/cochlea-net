#!/bin/bash
#SBATCH --job-name=segment-unet-SGN
#SBATCH --time 06:00:00             # for gerbil up to ~30 hours

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 400G

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

# directory containing prediction.zarr
OUTPUT_FOLDER=$1

export OUTPUT_FOLDER

export MIN_SIZE=1000
export BOUNDARY_DISTANCE_THRESHOLD=0.5

echo "Output directory: ${OUTPUT_FOLDER}"

cmd_array=(	'import sys,os;'
	'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
	'import unet_prediction;'
	'unet_prediction.run_unet_segmentation_slurm(output_folder=os.environ["OUTPUT_FOLDER"],'
	'min_size=os.environ["MIN_SIZE"])')
cmd="${cmd_array[*]}"
python -c "$cmd"

