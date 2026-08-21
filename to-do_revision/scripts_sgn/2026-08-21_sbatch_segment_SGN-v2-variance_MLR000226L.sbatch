#!/bin/bash
#SBATCH --job-name=segment-unet-SGN_M-AMD-000126-L
#SBATCH --time 06:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 12
#SBATCH --mem 256G

source ~/.bashrc
micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script
#python myprogram.py $SLURM_ARRAY_TASK_ID

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts
export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_000126_L/SGN_v2

export MIN_SIZE=1000

VERSIONS=(1 2 3 4)

COCHLEA="M_LR_000226_L"

for VERSION in "${VERSIONS[@]}" ; do
	export SCRIPT_DIR=$SCRIPT_REPO/scripts
	export INPUT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/"$COCHLEA"/PV.ome.zarr

	export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/SGN_v2-"$VERSION"
	export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/v2-"$VERSION"_cochlea_distance_unet_SGN_supervised
	export PREDICTION_INSTANCES=10
	export INPUT_KEY=s0

	echo "Input directory: ${INPUT}"
	echo "Output directory: ${OUTPUT_FOLDER}"
	echo "Model: ${MODEL}"

	cmd_array=(	'import sys,os;'
		'sys.path.insert(0,os.environ["SCRIPT_DIR"]);'
		'import unet_prediction;'
		'unet_prediction.run_unet_segmentation_slurm(output_folder=os.environ["OUTPUT_FOLDER"],'
		'min_size=os.environ["MIN_SIZE"])')
	cmd="${cmd_array[*]}"
	python -c "$cmd"
done
