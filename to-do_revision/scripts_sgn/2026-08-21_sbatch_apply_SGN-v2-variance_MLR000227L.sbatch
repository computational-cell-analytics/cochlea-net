#!/bin/bash
#SBATCH --job-name=apply-unet-IHC_M-LR-000227_L
#SBATCH -t 04:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -c 1
#SBATCH --mem 128G
#SBATCH -a 0-9

source ~/.bashrc
micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# Run the script

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

VERSIONS=(1 2 3 4)

COCHLEA="M_LR_000227_L"

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
		'unet_prediction.run_unet_prediction_slurm(input_path=os.environ["INPUT"],'
		'output_folder=os.environ["OUTPUT_FOLDER"],model_path=os.environ["MODEL"],'
		'input_key=os.environ["INPUT_KEY"],'
		'prediction_instances=os.environ["PREDICTION_INSTANCES"])')
	cmd="${cmd_array[*]}"
	python -c "$cmd"
done
