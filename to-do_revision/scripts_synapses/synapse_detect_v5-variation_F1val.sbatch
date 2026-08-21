#!/bin/bash
#SBATCH --job-name=synapse-detect_F1val
#SBATCH -t 00:45:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 1
#SBATCH --mem 24G

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

CROP_NAMES=("MLR226L_Vglut3Ctbp2_apex1_p50-140"  "MLR226L_Vglut3Ctbp2_apex2_p1-134"  "MLR226L_Vglut3Ctbp2_base1_reslice_p122-256"  "MLR226L_Vglut3Ctbp2_base2_p40-130"  "MLR226L_Vglut3Ctbp2_mid1_resliced_p70-185"  "MLR226L_Vglut3Ctbp2_mid2_p65-195")
export INPUT_KEY=raw

INPUT_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v5/images"
OUTPUT_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/val_synapses/v5-f1"
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v5-f1.pt

for name in  "${CROP_NAMES[@]}" ; do
	export INPUT_PATH="$INPUT_DIR"/"$name".zarr
	export OUTPUT_FOLDER="$OUTPUT_DIR"/"$name"

	if [ ! -d "$OUTPUT_FOLDER" ] ; then
		mkdir -p "$OUTPUT_FOLDER"
	fi

	echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
	echo "MODEL $MODEL"

	python ~/flamingo-tools/scripts/synapse_marker_detection/run_prediction.py \
		--input $INPUT_PATH \
		--input_key $INPUT_KEY \
		--output_folder $OUTPUT_FOLDER \
		--model $MODEL
done

OUTPUT_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/val_synapses/v5-f2"
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v5-f2.pt

for name in  "${CROP_NAMES[@]}" ; do
	export INPUT_PATH="$INPUT_DIR"/"$name".zarr
	export OUTPUT_FOLDER="$OUTPUT_DIR"/"$name"

	if [ ! -d "$OUTPUT_FOLDER" ] ; then
		mkdir -p "$OUTPUT_FOLDER"
	fi

	echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
	echo "MODEL $MODEL"

	python ~/flamingo-tools/scripts/synapse_marker_detection/run_prediction.py \
		--input $INPUT_PATH \
		--input_key $INPUT_KEY \
		--output_folder $OUTPUT_FOLDER \
		--model $MODEL
done

OUTPUT_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/val_synapses/v5-f3"
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v5-f3.pt

for name in  "${CROP_NAMES[@]}" ; do
	export INPUT_PATH="$INPUT_DIR"/"$name".zarr
	export OUTPUT_FOLDER="$OUTPUT_DIR"/"$name"

	if [ ! -d "$OUTPUT_FOLDER" ] ; then
		mkdir -p "$OUTPUT_FOLDER"
	fi

	echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
	echo "MODEL $MODEL"

	python ~/flamingo-tools/scripts/synapse_marker_detection/run_prediction.py \
		--input $INPUT_PATH \
		--input_key $INPUT_KEY \
		--output_folder $OUTPUT_FOLDER \
		--model $MODEL
done

OUTPUT_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/val_synapses/v5-f4"
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v5-f4.pt

for name in  "${CROP_NAMES[@]}" ; do
	export INPUT_PATH="$INPUT_DIR"/"$name".zarr
	export OUTPUT_FOLDER="$OUTPUT_DIR"/"$name"

	if [ ! -d "$OUTPUT_FOLDER" ] ; then
		mkdir -p "$OUTPUT_FOLDER"
	fi

	echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
	echo "MODEL $MODEL"

	python ~/flamingo-tools/scripts/synapse_marker_detection/run_prediction.py \
		--input $INPUT_PATH \
		--input_key $INPUT_KEY \
		--output_folder $OUTPUT_FOLDER \
		--model $MODEL
done

