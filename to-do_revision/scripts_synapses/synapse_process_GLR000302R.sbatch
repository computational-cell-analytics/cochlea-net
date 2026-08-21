#!/bin/bash
#SBATCH --job-name=synapse-marker_G-LR-000302-R
#SBATCH -t 10:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 8                    # cores for the prefetch workers and the peak detection
#SBATCH --mem 200G
#SBATCH --constraint=inet       # for reading the IHC segmentation from S3

# Prediction, peak detection and matching to the IHCs in one call.
# The CTBP2 image data is read from a local file, the IHC segmentation from the S3 bucket.
# The IHC segmentation restricts the inference to the region around the IHCs, which skips
# the background blocks and is much faster than predicting the full volume.

source ~/.bashrc
micromamba activate sam

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

SCRIPT_REPO=/user/schilling40/u15000/flamingo-tools
cd "$SCRIPT_REPO"/flamingo_tools/segmentation/ || exit

export SCRIPT_DIR=$SCRIPT_REPO/scripts

COCHLEA="G_LR_000302_R"

# Local CTBP2 image data.
export INPUT_PATH=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/"$COCHLEA"/GLR_302R_CTBP2_fused.n5
export INPUT_KEY="setup0/timepoint0/s0"

# IHC segmentation on the S3 bucket, given as a path relative to the bucket.
export MASK_PATH="$COCHLEA"/images/ome-zarr/IHC_v11.ome.zarr
export MASK_INPUT_KEY="s4"
export MASK_KEY="s0"

export OUTPUT_FOLDER=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/"$COCHLEA"/synapses_v3
export MODEL=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/Synapses/synapse_detection_model_v3.pt
export MAX_DISTANCE=8

echo "OUTPUT_FOLDER $OUTPUT_FOLDER"
echo "MODEL $MODEL"

if [ ! -d "$OUTPUT_FOLDER" ] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

python ~/flamingo-tools/scripts/synapse_marker_detection/marker_detection.py \
	--input $INPUT_PATH \
	--input_key $INPUT_KEY \
	--output_folder $OUTPUT_FOLDER \
	--mask $MASK_PATH \
	--mask_input_key $MASK_INPUT_KEY \
	--mask_key $MASK_KEY \
	--model $MODEL \
	--max_distance $MAX_DISTANCE \
	--s3_mask
