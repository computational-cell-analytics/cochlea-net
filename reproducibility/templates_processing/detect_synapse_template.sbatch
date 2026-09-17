#!/bin/bash
#SBATCH --job-name=synapse-detect
#SBATCH -t 16:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p grete:shared         # the partition
#SBATCH -G A100:1               # For requesting 1 A100 GPU.
#SBATCH -A nim00007
#SBATCH -c 8
#SBATCH --mem 400G
#SBATCH --constraint=inet		# for downloading models

# Runs the whole synapse detection as a single job. To split the prediction over a slurm array
# instead, use mean_std_synapse_template.sbatch, apply_synapse_template.sbatch and
# detect_synapse_peaks_template.sbatch.

source ~/.bashrc
micromamba activate micro-sam_gpu

# Print out some info.
echo "Submitting job with sbatch from directory: ${SLURM_SUBMIT_DIR}"
echo "Home directory: ${HOME}"
echo "Working directory: $PWD"
echo "Current node: ${SLURM_NODELIST}"

# data in n5 format, e.g. GEK11L_PV_GFP_01_fused.n5
DATA=$1
# channel in n5 folder, e.g. setup2/timepoint0/s0
INPUT_KEY=$2
# output folder
OUTPUT_FOLDER=$3
# IHC segmentation. Without it the prediction runs on the full volume, which is far more
# expensive, and the detections are not matched to the IHCs.
MASK_PATH=$4

if ! [[ -f $OUTPUT_FOLDER ]] ; then
	mkdir -p "$OUTPUT_FOLDER"
fi

echo "Detecting synapses of ""$DATA"
echo "IHC segmentation: ${MASK_PATH:-none, predicting on the full volume}"

args=(--input_path "$DATA" --input_key "$INPUT_KEY" --output_folder "$OUTPUT_FOLDER")
if [[ -n $MASK_PATH ]] ; then
	args+=(--mask_path "$MASK_PATH" --mask_input_key s4 --mask_key s0 --max_distance 3.0)
fi

flamingo_tools.run_detection "${args[@]}"
