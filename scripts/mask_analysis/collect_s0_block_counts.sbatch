#!/bin/bash
#SBATCH --job-name=mask-s0-counts
#SBATCH -t 08:00:00
#SBATCH --mail-type=FAIL
#SBATCH -p standard96s:shared
#SBATCH -A nib00064
#SBATCH -c 16
#SBATCH --mem 32G
#SBATCH --array=0-192%20
#SBATCH --output=slurm-%A_%a.out

# One task per dataset of mask_datasets.json (193 datasets, task ids 0-192).
# Before submitting:
#   1. Copy mask_threshold_analysis/mask_datasets.json and mask_threshold_analysis/mask_jobs.csv
#      next to collect_s0_block_counts.py in SCRIPT_DIR.
#   2. Set SCRIPT_REPO and OUTPUT_FOLDER below.
# The stain is read from the MoBIE copy on /mnt/vast-nhr if it is mounted, else from S3.
# Copy OUTPUT_FOLDER back to mask_threshold_analysis/s0_counts/ when all tasks are done.

trap 'exit 143' TERM
trap 'echo "Exit code of the job: $?"' EXIT

source ~/.bashrc
micromamba activate cochlea-net

SCRIPT_REPO=/user/schilling40/u32149/cochlea-net
SCRIPT_DIR="$SCRIPT_REPO"/scripts/mask_analysis
OUTPUT_FOLDER=/projects/extern/nhr/nhr_ni/nib00064/dir.project/moser/cochlea-lightsheet/mask_analysis/s0_counts

python "$SCRIPT_DIR"/collect_s0_block_counts.py --task_id "$SLURM_ARRAY_TASK_ID" \
	--output_folder "$OUTPUT_FOLDER" --n_threads 16
