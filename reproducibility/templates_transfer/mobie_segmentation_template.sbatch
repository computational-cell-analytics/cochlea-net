#!/bin/bash
#SBATCH --job-name=mobie_segm
#SBATCH -t 01:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 36
#SBATCH --mem 180G

source ~/.bashrc
micromamba activate mobie

# Run the script

# name of cochlea, as it appears in MoBIE and the NHR
COCHLEA=$1
# data in n5 format, e.g. IHC_v5/segmentation.zarr
DATA=$2
# segmentation name, as it appears in MoBIE, e.g. SGN_v2
CHANNEL_NAME=$3

MOBIE_PROJECT="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"

SEGMENTATION_KEY="segmentation"
RESOLUTION="[0.38,0.38,0.38]"
SCALE_FACTORS="[[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]"
CHUNKS="[64,64,64]"

echo "Adding segmentation data ""$DATA"" to MoBIE project""$MOBIE_PROJECT"" as ""$CHANNEL_NAME"

mobie.add_segmentation --input_path "$DATA" --input_key "$SEGMENTATION_KEY" --root "$MOBIE_PROJECT" \
	--dataset_name "$COCHLEA" --name "$CHANNEL_NAME" --resolution "$RESOLUTION" \
	--scale_factors "$SCALE_FACTORS" --chunks "$CHUNKS"
