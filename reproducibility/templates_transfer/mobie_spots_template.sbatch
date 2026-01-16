#!/bin/bash
#SBATCH --job-name=mobie_spots
#SBATCH -t 00:10:00             # estimated time, adapt to your needs
#SBATCH --mail-user=martin.schilling@med.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007
#SBATCH -c 12
#SBATCH --mem 16G

source ~/.bashrc
micromamba activate mobie

# Run the script

# name of cochlea, as it appears in MoBIE and the NHR
COCHLEA=$1
# synapse detection table
TABLE_PATH=$2
# segmentation name, as it appears in MoBIE, e.g. synapses_v3 or synapses_v3_ihc_v4
SPOT_NAME=$3

MOBIE_PROJECT="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"

echo "Adding spot data ""$TABLE_PATH"" to MoBIE project""$MOBIE_PROJECT"" as ""$SPOT_NAME"

mobie.add_spots --input_table "$TABLE_PATH" --root "$MOBIE_PROJECT" \
	--dataset_name "$COCHLEA" --name "$SPOT_NAME"
