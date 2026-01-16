#!/bin/bash

MOBIE_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet
# e.g. M_AMD_Runx1_L
COCHLEA=$1
# e.g. SGN_v2
SEG_CHANNEL=$2

export BUCKET_NAME="cochlea-lightsheet"
export SERVICE_ENDPOINT="https://s3.fs.gwdg.de"

mobie.add_remote_metadata -i $MOBIE_DIR -s $SERVICE_ENDPOINT -b $BUCKET_NAME

rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA"/dataset.json cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/dataset.json
rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA"/images/ome-zarr/"$SEG_CHANNEL".ome.zarr cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/images/ome-zarr/"$SEG_CHANNEL".ome.zarr
# TODO enable to also sync the whole thing and project.json
# take care that segmentation tables containing evaluations (tonotopic mapping, marker labels, etc.) might be overwritten
rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA"/tables/"$SEG_CHANNEL"/default.tsv cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/tables/"$SEG_CHANNEL"/default.tsv
