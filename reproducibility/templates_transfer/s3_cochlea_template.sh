#!/bin/bash

MOBIE_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet
# e.g. M_AMD_Runx1_L
COCHLEA=$1

export BUCKET_NAME="cochlea-lightsheet"
export SERVICE_ENDPOINT="https://s3.fs.gwdg.de"
mobie.add_remote_metadata -i $MOBIE_DIR -s $SERVICE_ENDPOINT -b $BUCKET_NAME

rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA" cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"
