#!/bin/bash

MOBIE_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet
# e.g. G_EK_000233_L
COCHLEA=$1
# e.g. synapse_v3
SYNAPSE=$2

rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA"/dataset.json cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/dataset.json
rclone --progress copyto "$MOBIE_DIR"/"$COCHLEA"/tables/"$SYNAPSE"/default.tsv cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/tables/"$SYNAPSE"/default.tsv
