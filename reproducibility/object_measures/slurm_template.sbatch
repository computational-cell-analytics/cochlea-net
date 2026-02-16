#!/bin/bash
#SBATCH -t 04:00:00             # estimated time, adapt to your needs
#SBATCH --mail-user=constantin.pape@informatik.uni-goettingen.de # change this to your mailaddress
#SBATCH --mail-type=FAIL        # send mail when job begins and ends

#SBATCH -p standard96s:shared         # the partition
#SBATCH -A nim00007

#SBATCH -c 4
#SBATCH --mem 12G
#SBATCH --constraint=inet

source ~/.bashrc
micromamba activate micro-sam_gpu

SCRIPT=/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/reproducibility/object_measures/repro_object_measures.py
SCRIPT=/user/schilling40/u15000/flamingo-tools/reproducibility/object_measures/repro_object_measures.py

python $SCRIPT --json $1 --output $2 --s3
