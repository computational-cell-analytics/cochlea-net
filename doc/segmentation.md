# Processing - Segmentation, detection, and object measures

The segmentation can be performed locally or using Slurm and the GWDG resources.

### Locally

The data can be processed locally using the command line interface.

#### SGN
```bash
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup0/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/SGN_v2"
MODEL_TYPE=SGN
flamingo_tools.run_segmentation  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_type "$MODEL_TYPE" \
    --min_size 1000
```

#### IHC
```bash
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup1/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/IHC_v4b"
MODEL_TYPE=IHC
flamingo_tools.run_segmentation  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_type "$MODEL_TYPE" \
    --min_size 1000
```

#### Synapses
Synapse detection of a CTBP2 stain:
```bash
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v3"
flamingo_tools.run_detection  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER"
```

Synapse detection matched to the IHC segmentation (requires IHC segmentation as an input):
```bash
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v3"
MASK_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/IHC_v4b.ome.zarr"
MASK_KEY="s0"
flamingo_tools.run_detection  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --mask_path "$MASK_PATH" \
    --mask_key "$MASK_KEY" \
    --output_folder "$OUTPUT_FOLDER"
```

#### Object measures
The calculation of object measures requires a segmentation, e.g. SGN.

```bash
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"

IMAGE_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/PV.ome.zarr"
SEG_TABLE="$MOBIE_DIR""/M_AMD_N162_L/tables/SGN_v2/default.tsv"
SEG_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/SGN_v2.ome.zarr"
OUTPUT="M-AMD-N162-L_PV_SGN-v2_object-measures.tsv"

flamingo_tools.object_measures  --image_paths "$IMAGE_PATH" \
    -t "$SEG_TABLE" \
    -s "$SEG_PATH" \
    --output "$OUTPUT"
```
or with data on the S3 bucket
```bash
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"

IMAGE_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/PV.ome.zarr"
SEG_TABLE="$MOBIE_DIR""/M_AMD_N162_L/tables/SGN_v2/default.tsv"
SEG_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/SGN_v2.ome.zarr"
OUTPUT="M-AMD-N162-L_PV_SGN-v2_object-measures.tsv"

flamingo_tools.object_measures  --image_paths "$IMAGE_PATH" \
    -t "$SEG_TABLE" \
    -s "$SEG_PATH" \
    --output "$OUTPUT" \
    --s3
```

### Using Slurm
Slurm is the batch system for submitting jobs on the GWDH HPC cluster: https://docs.hpc.gwdg.de/how_to_use/slurm/index.html

Because it is more efficient to split the network prediction into multiple jobs, the processing workflow is divided into three steps:
* Mask the image data based on intensity and calculate the mean and standard deviation of the intensity
* Apply CochleaNet
* Segment the prediction of CochleaNet

#### SGN
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup0/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/SGN_v2"

# --- Masking and calculating mean and standard deviation ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/mean_std_SGN_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER

# --- Applying CochleaNet ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/apply_unet_SGN_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER

# --- Segmenting prediction ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/segment_unet_SGN_template.sbatch $OUTPUT_FOLDER
```
or for the full workflow without splitting up the prediction step
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_R/MAMD_N162R_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup0/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_R/SGN_v2"

sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/process_SGN_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER
```

#### IHC
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
# --- Masking and calculating mean and standard deviation ---
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup1/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/IHC_v4b"
# --- Masking and calculating mean and standard deviation ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/mean_std_IHC_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER

# --- Applying CochleaNet ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/apply_unet_IHC_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER

# --- Segmenting prediction ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/segment_unet_IHC_template.sbatch $OUTPUT_FOLDER
```

or for the full workflow without splitting up the prediction step
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_R/MAMD_N162R_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup1/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_R/IHC_v4b"

sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/process_IHC_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER
```

#### Synapses
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_R/MAMD_N162R_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_R/synapses_v3"
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/detect_synapse_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER
```

```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"

INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v3"
MASK_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/IHC_v4b.ome.zarr"
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/detect_synapse_marker_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER $MASK_PATH
```
