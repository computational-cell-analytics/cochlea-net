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
Synapse detection of a CTBP2 stain, matched to an IHC segmentation. This is the standard case:
the segmentation is dilated into a mask, and the prediction only runs on the blocks inside it.
On a full cochlea that is about 3 % of the volume.
```bash
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v3"
MASK_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/IHC_v4b.ome.zarr"
flamingo_tools.run_detection  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --mask_path "$MASK_PATH" \
    --output_folder "$OUTPUT_FOLDER"
```
This writes `synapse_detection.tsv` with all detections, and `synapse_detection_filtered.tsv` with
the detections matched to an IHC and closer to it than `--max_distance` (3 micrometer by default).

The two mask keys select different resolutions of the same IHC segmentation: `--mask_input_key`
(default `s4`) is the downscaled level that builds the inference mask, and `--mask_key`
(default `s0`) is the full-resolution level used to match the detections. The mask is dilated by
`--dilation_iterations` (default 4) voxels of `--mask_input_key`, which is 24 micrometer at `s4`.
The dilation has to stay above `--max_distance`, otherwise a synapse near the border of an IHC is
never predicted. The command warns when it does not.

Without an IHC segmentation the prediction falls back to the full volume, which is far more
expensive, and the detections are not matched. The command warns about this too.
```bash
flamingo_tools.run_detection  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER"
```

The image data and the segmentation are resolved independently, so the CTBP2 data can be read from
a local file while the IHC segmentation is read from the S3 bucket:
```bash
flamingo_tools.run_detection  --input_path "$INPUT_PATH" --input_key s0 \
    --mask_path "$COCHLEA"/images/ome-zarr/IHC_v11.ome.zarr --s3_mask \
    --output_folder "$OUTPUT_FOLDER" --checkpoint_path "$MODEL" --max_distance 8
```
Add `--s3_input` to read the image data from the bucket as well.

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
The synapse prediction is split up in the same way. The peak detection replaces the segmentation
as the third step, and needs no GPU.

```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v5"

# --- Calculating mean and standard deviation ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/mean_std_synapse_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER

# --- Applying CochleaNet ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/apply_synapse_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER

# --- Detecting the synapse markers ---
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/detect_synapse_peaks_template.sbatch $OUTPUT_FOLDER
```

Submit each step only after the previous one finished. The first step writes `mean_std.json`, which
all tasks of the prediction array read so that they normalize the input identically.
The size of the array (`#SBATCH -a`) and `PREDICTION_INSTANCES` in
`apply_synapse_template.sbatch` must match.

Pass an IHC segmentation to the first step. It is dilated into the mask that restricts the
prediction to the region around the IHCs, which is the standard case: on a full cochlea it skips
about 97 % of the blocks. The synapses are matched to the IHCs within `MAX_DISTANCE` later on, so
this discards no detection that survives that filter. Without a segmentation the prediction falls
back to the full volume, and the first step warns about it.

```bash
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
MASK_PATH="$MOBIE_DIR""/M_AMD_N162_L/images/ome-zarr/IHC_v4b.ome.zarr"

sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/mean_std_synapse_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER $MASK_PATH
```

Pass the same segmentation at full resolution to the third step to also write
`synapse_detection_filtered.tsv`, with the detections matched to the IHCs.

```bash
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/detect_synapse_peaks_template.sbatch $OUTPUT_FOLDER $MASK_PATH s0
```

or for the full workflow without splitting up the prediction step. The fourth argument is the IHC
segmentation; omit it to predict on the full volume.
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
MOBIE_DIR="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_R/MAMD_N162R_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup2/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_R/synapses_v3"
MASK_PATH="$MOBIE_DIR""/M_AMD_N162_R/images/ome-zarr/IHC_v4b.ome.zarr"
sbatch "$SCRIPT_DIR"/reproducibility/templates_processing/detect_synapse_template.sbatch $DATA $INPUT_KEY $OUTPUT_FOLDER $MASK_PATH
```
