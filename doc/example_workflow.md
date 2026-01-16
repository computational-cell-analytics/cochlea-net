# Workflow for the segmentation and analysis of a cochlea

The following code snippets will show how a single cochlea is processed.
Some of these steps might be skipped because they include the transfer from the UKON100 to the GWDG servers.

## Transfer from UKON100 to the NHR

These steps can be skipped if the transfer from UKON100 is not necessary or desirable.

Data is transferred from from UKON100 to the NHR, specifically *Grete10*.
Note that some steps refer to the AcademicID (e.g. *schilling40*) others to the Username (e.g. *u15000*).
See `scripts/data_transfer/README.md` for more information.

* Log in to NHR login node
```bash
ssh -i ~/.ssh/id_rsa_scc u15000@glogin10.hpc.gwdg.de
```
* Navigate to the target directory. The data will be copied to this location.
* Log in via smbclient
```bash
smbclient //wfs-medizin.top.gwdg.de/ukon-all$/ukon100 -U GWDG/schilling40
```
* Navigate to the location of the data
```bash
cd UKON100\archiv\imaging\Lightsheet\Huiskengroup_CTLSM\2025\Aleyna\PELCO\PELCOfHC2\M_AMD_000N162_L_PELCOfHC\3_fused
```
* Toggle recursive copying and copying without a prompt for each file.
```bash
recurse
prompt
```
* Copy the data to the NHR
```bash
mget MAMD_N162L_Vglut3_PV_CTBP2_PELCOfHC_fused.n5
```

## Processing - Segmentation, detection, and object measures

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

## Adding data to the MoBIE project

Templates for the transfer into the MoBIE data format can be found in `reproducibility/templates_transfer`.
Both, the scripts as well as the command line usage require the installation of [MoBIE utils](https://github.com/mobie/mobie-utils-python).
Those scripts are designed for the usage with Slurm ([GWDG Doc](https://docs.hpc.gwdg.de/how_to_use/slurm/index.html)) and need some adjustments by the user, e.g. the activation of the working environment, the location of the MoBIE project, and the mail address.

### Image data

This example call shows the addition of the **PV** image data of the **M_AMD_N162_L** cochlea, which is located in `setup0/timepoint0/s0`. Currently, adding data in N5 format is not straightforward. This issue will be addressed in the future.
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
COCHLEA="M_AMD_N162_L"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
CHANNEL_NAME="PV"
INPUT_KEY="setup0/timepoint0/s0"
# submit the job to Slurm
sbatch "$SCRIPT_DIR"/reproducibility/templates_transfer/mobie_image_template.sbatch $COCHLEA $DATA $CHANNEL_NAME $INPUT_KEY
# or run the job locally
bash "$SCRIPT_DIR"/reproducibility/templates_transfer/mobie_image_template.sbatch $COCHLEA $DATA $CHANNEL_NAME $INPUT_KEY
```

### Segmentation data

This example call shows the addition of the **SGN_v2** segmentation data of the **M_AMD_N162_L** cochlea.
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
COCHLEA="M_AMD_N162_L"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/IHC_v4b/segmentation.zarr"
CHANNEL_NAME="IHC_v4b"
# submit the job to Slurm
sbatch "$SCRIPT_DIR"/reproducibility/templates_transfer/mobie_segmentation_template.sbatch $COCHLEA $DATA $CHANNEL_NAME
# or run the job locally
bash "$SCRIPT_DIR"/reproducibility/templates_transfer/mobie_segmentation_template.sbatch $COCHLEA $DATA $CHANNEL_NAME
```

### Synapse data
This example call shows the addition of the **SGN_v2** segmentation data of the **M_AMD_N162_L** cochlea.
The job can always be run locally because it is very fast.
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
COCHLEA="M_AMD_N162_L"

# data from synapse detection
TABLE_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v3/synapse_detection.tsv"
SPOT_NAME="synapse_v3"
bash "$SCRIPT_DIR"/reproducibility/templates_transfer/mobie_spots_template.sbatch $COCHLEA $TABLE_PATH $SPOT_NAME

# filtered synapse data
TABLE_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/synapses_v3_ihc_v4b/synapse_detection_filtered.tsv"
SPOT_NAME="synapse_v3_ihc_v4b"
bash "$SCRIPT_DIR"/reproducibility/templates_transfer/mobie_spots_template.sbatch $COCHLEA $TABLE_PATH $SPOT_NAME
```

## Transfer to S3 bucket

For the transfer to the S3 bucket, the `dataset.json` within a dataset needs to be modified.
**Tip:** You can set variables like `BUCKET_NAME` or `SERVICE_ENDPOINT` within your bash_rc.
```bash
BUCKET_NAME="cochlea-lightsheet"
SERVICE_ENDPOINT="https://s3.fs.gwdg.de"
MOBIE_DIR=<enter_your_mobie_directory>
mobie.add_remote_metadata -b $BUCKET_NAME -s $SERVICE_ENDPOINT  -i $MOBIE_DIR
```
This will set the paths within the `dataset.json` to the locations they will have on the S3 bucket.
The files can then be transferred using `rclone`.
**Tip:** Most of the scripts handling the data transfer already include this step.

### Copying data to the S3 bucket

Scripts for the copying of data are located under `reproducibility/templates_transfer`.
They cover the copying of the whole dataset (`s3_cochlea_template.sh`), segmentation data (`s3_cochlea_template.sh`), and synapse detections (`s3_synapse_template.sh`).
Special care should be taken during the copying process to avoid overwriting existing files, particularly segmentation tables that contain information added during analysis.

### Handling dataset names in the project
When datasets are added to a local MoBIE project, their dataset name is added to the `project.json` file.
However, the file located on the S3 bucket will likely feature a lot more datasets than your local version, so the local file can not just be copied over without losing the access to the other datasets on the S3 bucket.
To circumvent this, the file of the S3 bucket can be downloaded, updated and re-uploaded.
```bash
# 1) Download remote project.json file.
rclone copyto cochlea-lightsheet:cochlea-lightsheet/project.json project_remote.json
# 2) Edit project_remote.json by adding the new dataset name.

# 3) Upload the new version.
rclone copyto project_remote.json cochlea-lightsheet:cochlea-lightsheet/project.json
```

## Analysis

### Labeling components
The component labeling for SGNs is quite reliable. At this point, an interactive approach using a Jupyter Notebook might be preferable for IHCs.
```bash
# locally
flamingo_tools.label_components --input "$MOBIE_DIR"/M_AMD_N162_L/tables/SGN_v2/default.tsv --s3 -o M_AMD_N162_L_v2.tsv --cell_type sgn
# access segmentation table on S3 bucket
flamingo_tools.label_components --input M_AMD_N162_L/tables/SGN_v2/default.tsv --s3 -o M_AMD_N162_L_v2.tsv --cell_type sgn
# check with napari
flamingo_tools.label_components --input M_AMD_N162_L/tables/SGN_v2/default.tsv --s3 -o M_AMD_N162_L_v2.tsv --cell_type sgn --napari
```

The following approach could be used interactively in cases where several parameters have to be tested to find a good labeling.
The example features the labeling of an IHC segmentation but it can also be used for SGNs.
```python
import pandas as pd
import flamingo_tools.postprocessing.label_components as label_components

table_path = "M_AMD_N162_L/tables/IHC_v4b/default.tsv"
output_path = None

# overwrite input table if no output_path is specified
if output_path is None:
    output_path = table_path

# read segmentation table
tsv_table = pd.read_csv(table_path, sep="\t")

scale_factor = 20
# default values
min_component_length = 50,
max_edge_distance = 30,
tsv_table = label_components.label_components_ihc(tsv_table, max_edge_distance=max_edge_distance, min_component_length=min_component_length)
# for SGN
#tsv_table = label_components.label_components_sgn(tsv_table, max_edge_distance=max_edge_distance, min_component_length=min_component_length)

component_labels = list(tsv_table["component_labels"])
centroids = list(zip(tsv_table["anchor_x"], tsv_table["anchor_y"], tsv_table["anchor_z"]))
array_downscaled = label_components.downscaled_centroids(centroids=centroids, scale_factor=scale_factor, component_labels=component_labels, downsample_mode="components")
image_downscaled = label_components.downscaled_centroids(centroids, scale_factor=scale_factor, downsample_mode="accumulated")

# 1) View down-scaled labeled image and segmentation in napari.
# 2) Adjust the parameters of min_component_length and max_edge_distance until the IHC segmentation consists of individual components.
# 3) Note the components making up the IHC segmentation for later use (tonotopic mapping, quantifying IHCs).
viewer = napari.Viewer()
viewer.add_image(image_downscaled, name='3D Volume')
viewer.add_labels(array_downscaled, name="components")
napari.run()

# calculate the number of segmentation instances within the components
print(len(tsv_table[tsv_table["component_labels"] == 1])) # number of IHCs in largest component
comp_list = [1, 3, 9]
print(len(tsv_table[tsv_table["component_labels"].isin(comp_list)])) # number of IHCs in selection of components

# save modified segmentation table, the column "component_labels" has been added
tsv_table.to_csv(output_path, sep="\t", index=False)
```
