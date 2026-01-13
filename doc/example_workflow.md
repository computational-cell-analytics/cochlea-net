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

## Segmentation

The segmentation can be performeed locally or using Slurm and the GWDG resources.

### Locally

The data can be processed locally using the command line interface.

#### SGN
```bash
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup0/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/SGN_v2"
TYPE=SGN
flamingo_tools.run_segmentation  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_type "$TYPE" \
    --min_size 1000
```

#### IHC
```bash
INPUT_PATH="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/M_AMD_N162_L/MAMD_N162L_PV_Vglut3_CTBP2_fused.n5"
INPUT_KEY="setup1/timepoint0/s0"
OUTPUT_FOLDER="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/IHC_v4b"
TYPE=IHC
flamingo_tools.run_segmentation  --input_path "$INPUT_PATH" \
    --input_key "$INPUT_KEY" \
    --output_folder "$OUTPUT_FOLDER" \
    --model_type "$TYPE" \
    --min_size 1000
```

#### Synapses


### Using Slurm
Because it is more efficient to split the application of the network into multiple jobs, the segmentation workflow is divided into three steps:
* Masking the image data based on intensity and calculating the mean and standard deviation of the intensity
* Applying CochleaNet
* Segmenting the prediction of CochleaNet

#### SGN
```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
COCHLEA="M_AMD_N162_L"
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

#### IHC

```bash
SCRIPT_DIR="/user/schilling40/u15000/flamingo-tools"
# --- Masking and calculating mean and standard deviation ---
COCHLEA="M_AMD_N162_L"
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

## Transfer to MoBIE format

Templates for the transfer into the MoBIE data format can be found in `reproducibility/templates_transfer`.
Both, the scripts as well as the command line usage require the installation of [MoBIE utils](https://github.com/mobie/mobie-utils-python).
Those scripts are designed for the usage with Slurm ([GWDG Doc](https://docs.hpc.gwdg.de/how_to_use/slurm/index.html)) and need some adjustments by the user, e.g. the activation of the working environment, the location of the MoBIE project, and the mail address.

### Image data

This example call shows the transfer of the **PV** image data of the **M_AMD_N162_L** cochlea, which is located in `setup0/timepoint0/s0`.
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

This example call shows the transfer of the **SGN_v2** segmentation data of the **M_AMD_N162_L** cochlea, which is located in `setup0/timepoint0/s0`.
This would be an example call of the function which transfers the data of `setup0/timepoint0/s0` which contains the PV channel of the M_AMD_N162_L cochlea.
```bash
COCHLEA="M_AMD_N162_L"
DATA="/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/M_AMD_N162_L/SGN_v2/segmentation.zarr"
CHANNEL_NAME="SGN_v2"
# submit the job to Slurm
sbatch reproducibility/templates_transfer/mobie_segmentation_template.sbatch $COCHLEA $DATA $CHANNEL_NAME
# or run the job locally
bash reproducibility/templates_transfer/mobie_segmentation_template.sbatch $COCHLEA $DATA $CHANNEL_NAME
```
