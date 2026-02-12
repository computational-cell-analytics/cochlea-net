# Transfer of data on MoBIE format to S3 bucket

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
