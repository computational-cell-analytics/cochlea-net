# Methods for the analysis of a segmented cochlea

The following code snippets will show how a single cochlea is analyzed.

## Labeling components
The component labeling for SGNs is quite reliable.
```bash
# locally
flamingo_tools.label_components --input "$MOBIE_DIR"/M_AMD_N162_L/tables/SGN_v2/default.tsv -o M_AMD_N162_L_v2.tsv --cell_type sgn
# access segmentation table on S3 bucket
flamingo_tools.label_components --input M_AMD_N162_L/tables/SGN_v2/default.tsv --s3 -o M_AMD_N162_L_v2.tsv --cell_type sgn
# check with napari
flamingo_tools.label_components --input M_AMD_N162_L/tables/SGN_v2/default.tsv --s3 -o M_AMD_N162_L_v2.tsv --cell_type sgn --napari
```

Labeling IHC components may require tuning of the `--max_edge_distance` and `--min_component_length` parameters and is usually an iterative process.

```bash
# locally
flamingo_tools.label_components --input "$MOBIE_DIR"/M_AMD_N162_L/tables/IHC_v4b/default.tsv -o M_AMD_N162_L_v4b.tsv --cell_type ihc
# access segmentation table on S3 bucket
flamingo_tools.label_components --input M_AMD_N162_L/tables/IHC_v4b/default.tsv --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc
# check with napari
flamingo_tools.label_components --input M_AMD_N162_L/tables/IHC_v4b/default.tsv --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc --napari
```

### Example

A typical process could look like this:
```bash
flamingo_tools.label_components --input M_AMD_N162_L/tables/IHC_v4b/default.tsv --napari --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc
```
![Step1](img/component_label_ihc_01.png)
For the most part, the IHCs are correctly segmented, but there are some gaps between the labeled components.
The gap between components 2 (light blue) and 3 (light violet) is small, and a larger section between components 3 and 4 (dark violet) has been segmented but is not registered as a connected component. We can resolve these issues by increasing the `max_edge_distance` (default: 30 µm). Setting it to 50 or 70 µm is usually a good choice.

```bash
flamingo_tools.label_components --input M_AMD_N162_L/tables/IHC_v4b/default.tsv --napari --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc --max_edge_distance 70 --force
```
![Step2](img/component_label_ihc_02.png)
The gap between components 2 and 3 has closed, and components 2, 3, and 4, as well as the section between 3 and 4, have fused into a single component.
However, there are isolated IHC instances at the top and bottom of the main component which we can see in the white and grey overlay.
There are too few IHCs to register as a connected component, so we decrease the minimum component length (default: 50 instances).
It can be set as low as two, but estimating the number of IHCs can help keep the representation clear and concise.
```bash
flamingo_tools.label_components --input M_AMD_N162_L/tables/IHC_v4b/default.tsv --napari --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc --max_edge_distance 70 --min_component_length 20 --force
```
![Step3](img/component_label_ihc_03.png)
As we can see, the IHC spiral consists of components 1, 4, and 5. You can view the label by hovering over the component in Napari.
We can find the total number of IHCs by adding the parameter `-c` to our previous command.
Although the order of components 4, 1, and 5 does not matter in this instance, it is useful to keep in mind because it will be necessary for tonotopic mapping.
```bash
flamingo_tools.label_components --input M_AMD_N162_L/tables/IHC_v4b/default.tsv --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc --max_edge_distance 70 --min_component_length 20 -c 4 1 5 --force
```
Output of the terminal:
```
Total IHCs: 926
Component 4 has 21 instances.
Component 1 has 511 instances.
Component 5 has 21 instances.
Custom component(s) have 553 IHCs.
```
Because we can not be sure, if all the components we selected are indeed IHCs, the components should be verified by visualizing them in MoBIE.

## Tonotopic mapping

The tonotopic mapping command is quite similar to the component labeling command, but it does require some additional information.
Specifically, the animal type must be specified because the parameters of the Greenwood function, which is used for frequency mapping, differ between mice and gerbils.
If the segmentation consists of multiple connected components, they must be in the same consecutive order as in the cochlear volume.
```bash
flamingo_tools.tonotopic_mapping --input M_AMD_N162_L_v4b.tsv --s3 -o M_AMD_N162_L_v4b.tsv --cell_type ihc --animal mouse --max_edge_distance 70  -c 4 1 5 --force
```
We can use the table we created after labeling the components as the input because the function only adds new columns without changing the existing ones.

## Intensity annotation

For the analysis of optogenetic therapy or the identification of SGN subtypes, the intensity of stains within SGN segmentation is evaluated.
Although evaluation is performed on the entire cochlear volume, manual thresholding on subvolumes is sufficient and the results can be extrapolated to the entire volume.

1) Multiple equidistant crops are extracted along the segmentation.
2) Object measures of the stains are calculated for each instance of the segmentation.
3) The crops are manually thresholded to classify positive and negative instances.
4) Finally, based on the manual thresholding, the entire cochlear volume is analyzed.

### 1 - Extraction of regions of interest (ROI) blocks

Sub-volumes in form of blocks are extracted to determine thresholds for positive/negative instances.

Create a JSON dictionary which will be used as the input for extracting central crops.
```bash
flamingo_tools.json_block_extraction -o M_AMD_N162_L.json -d M_AMD_N162_L -i PV SGN_v2 --cell_type sgn -c 1 -n 6 --roi_halo 256 256 64 -s SGN_v2
```
Provide this JSON file as input to the function
```bash
# s3 cluster
flamingo_tools.extract_central_blocks --input M_AMD_N162_L.json -o <output_dir> --s3
# local MoBIE project
flamingo_tools.extract_central_blocks --input M_AMD_N162_L.json -o <output_dir> --mobie_dir <mobie_project_dir>
```

#### Step-by-Step
The process can also be performed step-by-step.
This function provides coordinates that are equally spaced along the center of Rosenthal's canal (for SGN) or along the inner hair cell (IHC) segmentation.
```bash
flamingo_tools.equidistant_centers -i M_AMD_N162_L/tables/SGN_v2/default.tsv -o M_AMD_N162_L_crop.json -n 6 --s3
```
This command creates a JSON dictionary with six center coordinates which can be used with the function `flamingo_tools.extract_block`:
```bash
flamingo_tools.extract_block --input M_AMD_N162_L/images/ome-zarr/SGN_v2.ome.zarr -o <output_dir> --json_info M_AMD_N162_L_crop.json --s3
# or for a single crop center
flamingo_tools.extract_block --input M_AMD_N162_L/images/ome-zarr/SGN_v2.ome.zarr -o crop.tif -c x y z --s3
```

### 2 - Calculation of object measures

Before performing the thresholding in Napari, the intensity of the stain within the SGN segmentation is calculated.
The function `flamingo_tools.object_measures` can also take the same JSON file which was used for the block extraction as an input.
When working locally, the MoBIE directory can be given via the argument `--mobie_dir` to create the files directly in the appropriate locations,
e.g. under `<mobie_project>/<cochlea>/tables/<seg_name>/<stain>_<seg_name>_object-measures.tsv`.
If an output directory is specified, the output is given as:
`<cochlea>_<stain>_<seg_name>_object-measures.tsv`
```bash
# for creating the files in the MoBIE project
flamingo_tools.object_measures --mobie_dir <mobie_dir> --json_info M_AMD_N162_L.json

# for creating the files in an output directory reading the files from the S3 bucket
flamingo_tools.object_measures -o <output_dir> --json_info M_AMD_N162_L.json --s3

# calculate the PV intensity object measures without a JSON dictionary
flamingo_tools.object_measures -o M-AMD-N162-L_PV_SGN-v2_object-measures.tsv \
    -i M_AMD_N162_L/images/ome-zarr/PV.ome.zarr \
    --seg_table M_AMD_N162_L/tables/SGN_v2/default.tsv \
    --seg_path M_AMD_N162_L/images/ome-zarr/SGN_v2.ome.zarr \
    --s3
```

#### 2a - For subtype analysis
You may want to calculate the ratio of subtype stains to a reference stain, e.g. the ratio of Calb1 and Ntng1 to PV.
You can use the script `scripts/measurements/sgn_subtype_ratio.py` for this.
Before executing the script, the relevant parameters for the cochlea can be added to `flamingo_tools/postprocessing/sgn_subtype_utils.py` to increase the reproducibility of the process.

### 3 - Manual intensity thresholding
For the annotation of GFP, the crops of PV, GFP, and SGN are needed.
The crop files are expected to have the format `<cochlea>_crop_xxx-yyy-zzz_<image_channel>.tif`.
The annotation tool in Napari is called using the common prefix `<cochlea>_crop_xxx-yyy-zzz` of all crops.
```bash
python /path/to/cochlea-net-repository/scripts/intensity_annotation/gfp_annotation.py --meas_table <path_to_object_measures> --prefix <common_prefix>
```
For each crop, analysis requires two segmentation representations, which separate the instances into two groups through thresholding.
1) The first should separate the clearly negative instances from all instances, which might be seen as positive.
It should be named `<cochlea>_crop_<crop-coords>_<stain>_allWeakPositiveIncluded_<suffix>.tif`.
2) The second should separate the clearly positive instances from all instances, which might be seen as negative.
It should be named `<cochlea>_crop_<crop-coords>_<stain>_allNegativeExcluded_<suffix>.tif`.

Note: While the suffixes can be chosen freely, the other components are essential.

Based on these two files, the next processing step will calculate a threshold as a mean value between both groups - the clearly negative and the clearly positive instances.

### 4 - Analysing the marker annotation
All thresholding files should be placed in the same directory.
This directory is passed as the `-a, --annotation_dirs` argument.
If multiple annotators worked on the same crops, multiple annotation directories can be passed.
The function will check within all passed directories for annotations of the specified cochleae.
It is not required that every annotator has annotated every crop.
A summary of the intensities for all crops and annotators will be created as `<cochlea>_<stain>_<seg_name>_annotations.tsv`.

```bash
python /path/to/cochlea-net-repository/flamingo-tools/scripts/measurements/eval_marker_annotations.py -c M_LR_000143_L \
    -o /path/to/output_dir -t /optional/path/to/output_dir \
    -a /path/to/annotation/results/Results{LR,AMD,EK} \
    --seg_name SGN_v2 --marker_name GFP --s3
```
This command is equivalent to the one above, but specifies the input paths explicitly.
```bash
python ~/flamingo-tools/scripts/measurements/eval_marker_annotations.py -c M_LR_000143_L \
    -o /path/to/output_dir -t /optional/path/to/output_dir \
    -a /path/to/annotation/results/Results{LR,AMD,EK} \
    --seg_data M_LR_000143_L/images/ome-zarr/SGN_v2.ome.zarr \
    --seg_table M_LR_000143_L/tables/SGN_v2/default.tsv \
    --meas_table M_LR_000143_L/tables/SGN_v2/GFP_SGN-v2_object-measures.tsv \
    --seg_name SGN_v2 --marker_name GFP --s3
```
The analysis can also be performed locally. If the paths are as expected (see previous command), passing the `--mobie_dir` argument is sufficient:
```bash
python ~/flamingo-tools/scripts/measurements/eval_marker_annotations.py -c M_LR_000143_L \
    -o /path/to/output_dir -t /optional/path/to/output_dir \
    -a /path/to/annotation/results/Results{LR,AMD,EK} \
    --mobie_dir <local_mobie_dir> \
    --seg_name SGN_v2 --marker_name GFP
```
If no output directory is passed, the output will be saved as a table in `<mobie_project>/<cochlea>/tables/<seg_name>/<marker_name>_<seg_name>.tsv`.

### 4a - Subtype analysis
The same functionality applies to subtype analysis as to marker annotation.
```bash
python ~/flamingo-tools/scripts/measurements/eval_subtype_annotations.py -c M_LR_N152_L \
    -o /path/to/output_dir -t /optional/path/to/output_dir \
    -a /path/to/annotation/results/Results{LR,AMD,EK} \
    --s3
```
