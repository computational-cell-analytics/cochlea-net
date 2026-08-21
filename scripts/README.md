# Segmentation for large lightsheet volumes


## Installation

Needs [torch-em](https://github.com/constantinpape/torch-em) in the python environment. See [here](https://github.com/constantinpape/torch-em?tab=readme-ov-file#installation) for installation instructions. (If possible use `mamba` instead of `conda`.)
After setting up the environment you also have to add support for the MoBIE python library via
```
conda install -c conda-forge mobie_utils
```


## Training

Contains the scripts for training a U-Net that predicts foreground probabilties and normalized object distances.
It also contains a documentation for how to run training on new annotated data.


## Prediction

Contains the scripts for running segmentation for a large volume with a distance prediction U-Net, postprocessing the segmentation
and exporting the segmentation result to MoBIE

To run the full segmentation workflow, including the export to MoBIE you can use the `segmentation_workflow.py` script as follows:
```
python segmentation_workflow.py -i /path/to/volume.xml -o /path/to/output_folder -s /path/to/seg_results --scale 0 --mobie_dataset name_in_mobie --model /path/to/model.pt
```

Here, `-i` must point to the xml file of the fused data exported from BigSticher, `-o` indicates the output folder where the MoBIE project with the semgentation result will be saved, `-s` is a folder where intermediate segmentation results will be saved, `--scale` indicates the scale to use for the segmentation, `--mobie_dataset` the (optional) name of the data in MoBIE and `--model` the path to the segmentation model.

### Individual Workflow Steps

You can also run individual steps of the workflow, like prediction and segmentation: 

You can run it like this for an input volume that is stored in n5, e.g. the fused export from bigstitcher:
```
python run_prediction_distance_unet.py -i /path/to/volume.n5 -k setup0/timepoint0/s0 -m /path/to/model -o /path/to/output_folder
```
Here, `-i` specifies the input filepath, `-o` the folder where the results are saved and `-k` the internal path in the n5 file.
The `-m` argument specifies the filepath to the model for prediction. You need to give the path to the folder that contains the checkpoint (the `best.pt` file).

You can also run the script for a tif file. In this case you don't need the `-k` parameter:
```
python run_prediction_distance_unet.py -i /path/to/volume.tif -m /path/to/model -o /path/to/output_folder
```

The result will be stored as `segmentation.zarr` in the output folder, where also the intermediate files `prediction.zarr` and `seeds.zarr` are saved.

In order to downsample the segmentation on the fly for the segmentation process you can use the argument `--scale`.
E.g. run the command 
```
python run_prediction_distance_unet.py ... --scale 2
```
to downsample the input by a factor of 2. Note that the segmentation result will be automatically rescaled back to the full shape of the input at the end.

In addition, the script `postprocess_seg.py` can be used to filter out false positive nucleus segmentations from regions in the segmentation with a low density of segmented nuclei.

You can use the script `to_tif.py` to convert the zarr object to a tif volume for easier viewing (won't work for large volumes!).


## Troubleshooting an n5 input

If a prediction fails with `'Group' object has no attribute 'shape'`, the key points to a
directory without valid array metadata. This happens when a container was copied incompletely.
Check the container with:
```
python check_n5_container.py /path/to/volume.n5 -k setup0/timepoint0/s0
```
The script reports the structure, the chunk files that are truncated or absent, and whether the
installed readers can open the format at all. It exits non-zero when it finds a problem.
