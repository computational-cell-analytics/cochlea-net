# CochleaNet

CochleaNet is a tool for the analysis of cochleae imaged in light-sheet microscopy.
Its main components are:
- A deep neural network for segmenting spiral ganglion neurons (SGNs) from parvalbumin (PV) staining.
- A deep neural network for segmenting inner hair cells (IHCs) from VGlut3 staining.
- A deep neural network for detecting ribbon synapses from CtBP2 staining.

In addition, it contains functionality for data pre-processing and different kinds of measurements based on the network predictions, including:
- Analyzing the tonotopic mapping of SGNs and IHCs in the cochlea.
- Validating gene therapies and optogentic therapies (based on additional fluorescent stainings).
- Analyzing SGN subtypes (based on additional fluorescent staining).
- Visualizing segmentation results and derived analyses in [MoBIE](https://mobie.github.io/).

The networks and analysis methods were primarily developed for high-resolution isotropic data from a [custom light-sheet microscope](https://www.nature.com/articles/s41587-025-02882-8).
The networks work best for the respective fluorescent stains they were trained on, but will work for similar stains.
For example, we have successfully applied the network for SGN segmentation on a calretinin (CR) stain and the network for IHC segmentation on a Myosin VII A stain. 
In addition, CochleaNet provides networks for the segmentation of SGNs and IHCs in anisotropic data from a [commercial light-sheet microscope](https://www.miltenyibiotec.com/DE-en/products/macs-imaging-and-spatial-biology/ultramicroscope-platform.html).

For more information on CochleaNet, check out our [preprint](https://doi.org/10.1101/2025.11.16.688700).

## Installation

CochleaNet can be installed via `conda` (or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)).
To install it:
- Download the CochleaNet github repository:
```
git clone https://github.com/computational-cell-analytics/cochlea-net
```
- Go to the directory:
```
cd cochlea-net
```
- Create an environment with the required dependencies:
```
conda env create -f environment.yaml
```
- Activate the environment:
```
conda activate cochlea-net
```
- Install the cochlea-net package:
```
pip install .
```
- (Optional): if you want to use the napari plugin you have to install napari:
```
conda install -c conda-forge napari pyqt
```

## Usage

CochleaNet can be used via:
- The [napari plugin](#napari-plugin): enables prediction with the pre-trained CochleaNet deep neural networks.
- The [command line interface](#command-line-interface): enables data conversion, model prediction, and selected analysis workflows for large image data.
- The [python library](#python-library): implements CochleaNet's functionality and can be used to implement flexible prediction and data analysis workflows for large image data. 

**Note: the napari plugin was not optimized for processing large data. Please use the CLI or python library for processing large data.**

### Napari Plugin

The plugins for segmentation (SGNs and IHCS) and detection (ribbon synapses) is available under `Plugins->CochleaNet->Segmentation/Detection` in napari:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/cochlea-net/refs/heads/master/doc/img/cochlea-net-plugin-selection.png" alt="The CochleaNet plugins available in napari.">


The segmentation plugin offers the choice of different models under `Select Model:` (see [Available Models](#available-models) for details). `Image data` enables the choice which image data (napari layer) the model is applied to.
The segmentation is started by clicking the `Run Segmentation` button. After the segmentation has finished, a new segmentation layer with the result (here `IHC`) will be added:
<img src="https://raw.githubusercontent.com/computational-cell-analytics/cochlea-net/refs/heads/master/doc/img/cochlea-net-plugin-segmentation.png" alt="The CochleaNet segmentation plugin." width="768">

The detection model works similarly. It currently provides the model for synapse detection. The predictions are added as a point layer (`Synapses`):
<img src="https://raw.githubusercontent.com/computational-cell-analytics/cochlea-net/refs/heads/master/doc/img/cochlea-net-plugin-detection.png" alt="The CochleaNet detection plugin." width="768">

For more information on how to use napari, check out the tutorials at [www.napari.org](https://napari.org/stable/).

**To use the napari plugin you have to install `napari` and `pyqt` in your environment. See [installation](#installation) for details.**

### Command Line Interface

The command line interface provides the following commands:

`flamingo_tools.convert_data`: Convert data from a flamingo microscope into the [bdv.n5 format](https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md) (compatible with  [BigStitcher](https://imagej.net/plugins/bigstitcher/)) or into [ome.zarr format](https://ngff.openmicroscopy.org/). You can use this command as follows:
```bash
flamingo_tools.convert_data -i /path/to/data -o /path/to/output.n5 --file_ext .tif
```
Use `--file_ext .raw` instead if the data is stored in raw files. By default, the data will be exported to the n5 format. It can be opened with BigDataViewer via `Plugins->BigDataViewer->Open XML/HDF5` or with BigStitcher as described [here](https://imagej.net/plugins/bigstitcher/open-existing).

`flamingo_tools.run_segmentation`: To segment cells in volumetric light microscopy data.

`flamingo_tools.run_detection`: To detect synapses in volumetric light microscopy data.

For more information on any of the command run `flamingo_tools.<COMMAND> -h` (e.g. `flamingo_tools.run_segmentation -h`) in your terminal.

### Python Library

CochleaNet's functionality is implemented in the `flamingo_tools` python library. It implements:
- `measurements`: functionality to measure morphological attributes and intensity statistics for segmented cells.
- `mobie`: functionality to export flamingo image data or segmentation results to a MoBIE project.
- `segmentation`: functionality to apply segmentation and detection models to large volumetric image data.
- `training`: functionality to train segmentation and detection networks.


## Available Models

CochleaNet provides five different models:
- `SGN`: for segmenting spiral ganglion neurons (SGNs) in high-resolution, isotropic light-sheet microscopy data.
    - This model was trained on image data with parvalbumin (PV) stain, with a voxel size of 0.38 micrometer.
- `IHC`: for segmenting inner hair cells (IHCs) in high-resolution, isotropic light-sheet microscopy data.
    - This model was trained on image data with Vglut3 stain, with a voxel size of 0.38 micrometer.
- `Synapses`: for detecting afferent ribbon synapses in high-resolution isotropic light-sheet microscopy data.
    - This model was trained on image data with CtBP2 stain, with a voxel size of 0.38 micrometer.
- `SGN-lowres`: for segmenting SGNS in lower-resolution, anisotropic light-sheet microscopy data.
    - This model was trained on image data with PV stain, with a voxel size of 0.76 X 0.76 X 3.0 micrometer.
- `SGN-lowres`: for segmenting SGNS in lower-resolution, anisotropic light-sheet microscopy data.
    - This model was trained on image data with Myosin VIIa stain, with a voxel size of 0.76 X 0.76 X 3.0 micrometer.
