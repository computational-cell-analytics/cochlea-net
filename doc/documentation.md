# CochleaNet

CochleaNet is a software tool for the analysis of cochleae imaged in light-sheet microscopy.
Its main components are:
- A deep neural network for segmenting spiral ganglion neurons (SGNs) from parvalbumin (PV) staining.
- A deep neural network for segmenting inner hair cells (IHCs) from VGlut3 staining.
- A deep neural network for detecting ribbon synapses from CtBP2 staining.

In addition, it contains functionality for data pre-processing and different kinds of measurements based on the network predictions, including:
- Analyzing the tonotopic mapping of SGNs and IHCs in the cochlea.
- Validating gene therapies and optogentic therapies (based on additional fluorescent stainings).
- Analyzing SGN subtypes (based on additional fluorescent staining).
- Visualizing segmentation results and derived analyses in [MoBIE](https://mobie.github.io/).

The networks and analysis methods were primarily developed for high-resolution isotropic data from a [custom light-sheet microscope](https://www.biorxiv.org/content/10.1101/2025.02.21.639411v2.abstract).
The networks will work best on the respective fluorescent stains they were trained on, but will work on similar stains. For example, we have successfully applied the network for SGN segmentation on a calretinin (CR) stain and the network for IHC segmentation on a myosin7a stain. 
In addition, CochleaNet provides networks for the segmentation of SGNs and IHCs in anisotropic data from a [commercial light-sheet microscope](https://www.miltenyibiotec.com/DE-en/products/macs-imaging-and-spatial-biology/ultramicroscope-platform.html).

For more information on CochleaNet, check out our [preprint](TODO).

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

**Note: the napari plugin was not optimized for processing large data. For processing large image data use the CLI or python library.**

### Napari Plugin

The napari plugin for segmentation (SGNs and IHCS) and detection (ribbon synapses) is available under `Plugins->CochleaNet->Segmentation/Detection` in napari:

The segmentation plugin offers the choice of different models under `Select Model:` (see [Available Models](#available-models) for details). `Image data` enables to choose which image data (layer) the model is applied to. The segmentation is started by clicking the `Run Segmentation` button. After the segmentation has finished, a new segmentation layer with the result (here `IHC`) will be added:

The detection model works similarly. It currently provides the model for synapse detection. The predictions are added as point layer (``):

TODO Video.
For more information on how to use napari, check out the tutorials at [www.napari.org](TODO).

**To use the napari plugin you have to install `napari` and `pyqt` in your environment.** See [installation](#installation) for details.

### Command Line Interface

TODO

### Python Library

TODO


## Available Models

TODO
