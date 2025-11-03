# CochleaNet

CochleaNet is a software tool for the analysis of cochleae imaged in light-sheet microscopy.
Its main components are:
- A deep neural network for segmenting spiral ganglion neurons (SGNs) from parvalbumin (PV) staining.
- A deep neural network for segmenting inner hair cells (IHCs) from VGlut3 staining.
- A deep neural network for detecting ribbon synapses from CtBP2 staining.

In addition, it contains functionality for different kinds of measurements based on network predictions, including:
- Analyzing the tonotopic mapping of SGNs and IHCs in the cochlea.
- Validating gene therapies and optogentic therapies (based on additional fluorescent stainings).
- Analyzing SGN subtypes (based on additional fluorescent staining).
- Visualizing segmentation results and derived analysis in [MoBIE](https://mobie.github.io/). 

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

### Napari Plugin

### Command Line Interface

### Available Models

### Python Library
