# Baselines for the Segmentation of IHCs and SGNs

Other networks/methods have been evaluated on SGN and IHC crops to compare the proposed method to the state of the art and justify the development of said method.
Cellpose 3, Cellpose-SAM, micro-sam, and the distance U-Net were evaluated for both SGNs and IHCs.
Additionally, Stardist was used for SGN segmentation.
HCAT, a specialized tool for the segmentation of IHCs, was omitted because it is a purely 2D network and, therefore, does not fit the use case.

## Micro-sam

Follow [instructions](https://computational-cell-analytics.github.io/micro-sam/micro_sam.html).

## Cellpose-SAM

Instalation:
```
micromamba create --name cellpose-sam python=3.10
micromamba activate cellpose-sam
python -m pip install cellpose
python -m pip install cellpose --upgrade
```

The script is adaapted from the [example Jupyter notebook online](https://github.com/MouseLand/cellpose/blob/main/notebooks/run_Cellpose-SAM.ipynb).

## Cellpose 3

Installation:
```
micromamba create --name cellpose3 python=3.10
micromamba activate cellpose3
python -m pip install cellpose==3.1.1.2
```

The script is adapted from the [example Jupyter notebook online](https://github.com/MouseLand/cellpose/blob/main/notebooks/run_cellpose3.ipynb).

## Stardist

Installation:
```
micromamba create -n stardist python=3.11 napari pyqt stardist-napari -y
micromamba activate stardist
python3.11 -m pip install 'tensorflow[and-cuda]'
python3.11 -m pip install stardist
python3.11 -m pip install 'numpy<2'
```

## Baseline for nucleus segmentation

NIS3d (light-sheet imaged data) with annotated nuclei.
Data was retrieved using this [torch-em script](https://github.com/constantinpape/torch-em/blob/main/torch_em/data/datasets/light_microscopy/nis3d.py).
The following summary is adapted from the same script:
* The NIS3D dataset contains fluorescence microscopy volumetric images of
multiple species (drosophila, zebrafish, etc) for nucleus segmentation.
* The dataset is from the [publication](https://proceedings.neurips.cc/paper_files/paper/2023/hash/0f2cd3d09a132757555b602e2dd43784-Abstract-Datasets_and_Benchmarks.html)
* The original codebase for downloading the data and other stuff is located [here](https://github.com/yu-lab-vt/NIS3D)
* The dataset is open-sourced at [zenodo](https://zenodo.org/records/11456029).

The dataset contains volumetric images for two samples of drosophilia, zebrafish, and mouse.
One sample each is used for training. The other samples are split for validation and testing.
