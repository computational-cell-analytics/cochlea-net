# CochleaNet

CochleaNet is a tool for the analysis of cochleae in light-sheet microscopy data.
Its main components are:
- A deep neural network that segments spiral ganglion neurons (SGNs) in a parvalbumin (PV) stain.
- A deep neural network that segments inner hair cells (IHCs) in a VGlut3 stain.
- A deep neural network that detects ribbon synapses in a CtBP2 stain.

CochleaNet also contains functions for data pre-processing and for measurements on the network predictions:
- Analysis of the tonotopic mapping of SGNs and IHCs in the cochlea.
- Validation of gene therapies and optogenetic therapies, with additional fluorescent stains.
- Analysis of SGN subtypes, with additional fluorescent stains.
- Visualization of segmentation results and derived analyses in [MoBIE](https://mobie.github.io/).

The networks and analysis methods are primarily for high-resolution isotropic data from a [custom light-sheet microscope](https://www.nature.com/articles/s41587-025-02882-8).
The networks work best for the fluorescent stains of their training data, but they also work for similar stains.
For example, the SGN network segments a calretinin (CR) stain, and the IHC network segments a Myosin VIIa stain.
CochleaNet also provides networks that segment SGNs and IHCs in anisotropic data from a [commercial light-sheet microscope](https://www.miltenyibiotec.com/DE-en/products/macs-imaging-and-spatial-biology/ultramicroscope-platform.html).

For more information on CochleaNet, see our [preprint](https://doi.org/10.1101/2025.11.16.688700).

## Installation

Install CochleaNet with `conda` (or [micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html)):
- Download the CochleaNet GitHub repository:
```
git clone https://github.com/computational-cell-analytics/cochlea-net
```
- Go to the directory:
```
cd cochlea-net
```
- Create an environment with the necessary dependencies:
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
- (Optional) To use the napari plugin, install napari:
```
conda install -c conda-forge napari pyqt
```
- (Optional) To train the synapse detection with the flow channels, install spotiflow:
```
pip install ".[flow]"
```
## Available Models

CochleaNet provides four segmentation models:
- `SGN`: segments spiral ganglion neurons (SGNs) in high-resolution, isotropic light-sheet microscopy data.
    - Training data: PV stain, voxel size 0.38 micrometer.
- `IHC`: segments inner hair cells (IHCs) in high-resolution, isotropic light-sheet microscopy data.
    - Training data: Vglut3 stain, voxel size 0.38 micrometer.
- `SGN-lowres`: segments SGNs in lower-resolution, anisotropic light-sheet microscopy data.
    - Training data: PV stain, voxel size 0.76 X 0.76 X 3.0 micrometer.
- `IHC-lowres`: segments IHCs in lower-resolution, anisotropic light-sheet microscopy data.
    - Training data: Myosin VIIa stain, voxel size 0.76 X 0.76 X 3.0 micrometer.

CochleaNet provides one detection model:
- `Synapses`: detects afferent ribbon synapses in high-resolution, isotropic light-sheet microscopy data.
    - Training data: CtBP2 stain, voxel size 0.38 micrometer.

The models are available on BioImage.IO:
- [SGN](https://bioimage.io/#/artifacts/knowledgeable-mouse),
- [IHC](https://bioimage.io/#/artifacts/content-fish),
- [Synapses](https://bioimage.io/#/artifacts/loyal-ant),
- [SGN-lowres](https://bioimage.io/#/artifacts/nervous-shark), and
- [IHC-lowres](https://bioimage.io/#/artifacts/confused-cow).

CochleaNet downloads a model from BioImage.IO when you use it for the first time.

**Note: To run the segmentation in an environment without internet access, download the models first. Then copy them to the `Model directory` of CochleaNet. The command `flamingo_tools.info` shows the model directory.**

Use `flamingo_tools.download_model` after the installation. It downloads all models or the selected models:
```bash
# Download all models
flamingo_tools.download_model
# Download a single model
flamingo_tools.download_model -m SGN
```

To examine the downloaded models, run:
```bash
flamingo_tools.info
```
The output is similar to:
```
Version: 0.1.0
Model directory: /home/pape/.cache/cochlea-net/models
Downloaded models: ['SGN-knowledgeable-mouse-v0.pt', 'IHC-content-fish-v0.pt', 'Synapses-loyal-ant-v0.pt', 'SGN-lowres-nervous-shark-v0.pt', 'IHC-lowres-confused-cow-v0.pt']
```
If you copy the models to the `Model directory` of another machine, run `flamingo_tools.info` on that machine to make sure that it finds the models.

## Usage

Use CochleaNet with:
- The [napari plugin](#napari-plugin): prediction with the pre-trained CochleaNet deep neural networks.
- The [command line interface](#command-line-interface): data conversion, model prediction, and analysis workflows for large image data.
- The [python library](#python-library): the functions of CochleaNet, for your own prediction and analysis workflows for large image data.
- The [HPC pipelines](#hpc-pipelines): processing of whole cochleae as Slurm jobs on the GWDG cluster.

**Note: The napari plugin is not optimized for large data. Use the CLI or the python library to process large data.**

### Napari Plugin

The plugins for segmentation (SGNs and IHCs) and detection (ribbon synapses) are in napari under `Plugins->CochleaNet->Segmentation/Detection`:

<img src="https://raw.githubusercontent.com/computational-cell-analytics/cochlea-net/refs/heads/master/doc/img/cochlea-net-plugin-selection.png" alt="The CochleaNet plugins available in napari." width="384">


Select the model under `Select Model:` (see [Available Models](#available-models) for details).
Select the image data (napari layer) for the model under `Image data`.
Click `Run Segmentation` to start the segmentation. When the segmentation is complete, the plugin adds a segmentation layer with the result (here `IHC`):
<img src="https://raw.githubusercontent.com/computational-cell-analytics/cochlea-net/refs/heads/master/doc/img/cochlea-net-plugin-segmentation.png" alt="The CochleaNet segmentation plugin." width="768">

Select `Show intermediate outputs` in the advanced settings to add the foreground, center-distance, and boundary-distance predictions as image layers.

The detection plugin works in the same way. It currently provides the model for synapse detection. The plugin adds the predictions as a point layer (`Synapses`):
<img src="https://raw.githubusercontent.com/computational-cell-analytics/cochlea-net/refs/heads/master/doc/img/cochlea-net-plugin-detection.png" alt="The CochleaNet detection plugin." width="768">

Select `Show intermediate outputs` in the advanced settings to also add the synapse heatmap as an image layer.

For more information on napari, see the tutorials at [www.napari.org](https://napari.org/stable/).

**The napari plugin needs `napari` and `pyqt` in your environment. See [installation](#installation) for details.**

### Command Line Interface

The command line interface provides these commands for data conversion and prediction:

`flamingo_tools.convert_data` converts data from a flamingo microscope:
```bash
flamingo_tools.convert_data -i /path/to/data -o /path/to/output.n5 --file_ext .tif
```
Use `--file_ext .raw` for data in raw files. The extension of the output path (`-o`) sets the output data format:
- `.n5` writes the [bdv.n5 format](https://github.com/bigdataviewer/bigdataviewer-core/blob/master/BDV%20N5%20format.md). Open it with BigDataViewer via `Plugins->BigDataViewer->Open XML/HDF5`, or with [BigStitcher](https://imagej.net/plugins/bigstitcher/) as described [here](https://imagej.net/plugins/bigstitcher/open-existing).
- `.ome.zarr` writes the [ome.zarr format](https://ngff.openmicroscopy.org/).

`flamingo_tools.run_segmentation` segments cells in volumetric light microscopy data:
```bash
flamingo_tools.run_segmentation -i /path/to/data.tif -o /path/to/output_folder -m SGN
```
`-m` selects the model. See also [available models](#available-models).
Use `--checkpoint_path` (`-c`) for a custom trained model.

`flamingo_tools.run_detection` detects synapses in volumetric light microscopy data. Its usage is similar to `flamingo_tools.run_segmentation`.
See the [segmentation documentation](https://github.com/computational-cell-analytics/cochlea-net/blob/master/doc/segmentation.md) for examples of both commands.

The command line interface also provides commands for the analysis of segmentation tables:
- `flamingo_tools.label_components`: finds the connected components of a segmentation.
- `flamingo_tools.tonotopic_mapping`: assigns a frequency to each instance.
- `flamingo_tools.object_measures`: measures the morphology and the intensity of each instance.
- `flamingo_tools.json_block_extraction`, `flamingo_tools.equidistant_centers`, `flamingo_tools.extract_central_blocks` and `flamingo_tools.extract_block`: extract blocks along the cochlea, for example for the intensity annotation.
- `flamingo_tools.sgn_density`: calculates the SGN density at positions along Rosenthal's canal.
- `flamingo_tools.table_info`: finds the table row closest to a value of a column, for example a frequency.

See the [analysis documentation](https://github.com/computational-cell-analytics/cochlea-net/blob/master/doc/analysis.md) and the [SGN density documentation](https://github.com/computational-cell-analytics/cochlea-net/blob/master/doc/sgn_density.md) for examples.

Run `flamingo_tools.<COMMAND> -h` (for example `flamingo_tools.run_segmentation -h`) to show all options of a command.

### Python Library

The `flamingo_tools` python library contains the functions of CochleaNet:
- `flamingo_tools.measurements`: measures morphological attributes and intensity statistics of segmented cells.
- `flamingo_tools.mobie`: exports flamingo image data or segmentation results to a MoBIE project.
- `flamingo_tools.postprocessing`: labels connected components and maps the tonotopy of segmented cells.
- `flamingo_tools.segmentation`: applies segmentation and detection models to large volumetric image data.
- `flamingo_tools.training`: trains segmentation and detection networks.

### HPC Pipelines

The repository [slurm-workflows](https://github.com/computational-cell-analytics/slurm-workflows) holds the Slurm templates and pipelines that process whole cochleae with CochleaNet on the GWDG cluster.
Its project `cochlea-net` provides pipelines for:
- the conversion to the MoBIE data format and the transfer to the S3 bucket,
- the segmentation of IHCs,
- the segmentation of SGNs, and
- the detection of ribbon synapses.

Each user keeps the account-specific values in a settings file. A parameter file per cochlea holds the values of a job.
The script `scripts/deploy_process.py` fills the templates with these values and submits the steps of a pipeline as a chain of Slurm jobs.
The repository also archives the metadata of each job.

The documentation of slurm-workflows explains:
- [Getting started](https://github.com/computational-cell-analytics/slurm-workflows/blob/main/doc/hpc_01_getting_started.md): the HPC project, the SSH connection, and the environment on the cluster.
- [Data transfer](https://github.com/computational-cell-analytics/slurm-workflows/blob/main/doc/hpc_02_data_transfer.md): the transfer from and to UKON100 and the S3 bucket.
- [Processing pipelines](https://github.com/computational-cell-analytics/slurm-workflows/blob/main/doc/hpc_03_processing_pipelines.md): the settings file, the parameter files, and the commands that run a pipeline.

After a pipeline, continue with the [analysis](https://github.com/computational-cell-analytics/cochlea-net/blob/master/doc/analysis.md) of the segmentation.
