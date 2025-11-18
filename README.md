# CochleaNet

CochleaNet is a software for the analysis of cochleae imaged in light-sheet microscopy. It is based on deep neural networks for the segmentation of spiral ganglion neurons, inner hair cells, and the detection of ribbon synapses.
It was developed for imaging data from (clear-tissue) [flamingo microscopes](https://huiskenlab.com/flamingo/) and is also applicable to data from commercial microscopes.

In addition to the analysis functionality, CochleaNet implements data pre-processing to convert data from flamingo microscopes into a format compatible with [BigStitcher](https://imagej.net/plugins/bigstitcher/) and to export image data and segmentation results to [ome.zarr](https://www.nature.com/articles/s41592-021-01326-w) and [MoBIE](https://mobie.github.io/).
This functionality is applicable to any imaging data from flamingo microscopes, not only clear-tissue data or cochleae. We aim to also extend the segmentation and analysis functionality to other kinds of samples imaged in the flamingo in the future.

For installation and usage instructions, check out [the documentation](https://computational-cell-analytics.github.io/cochlea-net/). For more details on the underlying methodology check out [our preprint](https://doi.org/10.1101/2025.11.16.688700).
