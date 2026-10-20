# CochleaNet Ribbon Synapse Detection

## Description

This 3D U-Net detects ribbon synapses in light-sheet microscopy volumes of cleared cochleae. The
input is the CtBP2 channel at 0.38 um isotropic voxel size. The network was trained on annotated
CtBP2-stained cochleae.

The network predicts one channel, `synapse_heatmap`. Each synapse gives a Gaussian peak
(sigma = 1 voxel) with a height of about 4. The output has no activation, so it is not limited to
a fixed range.

The network starts with an instance normalization. Thus, the prediction does not depend on a global
intensity scale.

## Intended Use

Use the model to count the ribbon synapses per inner hair cell (IHC). To get the synapse positions
from the prediction, do these steps:

1. Find the local maxima of the heatmap, for example with
   `skimage.feature.peak_local_max(heatmap, min_distance=2, threshold_abs=0.5)`.
2. Optional: keep only the synapses within 3 um of an IHC segmentation, for example from the
   CochleaNet IHC Segmentation model.

CochleaNet implements this pipeline for full cochleae (`flamingo_tools.run_detection`).

Limitations:

- Data at a different voxel size must be rescaled to about 0.38 um first.
- Synapses that touch can give one merged peak. Then the count is too low.
- Predict large volumes in tiles. A halo of (16, 64, 64) voxels along (z, y, x) prevents tile-border
  artifacts.

## Training Data

The training data for this model is the archive `synapses_v1.zip` in the Zenodo record
https://doi.org/10.5281/zenodo.23478267 (MIT license).

## Validation

The package contains a CtBP2 sample volume (64 x 256 x 256 voxels) as test input and the matching
network output. `bioimageio test` reproduces this output. The CochleaNet preprint gives the
quantitative evaluation against manual annotations.

## Citation

Roos et al. CochleaNet: deep learning-based image analysis for cochlear connectomics and gene
therapy. bioRxiv 2025. https://doi.org/10.1101/2025.11.16.688700

Roos et al. CochleaNet training data for SGN segmentation, IHC segmentation and ribbon synapse
detection. Zenodo 2026. https://doi.org/10.5281/zenodo.23478267
