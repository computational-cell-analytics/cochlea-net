# CochleaNet IHC Segmentation

## Description

This 3D U-Net segments inner hair cells (IHCs) in light-sheet microscopy volumes of cleared
cochleae. The input is the VGlut3 or Myosin VIIa channel at 0.38 um isotropic voxel size. The
network was trained on annotated VGlut3-stained mouse and gerbil cochleae.

The network predicts three channels in the range [0, 1]:

1. `foreground`: the probability that a voxel belongs to an IHC.
2. `center_distance`: the normalized distance to the center of the IHC.
3. `boundary_distance`: the normalized distance to the boundary of the IHC.

The network starts with an instance normalization. Thus, the prediction does not depend on a global
intensity scale.

## Intended Use

Use the model to segment the IHCs along the organ of Corti, then count them and assign ribbon
synapses to them. To get IHC instances from the prediction, do these steps:

1. Smooth the `center_distance` and `boundary_distance` channels with a Gaussian filter
   (sigma = 1 voxel).
2. Make the mask: `foreground > 0.5`.
3. Make the seeds: label the connected components of
   `(center_distance < 0.7) & (boundary_distance < 0.6)` inside the mask.
4. Run a seeded watershed on `boundary_distance` inside the mask.
5. Remove the objects smaller than 1000 voxels.

A grid search on the validation data selected these thresholds. CochleaNet implements this
pipeline for full cochleae (`flamingo_tools.run_segmentation`).

Limitations:

- Data at a different voxel size must be rescaled to about 0.38 um first.
- The training data contains only VGlut3-stained cochleae. Check the results on Myosin VIIa data
  before you use them.
- Predict large volumes in tiles. A halo of (16, 32, 32) voxels along (z, y, x) prevents tile-border
  artifacts.

## Training Data

The training data for this model is the archive `IHC_v1.zip` in the Zenodo record
https://doi.org/10.5281/zenodo.23478267 (MIT license).

## Validation

The package contains a VGlut3 sample volume (64 x 256 x 256 voxels) as test input and the matching
network output. `bioimageio test` reproduces this output. The CochleaNet preprint gives the
quantitative evaluation against manual annotations.

## Citation

Roos et al. CochleaNet: deep learning-based image analysis for cochlear connectomics and gene
therapy. bioRxiv 2025. https://doi.org/10.1101/2025.11.16.688700

Roos et al. CochleaNet training data for SGN segmentation, IHC segmentation and ribbon synapse
detection. Zenodo 2026. https://doi.org/10.5281/zenodo.23478267
