# CochleaNet SGN Segmentation

## Description

This 3D U-Net segments spiral ganglion neurons (SGNs) in light-sheet microscopy volumes of cleared
cochleae. The input is the parvalbumin (PV) or calretinin (CR) channel at 0.38 um isotropic voxel
size. The network was trained on annotated PV-stained mouse cochleae.

The network predicts three channels in the range [0, 1]:

1. `foreground`: the probability that a voxel belongs to an SGN.
2. `center_distance`: the normalized distance to the center of the SGN.
3. `boundary_distance`: the normalized distance to the boundary of the SGN.

The network starts with an instance normalization. Thus, the prediction does not depend on a global
intensity scale.

## Intended Use

Use the model to segment SGNs in Rosenthal's canal, then count them and measure their density along
the tonotopic axis. To get SGN instances from the prediction, do these steps:

1. Smooth the `center_distance` channel with a Gaussian filter (sigma = 2 voxels).
2. Make the mask: `foreground > 0.5`.
3. Make the seeds: label the connected components of
   `(center_distance < 0.4) & (boundary_distance < 0.5)` inside the mask.
4. Run a seeded watershed on `boundary_distance` inside the mask.
5. Remove the objects smaller than 1000 voxels.

CochleaNet implements this pipeline for full cochleae
(`flamingo_tools.run_segmentation`, model `SGN`).

Limitations:

- Data at a different voxel size must be rescaled to about 0.38 um first.
- The training data contains only PV-stained mouse cochleae. Check the results on other species or
  markers before you use them.
- Predict large volumes in tiles. A halo of (16, 32, 32) voxels along (z, y, x) prevents tile-border
  artifacts.

## Training Data

The training data for this model is the archive `SGN_v1.zip` in the Zenodo record
https://doi.org/10.5281/zenodo.23478267 (MIT license).

## Validation

The package contains a PV sample volume (64 x 256 x 256 voxels) as test input and the matching
network output. `bioimageio test` reproduces this output. The CochleaNet preprint gives the
quantitative evaluation against manual annotations.

## Citation

Roos et al. CochleaNet: deep learning-based image analysis for cochlear connectomics and gene
therapy. bioRxiv 2025. https://doi.org/10.1101/2025.11.16.688700

Roos et al. CochleaNet training data for SGN segmentation, IHC segmentation and ribbon synapse
detection. Zenodo 2026. https://doi.org/10.5281/zenodo.23478267
