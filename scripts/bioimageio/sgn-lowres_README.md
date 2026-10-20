# CochleaNet Low-Resolution SGN Segmentation

## Description

This anisotropic 3D U-Net segments spiral ganglion neurons (SGNs) in lower-resolution light-sheet
microscopy volumes of cleared cochleae. The input is the parvalbumin (PV) channel at a voxel size of
1.887779 x 1.887779 x 3.0 um (x, y, z). The network was trained on annotated PV-stained mouse and
marmoset cochleae from a LaVision UltraMicroscope.

The network predicts three channels in the range [0, 1]:

1. `foreground`: the probability that a voxel belongs to an SGN.
2. `center_distance`: the normalized distance to the center of the SGN.
3. `boundary_distance`: the normalized distance to the boundary of the SGN.

The network starts with an instance normalization. Thus, the prediction does not depend on a global
intensity scale.

## Intended Use

Use the model to segment SGNs in Rosenthal's canal in lower-resolution data of full cochleae, then
count them. To get SGN instances from the prediction, do these steps:

1. Make the mask: `foreground > 0.5`.
2. Make the seeds: label the connected components of `boundary_distance < 0.5` inside the mask.
3. Run a seeded watershed on `boundary_distance` inside the mask.
4. Remove the objects smaller than 100 voxels.

CochleaNet implements this pipeline for full cochleae
(`flamingo_tools.run_segmentation`, model `SGN-lowres`, option `--min_size 100`).

Limitations:

- Data at a different voxel size must be rescaled to about 1.89 x 1.89 x 3.0 um first.
- The training data contains PV-stained crops of one mouse and one marmoset cochlea. Check the
  results on other samples before you use them.
- Predict large volumes in tiles. A halo of (8, 32, 32) voxels along (z, y, x) prevents tile-border
  artifacts.

## Training Data

The training data for this model is the archive `SGN-lowres_v1.zip` in the Zenodo record
https://doi.org/10.5281/zenodo.23478267 (MIT license).

## Validation

The package contains a PV sample volume (32 x 128 x 128 voxels) as test input and the matching
network output. `bioimageio test` reproduces this output.

## Citation

Roos et al. CochleaNet: deep learning-based image analysis for cochlear connectomics and gene
therapy. bioRxiv 2025. https://doi.org/10.1101/2025.11.16.688700

Roos et al. CochleaNet training data for SGN segmentation, IHC segmentation and ribbon synapse
detection. Zenodo 2026. https://doi.org/10.5281/zenodo.23478267
