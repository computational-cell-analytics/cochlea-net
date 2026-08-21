# SGNs

## Network training

No network training required, the variations already exist in `/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN` as `v2-1_cochlea_distance_unet_SGN_supervised`, `v2-2_cochlea_distance_unet_SGN_supervised`, `v2-3_cochlea_distance_unet_SGN_supervised`, and `v2-4_cochlea_distance_unet_SGN_supervised`.
They have been renamed recently for an easier application for the full volume validation cochleae.

## Network application

The `SGN_v2` network variations need to be applied on the cochleae used for validation. Those are `M_AMD_000058_L`, `M_LR_000169_R`, `M_LR_000226_L`, `M_LR_000227_L`, and `M_LR_000227_R`.
The processing consists of three steps:
1) Calculating the mean and standard deviation together with an intensity mask.
2) Application of the network to get maps for foreground, boundary distance, and center distance.
3) Seeded watershed based on the maps.

Step 1 has been performed for all cochleae for an absolute intensity threshold of 200. It had to be done only once for each cochleae and has been copied to the respective folders. The folders are located in the cochlea folder in `/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions` as `SGN_v2-1`, `SGN_v2-2` ...
Step 2 has only been performed for `M_LR_000169_R` and has to be performed for all the others.
Step 3 has not been performed for any of the cochleae.

The scripts in `scripts_sgn` should cover all cases for step 2, e.g. `2026-08-21_sbatch_apply_SGN-v2-variance_MAMD000058L.sbatch`.

An example script for the segmentation step has been included in form of `2026-08-21_sbatch_segment_SGN-v2-variance_MLR000226L.sbatch`.
It has to be adapted to the user and I did not test it myself.


## Post-processing

* transfer synapse detection to MoBIE
* transfer to S3 bucket

The components can be labeled using
```bash
# for a single cochlea
flamingo_tools.label_components -i <cochlea>/tables/SGN_v2-1/default.tsv --s3 -o <output_dir>/<cochlea>_SGN_v2-1.tsv --cell_type sgn --force -c 1
flamingo_tools.label_components -i <cochlea>/tables/SGN_v2-2/default.tsv --s3 -o <output_dir>/<cochlea>_SGN_v2-2.tsv --cell_type sgn --force -c 1
flamingo_tools.label_components -i <cochlea>/tables/SGN_v2-3/default.tsv --s3 -o <output_dir>/<cochlea>_SGN_v2-3.tsv --cell_type sgn --force -c 1
flamingo_tools.label_components -i <cochlea>/tables/SGN_v2-4/default.tsv --s3 -o <output_dir>/<cochlea>_SGN_v2-4.tsv --cell_type sgn --force -c 1
```
They have to be copied to the S3 bucket afterwards.

## Network validation

Afterwards, the accuracy can be calculated using

```bash
python ~/flamingo_tools/scripts/validation/synapses/run_evaluation.py --segmentation_name SGN_v2-1 -o ~/flamingo-tools/reproducibility/model_accuracy/
python ~/flamingo_tools/scripts/validation/synapses/run_evaluation.py --segmentation_name SGN_v2-2 -o ~/flamingo-tools/reproducibility/model_accuracy/
python ~/flamingo_tools/scripts/validation/synapses/run_evaluation.py --segmentation_name SGN_v2-3 -o ~/flamingo-tools/reproducibility/model_accuracy/
python ~/flamingo_tools/scripts/validation/synapses/run_evaluation.py --segmentation_name SGN_v2-4 -o ~/flamingo-tools/reproducibility/model_accuracy/
```
The accuracy values will be written into `reproducibility/model_accuracy/SGN_3D.json`.
From there they can be read by `plot_fig2.py`.
