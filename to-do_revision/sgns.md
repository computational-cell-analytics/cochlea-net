# SGNs

Estimating the variability of the `SGN_v2` model by applying four variants that differ only in the
training seed. The whole experiment now lives in the workspace and does not touch MoBIE or S3; the
scripts and the detailed instructions are in `scripts_sgn_variance/` (read its README first).

## Network training

No network training required, the variations already exist in
`/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN` as
`v2-1_cochlea_distance_unet_SGN_supervised` … `v2-4_cochlea_distance_unet_SGN_supervised`.
They have been renamed recently for an easier application for the full volume validation cochleae.
Their `split.json` is byte-identical, so the training data and the split are the same and only the
seed differs.

## Network application

The `SGN_v2` network variations need to be applied on the cochleae used for validation. Those are
`M_AMD_000058_L`, `M_LR_000169_R`, `M_LR_000226_L`, `M_LR_000227_L`, and `M_LR_000227_R`.
The processing consists of three steps:
1) Calculating the mean and standard deviation together with an intensity mask.
2) Application of the network to get maps for foreground, boundary distance, and center distance.
3) Seeded watershed based on the maps.

Step 1 was performed on vast for all cochleae with an absolute intensity threshold of 200, once per
cochlea, and copied into the version folders. The staging job copies those results into the
workspace and checks that the mask still holds the expected number of in-mask blocks.

Step 2 had been performed for `M_LR_000169_R` only. Those four predictions are converted into the
new sharded layout rather than recomputed; the other 16 are predicted from scratch.

Step 3 had not been performed for any of the cochleae.

Everything is written to
`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/SGN-v2-variance/<cochlea>/SGN_v2-<n>/`.
`predictions.zarr` is a sharded zarr v3 array: one file holds 4 x 4 x 4 prediction blocks and all
three channels, which takes the experiment from ~740k files down to at most ~12k. That requires each
slurm array task to own whole shards, because a shard is a single file and a partial shard write
loses the concurrent writer's chunks — see the README for why the predictions are nevertheless
identical to a whole-volume run, and for the self-test that proves it.

## Post-processing

No MoBIE, no S3 and no multiscale pyramid: this is a validation experiment and the only consumer of
the segmentation is the accuracy number. `scripts_gerbil/create_table_and_components.py` computes
the segmentation table and the connected components locally, writing `default.tsv` and
`default_components.tsv` next to `segmentation.zarr`.

The predictions are deleted once the tables exist (`cleanup_predictions.sh`); they are 190-330 GB
each and nothing downstream needs them.

## Network validation

The accuracy is computed by the SGN evaluation script, reading the segmentation and the component
labels from the workspace via `--local_root`:

```bash
python ~/Work/my_projects/flamingo-tools/scripts/validation/SGNs/run_evaluation.py \
	--segmentation_name SGN_v2-1 \
	--local_root /mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/SGN-v2-variance \
	--cache_folder <workspace>/eval_cache/SGN_v2-1 \
	-o ~/Work/my_projects/flamingo-tools/reproducibility/model_accuracy/
```

`2026-08-23_sbatch_evaluate_SGN-v2-variance.sbatch` runs this for all four variants. Two things it
takes care of that are easy to get wrong by hand: the four runs must be sequential, because
`update_json` rewrites the whole file, and each version needs its own `--cache_folder`, because the
cache file names do not contain the segmentation name.

The accuracy values are written into `reproducibility/model_accuracy/SGN_3D.json` as `v2-1` … `v2-4`
next to the existing `v2` reference (precision 0.887 / recall 0.880 / F1 0.884). `check_results.py`
sanity-checks them. Reading them into a figure is still to be done: `plot_fig2.py` currently reads
only the `v2` key.
