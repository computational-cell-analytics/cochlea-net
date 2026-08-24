# Ribbon synapse detection for the wild-type gerbils

Synapse detection (`synapse_detection_model_v3.pt`) for the four wild-type gerbil cochleae of
figure 5. Two of them were already finished; this folder covers the two that were not.

**No prediction has been run yet.** The inputs are staged and the chain is validated with
`--dry_run`, but no `predictions.zarr` exists for either cochlea.

One exception to be aware of: `synapses-v3/G_LR_000301_L/` already holds a `mask.zarr` and a
`mean_std.json` from a preprocessing job that ran on 2026-08-24 (4.5 min, job 15469008) before
the chain was cancelled. They are valid and 220 KB in total, so `submit_all.sh G_LR_000301_L`
will report `preprocess: already done, skipping` and go straight to the prediction. Delete that
folder if you want the chain to start from scratch.

Everything is written to
`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/synapses-v3/<cochlea>/`.
Nothing is written into the existing `predictions/<cochlea>/synapses_v3` folders on vast, so
the current G_LR_000302_R result stays in place until the new one has been compared against it.

## State of the four cochleae

| cochlea | detections | helix covered | syn/IHC | state |
|---|---|---|---|---|
| `G_EK_000233_L` | 395,523 | yes, all 20 bins at 6-9 um | 16.4 | finished |
| `G_LR_000301_R` | 18,149 | yes, all bins at 5-20 um | 13.1 | finished |
| `G_LR_000302_R` | 7,230 | **no, 13 bins at 105-682 um** | 5.06 | re-run |
| `G_LR_000301_L` | - | - | - | never run |

`G_LR_000302_R` was the reason this folder exists. Its detection is not merely noisy, it is
truncated: 583 of its 930 counted IHCs got zero synapses, and in a contiguous 13-bin stretch of
the helix the nearest detection of any kind is 105 to 682 micrometer away, meaning those blocks
were never predicted. The v5 model on the same IHC_v11 mask covers the same components at 0.80
to 0.97, so the mask is fine and it is the v3 pass that did not finish. Its numbers are already
published -- the `synapse_v3_ihc_v11` table on S3, the `syn_per_IHC` column of its S3 IHC table,
and `ihc_counts_v11/ihc_count_G_LR_000302_R.tsv` -- and all three have to be replaced.

`G_LR_000301_L` has no synapse detection and is not on S3 at all. The detection itself can run
entirely from the workspace, see below. Its *downstream* count cannot yet: `measure_synapses.py`
reads the tables through S3 and needs `length[um]` and `frequency[kHz]`, and this cochlea has no
tonotopic mapping and no decided IHC component list (its largest IHC_v11 component holds 252 of
973 IHCs). It is also still missing from `wt_gerbil` in `scripts/figures/util.py`, which lists
three cochleae, and has no `IHC` entry in `VALUE_DICT`.

## Inputs

Both CTBP2 channels live in the workspace, so the chains can be submitted as they are.
`G_LR_000301_L` was transferred from the UKON archive as a fused n5. `G_LR_000302_R` had to be
copied back from S3 because its raw data is no longer on vast -- `predictions/G_LR_000302_R/`
still exists but the data folder `cochlea-lightsheet/G_LR_000302_R/` is empty. That copy is
complete and verified against the remote at 680,267 objects and 170,080,345,045 bytes; if it
ever needs redoing, `rclone copy` skips what is already there:

```bash
WS=/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools
rclone copy cochlea-lightsheet:cochlea-lightsheet/G_LR_000302_R/images/ome-zarr/CTBP2.ome.zarr \
	"$WS/G_LR_000302_R/CTBP2.ome.zarr" --transfers 32 --checkers 32
```

Its IHC_v11 pyramid was only on S3 as well and was copied in too, which is cheap (52 MiB).
`G_LR_000301_L`'s mask is the IHC_v11 segmentation produced in this workspace by
`scripts_gerbil/`.

The mask that restricts the prediction is derived from the IHC_v11 segmentation at `s4`,
binarized and dilated by a 9^3 structure element, so about 55 micrometer around the IHCs. The
detections are matched against the same segmentation at `s0`.

## Running it

```bash
bash submit_all.sh --dry_run G_LR_000301_L     # check the chain first
bash submit_all.sh G_LR_000301_L
python check_coverage.py G_LR_000301_L         # when it is done
```

Three stages, chained with slurm dependencies:

1. `2026-08-24_sbatch_preprocess_syn-v3.sbatch` - build `mask.zarr` from the IHC segmentation
   and compute `mean_std.json` over the masked volume. One CPU job. It has to finish before the
   array starts: every task must normalize identically, and computing the statistics per task
   would read the volume ten times over.
2. `2026-08-24_sbatch_apply_syn-v3.sbatch` - the prediction, a GPU array of ten tasks each
   owning a tenth of the blocks of the shared `predictions.zarr`.
3. `2026-08-24_sbatch_detect_syn-v3.sbatch` - peak detection and matching to the IHCs, on CPU.
   Writes `synapse_detection.tsv` and `synapse_detection_filtered.tsv`.

`MAX_DISTANCE=8` for the matching, which is what the three finished cochleae used (their
`synapse_detection_filtered.tsv` tops out at 7.99 um). `measure_synapses.py` re-filters to 3 um
when it counts, so keeping 8 leaves that cutoff free to change without re-running anything.

## Do not resubmit the array on top of a partial prediction

This is the trap that let the truncated G_LR_000302_R result pass as finished, and it is worth
stating plainly because nothing in the output betrays it. What interrupted that run is not
recorded anywhere I can read, but the reason nobody noticed is structural.

`run_synapse_prediction_slurm` has no skip-existing check, deliberately: `predictions.zarr` is
created by whichever array task starts first, so skipping on its existence would leave every
other task's blocks empty. The consequence is that a resubmission after a timeout does not
resume -- and worse, the single-job entry point `marker_detection` skips the prediction outright
when `synapse_detection.tsv` already exists, so a rerun of that path goes straight to detecting
maxima in whatever the previous attempt left behind. Either way the result is a complete-looking
table covering part of the cochlea.

`submit_all.sh` therefore refuses to submit when `predictions.zarr` already exists. Pass
`--reset` to delete it and the tables derived from it, and start over.

## check_coverage.py

The guard that would have caught this. For every IHC it asks how far away the nearest detection
of any kind is -- a few micrometer around a predicted IHC, hundreds around one in an unpredicted
block -- and reports that per component and along the main axis of the largest component.

The verdict keys on the median distance, not on the fraction of IHCs with a detection nearby,
because those two measure different things. `G_LR_000301_R` has a stretch where only half the
IHCs carry a detection within 20 um, but the median there is 10 um: it was predicted and simply
has fewer ribbons. Only a median in the hundreds is a pipeline failure. A hole also has to span
at least two neighbouring bins, since an array task owns a contiguous range of blocks -- the
single 51 um bin of `G_LR_000301_R`, whose neighbours are at 12 and 9 um, is weak signal.

Validated against all three existing results: `G_EK_000233_L` and `G_LR_000301_R` pass,
`G_LR_000302_R` fails. It exits non-zero on failure, so it can gate the steps below.

## After the detection

Not done by these scripts, and in this order:

1. `check_coverage.py` on the new result, and `--compare_old` for G_LR_000302_R.
2. Transfer to MoBIE and S3 (`reproducibility/templates_transfer/`). G_LR_000301_L needs the
   whole cochlea uploaded, not only the synapses.
3. Decide the IHC component list for G_LR_000301_L and add it to `SYNAPSE_DICT` in
   `flamingo_tools/postprocessing/synapse_per_ihc_utils.py`; it needs a tonotopic mapping first.
4. `scripts/measurements/measure_synapses.py -c <cochlea> -o <ihc_counts_v11>`, which replaces
   `ihc_count_G_LR_000302_R.tsv`.
5. `scripts/synapse_marker_detection/add_synapse_per_ihc.py` and upload the IHC table, which
   replaces the `syn_per_IHC` column of G_LR_000302_R.
6. Add `G_LR_000301_L` to `wt_gerbil` in `scripts/figures/util.py` and its IHC count to
   `VALUE_DICT`, so figure 5 shows four wild types rather than three.
