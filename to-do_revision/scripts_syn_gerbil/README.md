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
bash submit_all.sh --dry_run G_LR_000301_L      # check the chain first
bash submit_all.sh --preemptible G_LR_000301_L  # or without, for a whole A100
python verify_prediction.py G_LR_000301_L       # after the array: did every task finish
python check_coverage.py G_LR_000301_L          # after the detection: is the helix covered
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

## How much work this actually is

Worth knowing before sizing anything, because the mask makes it much smaller than the volume
suggests. `G_LR_000301_L` is 4951 x 5762 x 6308 voxels, which is 44,850 prediction blocks of
(64, 256, 256) -- but only **1,238 of them (2.8 %) overlap the dilated IHC mask** and get a
forward pass. Split over ten tasks that is 102 to 139 blocks each, so a task is a few minutes
of GPU work and a few more walking the 4,485 blocks it was assigned to find them.

That is why the walltime requests are an hour rather than the four I first guessed, and it is
what makes the preemptible queue attractive: a preempted task loses minutes, not hours.

## Which GPU, and the preemptible queue

`submit_all.sh --preemptible` moves the prediction to `grete:preemptible`, which is usually
free while `grete:shared` is not. The slice comes from `PREEMPTIBLE_SLICE` in `common.sh` and
`--slice` overrides it.

**`1g.10gb` does not work.** This was measured, not estimated, and it is the one result worth
remembering:

| slice | visible | peak device use | s/block | outcome |
|---|---|---|---|---|
| `1g.10gb` | 9.5 GiB | - | - | **CUDA OOM inside the first in-mask block** |
| `1g.20gb` | 19.5 GiB | 17.6 GiB tensors + a context per worker | 1.8 - 2.8 | fits, under 2 GiB spare |
| `3g.40gb` | 39.5 GiB | same | same or better | fits comfortably, 3x the compute |

`1g.10gb` and `1g.20gb` are both one compute slice of an A100 and differ only in framebuffer,
so this is not a speed trade-off -- it is purely whether the forward pass fits. It does not fit
in 10 GiB: the block is (64, 256, 256) with a (16, 64, 64) halo, so the padded input is
96 x 384 x 384 = 14.2 M voxels, three and a half times the 160^3 of the SGN model that the
existing `1g.10gb` recommendation in `scripts_sgn_variance/` came from. Do not carry that
recommendation over; `submit_all.sh` rejects `1g.10gb` outright.

Two details behind the numbers:

* The 17.6 GiB is `torch.cuda.max_memory_allocated`, i.e. the main process's tensors. On top of
  it each prefetch worker holds its own CUDA context on the same slice -- the OOM listed nine
  processes at about 188 MiB each -- which is why `1g.20gb` was seen at 19.3 of 19.5 GiB. The
  benchmark now samples `cuda.mem_get_info` to report device-level occupancy directly.
* `PYTORCH_ALLOC_CONF=expandable_segments:True` did not help: allocated stayed at 17.55 GiB and
  reserved went *up*, to 19.30 GiB. Reserved is the caching allocator growing into whatever is
  free, not a requirement.

Availability cuts the other way and is worth checking before choosing. There are only eight
`3g.40gb` slices (`ggpu158`, `ggpu192`) and eight `1g.20gb` ones (`ggpu137`, `ggpu159`). Both
`1g.20gb` test jobs started within a minute, while a `3g.40gb` request sat at the top of the
queue with an estimated start ten hours out. If `3g.40gb` is not moving, `--slice 1g.20gb`
works -- it just has little room to spare, so pair it with fewer cores to cut worker contexts.

`benchmark_slice.py` is what produced all of this. It runs the real prediction path on a slice
of the real volume, writing to a scratch folder with the mask and normalization symlinked in so
it cannot touch the production output:

```bash
sbatch --gpus=1g.20gb:1 -J bench-20gb 2026-08-24_sbatch_benchmark_slice.sbatch
sbatch --gpus=3g.40gb:1 -J bench-40gb 2026-08-24_sbatch_benchmark_slice.sbatch
```

## Preemption safety

Preemption is safe here, but not by accident. Three things have to hold, and two of them are
properties of the library that a future change could break:

1. **One block is one chunk.** `prediction_impl` sets `output_chunks = block_shape`, and says
   so: "the chunks are aligned with block_shape, so each chunk has exactly one writer". Two
   tasks therefore never touch the same chunk file, and a killed task cannot corrupt another's
   work -- unlike the sharded SGN layout, where a partial shard write loses a co-writer's
   blocks.
2. **The block assignment is reproducible.** It is a permutation under seed 1234 split by
   `np.array_split`, so a requeued task redoes exactly its own share.
3. **`--requeue` is set on the apply job.** This is the part that is not automatic. A preempted
   task that is not requeued leaves its blocks empty, the detection step then writes a
   complete-looking table from a partial volume, and nothing in the output says so. That is
   precisely the G_LR_000302_R failure. There is no resume inside the library, so a requeued
   task starts its share over -- which costs minutes, per the section above.

The detection depends on `afterok` of the whole array, so it cannot start on a partial volume.
If a task is killed for good the detection sits in `DependencyNeverSatisfied`. That is the
intended outcome: run `verify_prediction.py`, resubmit the tasks it names, then submit the
detection by hand.

## verify_prediction.py

The mechanical check, to be run after the array: did every task finish? The authority is a
receipt, `tasks/task_<id>_of_<n>.json`, written by the apply job only after the prediction
returns. Missing receipts are exactly the tasks to resubmit, and the script prints the
`sbatch --array=...` line for them.

It deliberately does *not* decide this from the data. One block being one chunk makes missing
chunks look like they should pinpoint missing blocks, and they nearly do -- the obstacle is
knowing which blocks *should* have a chunk, which means replicating the skip rule in
`_prepare_block_input`. That rule tests the inner block of `ResizedVolume(mask, shape, order=0)`,
and a floor/ceil approximation of its nearest-neighbour rounding disagreed with the real thing
on 2 of 26 blocks of a test task, in both directions. The chunk count is reported for
information only and never gated on.

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
