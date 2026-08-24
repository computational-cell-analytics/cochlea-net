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
| `G_LR_000301_L` | 4,298 | **no, 11 bins at 90-437 um** | ~4.4 | **run, and the data is the limit** |

`G_LR_000302_R` was the reason this folder exists. Its detection is not merely noisy, it is
truncated: 583 of its 930 counted IHCs got zero synapses, and in a contiguous 13-bin stretch of
the helix the nearest detection of any kind is 105 to 682 micrometer away, meaning those blocks
were never predicted. The v5 model on the same IHC_v11 mask covers the same components at 0.80
to 0.97, so the mask is fine and it is the v3 pass that did not finish. Its numbers are already
published -- the `synapse_v3_ihc_v11` table on S3, the `syn_per_IHC` column of its S3 IHC table,
and `ihc_counts_v11/ihc_count_G_LR_000302_R.tsv` -- and all three have to be replaced.

`G_LR_000301_L` has now been run, and its result is partial for a reason that no amount of
re-running will fix: **the CTBP2 channel has no punctate signal over about a third of the
helix.** The prediction array finished (all five receipts present) and the mask covers 967 of
its 973 IHCs, including 290 of the 295 that ended up with no detection nearby -- so those
blocks were predicted and the model correctly found nothing. The raw data says the same thing:

| region | mean | p99.9 | max |
|---|---|---|---|
| around IHCs with detections | 133.9 | 307.8 | 531 |
| around IHCs without | 144.0 | 240.8 | 303 |

Slightly *higher* background, no puncta -- a maximum 1.6x the mean where ribbons would be 4x.
Its `mean_std.json` shows the same at whole-volume scale: std 28.9, against 210.5 for
`G_LR_000302_R`. So 4,298 detections and roughly 4.4 per IHC is what this staining supports,
against 13 to 16 for the two good cochleae. **This is a data quality question for whoever
acquired it, not a processing one**, and the cochlea may not be usable for the syn/IHC panel.

`G_LR_000301_L` is also not on S3 at all. The detection itself can run
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
forward pass. Walking the rest to find them is free within the measurement: the benchmark's
72.2 s is fully accounted for by its 26 in-mask blocks at 2.78 s each.

So the array is **five tasks, not ten**. Ten put 102 to 139 in-mask blocks in each, about five
minutes of work, which is mostly slurm overhead. Five gives 227 to 269 blocks, 11 to
12.5 minutes -- still short enough that a preemption costs one requeue of that, which is why
the walltime is 1 h rather than the four I first guessed.

## Which GPU, and the preemptible queue

`submit_all.sh --preemptible` moves the prediction to `grete:preemptible`, which is usually
free while `grete:shared` is not. The slice comes from `PREEMPTIBLE_SLICE` in `common.sh` and
`--slice` overrides it.

**`1g.10gb` does not work.** This was measured, not estimated, and it is the one result worth
remembering:

| slice | visible | peak device use | s/block | outcome |
|---|---|---|---|---|
| `1g.10gb` | 9.5 GiB | - | - | **CUDA OOM inside the first in-mask block** |
| `1g.20gb` | 19.5 GiB | 18.10 GiB, 1.40 spare | 2.78 | **the default** |
| `3g.40gb` | 39.5 GiB | 18.10 GiB expected | 2.78 or better | more room, but too scarce to wait for |

`1g.10gb` and `1g.20gb` are both one compute slice of an A100 and differ only in framebuffer,
so this is not a speed trade-off -- it is purely whether the forward pass fits. It does not fit
in 10 GiB: the block is (64, 256, 256) with a (16, 64, 64) halo, so the padded input is
96 x 384 x 384 = 14.2 M voxels, three and a half times the 160^3 of the SGN model that the
existing `1g.10gb` recommendation in `scripts_sgn_variance/` came from. Do not carry that
recommendation over; `submit_all.sh` rejects `1g.10gb` outright.

The requirement is **18.10 GiB of device occupancy**: 17.55 GiB of tensors
(`torch.cuda.max_memory_allocated`) plus about 0.55 GiB of CUDA context. Read it off
`cuda.mem_get_info`, which the benchmark samples on a thread, rather than off the per-process
torch counters.

**Requesting fewer cores does not buy headroom.** The OOM on `1g.10gb` listed nine processes
holding ~188 MiB each, which looked like one CUDA context per prefetch worker and therefore
like something a smaller `-c` would reclaim. It is not: occupancy is flat at 18.10 GiB whether
the job asks for 8, 4 or 2 cores. All that changes is the prefetch throughput, so keep `-c 8`:

| cores | s/block | peak device use |
|---|---|---|
| 8 | 2.78 | 18.10 GiB |
| 4 | 3.00 | 18.10 GiB |
| 2 | 5.76 | 18.10 GiB |

Nodes are interchangeable, for what it is worth: `ggpu137` and `ggpu159` both give 2.78 s per
block at `-c 8`. Which is what makes the next result trustworthy.

### expandable_segments: 1.6x faster, 1.3 GiB more occupancy (not used)

`PYTORCH_ALLOC_CONF=expandable_segments:True` is a real speedup and a real cost, measured on
one node so the comparison is clean:

| allocator | s/block | peak device use on `1g.20gb` |
|---|---|---|
| default | 2.78 | 18.10 GiB, 1.40 spare |
| `expandable_segments:True` | **1.76** | **19.42 GiB, 0.08 spare (100 %)** |

1.76 s per block reproduced exactly across two runs, so the 1.6x is not noise. But it takes a
`1g.20gb` slice to 80 MiB of spare framebuffer, which is no margin at all -- **do not use it
there**. On slices with room it is free speed.

Note what this says about `max_memory_reserved`, which rose to 19.30 GiB in the same runs. With
the default allocator, reserved (17.98) overstates occupancy (18.10) by nothing much and is
mostly caching. With expandable segments, reserved is essentially all real occupancy
(19.30 against 19.42). So reserved is neither a requirement nor safe to ignore; measure
occupancy.

The knob is `SYN_ALLOC_CONF` in `common.sh`, passed to the apply job as `PYTORCH_ALLOC_CONF`
when non-empty. It is **deliberately empty**: the default slice is `1g.20gb`, and this is
exactly the slice it must not be used on. Trading 1.32 of 1.40 GiB of margin for four minutes
per task is not worth it, and running a preemptible job at 100 % of framebuffer invites an OOM
that looks like a preemption. Only set it if the slice changes to something with real room:

```bash
# in common.sh, only with --slice 3g.40gb or a whole A100
SYN_ALLOC_CONF=expandable_segments:True
```

**`1g.20gb` is the default, not `3g.40gb`.** Its 1.40 GiB of spare framebuffer is enough --
the working set is a fixed 18.10 GiB, it does not grow with the volume -- and the extra room on
a `3g.40gb` slice buys nothing that the prediction needs. What it costs is availability: there
are only eight `3g.40gb` slices (`ggpu158`, `ggpu192`), they are the scarcest thing on the
partition, and a request for one sat at the top of the queue with an estimated start ten hours
out while `1g.20gb` jobs were starting within the minute. Waiting ten hours to make a
twelve-minute task marginally faster is the wrong trade. `--slice 3g.40gb` is there if they
ever free up.

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

It reads the prediction array's receipts before concluding anything, because a large distance
has two causes it cannot otherwise separate: blocks that were never predicted, and blocks that
were predicted and hold no ribbons. `G_LR_000301_L` is the case that forced this -- a third of
its helix uncovered with a complete set of receipts, which is the image and not the pipeline.
Without the receipt check the script told me to delete a perfectly good prediction and re-run.

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
