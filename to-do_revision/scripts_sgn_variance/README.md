# SGN v2 seed variance

Applies the four SGN v2 seed variants (`v2-1` … `v2-4`, identical training data and split, different
seed) to the five F1-validation cochleae and measures the spread in accuracy caused by the seed.

Everything this experiment writes lives in the workspace
`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/SGN-v2-variance`. The raw data, the
models and the predictions of the earlier runs stay read-only on vast. Nothing goes through MoBIE or
S3: the segmentation table is computed locally with `../scripts_gerbil/create_table_and_components.py`
and the evaluation reads the segmentation straight from the workspace.

The one exception is `reproducibility/model_accuracy/SGN_3D.json`, which stays in the repository:
it is the durable, committed product, and the workspace expires.

## Layout

```
$WS/<cochlea>/mask.zarr             copied from vast, one per cochlea rather than per version
$WS/<cochlea>/mean_std.json         copied from vast
$WS/<cochlea>/shard_manifest.json   in-mask blocks per shard, and the shard -> task assignment
$WS/<cochlea>/SGN_v2-<n>/predictions.zarr     sharded, deleted once the tables exist
$WS/<cochlea>/SGN_v2-<n>/seeds.zarr           watershed intermediate, deleted with the prediction
$WS/<cochlea>/SGN_v2-<n>/segmentation.zarr    key "segmentation"
$WS/<cochlea>/SGN_v2-<n>/default.tsv
$WS/<cochlea>/SGN_v2-<n>/default_components.tsv
$WS/eval_cache/SGN_v2-<n>/                    one cache folder per version, see below
```

## Why the prediction is sharded, and why one task owns whole shards

`predictions.zarr` is a zarr v3 array with inner chunks `(1, 128, 128, 128)` and shards
`(3, 512, 512, 512)`: one file holds 4 x 4 x 4 prediction blocks and all three output channels.
Unsharded, this experiment would be ~740k files (measured: 152 files per GB); sharded it is at most
11.8k. The workspace shares a project inode quota, which is the actual constraint.

A shard is one file, and zarr writes a *partial* shard by loading it, merging, and rewriting it. Two
writers that both load before either renames silently discard each other's chunks, and
`torch_em`'s in-process guard cannot see the ten slurm array tasks. So a shard may never be filled
by more than one writer. `sgn_variance.py predict` therefore gives each array task whole shards
(from the manifest), predicts the blocks of one shard into memory, and writes the shard in a single
complete write.

Two consequences worth remembering:

- **The predictions are unchanged.** `predict_with_halo_pipelined` reads every block's halo from the
  global input regardless of `roi`, and the shard grid is aligned with the block grid, so the block
  bounding boxes are exactly those of a whole-volume run. `sgn_variance.py selftest` proves this by
  predicting one shard both ways and comparing.
- **Counting shard files is a sufficient completeness check.** A shard is written once, atomically
  (zarr writes a temp file and renames), so a shard file that exists is a finished shard. The check
  is `number of shard files == the manifest's n_shards_in_mask`.

The watershed outputs (`seeds.zarr`, `segmentation.zarr`) are deliberately **not** sharded: they go
through `elf.parallel`, which has no shard-exclusive write routing, and `size_filter` reads and
writes `segmentation.zarr` in place from 16 threads.

## Order

Run one cochlea at a time, smallest first. Four predictions of the largest cochlea are 1.3 TB.

```bash
cd ~/Work/my_projects/flamingo-tools/to-do_revision/scripts_sgn_variance

# 0. once, before anything. The workspace expires 2026-09-20 and has 2 extensions left.
ws_extend flamingo-tools 30
df -h /mnt/lustre-rzg/workspaces/ws/nim00007

# 1. stage the inputs and create the empty sharded arrays for all 5 cochleae.
#    Fails if a copied mask does not reproduce the in-mask block count of the reference runs.
sbatch 2026-08-23_sbatch_stage_SGN-v2-variance.sbatch

# 2. prove that per-shard prediction reproduces a whole-volume run. Run this before step 3.
sbatch 2026-08-23_sbatch_selftest_SGN-v2-variance.sbatch M_LR_000226_L 1

# 3. per cochlea: predict (or convert), watershed, table. Submits the chain with dependencies.
#    --preemptible puts the prediction on a MIG slice, which is usually schedulable at once.
bash submit_all.sh --preemptible M_LR_000226_L
#    ... then, once the table jobs are done:
sacct -j <watershed job id> --format=JobID,State,Elapsed,ReqMem,MaxRSS
bash cleanup_predictions.sh M_LR_000226_L            # dry run, review
bash cleanup_predictions.sh --delete M_LR_000226_L
df -h /mnt/lustre-rzg/workspaces/ws/nim00007
#    ... repeat, in this order:
#    M_LR_000226_L (9627 blocks) -> M_AMD_000058_L (11380) -> M_LR_000227_L (12064)
#    -> M_LR_000227_R (12309) -> M_LR_000169_R (16327, converted rather than predicted)

# 4. evaluate all four variants, sequentially, into reproducibility/model_accuracy/SGN_3D.json
sbatch 2026-08-23_sbatch_evaluate_SGN-v2-variance.sbatch
git -C ~/Work/my_projects/flamingo-tools diff reproducibility/model_accuracy/SGN_3D.json

# 5. last, and only after reviewing the results: remove the old outputs on vast.
bash remove_vast_outputs.sh
bash remove_vast_outputs.sh --delete
```

`M_LR_000169_R` is the one cochlea whose predictions already exist. `submit_all.sh` converts them
into the sharded layout with `bioimage_py.copy` instead of recomputing them, and verifies whole
shards against the originals before anything may be deleted.

To reclaim the 1.3 TB those originals occupy without waiting for the whole experiment, use
`remove_converted_source.sh` (dry run by default). It removes only `predictions.zarr` on vast, keeps
`mask.zarr` and `mean_std.json`, and refuses unless every expected shard is present and a sample of
whole shards is bit-for-bit equal to the original. `--n_samples` raises the sample size; each sample
reads about 1.6 GB from either side.

**Order matters.** `remove_converted_source.sh` compares the converted array against the original,
so it has to run *before* `cleanup_predictions.sh`, which deletes the converted array. Once that
array is gone the comparison is impossible for good, and the only way through is `--accept_derived`,
which accepts `segmentation.zarr` and `default_components.tsv` as evidence instead: the watershed
reads every voxel of every shard to produce them, so their existence shows the converted array was
complete and readable. Sound, but a different argument, so it is opt-in rather than a silent
fallback.

The safe sequence for the converted cochlea:

```bash
bash submit_all.sh M_LR_000169_R                       # watershed + table
# check the component-1 count against the reference, then:
bash remove_converted_source.sh --delete               # vast, needs the converted array
bash cleanup_predictions.sh --delete M_LR_000169_R     # workspace, removes it
```

## Where to run the prediction

`grete:shared` is often fully allocated -- every A100 busy and most nodes draining -- and a
prediction submitted there can sit for many hours. `grete:preemptible` has no whole A100s, only MIG
slices, but the model is small enough that this barely matters: measured on `M_LR_000226_L`,
**0.62 s per block on a `1g.20gb` slice and 0.72 s on a `1g.10gb` one**, i.e. 10 to 15 minutes for a
task's ~1000 blocks. Peak GPU memory is under 10 GB, so the smallest slice is enough, and those are
the plentiful ones.

Preemption is cheap for this stage by construction: a shard is written in a single atomic write and
`--skip_existing` resumes at the next unwritten shard, so a kill costs at most one shard, about 20 s.
`submit_all.sh --preemptible` sets `--partition=grete:preemptible --gpus=1g.10gb:1` on the prediction
only; the watershed and the table are CPU jobs and stay where they are.

## Two constraints that are easy to get wrong

- **The evaluation must not be an array job.** `json_util.update_json` reads `SGN_3D.json`, updates
  the top-level keys and rewrites the whole file, so four concurrent writers would keep one result.
  The evaluate script loops the versions sequentially for this reason.
- **One evaluation cache folder per version.** The cache file name is `<cochlea>_<slice>.tif` with no
  version in it, and a cache hit short-circuits before the segmentation is read, so a shared folder
  would score the first version four times. `check_results.py` fails if all four F1 scores are
  identical, which is what that mistake looks like.

## Sizing

Keep the walltime requests tight. `standard96s:shared` is regularly saturated, and a job that asks
for 12 h cannot be backfilled into the gaps a 2-3 h job fits into, so an over-generous limit costs
real queue time on every submission.

| stage | limit | basis |
|---|---|---|
| stage | 1 h | measured 1:25 and 2:23 |
| convert | 4 h | measured 1:49 to 1:56 per version, two at a time |
| apply | 2 h | measured 0.62 s/block on a 1g.20gb slice, 0.72 s on 1g.10gb -> 10-15 min per task |
| watershed | 3 h | gerbil 53 min at 676 GB; these are 192-326 GB, four tasks contend for lustre |
| table | 1.5 h | gerbil 13 min at 2.5 GB MaxRSS over a larger volume |
| evaluate | 2 h | not measured; 48 single-plane slice evaluations |
| selftest | 30 min | measured 1:32 and 1:54 |

Memory: the four watersheds of `M_LR_000169_R` peaked at **161, 229, 254 and 229 GiB**, so the
request is 300G. Two things that follow. The peak varies by 90 GiB between versions of the *same*
volume, so it is not predictable from a single run. And it does not track the in-mask block count --
this cochlea has 16,327 blocks and peaked higher than the gerbil's 189 GB at 34,248 -- so a smaller
cochlea is not automatically cheaper. Measure per cochlea with `sacct -o MaxRSS`. Apply needs 64 GB
and the table job 2.5 GB, both from the gerbil run.

Per-cochlea in-mask 128³ blocks, shard grid, and the size of one prediction:

| cochlea | blocks | shard grid | shards | one prediction |
|---|---|---|---|---|
| M_AMD_000058_L | 11,380 | 8x8x6 | 384 | ~227 GB |
| M_LR_000169_R | 16,327 | 10x8x12 | 960 | 326 GB |
| M_LR_000226_L | 9,627 | 8x8x6 | 384 | ~192 GB |
| M_LR_000227_L | 12,064 | 8x8x9 | 576 | ~241 GB |
| M_LR_000227_R | 12,309 | 9x8x9 | 648 | ~246 GB |

## Verification

- `sgn_variance.py selftest` — per-shard prediction is identical to a whole-volume run.
- Staging fails unless the mask reproduces the block count in the table above.
- The watershed refuses to start unless the shard count matches the manifest.
- `cleanup_predictions.sh` refuses to delete unless component 1 holds at least 1000 objects and at
  least 25% of them. Gate on the size of component 1, not on its share: see below.
- **The F1 score is computed only over the objects inside the component.** Not over all
  segmented objects. `run_evaluation.py` resolves the component list per cochlea and version and
  passes it to `fetch_data_for_evaluation`, which zeroes every voxel whose label is not in that
  component before the matching runs, then relabels what is left. For this experiment the list is
  `[1]` for all five cochleae and all four versions, and the lookup defaults to `[1]` when an entry
  is missing, so the filter can never be skipped by accident. Objects outside component 1 are
  therefore neither true positives, false positives, nor false negatives -- they are not scored at
  all. Verified on the three `M_LR_000169_R` slices for `SGN_v2-1`: filtering to component 1 drops
  5 of 679 objects on those planes and moves F1 from 0.864 to 0.867, all of it through `fp`
  (111 against 116); `tp` and `fn` are identical either way.
- After the watershed, compare the size of **component 1**, not the total object count. On
  `M_LR_000169_R` the four seeds gave 11,186 / 11,383 / 11,235 / 11,265 objects in component 1
  against a reference `SGN_v2` of 10,973 -- within 2 to 4%, and agreeing with each other to 1.7%.
  Their totals, in contrast, ran from +9% to +42% of the reference, and the share in component 1
  from 72% to 92% against the reference's 98%. The seeds find the same helix and differ in how many
  strays they produce away from it, which is exactly what the component step discards -- and, per
  the point above, those strays are never scored. They are also spread through the whole 3D volume,
  so only a handful of them intersect any given annotated plane; that is why a 42% difference in the
  total object count moves the F1 of a slice by 0.003. Reference component-1 counts have to be read
  off the S3 tables per cochlea; the local MoBIE copies carry no `component_labels` column.
- `check_results.py` checks the accuracy: 12 crops per variant, `tp + fn` per crop identical to the
  reference (it is the annotation count, so it must match for any segmentation), and F1 within 0.03
  of the published 0.884. The completed IHC seed-variance experiment spread over about 0.01 F1.

## Files

| file | purpose |
|---|---|
| `sgn_variance.py` | Geometry, manifest, array creation, per-shard prediction, conversion, verification, self-test. One module so those cannot drift apart. |
| `common.sh` | Paths, cochlea and version lists, expected block counts. Sourced by every sbatch. |
| `check_results.py` | Sanity-checks the four accuracy entries in `SGN_3D.json`. |
| `submit_all.sh` | Submits the per-cochlea chain with dependencies. `--preemptible` for a MIG slice, `--dry_run` to print instead of submit. |
| `cleanup_predictions.sh` | Deletes `predictions.zarr` in the workspace once the tables are complete and plausible. |
| `remove_converted_source.sh` | Deletes the unsharded `predictions.zarr` on vast for a converted cochlea, after verifying the conversion. Run before `cleanup_predictions.sh`. |
| `check_derived_products.py` | Checks a segmentation and its component table, for `remove_converted_source.sh --accept_derived`. |
| `remove_vast_outputs.sh` | Removes the old vast output folders entirely, behind four gates. |
