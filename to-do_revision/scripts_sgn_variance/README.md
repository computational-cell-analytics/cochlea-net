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
bash submit_all.sh M_LR_000226_L
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

## Two constraints that are easy to get wrong

- **The evaluation must not be an array job.** `json_util.update_json` reads `SGN_3D.json`, updates
  the top-level keys and rewrites the whole file, so four concurrent writers would keep one result.
  The evaluate script loops the versions sequentially for this reason.
- **One evaluation cache folder per version.** The cache file name is `<cochlea>_<slice>.tif` with no
  version in it, and a cache hit short-circuits before the segmentation is read, so a shared folder
  would score the first version four times. `check_results.py` fails if all four F1 scores are
  identical, which is what that mistake looks like.

## Sizing

Measured on the gerbil G301L run (a 676 GB prediction, roughly twice these):

| stage | time | memory |
|---|---|---|
| mask + mean/std | 7 min | small |
| apply | ~12 min per array task | 64 GB |
| watershed | 53 min | 189 GB MaxRSS of 400 GB requested |
| table + components | 13 min | 2.5 GB MaxRSS |

The watershed here requests 256 GB: the gerbil figure was measured at 34,248 in-mask blocks and our
largest cochlea has 16,327, so 90-120 GB is expected. Check `sacct -o MaxRSS` after the first
cochlea and raise it to 400 GB if it comes close.

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
- `cleanup_predictions.sh` refuses to delete unless the components table has at least 1000 rows and
  at least 90% of the objects in component 1 (the gerbil reference is 99.4%).
- After the watershed, `default.tsv` should hold about as many objects as the reference `SGN_v2`
  segmentation of that cochlea: 10,599 / 11,170 / 11,330 / 10,416 / 13,868 in the order above.
  A wildly different count is a broken prediction, not seed variance.
- `check_results.py` checks the accuracy: 12 crops per variant, `tp + fn` per crop identical to the
  reference (it is the annotation count, so it must match for any segmentation), and F1 within 0.03
  of the published 0.884. The completed IHC seed-variance experiment spread over about 0.01 F1.

## Files

| file | purpose |
|---|---|
| `sgn_variance.py` | Geometry, manifest, array creation, per-shard prediction, conversion, verification, self-test. One module so those cannot drift apart. |
| `common.sh` | Paths, cochlea and version lists, expected block counts. Sourced by every sbatch. |
| `check_results.py` | Sanity-checks the four accuracy entries in `SGN_3D.json`. |
| `submit_all.sh` | Submits the per-cochlea chain with dependencies. |
| `cleanup_predictions.sh` | Deletes `predictions.zarr` once the tables are complete and plausible. |
| `remove_vast_outputs.sh` | Removes the old vast outputs, behind four gates. |
