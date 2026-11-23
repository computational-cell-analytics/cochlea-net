# Input mask of the U-Net prediction

The input mask decides which blocks of a volume the SGN and IHC U-Nets predict. A skipped block
predicts 0, so the mask must keep every block with cells. This page summarizes the analysis
behind the mask rule of 2026-11 and the halo fix of the same change.

The full record, the scripts and the key outputs are at the tag
[`input-mask-analysis`](https://github.com/computational-cell-analytics/cochlea-net/tree/input-mask-analysis/scripts/mask_analysis)
(commit `3819750`). To read the record or rerun the analysis, check out the tag in a separate
worktree:

```bash
git worktree add ../input-mask-analysis input-mask-analysis
```

## The rules

`find_mask` in `flamingo_tools/segmentation/unet_prediction.py` writes `mask.zarr` and
`mask_statistics.json` (the rule, the background level, the threshold and the kept fraction).

**Default rule** (`compute_mask_blocks`, for `seg_class` 'sgn' and 'ihc'). The rule reads the
input at 8x lower resolution: the pyramid level `s3` for the input `s0`, or 8^3 mean pooling of
the input if that level does not exist or does not average the input. b is the median of the
non-zero voxels, which is the background level of a whole cochlea. A 128^3 block is kept with
these rules (`MASK_RULES`):

| class | rule | datasets >= 99.5% complete | kept blocks, median | GPU blocks vs the old rule |
|---|---|---|---|---|
| SGN | block p95 > 1.08 x b | 114 of 114 | 45% | 0.46 |
| IHC | block p99 > 1.75 x b, plus all 26 neighbours | 75 of 75 | 11.9% | 2.05 |

"Complete" means that the U-Net predicts every block that the bounding box of a relevant object
touches. The rules were fitted on 100 SGN and IHC segmentations from the S3 bucket and validated
on 89 others, split by animal.

**Legacy rule** (the rule before 2026-11). It applies with `legacy_mask=True` (`--legacy_mask`),
and for every other `seg_class`, which includes the low-resolution LaVision models. A block of 2 x
the input chunks is kept if its p95 (SGN) or p99 (IHC) at full resolution is above
`absolute_threshold`, or above an adaptive threshold. An `absolute_threshold` without the legacy
rule raises a `ValueError`.

## Findings

- The old IHC rule (p99 > 150, no dilation) cut IHCs at block faces. Only a median of 91.1% of
  the relevant IHCs per dataset were complete (worst dataset 76.0%). Dilation by all 26
  neighbours fixes this.
- The old SGN rule (p95 > 100) kept 95% of the blocks, because the background level b is 100-137
  in all datasets. The new SGN rule keeps less than half of these blocks.
- A threshold relative to b generalizes better across protocols than an absolute threshold. The
  dimmest protocols set the factors.
- The s3 level gives the same result as the full-resolution data and reads 1/512 of it.
- Some old references were probably made with masks that were too strict:
  - IHC references with p95 > 200, p99 > 175 or p99 > 300 probably miss IHCs.
  - In 21 SGN references, the old rule skipped at least 10 more blocks with neuron-like signal
    next to the segmented neurons than the background rate explains (for example
    M_AMD_000088_L, M_AMD_000115_L, M_AMD_C202_R). This is a proxy, not a count of neurons.
  - For 7 SGN references, the mask rule in the job archive cannot have made the reference.

## Halos

Many cells were split at the faces of the 128^3 prediction blocks, also next to predicted
blocks. Two halos caused this:
- The prediction halo had only 16 voxels in z, so the center distances jump at the z faces.
  The default of `_get_device_and_tiling` is now (32, 32, 32). This costs about 20% more GPU
  time. It applies to every prediction without an explicit halo.
- The block-wise watershed used the halo (2, 8, 8), so a block did not see the seed of a cell
  that crosses its face. `distance_watershed_implementation` now uses `halo=(16, 32, 32)`.

On two real crops, the face splits went from 9 to 2 (SGN) and from 13 to 3 (IHC). The control
planes inside the blocks have 0 and 3.

## Limits and open items

- The analysis used whole cochleae only. In a small crop that cells fill for the most part, b is
  not the background level. Use `--disable_masking` for such a crop.
- A new, dimmer protocol can need a lower factor (`--background_factor`). Check the threshold in
  `mask_statistics.json`.
- Change `MASK_RULES` only after a rerun of the evaluation from the tag. Every validation dataset
  must keep 99.5% of its relevant objects complete.
- Open: a cluster run with the new defaults on M_AMD_000126_R (IHC) and M_AMD_000115_L (SGN). It
  confirms the halo fix on a full cochlea (`audit_block_cuts.py` at the tag) and shows whether
  the old SGN reference misses neurons.
