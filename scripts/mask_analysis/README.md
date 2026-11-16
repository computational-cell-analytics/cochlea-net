# Input mask analysis

This is the full record of the analysis of the input mask of the U-Net prediction. The work was
done on the branch `mask_threshold_analysis` from 2026-11-12 to 2026-11-16. The tag
`input-mask-analysis` holds this record, the scripts and the key outputs. After the tag, master
keeps only the code of the new rule and the summary in `doc/input_mask.md`.

Data: 193 segmentations from S3 (114 SGN and 75 IHC with a reference, 4 failure cases). The
features were collected on 2026-11-13, and the exact s0 statistics of the HPC job were added on
2026-11-16. All numbers for the production rule and the new rules are exact s0 values.

## 1. Question

`find_mask` (`flamingo_tools/segmentation/unet_prediction.py`) decides which blocks the U-Net
predicts. A skipped block predicts 0, so the mask must keep every block with cells. The rule
before 2026-11 (now the legacy rule):
- mask blocks of 2 x the input chunks;
- a block is kept if its p95 (SGN) or p99 (IHC) at full resolution is above an absolute
  threshold, or above an adaptive threshold (`relative_threshold` x the median p99.9 of 16 random
  blocks, clipped to 100-200 for SGN and 250-400 for IHC);
- no dilation.

The production runs used fixed thresholds: SGN p95 > 100 and IHC p99 > 150 in 2026-11. The
older reference segmentations on S3 used other rules, for example p95 > 200 or p99 > 300.

IHCs were cut at block faces in some segmentations. The questions were:
- Q1: Is the current rule good enough?
- Q2: Do some datasets need lower thresholds?
- Q3: Does dilation fix the missing neighbour blocks?
- Q4: Is another method necessary?

The aim was a rule that generalizes across protocols and saves GPU time.

## 2. Decision and implementation

Decision (2026-11-16): Outcome B with the halo fix.

| class | rule | datasets >= 99.5% | complete, worst | kept, median | blocks vs production |
|---|---|---|---|---|---|
| SGN | block p95 at s3 > 1.08 x b | 114 of 114 | 99.6% | 45% | 0.46 |
| IHC | block p99 at s3 > 1.75 x b, plus all 26 neighbours | 75 of 75 | 99.8% | 11.9% | 2.05 |

b is the median of the non-zero voxels, which is the background level. s3 is the 8x downsampled
pyramid level. Section 4 (Q4) explains the columns.

Outcome A was not chosen. It keeps IHC p99 > 150 at s0 and adds D26 dilation. It costs 37% more
GPU time than the new IHC rule (2.81 vs 2.05 times the production blocks), and it saves nothing
for SGN.

**Implementation** in `flamingo_tools/segmentation/unet_prediction.py`:
- `MASK_RULES` holds the percentile, the background factor and the dilation of each class.
  `MASK_BLOCK = (128, 128, 128)` is the GPU prediction block, for every input format.
- `compute_mask_blocks` reads the pyramid level 3 steps below `input_key` (`s0` -> `s3`, also
  `setupX/timepoint0/s0`). A 128^3 block is 16^3 voxels there.
- `_matches_mean_pooling` compares one background block of that level with 8^3 mean pooling of
  s0. A subsampled pyramid, or an input without a pyramid, falls back to mean pooling of s0
  (`_mean_pool`). The fallback reads all of s0.
- `_fit_grid` gives the blocks of a last slab that the s3 level does not cover the decision of
  their neighbours.
- `find_mask` applies the default rule for `seg_class` 'sgn' and 'ihc'. The legacy rule
  (`_legacy_mask_blocks`) applies if `legacy_mask=True`, or for every other `seg_class` (None,
  'sgn_low', 'ihc_low').
  - `absolute_threshold` without the legacy rule raises a `ValueError`. Old Slurm parameters with
    a number therefore fail at the mask step and do not silently change the rule.
  - `background_factor` with the legacy rule raises a `ValueError`.
- Every run writes `mask_statistics.json`: the rule, b, the threshold, the low-res source and the
  kept fraction.
- `model_utils.get_default_segmentation_settings("IHC-lowres")` now uses `seg_class` 'ihc_low',
  so both LaVision models keep the legacy rule. This analysis has no LaVision data.
- The halo fix of section 5: `_get_device_and_tiling` defaults to the prediction halo
  (32, 32, 32), and `distance_watershed_implementation` has the argument `halo=(16, 32, 32)`.
  The smoothing branch of `distance_watershed_implementation` used the name `halo` for the
  Gaussian halo. With `distance_smoothing > 0` (IHC: 1.0), it silently replaced the watershed
  halo. It is now `smoothing_halo`.

**Command line:**
- `flamingo_tools.run_segmentation`: `--legacy_mask` and `--background_factor`.
- The mask step of the Slurm pipelines (`run_unet_prediction_preprocess_slurm`):
  `--legacy_mask` and `--background_factor`. The templates pass strings: `legacy_mask` is off
  for "", "0", "false" and "no", and `background_factor` "None" uses the class default.
- `run_unet_prediction_slurm`: `--seg_class`, for a run without `mask.zarr`.

**slurm-workflows:** the branch `mask_rule_default` sets `<group>_masking_threshold` to "None"
and `<group>_legacy_mask` to "false" by default. Merge flamingo-tools first. The new templates
pass `legacy_mask`, which the old flamingo-tools does not accept.

**Tests:** `TestFindMask`, `TestPreprocessSlurmArguments` and `TestWatershedHalo` in
`test/test_segmentation/test_unet_prediction.py`, and the CLI checks in `test/test_cli.py`.

## 3. Data and method

- Datasets: the SGN and IHC segmentations on S3 with component labels. The split is frozen, by
  animal, and stratified by class and protocol (`mask_datasets.json`, seed 20261113): SGN 62
  train / 52 val, IHC 38 train / 37 val.
  - 4 SGN datasets have fewer relevant objects than `MIN_RELEVANT` and are failure cases
    (M_AMD_000085_L, M_AMD_000085_R, M_AMD_000086_L, M_AMD_N159_L).
  - 43 sources are excluded with a reason in the same file: LaVision, rerun or resized datasets
    (12), segmentations without component labels (28), no IHC stain (2), and no `dataset.json`
    (1).
- Reference objects: the relevant components of each segmentation. The source is the
  processing JSON, else `COCHLEA_DICT`, else component 1.
- Objects without stain signal are not targets: their s3 maximum is below 1.2 x b. 182 objects
  are excluded (`evaluation/excluded_objects.csv`):
  - 130 IHC objects in 19 datasets. 120 of them are within one block of the volume border, so
    they are probably U-Net hallucinations in empty blocks.
  - 52 dim SGNs in 7 datasets (1-23 of 5,000-12,000 per dataset).
  - With a floor of 1.0 or 1.4, the proposed rules do not change (`evaluation_floor*/`).
- Objects that end on a block face (`audit/face_hits.csv`) get their box extended across the
  face, because the true object continues there.
- A rule is a block score `s_b` and a threshold `T`: a block is predicted if `s_b > T`. An object
  is complete if all blocks that its bounding box touches are predicted. "Kept" is the fraction
  of all 128^3 blocks that the U-Net predicts, so it is proportional to the GPU time.
- Fit: per training dataset, the critical threshold is the highest threshold that keeps 99.5% of
  the relevant objects complete. The rule uses the minimum over the training datasets. The fit
  does not use the validation datasets. Results for 99% and 99.9% are in
  `evaluation/summary_t0.99.csv` and `summary_t0.999.csv`.
- Calibration against the logged production runs: `evaluation/calibration_logged_runs.csv`.

## 4. Results

All statements use the 128^3 prediction block grid. b is 100-137 in all datasets, because most
voxels of a cochlea volume are background.

### Q1: Is the current rule good enough?

**IHC (p99 > 150, no dilation): no.** The rule cuts IHCs at block faces. It loses almost no IHC
completely (at most 0.4% per dataset).

| IHC rule (exact, s0), 75 datasets | complete, median | complete, worst | datasets >= 99.5% | kept, median |
|---|---|---|---|---|
| p99 > 150 (production) | 91.1% | 76.0% | 0 | 2.6% |
| p99 > 200 | 80.0% | 68.1% | 0 | 1.2% |
| p99 > 300 | 70.7% | 28.2% | 0 | 0.8% |

At p99 > 300, one dataset loses 27.6% of its IHCs completely.

Two independent results agree:
- The audit of the existing segmentations (`audit/`,
  [audit_cutoffs.png](../../mask_threshold_analysis/figures/audit_cutoffs.png)) finds that 10.1% of
  the relevant IHCs (median over datasets) end on a block face next to a block without objects.
  For SGN, this is 1.2%.
- The block scores above give the same order of magnitude.

**SGN (p95 > 100): complete, but it is no mask.** The rule keeps a median of 95.1% of the blocks,
and 99.5% in the logged G_LR_000301_L run. Because b is 100-137, almost every block has p95 > 100.

**Older references with higher thresholds.** The missed-signal proxy
(`evaluation/missed_signal_proxy.csv`) counts excluded blocks with object-level signal next to
the relevant objects:
- IHC: the proxy is high for the references made with p95 > 200 (G_EK_000032_L: 1,238 blocks),
  p99 > 175 (G_LR_000301_R: 165) and p99 > 300 (G_LR_000302_R: 117, M_AMD_OTOF28_L: 86). These
  references probably miss IHCs. For the IHC references made with p99 > 150, the median is 8
  blocks.
- SGN: some references made with p95 > 200 probably miss neurons at the edge of Rosenthal's
  canal. Section 6 gives the details.

### Q2: Do some datasets need lower thresholds?

Yes, if the threshold is absolute. The highest safe threshold per dataset (the critical
threshold, [critical_thresholds.png](../../mask_threshold_analysis/figures/critical_thresholds.png))
depends on the protocol:

| class, score | critical threshold / b: min, median, max | dimmest protocols |
|---|---|---|
| SGN, block p95 | 1.09, 1.47, 2.5 | sgn_subtypes, PELCOfHC2longnoDCM, PELCOfDISCO, PELCOfHC2, undefined_mouse |
| IHC, block p99 + D26 | 1.77, 6.4, 12.7 | wt_gerbil, otof_mouse_pilot, PELCODISCO2, undefined_gerbil |

- A threshold relative to b spreads less than an absolute threshold (SGN p90/p10 1.67 instead
  of 1.98). Normalizations by the noise width or by high quantiles (`noise`, `contrast*`) do not
  generalize better.
- One rule per class, fitted on the training datasets, passes all validation datasets. No
  validation dataset needs a lower threshold than the training minimum.
- For IHC, one training dataset (wt_gerbil, 1.77 x b) sets the rule. Most IHC datasets tolerate
  4-12 x b. A rule per species could save more, but there are only 3 wt_gerbil datasets.

### Q3: Does dilation fix the missing neighbour blocks?

Yes, for IHC. The legacy rule has no dilation.

| IHC, p99 > 150 (exact, s0), 75 datasets | complete, worst | datasets >= 99.5% | kept, median | blocks vs production |
|---|---|---|---|---|
| no dilation (production) | 76.0% | 0 | 2.6% | 1.00 |
| 6 face neighbours (D6) | 99.2% | 71 | 9.1% | 1.93 |
| all 26 neighbours (D26) | 99.8% | 75 | 17.9% | 2.81 |

See [ihc_dilation.png](../../mask_threshold_analysis/figures/ihc_dilation.png). SGN does not need
dilation to reach the target. D6 removes the last misses (all datasets 100%) at 1.35 times the
blocks of the undilated rule.

Dilation does not fix all cut cells. Most cells that end on a block face end next to a predicted
block. Section 5 shows the cause.

### Q4: Is another method necessary?

No new principle is necessary. The change is a threshold relative to b, a different statistic
for SGN, and dilation for IHC (the table of section 2):
- The factors are the fitted values (1.086 and 1.77), rounded down.
- s3 is the 8x downsampled pyramid level.
  - 181 of the 193 S3 pyramids are exact 2x block means, so s3 is the 8^3 mean of s0.
  - 12 datasets (8 cochleae: G_LR_000301_R, G_LR_000302_R, M_AMD_000125_L, M_AMD_000137_L/R,
    M_AMD_N159_L, M_AMD_OTOF28_L/R) have levels of floor(shape / 2) with a bias of about 1.5 grey
    values. The rules pass on them too.
  - An input without a pyramid, or with a subsampled one, gets the s3 values by mean pooling of
    s0.
- "Blocks vs production" is the sum over all datasets (`evaluation/proposed_vs_production.csv`).
  - SGN: less than half of the current GPU blocks.
  - IHC: 2.05 times the current GPU blocks, but no cut IHCs. The median per dataset is 4.4,
    because the current rule keeps very few blocks in the small datasets.
- The mask step reads 1/512 of the data. The legacy step reads all of s0 (5-22 CPU minutes per
  cochlea on the cluster).
- Other candidates are worse at the same coverage (`evaluation/summary_t0.995.csv`):
  - maximum-based scores (`s3max_r*`);
  - the noise and contrast mappings;
  - one rule pooled over both classes (about 50% kept);
  - a fixed kept fraction per volume (`ratio`, the "experience" approach: 40-50% kept and
    validation failures).
- The same rules on the exact s0 percentiles give no saving (section 6).
- See [validation.png](../../mask_threshold_analysis/figures/validation.png) for the validation
  datasets.

## 5. Cut cells at block faces: the halos

The audit shows that a median of 19.5% of the relevant SGNs and 26.5% of the IHCs end on a
128-voxel block face. Chance is about 4.6%. The excess is about 3 times stronger in z than in y
and x. Most of these cells end next to a predicted block, so the input mask is not the cause:

| per dataset, median | SGN | IHC |
|---|---|---|
| ends on a block face, neighbour block predicted, no partner object | 11.5% | 12.0% |
| split at a block face, partner object on the other side | 6.1% | 3.4% |
| ends on a block face, neighbour block empty (mask-cut candidate) | 1.2% | 10.1% |

**Test on real data** (`block_seam_test.py`). Two crops of 2 x 1 x 1 blocks (G_EK_000076_R
SGN_v2, M_AMD_000126_R IHC_v11) around faces with many cuts, plus a ring of one block for
context. The test uses the production tiling: U-Net blocks of 128^3 on the CPU, the production
watershed on 128^3 chunks, and no input mask. The count is the number of label pairs with a
contact of 20 or more voxels across 7 block faces. 4 interior planes are the control: there, a
contact is a real contact between two cells.

| prediction halo | watershed halo | SGN face splits | IHC face splits |
|---|---|---|---|
| S3 reference segmentation | | 11 | 13 |
| (16, 32, 32), production | (2, 8, 8), production | 9 | 13 |
| (16, 32, 32) | (16, 32, 32) | 6 | 7 |
| (32, 32, 32) | (2, 8, 8) | 2 | 11 |
| (32, 32, 32) | (16, 32, 32) | 2 | 3 |
| control planes (all settings) | | 0 | 3 |

- The production settings reproduce the reference segmentations.
- Prediction seam: the U-Net sees 16 voxels of context in z, but 32 in y and x. At a block face,
  the center-distance prediction jumps 4-7 times more than between interior planes. In the SGN
  crop, this creates a seed only 1-2 planes thick at the face, and the watershed gives it its own
  label (1,700-3,000 voxels, so it passes `min_size`). A z halo of 32 removes most of these.
- Watershed halo: `distance_watershed_implementation` used `halo=(2, 8, 8)`. A block did not see
  the seed of a cell that crosses the face further away. This matters for the larger IHCs.
- Both changes together bring the face splits down to the level of the control planes.

**Fix:**
- Prediction halo (32, 32, 32): the input blocks are 192^3 instead of 160 x 192 x 192, so the
  GPU time increases by about 20% (CPU test: +24% for SGN).
- Watershed halo (16, 32, 32). This is a CPU step, and the test shows no measurable cost.
- Limits: two crops with 9-13 splits each. A run on a full cochlea with `audit_block_cuts.py`
  must confirm the fix.

## 6. Exact s0 check (HPC job)

`collect_s0_block_counts.py` read the full-resolution stain of all 193 datasets from S3. For
every 128^3 block, it stored p95, p99, p99.9, the maximum, the mean, and count(I > t) for 68
thresholds t (`s0_counts/`).

**Validation:**
- The s0 percentiles reproduce the kept blocks of all 15 logged production IHC runs exactly. The
  s3 estimate kept 0.73-0.99 times the logged blocks.
- At s0, the production rules keep more blocks than the s3 estimate (median 1.10 times for SGN
  p95 > 100 and 1.06 times for IHC p99 > 150). No conclusion changes.
- None of the 162 output folders of the reference runs still holds a `mask.zarr`. The comparison
  with the reference runs therefore uses the rule recorded in the job archive.
- 3 datasets (G_LR_000302_R SGN and IHC, M_AMD_OTOF28_L IHC) have a last slab of 1-3 planes that the s3 level
  does not cover. The evaluation crops the s0 grid to the s3 grid (`align_s0`).
- `compute_mask_blocks` reads the s3 level from S3 in 6-46 s per dataset. On 16 datasets, among
  them all 12 with a non-exact pyramid, it gives the same block grid as this evaluation.

**s0 or s3 statistic.** Rules fitted on the exact s0 percentiles keep about the same number of
blocks:

| | SGN, validation kept, median | IHC, validation kept, median |
|---|---|---|
| rule at s3 | 44.6% | 11.3% |
| rule at s0 | 45.8% | 11.0% |

The s0 statistic gives no saving, and s3 reads 1/512 of the data.

**Attribution of the old references** (`evaluation/reference_consistency.csv`). The rule that
made a reference cannot have lost its objects. 143 of 150 references agree with their recorded
rule. 7 SGN references do not:

| reference | recorded rule | own objects lost under that rule |
|---|---|---|
| M_AMD_000127_L | p95 > 200 | 100% |
| M_AMD_N75_L | p99 > 150 | 35.6% |
| M_AMD_000126_L | p95 > 200 | 31.3% |
| M_AMD_N159_L (failure case) | p95 > 200 | 18.4% |
| M_AMD_000126_R | p95 > 200 | 18.0% |
| M_AMD_N161_L | p95 > 200 | 2.8% |
| M_AMD_Roe310_L | p95 > 200 | 2.7% |

- For N75_L, a later job with the class `ihc_low` (p99.5 > 100) exists.
- For 126_L, 126_R and 127_L, a job of the next day ran with the class `sgn` and no threshold.
- A coarser 256^3 mask grid does not explain the losses; it loses more.
- The rule of these references is unknown, so the proxy skips them.

**Missed neurons in the old SGN references.** The s0 counts give a direct estimate:
- A block is neuron-like if it holds at least as many voxels above the object level as a small
  neuron. The object level is the 5th percentile of the object signal. A small neuron is the 10th
  percentile of the object sizes.
- The candidates are the blocks that the reference rule skipped next to the blocks of the
  relevant objects.
- The expected count is the neuron-like rate of skipped blocks far from all objects, times the
  number of candidates. The excess is the observed minus the expected count.

Results (`evaluation/missed_signal_proxy.csv`, 87 consistent SGN references):
- 21 references have an excess of 10 blocks or more, and 6 of 50 or more.
- The median excess is 4 blocks for p95 > 200 (38 references) and 0 blocks for p99 > 150
  (46 references).

| reference | rule | excess blocks | blocks touched by objects | candidates kept by the new rule |
|---|---|---|---|---|
| M_AMD_000088_L | p99 > 150 | 165 | 630 | 295 of 318 |
| M_AMD_000115_L | p95 > 200 | 150 | 625 | 218 of 218 |
| M_AMD_C202_R | p95 > 200 | 103 | 447 | 144 of 170 |
| G_EK_000074_L | p95 > 200 | 81 | 1,237 | 168 of 169 |
| G_EK_000032_L | p95 > 200 | 57 | 1,443 | 69 of 69 |
| M_AMD_000087_R | p99 > 150 | 55 | 649 | 63 of 89 |

- The criterion also counts bright tissue that is not a neuron. The background rate is up to 20%
  in some datasets, so this is a proxy, not a count of neurons.
- The new SGN rule keeps most of the candidate blocks. A run with the new rule on one of these
  cochleae (for example M_AMD_000115_L) shows whether the reference misses neurons.

## 7. Review (2026-11-16)

Checks before the tag:
- The numbers of sections 2, 4 and 6 agree with the CSV files in `evaluation/`: the block ratios
  (SGN 0.456, IHC 2.051), the worst datasets, the 7 inconsistent references, the 21 and 6 SGN
  references with an excess, and the 15 logged runs.
- `compute_mask_blocks` computes b and the block score as the evaluation does: the median of the
  integer histogram with interpolation inside the bin (`hist_quantile`), and the percentile of
  the zero-padded 16^3 block (`collect_lowres_features.py`).
- The datasets with a non-exact pyramid are 12 datasets of 8 cochleae, not 6 cochleae as the
  first report said. The shapes of `lowres_s3/` and `s0_counts/` confirm the 8 cochleae and the
  3 datasets with a last slab (G_LR_000302_R SGN and IHC, M_AMD_OTOF28_L IHC).

Notes on the reasoning:
- **Absolute or relative threshold for IHC.** On this data, the absolute variant (s3p99_D26 >
  188) keeps slightly fewer blocks than the relative rule (training median 11.2% vs 12.2%,
  validation 10.4% vs 11.3%), at the same coverage. b is 100-137 in all datasets, so the two
  variants are nearly equal. The relative rule is chosen because it spreads less for SGN
  (p90/p10 1.67 vs 1.98) and gives one rule form for both classes.
- **Failure case.** M_AMD_000085_L (484 relevant SGNs, a failed segmentation) keeps 96.3% of its
  objects complete under the new SGN rule. The failure cases are not part of the fit or the
  validation.
- **Small crops.** b is the background level only if most non-zero voxels are background, as in
  a whole cochlea. The analysis used whole cochleae only. In a small crop that cells fill for
  the most part, b is too high. Use `--disable_masking` for such a crop.
- **Other callers of the prediction halo.** The default (32, 32, 32) of `_get_device_and_tiling`
  applies to every `prediction_impl` call without a halo: the CPU path of the synapse prediction,
  `sgn_detection` and the gridsearch tools. They get more context at about 20% more compute.

Changes from the review: the `MASK_RULES` comment names segmentations instead of cochleae,
`find_mask` documents the crop limit, and `run_segmentation` and `run_unet_prediction` accept
`background_factor`.

## 8. Limits

- Reference bias: objects that the reference mask removed completely are not in the reference.
  The missed-signal proxy shows where this is probable. Only an unmasked run is a true reference.
  The new rules keep more blocks than the old high thresholds, so a run with the new rules also
  shows missed objects.
- The mask-cut candidates of the audit are an upper bound.
- The factors come from the protocols in the data. A new, dimmer protocol can need a lower
  factor (`--background_factor`). `mask_statistics.json` makes the threshold of every run
  visible.
- The rules apply to whole cochleae (section 7).
- The halo fix is validated on two crops only (section 5).

## 9. Open items

1. Cluster check with the new defaults: one IHC cochlea (M_AMD_000126_R, logged mask and seam
   crop) and one SGN cochlea with suspected misses (M_AMD_000115_L). Compare the kept blocks, the
   runtime, the number of objects and the face hits of `audit_block_cuts.py` with the reference.
2. Merge flamingo-tools, then the slurm-workflows branch `mask_rule_default`.

## 10. Scripts

All outputs go to `mask_threshold_analysis/` in the repository root. Run the scripts from the
repository root with the `cochlea-net` environment.

| Step | Script | Output | Runtime |
|---|---|---|---|
| 1 | `parse_mask_jobs.py` | `mask_jobs.csv`: the mask rule of every archived mask job | seconds |
| 2 | `mask_datasets.py` | `mask_datasets.json`: datasets and the frozen train/val split | ~15 min |
| 3 | `audit_block_cuts.py` | `audit/`: objects that end on the 128-voxel block grid | ~15 min |
| 4 | `collect_lowres_features.py` | `lowres_s3/`: s3 histograms, block scores, object signal | ~4 h over S3 |
| 5 | `collect_s0_block_counts.sbatch` | `s0_counts/`: exact s0 statistics, harvested masks (HPC) | HPC array job |
| 6 | `evaluate_mask_rules.py` | `evaluation/`: fitted rules, validation, calibration, s0 checks | minutes |
| 7 | `make_report_figures.py` | `figures/` | seconds |
| - | `block_seam_test.py` | cell splits at block faces for one crop and several halos (CPU) | ~10 min per prediction halo |

`mask_common.py` holds the shared paths and helpers.

- `mask_datasets.json` is the frozen split. The script refuses to overwrite it without `--force`.
- Step 1 reads the job archives in `~/Documents/job_archive*` and the Slurm logs in
  `~/Documents/output_slurm_NHR`.
- Step 5 runs on the cluster. Copy `mask_datasets.json` and `mask_jobs.csv` next to
  `collect_s0_block_counts.py`, set the paths in the sbatch file, and copy the output folder back
  to `mask_threshold_analysis/s0_counts/`. Step 6 uses these files when they exist.
- With the s0 counts, step 6 also checks each reference against its recorded rule
  (`reference_consistency.csv`), compares the new rules with the exact production rules
  (`proposed_vs_production.csv`), and estimates the neuron-like blocks that the old SGN rules
  skipped (`missed_signal_proxy.csv`).
- Step 6 options: `--no_s0` ignores the HPC counts, `--signal_floor` sets the minimum relative
  object signal of a target object (default 1.2), and `--output_dir` writes a variant run to
  another folder (for example `evaluation_floor1.0`).
- `block_seam_test.py` has the commands of section 5 in its docstring.
- To change `MASK_RULES`, rerun steps 4 to 6 on the current S3 data and keep the target of 99.5%
  complete relevant objects in every validation dataset.

## 11. Files

At the tag, `mask_threshold_analysis/` holds the key outputs:
- `mask_datasets.json` (the frozen split) and `mask_jobs.csv`;
- `figures/`: `audit_cutoffs.png`, `ihc_dilation.png`, `critical_thresholds.png`,
  `validation.png`;
- `evaluation/`: `summary_t*.csv` (all rules), `proposed_vs_production.csv`,
  `reference_consistency.csv`, `missed_signal_proxy.csv`, `calibration_logged_runs.csv`,
  `excluded_objects.csv`, `normalization.csv`, `failure_datasets.csv`;
- `audit/summary.csv`: the face hits per dataset;
- `evaluation_log.txt` and `lowres_s3_log.txt`.

These outputs are too large for the repository and stay on the analysis machine:
- `s0_counts/` (498 MB): the exact s0 block statistics of the HPC job;
- `lowres_s3/` (107 MB): the s3 features per dataset;
- `evaluation/results_t*.csv` and `critical_t*.csv` (per dataset), and `fixed_rules.csv`;
- `audit/face_hits.csv` (per object) and `audit/audit_log.txt`;
- the variant runs `evaluation_floor1.0/` and `evaluation_floor1.4/`.

Steps 1 to 6 recreate them.
