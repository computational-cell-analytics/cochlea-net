# Synapses

## Network training

`scripts/synapse_marker_detection/train_synapse_detection.py` trains the v11 recipe by default
since 2026-10-28:

- heatmap only, with `DetectionLoss` (mean squared error) as loss and as validation metric;
- `MinPointSampler(1, 0.8)`: a patch needs two annotations, or it passes with probability 0.2;
- fixed validation patches, seeded by `--random_state`;
- the fixed target: every isolated annotation has a peak of 4, and an annotation up to 3 sigma
  outside the patch keeps its Gaussian;
- a cosine learning rate schedule from 1e-4 to zero over 3e4 iterations. Use `latest.pt`;
- the `train_val_split.json` of the data folder. For `v11`, the val crops are the consensus
  crops apex1 and base1.

The patch shape is 40 x 112 x 112 and the batch size is 32. One run takes about 5.7 A100 hours.
`scripts_synapses/train_synapse_v11.sbatch` trains the three seeds v11-1 to v11-3. The training data
`training_data/synapses/training_data/v11` holds the 29 crops of `v7`, which is stated to equal
`v5` and `v6`, and the two consensus val crops. The model, the loaders, the loss and the trainer
are in `flamingo_tools/synapse_detection/training.py`. The raw input needs no normalization,
because the U-Net starts with an InstanceNorm (`TestModelNormalization`).

**Retrain v3, v5 or v6-1 with `--legacy_recipe`.** It restores the unweighted mean squared error
over every output channel as the validation metric, redrawn validation patches, no sampler
without `--use_flow`, the old target and the plateau schedule over 1e5 iterations. Add
`--use_flow` for the heatmap and flow model of v5. The archived scripts in `scripts_synapses/`
pass the flags.

**Retrain v7 to v10 from the snapshot.** Their options are removed. Check out the tag
`synapse-detection-investigation` for v7 and v10, and commit `4da9e70` for the masks of v8 and
v9. The tag also holds the full, dated record of the investigation. "Network variation" below is
its summary.

## Network application
Potentially relevant for a new synapse network.

The standard network is v11-2 `latest.pt`. The detection defaults (threshold 0.5,
`min_distance` 1) are in `flamingo_tools.model_utils.get_default_detection_settings`. The
`detect_synapses` template of slurm-workflows passes the threshold and keeps the default
`min_distance`.

An example script for cochlea `G_LR_000302_R` is `synapse_process_GLR000302R.sbatch`.
The volume can be masked based on an IHC segmentation, which can be local or on the S3 bucket.
`build_ihc_mask` dilates the segmentation by 4 voxels at `s4`, which is 64 voxels at full
resolution, so the mask does not cut off synapses at the border of the IHC segmentation.

**Prefer the three-stage slurm workflow over `marker_detection` for a whole cochlea.**
`synapse_process_GLR000302R.sbatch` runs prediction, peak detection and matching in one 10 h
job, and the `G_LR_000302_R` result it produced covers only part of the helix while looking
complete. Whatever interrupted that run, the reason it went unnoticed is structural:
`marker_detection` skips the prediction outright when `synapse_detection.tsv` already exists,
so a rerun detects maxima in whatever the previous attempt left behind.
`run_synapse_prediction_preprocess_slurm` / `run_synapse_prediction_slurm` /
`run_synapse_detection_slurm` split the same work into a preprocessing job, a prediction array
and a detection job. `to-do_revision/scripts_syn_gerbil/` wires them up for the wild-type
gerbils, including the coverage check that catches a truncated prediction; read its README
before applying the network to a new cochlea.


## Post-processing

* transfer synapse detection to MoBIE
* transfer to S3 bucket

### Calculate synapses near IHC components
This script reads the information of `synapse_per_ihc_utils.py`:
```bash
OUT_DIR=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/ihc_counts_v11/
python ~/flamingo-tools/scripts/measurements/measure_synapses.py -c <cochlea1> <cochlea2> ... -o "$OUT_DIR"
```

### Add `syn_per_IHC` to segmentation table
The column `syn_per_IHC` is read by some plot functions for figure 3. It has to be added to the segmentation table and needs to be uploaded to the S3 bucket.
```bash
python ~/flamingo-tools/scripts/synapse_marker_detection/add_synapse_per_ihc.py -c <cochlea> -o .
```
will produce a segmentation table in the current directory. It can be transferred to S3 with:
```bash
COCHLEA=G_LR_000302_R
rclone copyto "$COCHLEA"_v11_syn-per-ihc.tsv cochlea-lightsheet:cochlea-lightsheet/"$COCHLEA"/tables/IHC_v11/default.tsv
```

## Network variation

### Final state (2026-10-28)

The investigation is closed. Decision (Martin, 2026-10-28):

- The v11 recipe is the training default, see "Network training".
- v11-2 `latest.pt` is the standard network.
- Production detects at threshold 0.5 and `min_distance` 1. The values are in
  `flamingo_tools.model_utils.get_default_detection_settings`.

Test v11 (4 crops, 733 points), F1. `cons` is the setting that the gridsearch selects on the
consensus val crops:

| Model | (0.5, 2) | (0.5, 1) | `cons` |
|---|---:|---:|---|
| v5, v5-rep1 to v5-rep4, `best.pt` | 0.840 to 0.878, mean 0.861 | | 0.835 to 0.890, mean 0.868 |
| v7-latest | 0.881 | | (0.35, 1), 0.896 |
| v10 | 0.873 | | (0.15, 1), 0.901 |
| v11-1-latest | 0.882 | 0.894 | (0.10, 1), 0.883 |
| v11-2-latest | 0.889 | 0.907 | (0.20, 1), 0.908 |
| v11-3-latest | 0.888 | 0.902 | (0.10, 1), 0.900 |

### What was tried

F1 on the six consensus crops at (0.5, 2), from `reproducibility/model_accuracy/synapses.json`:

| Version | Change | F1 | Verdict |
|---|---|---:|---|
| v3, v3-1 to v3-7 | the released heatmap model and its seed replicates | 0.782 to 0.872 (IHC v4) | the seed spread is larger than every later recipe change |
| v5 | flow channels and the `MinPointSampler` | 0.886 (`v5-ihc11`) | v7 matches it without flow, so the gain probably comes from the sampler |
| v7 | heatmap only, `DetectionLoss` as metric, fixed val patches, the sampler | 0.876, latest 0.886 | the base of v8 to v11 |
| v8 | loss only in cubes of half width 16 around the annotations | 0.712 (P 0.627) | failed |
| v9 | loss without cubes around the unannotated CTBP2 spots | 0.875 | no gain |
| v10 | target sigma 0.7 instead of 1 | 0.877 | no gain at (0.5, 2) |
| v11-1 to v11-3 | fixed target, cosine schedule over 3e4 iterations, consensus val crops | see above | the standard |

### Key findings

1. **Close pairs limit the recall.** For 94 % of the false negatives of v7-latest, the matching
   gave a detection within 3 um to a neighbour. 42 % of the consensus points have a neighbour
   within 2 um. The IHC filter removes no true positive, so IHC context (as input or as loss
   mask) cannot help. A narrower target (v10) does not separate the pairs. `min_distance` 1
   instead of 2 helps every model by about 0.01 F1.
2. **The loss masks fail.** The test crops are 35 to 78 % zero from the reslicing. The cube mask
   of v8 removes the negative supervision at the borders between tissue and zero, and v8 fires
   there. The masked metric could not see it. The candidate mask of v9 changes nothing.
3. **The target of every model up to v10 depends on the patch.** `CsvHeatmapTransform` smoothed
   with the `nearest` mode and divided by the patch maximum, so in 80 to 85 % of the patches the
   peaks are 2.28 instead of 4. In 92 % of the patches, an annotation just outside the patch lost
   its Gaussian. The fixed target of v11 removes both defects.
4. **The plateau schedule collapses early.** Every run from v5 to v10 reached the floor of
   1.22e-8 within 100 to 200 of 1,000 epochs, so about 86 % of the iterations did nothing.
   `latest.pt` was never worse than `best.pt`. The cosine schedule over 3e4 iterations takes 5.6
   times less training.
5. **The old val crops are not representative.** In 20 of the 29 training crops, and in all three
   old val crops, the labels mark close pairs 2 to 3 times less often than the consensus. A
   selection on them chose too high a threshold (v3 and v5: -0.019 F1 on average). The new split
   uses the consensus crops apex1 and base1 (416 points) as val crops and apex2, mid1, mid2 and
   base2 (733 points) as test v11.
6. **The selection must filter by the IHCs.** A radius filter around the annotations hides the
   false positives far from them and selects too low a threshold (v9: -0.09 F1). The gridsearch
   now filters by the distance to the IHCs, as the evaluation does (`--ihc_dir`).
7. **The models are at human level.** Against the points that the other two annotators agree on,
   a held-out annotator gets F1 0.871 to 0.916 (mean 0.901), and v7-latest 0.897 to 0.918. The
   paired crop bootstrap of a model difference is about +-0.01.
8. **v11 has a small seed spread.** At (0.5, 2) on test v11, the v5 seeds have SD 0.016 (worst
   0.840) and the v11 seeds SD 0.004 (worst 0.882). Three seeds cannot prove a smaller spread.
9. **`min_distance` 1 gives the gain of the selection, the low threshold does not.** The v11 seeds
   select (0.10, 1), (0.20, 1) and (0.10, 1). At threshold 0.5, `min_distance` 1 gains +0.007 to
   +0.013 on the val crops and +0.012 to +0.018 on the test crops. The low threshold adds false
   positives that the two val crops do not show (v11-1 on base2: 37 at (0.10, 1), 3 at (0.5, 2)).
10. **Block shape and halo matter, the normalization does not.** Predict with
    `_PREDICTION_BLOCK_SHAPE` and `_PREDICTION_HALO`.

### The v11 decision rules

The rules were fixed before the results:

1. Select the setting with the gridsearch on the consensus val crops.
2. Score test v11 at the selected setting and at the median setting of the three seeds.
3. Baseline: v5, v5-rep1 to v5-rep4, v7 and v7-latest at their `cons` setting: mean 0.874,
   worst 0.835.
4. Make the recipe the default if the mean is at least the baseline mean minus 0.01, no seed is
   below the worst baseline seed, and the selected thresholds agree.
5. Choose the released seed by its val F1.

Rules 4.1 (mean 0.897) and 4.2 (worst 0.883) pass. Rule 4.3 fails on the threshold, but all three
seeds select `min_distance` 1. Production therefore uses (0.5, 1): every seed is within 0.008 of
its best val F1 there, at a precision of 0.97 or more. Rule 5 chooses v11-2 (val F1 0.925, then
v11-3 0.916 and v11-1 0.909).

### Reproduction

```bash
ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
DATA=$ROOT/training_data/synapses
IHC=$ROOT/trained_models/IHC/v11_cochlea_distance_unet_IHC_supervised_2026-07-20
# Train with the v11 recipe on the v11 data folder and its split.
python scripts/synapse_marker_detection/train_synapse_detection.py -m <name> --random_state 42 -s <save_root>
# Select the setting on the consensus val crops. --ihc_dir holds <crop>_ihc/segmentation.zarr.
python -m flamingo_tools.synapse_detection.gridsearch -m <checkpoint>/latest.pt \
    --ihc_dir $ROOT/predictions/val_synapses/production_2026-09-25/v7
# Score test v11 once, at the production setting. Use a new output root for every run.
OUT=$ROOT/predictions/val_synapses/production_<date>/<name>
python scripts/validation/synapses/prediction.py -o $OUT -i $DATA/test_data/v11/images \
    -g $DATA/test_data/v11/labels --model_synapse <checkpoint>/latest.pt --model_ihc $IHC
python scripts/validation/synapses/run_evaluation.py -p $OUT -o reproducibility/model_accuracy/ \
    -c $DATA/test_data/v11/images -r $ROOT/AnnotatedImageCrops/Synapses_2026-04/consensus_annotation
```

A registered version scores with `prediction.py -v <version>` and `run_evaluation.py -v <version>`.

### Gotchas

- `synapses.json` has two IHC eras. The entries written before 2026-09-24 were filtered against
  IHC v4. Compare only the `-ihc11` keys and v7 or later. The plain `v3` and `v5` keys stay on
  IHC v4 for Figure 2c and Supplementary Figure 2.
- Every `synapses.json` entry written before 2026-10-28 used `min_distance` 2. The entries do not
  store the setting.
- The `v11-*` entries score the six consensus crops, which include the v11 val crops apex1 and
  base1. Compare v11 with earlier models on the four test v11 crops only.
- `run_evaluation.py` refuses to replace an entry without `--overwrite`. `prediction.py` skips a
  crop that already has a prediction.
- The pickled exports (`synapse_detection_v*.pt` up to v10, `synapse_detection_model_*.pt`) fail
  in torch_em 0.10.7 and later. Use an env with torch_em 0.10.6 or earlier, or a trainer
  checkpoint.
- Use `consensus_annotation/` (1,149 points) as reference, not `test_data/v5/labels` (1,133
  consensus points). 1,123 of the 1,149 points lie within 3 um of the IHCs.
- The checkpoint folders `v5-f1` to `v5-f4` are seed replicates (`v5-rep1` to `v5-rep4`). The
  exports `synapse_detection_model_v5-f{1..4}.pt` should be the fold models `v5-fold1..4/best.pt`,
  but on the cluster they gave heatmaps identical to `v5-rep1` to `v5-rep4`. Compare them before
  you score a fold model again.
- Do not switch the branch of the cluster checkout while a job runs.
- Never select a setting on the test crops, and never on the old val crops.

### Open items

- Compare (0.5, 1) with (0.5, 2) on one whole cochlea (`syn_per_IHC`). Predict it with v11-2 and
  keep `predictions.zarr` (`--no-cleanup` in slurm-workflows). Then run only the detection stage,
  once per `min_distance`.
- Copy the v11 checkpoints to `trained_models/Synapses/synapse_detection_v11-<i>[-latest].pt`, the
  paths of `PREDICTION_DICT`, and add `synapses_v11` to the slurm-workflows project settings.
- Publish v11-2 on BioImage.IO (loyal-ant v1). Then bump `MODELS["Synapses"]` in
  `flamingo_tools/model_utils.py`. Until then, `get_model("Synapses")` gives v3.
- A detection-level validation metric. The voxelwise MSE does not rank detectors (Spearman rho
  0.39), but with the cosine schedule `latest.pt` does not depend on it.
- `v6-1` has no IHC v11 score. That the `v7` data equals the `v5` data is not shown.

### Settings of every model version

Common to every run up to v10: patch shape 40 × 112 × 112, batch size 32, learning rate 1e-4
with the torch_em `ReduceLROnPlateau` (factor 0.5, patience 5 epochs, see "Key findings"), 3,200
training and 160 validation samples per epoch, no augmentation, a Gaussian target of peak 4 and
sigma 1 voxel (v10: 0.7) divided by its patch maximum, and three validation crops. v11 differs
as given in its row. The `synapses.json` entries of the production
pipeline use threshold 0.5, `min_distance` 2, the IHC filter at 3 um and matching at 3 um on the
six consensus test crops.

| Version | Data | Output | Loss / validation metric | Sampler | Loss mask | Validation patches | Iterations | Export | Evaluation |
|---|---|---|---|---|---|---|---|---|---|
| v3 | `v3`, 15 crops | heatmap | MSE / MSE | none (passed, but dropped by the old upstream) | none | redrawn | 1e5 | not recorded | `v3`: production, IHC v4 (legacy pipeline: threshold 1.3); `v3-ihc11`: IHC v11 |
| v3_05t | v3 checkpoint | | | | | | | | old per-version pipeline at threshold 0.5 |
| v3-1 … v3-4 | `v3` | heatmap | MSE / MSE | none | none | redrawn | 1e5 | `best.pt` (epochs 750, 380, 848, 134) | production, IHC v4 |
| v3-3-best | v3-3 | | | | | | | `best.pt`, epoch 848 | production, IHC v4 |
| v3-5, v3-6, v3-7 | `v3` | heatmap | MSE / MSE | none | none | redrawn | 1e4 | v3-5: epoch 71 (`v3-5-epoch71`); v3-6: best, epoch 99; v3-7: best, epoch 43 | production, IHC v4; v3-7 not scored |
| v3-flow-1 | `v3` | heatmap + flow | MSE + 0.1 flow / MSE over 5 channels | `MinPointSampler(1)` | none | redrawn | 1e5 | best and latest | not scored |
| v4 | `v4`, 20 crops | heatmap + flow | upstream, not recorded in detail | `MinPointSampler(1)`, probably applied | none | redrawn | 1e5 | not recorded | legacy pipeline, IHC v4, never rescored |
| v5 | `v5`, 29 crops | heatmap + flow | MSE + 0.1 flow / MSE over 5 channels | `MinPointSampler(1)`; last 21 voxels per axis not trained | none | redrawn | 1e5 | `best.pt` | `v5`: legacy pipeline at 0.5, IHC v4; `v5-ihc11`: production, IHC v11 |
| v5_05t | v5 checkpoint | | | | | | | | old per-version pipeline at threshold 0.5 |
| v5-f1 … v5-f4 | `v5`, four other splits: the exports are `v5-fold1..4/best.pt` | heatmap + flow | as v5 | as v5 | none | redrawn | 1e5 | `best.pt` (epochs 31, 698, 681, 64) | `run_detection` at 0.5, IHC v4; `v5-f0` has the counts of v5 |
| v5-rep1 … v5-rep4 | `v5`, the v5 split; checkpoint folders `v5-f1..4` | heatmap + flow | as v5 | as v5 | none | redrawn | 1e5 | `best.pt` (epochs 278, 13, 109, 28) and `latest.pt` | test v11 only, at (0.5, 2) and `cons` (2026-10-26, in the snapshot) |
| v6-1 | `v6` = v5 data | heatmap | MSE / MSE | none | none | redrawn | 1e5 | not recorded | not scored |
| v7 | `v7` | heatmap | `DetectionLoss` / the same | `MinPointSampler(1)` | none | fixed, seed 42 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v8 | `v8` | heatmap | masked `DetectionLoss` / the same | `MinPointSampler(0)` | cubes of half width 16 around the annotations; last 33 voxels per axis not trained; commit `4da9e70` | fixed, seed 42 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v9 | `v7` | heatmap | masked `DetectionLoss` / the same | `MinPointSampler(1)` | cubes of half width 3 around the unannotated candidates, percentile 10; commit `4da9e70` | fixed, seed 42, other draws than v7 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v10 | `v7` | heatmap, target sigma 0.7 | `DetectionLoss` / the same | `MinPointSampler(1)` | none | fixed, seed 42, other draws than v7 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v11-1 … v11-3 | `v11` = `v7` + 2 consensus val crops; split file, the 3 old val crops train | heatmap, fixed target | `DetectionLoss` / the same | `MinPointSampler(1)` | none | fixed, seed 42 | 3e4, cosine learning rate | `best.pt` and `latest.pt` | production, IHC v11; selected on the consensus val crops: (0.10, 1), (0.20, 1), (0.10, 1); released: v11-2 `latest.pt` at (0.5, 1) |

`MinPointSampler(k)` accepts a patch with more than `k` annotations, and otherwise with
probability 0.2. v7 to v10 standardize the raw input, which is a no-op for this network; whether
the earlier runs did is irrelevant for the same reason. Gaps and conflicts in the record:

- The v3 split is 13/2 in the original 2025-06-15 script and 12/3 in the later notes.
- The v4 training date and its export are not recorded, and v4 was never rescored with the
  production settings.
- `v5_05t` (F1 0.888) and `v5` (0.871) are both at threshold 0.5. The cause of the difference is
  not recorded.
- The plain `v3-3` export (0.851) is not documented. `v3-3-best` is the epoch 848 `best.pt`.
- The commands of the v5 folds and of the v5 replicates are not recorded. The val crops of each
  run are in its checkpoint, in `init["val_dataset"]`.
- The thresholds behind `v5_f1val_threshold` and `v5_train_threshold` in
  `plot_supp_fig2.py` are not recorded.
- That the `v7` and `v8` data folders equal the v5/v6 data is stated, but not shown.
- `v6-1` has no score.

