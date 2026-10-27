# Synapses

## Network training

Since PR `#133` the synapse detection network is trained without flow per default. The flow option has to be selected by supplying a `--use_flow` flag.
The training data is located in separate `images` and `labels` directories, so the training/validation split has to be reproducible. The random state `--random_state 42` has to be passed for training the networks to test the variation to ensure the same split.
An example script for training `synapses_v3-1` is `train_synapse_v3-1.sbatch`.

The training commands of every run so far are in `scripts_synapses/`. The runs before v7 need
`--legacy_recipe`, and their scripts pass it. The mask options of v8 and v9 were removed on
2026-10-21. Check out commit `4da9e70` to retrain v8 or v9. The `v6` training data in
`/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v6`
is identical to the `v5` data.

### Training-data and loss diagnostics (2026-08-27)

The unusual v3 training dynamics are consistent with conflicting supervision from unlabeled
CTBP2 spots. The training volumes contain many bright CTBP2 candidates outside the annotated
IHC region. The model receives only the CTBP2 channel, so predicting a spot at such a location is
penalized as background even when the local intensity profile resembles a synapse.

#### Evidence in the v3 data

CTBP2 local maxima were detected with a threshold calibrated independently for each volume. The
results below use a conservative threshold: a candidate had to be at least as bright as the
median local maximum at annotated synapses. A candidate was considered annotation-matched when
it was within 4 voxels of an annotation.

For the seven v3 volumes that contain `raw_ihc`, an unmatched candidate was classified as
IHC-unsupported when the maximum IHC signal in a local neighborhood was below the fifth
percentile measured at annotated synapses. This is the direct evidence for off-target spots. For
the eight volumes without `raw_ihc`, candidates more than 40 voxels from every annotation were
counted separately. These spatially distant candidates are suggestive, but they cannot be called
off-target with the same confidence: some may be real synapses outside the manually annotated
extent.

| Median-intensity candidates | Train | Validation |
|---|---:|---:|
| Directly IHC-unsupported | 1,215 | 7 |
| More than 40 voxels from an annotation, IHC unavailable | 534 | 108 |
| Annotation-matched | 1,251 | 319 |

The result is not caused only by a permissive intensity threshold. At the 75th percentile of the
annotated intensities, the full v3 data still contained 1,209 IHC-unsupported or spatially distant
candidates versus 766 annotation-matched candidates.

The direct IHC evidence is highly stain/domain-specific. The three `m78l_*_cr-ctbp2` training
volumes contribute 1,205 of the 1,222 directly IHC-unsupported median-intensity candidates in the
full v3 split. All three CR volumes are in the training set. The issue is therefore severe in a
specific part of the training distribution rather than equally severe in every crop.

The extraction procedure explains why these objects are present. `extract_training_data` keeps
the complete non-zero field of view; it only removes terminal zero padding and does not crop to
the IHC or to the annotated region. It stores a second channel as `raw_ihc` when available, but
`train_synapse_detection.py` trains exclusively from `raw`.

#### How distinguishable are the off-target spots?

The strongest version of the hypothesis, that *all* off-target objects are indistinguishable in
the CTBP2 channel, is too strong. A five-fold held-volume classifier based on local CTBP2 patch
summaries achieved an ROC AUC of 0.94 for annotation-matched versus directly IHC-unsupported
candidates. Many off-target objects have broad, saturated, or otherwise recognizable profiles.

There is nevertheless a substantial hard subset. At a threshold retaining 90% of the annotated
candidates, the raw-only classifier also accepted 382 of 2,781 (13.7%) directly IHC-unsupported
candidates from the represented v3 volumes. When restricted to compact spot-shape profiles, it
accepted 736 of 2,781 (26.5%). A more accurate statement is therefore:

> The training data contain a large population of unlabeled CTBP2 candidates, including a
> substantial hard subset that cannot be rejected reliably from its local CTBP2 profile alone.

This classifier is a diagnostic proxy, not proof that the U-Net cannot learn additional context.
Manual review or checkpoint predictions at the candidate coordinates are needed to establish
which individual candidates are genuinely ambiguous.

#### Why these false negatives strongly affect MSE

`CsvHeatmapTransform` creates a Gaussian target with sigma 1 and peak value 4. For the training
patch shape `40 x 112 x 112`, one isolated, target-shaped prediction at an unlabeled position
contributes approximately:

| Predicted peak amplitude | MSE contribution |
|---:|---:|
| 0.5 | 0.00000277 |
| 1 | 0.0000111 |
| 2 | 0.0000444 |
| 4 | 0.000178 |

The best recorded v3-variant validation losses range from approximately 0.000112 to 0.000161.
Thus, one confident target-shaped prediction at an unlabeled spot can contribute more than the
entire reported validation loss. At the median-intensity candidate threshold, a uniformly sampled
v3 training patch contains an estimated 1.97 IHC-unsupported or spatially distant candidates on
average. Suppressing such candidates can lower voxelwise MSE while also suppressing true
synapses, which is consistent with the low-loss, low-recall behavior of conservative seeds such
as v3-1 and v3-2. This is a plausible mechanism, not yet a causal checkpoint-level demonstration.

#### Raw normalization

The raw input needs no normalization. The U-Net starts with an InstanceNorm on its input
(`ConvBlock3d` is norm, convolution, ReLU), so every training patch and every prediction block is
standardized inside the network. A global mean and standard deviation change the output by about
2e-5, and the mix of `uint8` and `uint16` crops does not matter. `TestModelNormalization` in
`test/test_segmentation/test_detection_dataset.py` pins this. The 2026-08-27 diagnosis that called
the missing normalization a bug is withdrawn. v7 to v10 standardized each crop anyway. That code
was removed on 2026-10-21.

#### Status of the recommended tests

1. Raw normalization: not needed, see above.
2. A fixed validation patch set: done on 2026-09-21. A detection-level validation metric: open.
3. Supervise the unannotated CTBP2 spots as *ignore*: done as the cube mask of v8 and the
   candidate-exclusion mask of v9. Neither helped, see "v7 against v8" and "v9 and the recall
   limit".
4. An ablation without the three `m78l_*_cr-ctbp2` volumes: not needed. v7 against v8 shows that
   the unannotated spots do not limit the recall.
5. Checkpoint predictions at the candidate coordinates: obsolete.

#### The training recipe since 2026-09-21

The training no longer imports czii-protein-challenge. The model, the loaders, the loss and the
trainer live in `flamingo_tools/synapse_detection/training.py`.

**Retraining v3 or v5.** Pass `--legacy_recipe`. It restores the deltas that otherwise make the
old recipes unreachable: the validation metric goes back to an unweighted mean squared error over
every output channel, which is what selected `best.pt` for v5, the validation patches are redrawn
on every epoch, and a heatmap-only run has no sampler. The archived scripts in
`scripts_synapses/` pass the flag. Without it, the script trains the v7 recipe: heatmap only,
`MinPointSampler` and fixed validation patches.

**Comparing the next runs.** Score models with `scripts/validation/synapses/run_evaluation.py`,
and train a fresh baseline with the same code rather than comparing against the shipped models.

**The validation patch set is fixed.** `DetectionDataset` takes a `patch_seed`, and the bounding
box for a given index is a pure function of `(patch_seed, index)`, so the same index gives the
same patch in every epoch and in every data loader worker. The training script derives the seed
from `--random_state`, the same number that already fixes the train and validation split, so the
validation data is reproducible across runs as well. The validation loader also stops shuffling,
because the trainer averages the metric over batches. Measured on a synthetic stand-in at the
production validation size, 160 patches in batches of 32, scored against a fixed prediction so
that only the patch draw moves: the metric spread over six passes was **15 % of its mean** before
the change and exactly zero after it.

**Still open: the detection-level metric.** The metric remains a voxelwise mean squared error, so
`best.pt` is now stable but still a proxy for detection F1, and the seed study already showed
that validation loss does not rank detectors (Spearman rho 0.39, p = 0.38). Keep scoring
candidates with `scripts/validation/synapses/run_evaluation.py`, and keep comparing `best.pt`
against `latest.pt`.

## Network application
Potentially relevant for a new synapse network.
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

The v3 seed replicates are trained and scored, see `scripts_synapses/README.md` and
`scripts_synapses/v3_training_dynamics_report.md`. `plot_fig2.py` reads their accuracy from
`reproducibility/model_accuracy/synapses.json`. These entries were scored against IHC v4.

### Evaluating v7 and v8, and the IHC v11 switch (2026-09-24)

`v7` and `v8` are the unmasked and masked pair of the 2026-09-21 recipe. Four entries are
registered, because the two `best.pt` were selected by different metrics, the masked criterion
for `v8` and the unmasked one for `v7`:

```bash
for VERSION in v7 v7-latest v8 v8-latest ; do
	python scripts/validation/synapses/prediction.py -v "$VERSION"
	python scripts/validation/synapses/run_evaluation.py -v "$VERSION" -o reproducibility/model_accuracy/
done
```

The cluster job for the rerun of 2026-09-25 is `scripts_synapses/synapse_detect_ihc11_F1val.sbatch`.

`run_evaluation.py` itself has no IHC dependency; it scores whatever
`synapse_detection_filtered.tsv` the prediction step produced. The IHC segmentation is predicted
per crop by `prediction.py` from the `raw_ihc` channel, and detections further than 3 um from it
are dropped.

**That IHC model changed with this evaluation.** `prediction.py` pinned the v4 IHC network and
now uses `v11_cochlea_distance_unet_IHC_supervised_2026-07-20`, which is the IHC version the rest
of the repository treats as current. Precision depends on the IHC segmentation, so **all 17
entries written into `synapses.json` before 2026-09-24 were scored against a v4 mask and are not
comparable to `v7` and `v8`.** Re-run prediction and evaluation for every baseline the comparison
needs. Pass `--model_ihc` to score against the v4 model again.

The rescored baselines have their own keys, `v3-ihc11` and `v5-ihc11`. The plain `v3` and `v5`
entries stay on IHC v4, because Figure 2c and Supplementary Figure 2 compare them with other
IHC v4 entries. A first rescore overwrote `v3`, which mixed the two eras in Figure 2c. Since
then, `run_evaluation.py` refuses to replace an existing entry without `--overwrite`. `v6-1` is
still open: it has no entry in `synapses.json` yet.

`v7` and `v8` are also not a single step from `v6-1`: it ran with no sampler and redrawn
validation patches. Its missing raw normalization makes no difference. And the pair is not a clean ablation of the mask
alone, see `to-do_revision/scripts_synapses/README.md`.

### v7 against v8, and the candidate-exclusion mask (2026-09-25)

The numbers below come from the 2026-09-24 predictions in `production_2026-09-24/`. The
per-crop figures were matched against the IHC-filtered consensus points, so they differ
slightly from `synapses.json`.

**v8 collapses in precision, not in recall.** Summed over the six test crops, v8 finds 945 true
positives against 918 for v7, but it produces 562 false positives against 28.

| Evidence | v7 | v8 |
|---|---|---|
| Peaks before the IHC filter | 1,012 | 3,146 (v8-latest: 4,491) |
| 99th percentile of the heatmap | 0.008–0.011 | 0.055–0.086 |
| Heatmap value at a false positive, median | 1.0–1.5 | 0.68–0.80 |
| Distance from a false positive to the nearest annotation, median | 2–8 um | 19–64 um |
| False positives within 10 voxels of a zero-valued voxel | 0–50 % | 34–94 %, 76 % or more in four crops |
| False positives within 3 um of a single-annotator point | 22 / 33 | 27 / 569 |
| Best F1 of a threshold sweep over the detections | 0.872 at 0.5 | 0.826 at 1.0 |

1. **The cube mask removes the negative supervision far from the annotations.** For five of the six test
   crops, 35–78 % of the crop is zero, from the reslicing. The borders between tissue and zero
   and the dim tissue are never inside a cube, so v8 never learned to predict zero there. Most
   false positives lie on these borders. Removing the detections within 5 voxels of a zero
   voxel raises v8 precision to 0.89 and leaves v7 at 0.97.
2. **The other v8 false positives are the unannotated CTBP2 spots.** There are 116 against 29
   for v7. Their median raw intensity is 771, against 1,148 at the true positives, and 68 % lie
   outside the IHC but within the 3 um filter distance. The mask worked as intended, but the
   IHC filter does not remove spots close to the IHC.
3. **The mask did not buy recall.** At matched precision, v8 has the lower recall: 0.735 at
   precision 0.94 (threshold 1.25), against 0.795 at precision 0.965 for v7. The v7 false
   positives are mostly points that one annotator marked, so v7 is close to the annotation
   ceiling. The hypothesis that the unannotated spots suppress recall is not supported.
4. **The masked metric could not see the failure.** It ignores every voxel outside the cubes,
   so `best.pt` of v8 was selected blind to the false positives, and `latest.pt` is worse again.
   The fixed validation patches do not help here: a masked recipe needs a detection-level
   metric, or a mask that covers almost all of the patch.

**Fixed: the patch sampling.** `DetectionDataset._sample_bounding_box` drew the patch start
from `[0, shape - patch_shape - 2 * halo)`, although the halo is clamped when the patch is
loaded. The last `2 * halo + 1` voxels of every axis never reached training: 33 for v8, 21 for
the flow models such as v5, and 1 for v7.

**v9: the candidate-exclusion mask.** `--ignore_percentile P` kept the loss on every voxel
except a cube of half width 3 voxels around each unannotated CTBP2 candidate. A candidate was a
local maximum of the smoothed crop that is more than 4 voxels from every annotation and at least
as bright as the `P`th percentile of the maxima at the annotations. v9 kept the data, the split
and the sampler of v7. It was trained after the patch sampling fix, so its fixed validation
patches are other random draws than those of v7. The option was removed on 2026-10-21. Check out
commit `4da9e70` to run the command again:

```bash
python $SCRIPT_DIR/train_synapse_detection.py -v v7 -m v9 --random_state 42 -s $SAVE_ROOT \
    --sampler minpoint --ignore_percentile 10
```

### v9 and the recall limit (2026-10-01)

| Entry | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| `v3-ihc11` | 0.961 | 0.794 | 0.869 | 912 | 37 | 237 |
| `v5-ihc11` | 0.962 | 0.821 | 0.886 | 943 | 37 | 206 |
| `v7` | 0.970 | 0.799 | 0.876 | 918 | 28 | 231 |
| `v7-latest` | 0.972 | 0.815 | 0.886 | 936 | 27 | 213 |
| `v9` | 0.974 | 0.794 | 0.875 | 912 | 24 | 237 |
| `v9-latest` | 0.974 | 0.792 | 0.874 | 910 | 24 | 239 |

**v9 gives no gain.** All six entries lie within the seed spread of one recipe: the v3 seeds
range from F1 0.782 to 0.872, with an SD of 0.016 to 0.031. So no entry is significantly better
than another.

**Neither error type depends on the IHC context.** The precision is at the annotation ceiling:
most v7 false positives are points that one annotator marked, see "v7 against v8". The recall
is limited by adjacent synapses that the detector merges into one peak. For the 193 false
negatives of `v7-latest`, matched against the IHC-filtered consensus points:

- 94 % have a detection within 3 um, which the matching assigned to a neighbouring annotation;
- the nearest other consensus point is closer than 1.5 um for 115, at 1.5 to 2 um for 54, at
  2 to 3 um for 18, and further for 6;
- 179 have a heatmap value of at least 0.5 within 4 voxels, so the network responds, but with
  one peak for two synapses;
- the IHC filter removes none of them. 26 of the 1,149 consensus points (2.3 %) lie further
  than 3 um from the IHC segmentation and cannot be found.

Adjacent synapses are common: 42 % of the consensus points have a neighbour within 2 um. Of
the 126 pairs closer than 1.5 um, both points are found for 11, one for 101 and none for 14. A
consensus point needs two annotators within 2 um (`consensus_annotations.py`), so the pairs are
real. An IHC channel as a second input, or an IHC-derived loss weight, cannot separate them.

**Recommendation: keep the v7 recipe as the standard.** That is heatmap only, the
`MinPointSampler` and the fixed validation patches, without a loss mask. `v7-latest` has the
best F1 of the new runs, equal to `v5-ihc11`, without the flow channels. The loss of v7 is the
heatmap mean squared error of v3, and its raw normalization is a no-op. So against `v6-1`, v7
changes the sampler and the validation patches. That v7 reaches the recall of v5 suggests that
the gain of v5 over `v6-1` came from the sampler and not from the flow. Two checks are open:
`v6-1` has no IHC v11 score, and that the `v7` data folder equals the v5/v6 data is not shown.

**A free gain: `min_distance` 1 instead of 2.** The peak detection suppresses every peak within
`min_distance` voxels of a higher one. Detected again with the production code on the existing
heatmaps, against the IHC-filtered consensus points:

| Entry | `min_distance` 2 (P / R / F1) | `min_distance` 1 (P / R / F1) |
|---|---|---|
| `v7-latest` | 0.966 / 0.809 / 0.881 | 0.961 / 0.836 / 0.894 |
| `v7` | 0.965 / 0.795 / 0.872 | 0.961 / 0.818 / 0.884 |

These values come from the test crops, so they must not select the setting. Select the
threshold and `min_distance` on the validation crops of the training run, then score the result
once on the test crops under a key of its own:

```bash
ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
MODEL=$ROOT/trained_models/Synapses/synapse_detection_v7-latest.pt
python -m flamingo_tools.synapse_detection.gridsearch -m $MODEL \
    -s $ROOT/training_data/synapses/training_data/v7/train_val_split.json
# With the selected values, here min_distance 1 and threshold 0.5:
OUT=$ROOT/predictions/val_synapses/production_2026-10-01/v7-latest_md1
python scripts/validation/synapses/prediction.py -o $OUT --min_distance 1 --threshold 0.5 \
    -i $ROOT/training_data/synapses/test_data/v5/images -g $ROOT/training_data/synapses/test_data/v5/labels \
    --model_synapse $MODEL --model_ihc $ROOT/trained_models/IHC/v11_cochlea_distance_unet_IHC_supervised_2026-07-20
python scripts/validation/synapses/run_evaluation.py -p $OUT -o reproducibility/model_accuracy/ \
    -c $ROOT/training_data/synapses/test_data/v5/images \
    -r $ROOT/AnnotatedImageCrops/Synapses_2026-04/consensus_annotation
```

`run_evaluation.py` without `--version` stores the result under the folder name,
`v7-latest_md1`. The whole-cochlea entry points still detect with `min_distance` 2. Pass the
option on to them once the validation selects another value.

**v10: a narrower target.** For two annotations 3 voxels (1.1 um) apart, the sigma-1
target falls only to 73 % of the peak between them, and the trained heatmap merges such pairs.
With sigma 0.7 it falls to 38 %. v10 was trained with this command. For the result, see the next
section:

```bash
python $SCRIPT_DIR/train_synapse_detection.py -v v7 -m v10 --random_state 42 -s $SAVE_ROOT \
    --sampler minpoint --sigma 0.7
```

A narrower target has fewer positive voxels, which can lower the recall instead. Score v10 at
the `min_distance` that its own validation selects.

### v10 and the narrower target (2026-10-21)

v10 is the v7 recipe with `--sigma 0.7`. Like v9, it was trained after the patch sampling fix.
The production scores from `synapses.json` (IHC v11):

| Entry | Precision | Recall | F1 | TP | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| `v7` | 0.970 | 0.799 | 0.876 | 918 | 28 | 231 |
| `v7-latest` | 0.972 | 0.815 | 0.886 | 936 | 27 | 213 |
| `v10` | 0.986 | 0.789 | 0.877 | 907 | 13 | 242 |
| `v10-latest` | 0.985 | 0.789 | 0.876 | 907 | 14 | 242 |

**At the production setting, v10 gives no gain.** It halves the false positives, but it loses 11
to 29 true positives. F1 does not change.

The diagnostics below detect the peaks again on the stored heatmaps in `production_2026-09-25/`
with the production code. They match the detections at 3 um against the 1,123 IHC-filtered
consensus points. At threshold 0.5 and `min_distance` 2, the detection reproduces the stored
detections exactly. The recall divides by 1,123. The `min_distance` table of 2026-10-01 divides
by all 1,149 points, so its recall is about 0.02 lower for the same detections. These values come
from the test crops. They describe the models but do not select a setting.

1. **The narrow target does not separate the adjacent synapses.** This was the purpose of v10.
   There are 131 consensus pairs closer than 1.5 um. The table counts the pairs with both points
   found at threshold 0.5. The last column gives the pairs with a separate local maximum in the
   heatmap for each point. Each maximum has a value above 0.1 and lies within 1.5 um of its point.

   | Entry | Both found, `min_distance` 2 | Both found, `min_distance` 1 | Two heatmap maxima |
   |---|---:|---:|---:|
   | `v7-latest` | 11 | 22 | 20 % |
   | `v10-latest` | 9 | 20 | 24 % |

   A probable reason is that the CTBP2 signal itself does not resolve these pairs. A narrower
   target cannot add that information. This reason is not tested.
2. **The narrow target lowers the heatmap values, so threshold 0.5 is strict for v10.** For 19 %
   of the v10 false negatives, no heatmap value within 4 voxels reaches 0.5. For `v7-latest`,
   this is 7 %. A threshold sweep from 0.05 to 2.0 gives:

   | Entry | Best F1 (threshold, `min_distance`) | Recall at precision >= 0.97, `min_distance` 2 | Recall at precision >= 0.98, `min_distance` 2 |
   |---|---|---|---|
   | `v7` | 0.903 (0.35, 1) | 0.790 (0.75) | 0.754 (1.1) |
   | `v7-latest` | 0.909 (0.35, 1) | 0.812 (0.65) | 0.792 (0.9) |
   | `v10` | 0.908 (0.20, 1) | 0.825 (0.30) | 0.810 (0.45) |
   | `v10-latest` | 0.909 (0.20, 1) | 0.831 (0.25) | 0.808 (0.45) |

   At its best threshold, v10 has the F1 of `v7-latest`. At matched high precision, `v10-latest`
   finds about 20 more synapses than `v7-latest`, which is 2 % recall. The gap to `v7` is larger,
   but `v7` is the weaker export of that run. This is the only advantage. It is smaller than the
   seed spread of one recipe (F1 SD 0.016 to 0.031 for the v3 seeds), and each recipe has one seed.
3. **`min_distance` 1 helps every model by about the same amount.** At threshold 0.5, F1 increases
   by 0.008 to 0.013: from 0.892 to 0.905 for `v7-latest`, and from 0.885 to 0.893 for
   `v10-latest`. The narrow target does not make `min_distance` 1 more useful.

**Recommendation: keep sigma 1 and the v7 recipe.** v10 has no advantage at the F1 optimum, and
it does not remove the recall limit that it was made for. If an application needs a
high-precision operating point, v10 at a lower threshold is an option. Select that threshold with
the gridsearch on the validation crops. `min_distance` 1 for `v7-latest`, selected on the
validation crops as shown above, is still the open gain.

### Target defect and v11 (2026-10-21)

The Gaussian target of every model up to v10 has two defects in `CsvHeatmapTransform`:

1. **The peak height depends on the patch.** The target is smoothed with the `nearest` border
   mode and divided by its patch maximum. An annotation on a face voxel of the loaded box gets
   a 1.75 times higher value, so every other peak of that patch falls from 4 to 2.28 (two faces:
   1.30, a corner: 0.74). On the 29 local v5/v7 training crops with the v7 sampler, the median
   peak of a patch is 2.28 in 80 to 85 % of the patches.
2. **Annotations just outside the patch lose their Gaussian.** The heatmap-only transform loads
   no halo since #133. In 92 % of the patches an annotation lies up to 2 voxels outside the
   patch, and the patch border is trained as 0 where its Gaussian reaches 2.43 (1 voxel) or 0.54
   (2 voxels).

Close pairs do not matter: only 1.3 % of the annotations have a neighbour within 2 voxels.

`--fixed_target` smooths with the `constant` mode, scales every isolated peak to exactly 4, and
loads a halo of `ceil(3 * sigma)` voxels. The default stays the old target, so v7 and v10 remain
retrainable. With the flag, the median peak is 4.0 in every patch of the local crops. The effect
on F1 is not known. Superseded: v11 is now the refined recipe with three seeds, see
"Rescoring result (2026-10-23)". The single-variable v11 proposed here was the v7 recipe with this
one change:

```bash
python $SCRIPT_DIR/train_synapse_detection.py -v v7 -m v11 --random_state 42 -s $SAVE_ROOT \
    --fixed_target
```

Its heatmap peaks near 4 instead of about 2.3, so threshold 0.5 is not comparable. Select the
threshold and `min_distance` with `python -m flamingo_tools.synapse_detection.gridsearch` on the
validation crops (default grid 0.05 to 2.95; on the consensus val crops with `--ihc_dir`, see
"Training robustness"), score once on the test crops, and then register
`v11` in `PREDICTION_DICT` and `_PRODUCTION_VERSIONS`.

### Training robustness (2026-10-21)

Every recent difference is smaller than the seed spread, and each recent recipe has one seed. So
no recipe change can be judged. This section collects the causes found before a seed study of
the next recipe. The measurements use the local copies of the crops and of the checkpoints.

**The training labels mark close pairs 2-3x less often than the consensus.** Share of the points
with a neighbour closer than 1.5 um and closer than 2 um:

| Labels | Points | < 1.5 um | < 2 um |
|---|---:|---:|---:|
| Consensus, IHC-filtered (six test crops) | 1,123 | 20.7 % | 42.4 % |
| The test annotators AMD, EK, LR, one at a time | 1,098-1,274 | 15-27 % | 42-48 % |
| Training: `*IHCribboncount*` and `*IHCribbons*` (9 crops) | 2,666 | 24.6 % | 50.3 % |
| Training: MLR226R, `mlr226r_*` and `m226r_*` (9 crops) | 1,558 | 9.7 % | 31.7 % |
| Training: MLR227L, `227l_*` and `mlr227-l_*` (8 crops) | 1,701 | 5.9 % | 26.2 % |
| Training: CR, `m78l_*` (3 crops) | 654 | 7.3 % | 29.1 % |
| v7-latest detections at threshold 0.5, `min_distance` 2 | 963 | 2.3 % | 24.1 % |

- The consensus is not denser than one annotator: 1,149 consensus points against 1,138 (AMD),
  1,098 (EK) and 1,274 (LR) on the same crops. So the 20 sparse training crops are the outlier.
  This is also true for MLR226R, the other ear of the test animal.
- The network gets conflicting targets for close pairs: dense in 9 crops, sparse in 20. This
  agrees with the recall limit in "v9 and the recall limit".
- All three val crops of v5 to v10 (`mlr227-l_0973…`, `MLR227-L_0403…`, `m78l_apexp2718…`) are
  in the sparse groups. `m78l_apexp2718_cr-ctbp2` (CR stain, 34 points) also supplies a third
  of the 160 val patches.
- Hypothesis (Martin): this mismatch makes a gridsearch on the val crops unreliable for the test
  crops. `scripts_synapses/rescore_operating_point.sbatch` tests it, see below.

**The models are at human level on the test crops.** As reference, take the points that the
other two annotators agree on (2 um), and score at 3 um after the IHC filter. The held-out
annotator gets F1 0.871 to 0.916 (mean 0.901). v7-latest gets 0.897 to 0.918 (mean 0.909), and
v5-ihc11 0.893 to 0.919. The models have the higher precision and the lower recall. The test
set has little headroom.

**The test set is precise, but it shows one cochlea.** For two fixed models, the paired crop
bootstrap of the F1 difference has a 95 % interval of about +-0.01 (v7-latest - v10-latest:
+0.007 [-0.004, 0.018]; v5-ihc11 - v3-ihc11: +0.019 [0.010, 0.026]). That is smaller than the
seed spread. All six crops come from MLR226L, and 9 training crops probably come from MLR226R.
So there is no animal-level independence that a consensus val crop could break.

**The learning rate collapses early in every run.** The trainer uses the torch_em default
`ReduceLROnPlateau(factor 0.5, patience 5)`, and an epoch has 100 iterations. Every `latest.pt`
ends at the floor of 1.22e-8 (1e-4 / 2^13). From the checkpoints:

| Run | `best.pt` epoch | Learning rate at `best.pt` | Floor reached |
|---|---:|---:|---|
| v7 | 58 | 1.25e-5 | at epoch 118 or earlier |
| v9 | 72 | 3.9e-7 | at epoch 102 or earlier |
| v10 | 95 | 1.6e-6 | at epoch 137 or earlier |
| v8 | 10 | 1e-4 | |
| v5 | 204 | 1.22e-8 | at epoch 204 or earlier |
| v5-rep2, v5-rep4 | 13, 28 | 1e-4, 2.5e-5 | |

After `best.pt`, the metric does not improve, so the learning rate halves every 6 epochs. About
86 % of the 1e5 iterations (about 27 of 32 A100 hours) run at 1.2e-8. The final weights depend
on when a noisy plateau detector fires, which differs between seeds. The `best.pt` and
`latest.pt` heatmaps of v9 correlate at 0.998, against 0.81 to 0.86 between runs.
`--lr_schedule cosine` decays the learning rate to zero over the whole run, independent of the
metric. Use `latest.pt` with it. The default stays the plateau schedule, so v7 to v10 stay
retrainable.

**The `v5-f1…f4` scores mix split and seed.** The exports
`synapse_detection_model_v5-f{1..4}.pt` are identical to `v5-fold{1..4}/best.pt`. Each fold has
other val crops, for example `4.1L_apex`, `6.2R_base` and `m78l_base` for fold 1. The checkpoint
folders `v5-f1…f4` are seed replicates with the v5 split. Their exports were never scored. They
are `v5-rep1…4` here, to keep the names apart.

**The operating point differs between runs.** The best threshold on the test crops is 0.20 for
v10, 0.35 for v7, 0.45 for v3 and 0.65 for v5. At the fixed threshold 0.5, each run is at
another point of its precision-recall curve. The v3 seed spread is mostly recall (0.66 to 0.80
at precision 0.95 to 0.98).

**A new split.** The paper is published, so two consensus crops become val crops:

- val: `MLR226L_Vglut3Ctbp2_apex1_p50-140`, `MLR226L_Vglut3Ctbp2_base1_reslice_p122-256`
  (416 points);
- test v11: `…apex2_p1-134`, `…mid1_resliced_p70-185`, `…mid2_p65-195`, `…base2_p40-130`
  (733 points);
- the three old val crops go back to training (29 training crops).

Code changes for the split:

- `read_csv_points` keeps only the `consensus` rows of a consensus CSV.
- `prediction.py` and `run_evaluation.py` pair each image with `<ref_root>/<crop>.csv`, so a
  folder with a subset of the crops scores that subset.
- `train_synapse_detection.py --split` takes a fixed split file.
- `gridsearch --ihc_dir` filters the detections by their distance to the IHCs, as the
  evaluation does (see the result below).
- `prediction_impl` loads a GPU checkpoint on a CPU node. On the local CPU, one small crop takes
  about 1 hour, so run the predictions on the cluster.

Use `consensus_annotation/` (1,149 points) as reference, not `test_data/v5/labels`, which is an
older copy with 1,133 consensus points.

Set up the data on the cluster:

```bash
ROOT=/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet
DATA=$ROOT/training_data/synapses
REF=$ROOT/AnnotatedImageCrops/Synapses_2026-04/consensus_annotation
VAL="MLR226L_Vglut3Ctbp2_apex1_p50-140 MLR226L_Vglut3Ctbp2_base1_reslice_p122-256"
TEST="MLR226L_Vglut3Ctbp2_apex2_p1-134 MLR226L_Vglut3Ctbp2_mid1_resliced_p70-185 \
MLR226L_Vglut3Ctbp2_mid2_p65-195 MLR226L_Vglut3Ctbp2_base2_p40-130"

# The open check: the v7 data equals the v5 data.
diff <(cd $DATA/training_data/v5 && find images labels -type f | sort | xargs md5sum) \
     <(cd $DATA/training_data/v7 && find images labels -type f | sort | xargs md5sum) && echo "v7 == v5"

mkdir -p $DATA/training_data/v11/images $DATA/training_data/v11/labels \
    $DATA/test_data/v11/images $DATA/test_data/v11/labels
ln -s $DATA/training_data/v7/images/*.zarr $DATA/training_data/v11/images/
ln -s $DATA/training_data/v7/labels/*.csv $DATA/training_data/v11/labels/
for n in $VAL; do
    ln -s $DATA/test_data/v5/images/$n.zarr $DATA/training_data/v11/images/
    ln -s $REF/$n.csv $DATA/training_data/v11/labels/
done
for n in $TEST; do
    ln -s $DATA/test_data/v5/images/$n.zarr $DATA/test_data/v11/images/
    ln -s $REF/$n.csv $DATA/test_data/v11/labels/
done
# Writes the same file as the local synapses_data/train_val_split.json.
python -c "
import json, os, sys
root, val = sys.argv[1], sys.argv[2:]
train = sorted(f[:-5] for f in os.listdir(root + '/images') if f[:-5] not in val)
json.dump({'train': train, 'val': val}, open(root + '/train_val_split.json', 'w'), indent='\t')
" $DATA/training_data/v11 $VAL
```

Then pull this branch on the cluster, check the checkpoint paths in
`scripts_synapses/rescore_operating_point.sbatch`, and submit it. For every model it selects the
setting on the consensus val crops and on the old val crops, and scores test v11 at (0.5, 2) and
at both selected settings.

**First result, from the stored heatmaps of the 8 production models.** The setting is selected
on the 2 consensus val crops, and test v11 is scored with the production IHC filter and matching.
This scoring reproduces `synapses.json` exactly: on all six crops, every model gets its stored
F1, and v7-latest its per-crop counts.

| Model | Test v11 at (0.5, 2) | Gridsearch, radius filter: setting, test F1 | Gridsearch, `--ihc_dir`: setting, test P / R / F1 | Best test F1 (diagnostic) |
|---|---:|---|---|---:|
| v3-ihc11 | 0.862 | (0.50, 1), 0.876 | (0.50, 1), 0.964 / 0.802 / 0.876 | 0.879 |
| v5-ihc11 | 0.878 | (0.35, 1), 0.886 | (0.50, 1), 0.955 / 0.834 / 0.890 | 0.893 |
| v7 | 0.869 | (0.15, 1), 0.859 | (0.20, 1), 0.889 / 0.872 / 0.880 | 0.890 |
| v7-latest | 0.881 | (0.35, 1), 0.896 | (0.35, 1), 0.951 / 0.847 / 0.896 | 0.898 |
| v9 | 0.875 | (0.15, 1), 0.785 | (0.55, 1), 0.978 / 0.802 / 0.882 | 0.885 |
| v9-latest | 0.874 | (0.20, 1), 0.851 | (0.40, 1), 0.971 / 0.813 / 0.885 | 0.885 |
| v10 | 0.873 | (0.10, 1), 0.873 | (0.15, 1), 0.956 / 0.853 / 0.901 | 0.903 |
| v10-latest | 0.873 | (0.10, 1), 0.888 | (0.10, 1), 0.907 / 0.869 / 0.888 | 0.905 |

- **The default gridsearch is unreliable on the consensus crops.** Its radius filter scores
  only the detections within 40 voxels of an annotation. This hides the false positives far
  from the annotations, so it selects thresholds of 0.10 to 0.20 and loses up to 0.09 F1 (v9).
  Without a filter it selects 0.6 to 0.8, which is too high. Only the IHC filter of the
  evaluation selects a setting that transfers to the test crops.
- **With `--ihc_dir`, the selection works.** The gridsearch then filters the detections by
  their distance to the IHCs, as the evaluation does. Every model gains 0.007 to 0.028 F1 over
  (0.5, 2) and is within 0.017 (median 0.003) of its best test F1. Every model selects
  `min_distance` 1.
- **The selection does not shrink the spread between these runs.** At the selected setting the
  8 models span F1 0.876 to 0.901 (SD 0.008), at (0.5, 2) 0.862 to 0.881 (SD 0.006). The ranking
  changes: v10 moves from the middle to the top.
- The close-pair hypothesis needs the old val crops, which have no stored heatmaps here. The
  cluster job scores them with the former practice (radius filter). Note that the radius filter
  can make that gridsearch unreliable by itself, not only the sparse close pairs.

### Rescoring result (2026-10-23)

`rescore_operating_point.sbatch` ran on 2026-10-22 on checkout `4da9e70` in the `micro-sam_gpu`
env. The local copy is `rescore_2026-10-21/` in the repository root. Most of the job failed:

- The `cons` arm failed for every model. `4da9e70` has no `gridsearch --ihc_dir` and no
  consensus-only `read_csv_points`.
- Every *export* (v3, v3-1…4, v5, v5-fold1…4, v7, v9, v10 and their `-latest`) failed. Each
  wrote an empty `predictions.zarr` and gave no gridsearch setting. The trainer checkpoints
  (`v5-latest`, `v5-rep1…4` best and latest) ran. The log `slurm-16801815.out` gives two causes:
  - The exports are pickled U-Nets. torch_em 0.10.7 and later reads
    `Decoder.explicit_skip_channels`, which a U-Net pickled with 0.10.6 or earlier does not
    have (`AttributeError`). `micro-sam_gpu` resolves to `micro-sam2` with torch_em 0.10.8.
    Locally, the v5-f1 export runs with 0.10.1 and 0.10.6 and fails with 0.10.7 and 0.10.8.
    The same is true for the pickled `get_model` files of this branch. On master, `get_model`
    loads the BioImage.io state-dict weights and is not affected. A trainer checkpoint builds
    a new model, so it is not affected either. `environment.yaml` allows `torch_em >=0.9`.
  - `synapse_detection_model_v3-{1..4}.pt` are not readable for the job account
    (`PermissionError`). They are removed from the rescoring.
- After the first failure, `IHC_DONE` pointed to a run without IHC output, and every later run got
  a broken `*_ihc` link.
- `test_data/v6/images` on the cluster has all six crops. The test v11 values below add up the
  per-crop counts of the four test crops.

The scored part is valid. The (0.5, 2) arm uses the prediction path of HEAD, and per crop
TP + FN equals the consensus counts (1,149 points). The `old` arm used the old gridsearch (grid 0.3
to 1.9, border handling on the unpadded crop). All its selected thresholds are inside that grid.
The script is fixed (HEAD, `cochlea-net` with torch_em 0.10.6 or earlier, 12 h, `IHC_DONE` only
from a run with IHC output, output root `rescore_2026-10-23`, test v11 with 4 crops, no v3-1…4).
The rerun must reproduce the 18 existing entries at the test v11 values below.

**Selection on the old val crops loses F1.** Test v11, F1:

| Run | (0.5, 2) | Old-selected (setting) | Change |
|---|---:|---:|---:|
| v5-latest | 0.881 | 0.877 (0.8, 2) | -0.004 |
| v5-rep1 / -latest | 0.840 / 0.841 | 0.825 / 0.827 (0.4, 2) | -0.015 / -0.014 |
| v5-rep2 / -latest | 0.852 / 0.858 | 0.816 (1.0, 1) / 0.833 (0.9, 2) | -0.036 / -0.025 |
| v5-rep3 / -latest | 0.876 / 0.877 | 0.844 / 0.845 (1.1, 2) | -0.032 / -0.032 |
| v5-rep4 / -latest | 0.860 / 0.877 | 0.843 / 0.880 (0.9, 1) | -0.017 / +0.003 |

In 8 of 9 runs, the old val crops select a threshold that is too high (precision 0.97 to 0.99, lower
recall). The mean change is -0.019. The consensus val crops with `--ihc_dir` gave +0.007 to +0.028
on the 8 production models (see above). This agrees with the close-pair hypothesis. Do not select
a setting on the old val crops.

**Seed spread of the v5 recipe.** v5 and v5-rep1…4 at (0.5, 2) on test v11: `best.pt` 0.840 to
0.878 (SD 0.016), `latest.pt` 0.841 to 0.881 (SD 0.017). On all six crops the SD is 0.030.
v5-rep1 has low recall (0.75) and is the outlier.

**`latest.pt` is never worse than `best.pt`.** Test v11, F1 at (0.5, 2), with the learning rate at
the `best.pt` epoch from the local checkpoints:

| Run | `best.pt` epoch | Learning rate | `best.pt` | `latest.pt` |
|---|---:|---:|---:|---:|
| v5-rep2 | 13 | 1e-4 | 0.852 | 0.858 |
| v5-rep4 | 28 | 2.5e-5 | 0.860 | 0.877 |
| v7 | 58 | 1.25e-5 | 0.869 | 0.881 |
| v10, v9, v5, v5-rep3, v5-rep1 | 95, 72, 204, 109, 278 | 1.6e-6 or less | | change -0.001 to +0.003 |

**Learning rate and early stopping.**

- The decay helps. When `best.pt` is from a high learning rate, the annealed `latest.pt` gains
  0.006 to 0.017. Below about 1e-6 nothing changes. The extra epochs confound this comparison.
- The plateau trigger does not help. It follows the noisy val MSE, reaches the floor at a
  seed-dependent epoch (102 to 204), and leaves about 86 % of the iterations at 1.2e-8.
- Early stopping (torch_em `early_stopping`, a patience of 80 epochs or more) would stop at the
  floor and give the same `latest.pt` at 20 to 30 % of the compute. But it keeps the dependency
  on the val MSE, which does not rank detectors (Spearman 0.39; the v5-rep2 `best.pt` at epoch
  13 is its weaker export).
- Use a fixed cosine decay over 3e4 iterations (300 epochs) and `latest.pt`. This covers the
  latest `best.pt` epoch (278) and the latest floor (204). The v3 runs with 1e4 iterations (F1
  0.80 to 0.84) did not differ from those with 1e5 (0.78 to 0.87). A run takes about 10 instead
  of 32 A100 hours.

**v11 is now the refined recipe, with three seeds.** The v7 recipe with `--fixed_target`,
`--lr_schedule cosine -n 30000` and the consensus val split above (the three old val crops go back
to training). This replaces the single-variable v11 of "Target defect and v11". The test set has
little headroom (human level, bootstrap +-0.01), so the purpose is a recipe without the known
defects: deterministic, about 3 times cheaper, and with a threshold that transfers. Three seeds can
show a large change of the spread, but not a small one. Train with
`scripts_synapses/train_synapse_v11.sbatch` (array `v11-1…3`).

Evaluation of each seed on `latest.pt`, with rules that are fixed before the results:

1. Select the setting with
   `python -m flamingo_tools.synapse_detection.gridsearch -m <ckpt>/latest.pt -s $DATA/training_data/v11/train_val_split.json -i $DATA/training_data/v11/images -l $DATA/training_data/v11/labels --ihc_dir $ROOT/predictions/val_synapses/production_2026-09-25/v7`.
2. Score test v11 at the selected setting, and once at the median setting of the three seeds.
   `scripts_synapses/evaluate_synapse_v11.sbatch` runs rules 1 and 2.
3. Baseline: v5, v5-rep1…4, v7 and v7-latest at their `cons` setting, see "Rescoring rerun
   (2026-10-26)": mean F1 0.874, worst seed 0.835 (v5-rep1).
4. Make the recipe the default when all of these are true:
   - the mean F1 is at least the baseline mean minus 0.01;
   - no seed is below the worst baseline seed;
   - the selected thresholds of the three seeds agree.
5. Choose the released seed by its val-crop F1, not by its test F1.

### Rescoring rerun (2026-10-26)

The rerun of `rescore_operating_point.sbatch` at HEAD in `cochlea-net` (local copy
`rescore_2026-10-23/`) scored only v3, v5, v5-latest and v5-rep1. At about 16:48 the code in the
cluster checkout changed:

- The entry `v5-rep1-old` (F1 0.002) is invalid. Its crops are apex1, apex2, base1 and base2: the
  4 test predictions were paired by index with the first 4 of the 6 reference CSVs, as
  `run_evaluation.py` on master does. Its predictions are identical to `v5-rep1-cons` (same
  setting), so its true F1 is 0.835.
- From v5-rep2 on, every run is empty, also at (0.5, 2). Master has no `prediction.py --threshold`
  and no `synapse_detection/gridsearch.py`.
- Probable cause: the checkout changed to master while the job ran. Do not switch the branch of the
  cluster checkout while a job runs.

The scored part is valid. v3 and v5 select (0.50, 1) and get the P/R/F1 of the stored-heatmap
analysis above. The per-crop counts of v5-latest (fixed and old) and v5-rep1 (fixed) equal those of
2026-10-21. The exports run, so the torch_em fix works.

**Local completion.** The 2026-10-21 copy has the heatmaps of all six crops for v5-latest,
v5-rep1…4 and v5-rep1…4-latest. The selection runs the gridsearch with `--ihc_dir` on these
heatmaps. The test score uses the production peak detection (with the flow correction), the
padding drop, the IHC filter and `run_evaluation.py`. This reproduces the cluster values of
`v5-latest-cons` and `v5-rep1-cons` per crop. The gridsearch scores the peaks without the flow
correction of the v5 models. For v5-rep1 this changes one match on the test crops.

Test v11, F1 (old: rows v3 to v5-rep1 from the rerun, the other rows from 2026-10-21 with the old
gridsearch):

| Run | (0.5, 2) | `cons`: setting, P / R / F1 | `old`: setting, F1 | cons - fixed |
|---|---:|---|---|---:|
| v3 | 0.862 | (0.50, 1), 0.964 / 0.802 / 0.876 | (0.90, 2), 0.851 | +0.014 |
| v5 | 0.878 | (0.50, 1), 0.955 / 0.834 / 0.890 | (0.75, 2), 0.876 | +0.012 |
| v5-latest | 0.881 | (0.50, 1), 0.949 / 0.838 / 0.890 | (0.80, 2), 0.877 | +0.009 |
| v5-rep1 | 0.840 | (0.45, 2), 0.940 / 0.752 / 0.835 | (0.45, 2), 0.835 | -0.005 |
| v5-rep2 | 0.852 | (0.55, 1), 0.924 / 0.797 / 0.856 | (1.0, 1), 0.816 | +0.004 |
| v5-rep3 | 0.876 | (0.45, 1), 0.932 / 0.846 / 0.887 | (1.1, 2), 0.844 | +0.011 |
| v5-rep4 | 0.860 | (0.50, 1), 0.944 / 0.808 / 0.871 | (0.9, 1), 0.843 | +0.011 |
| v5-rep1-latest | 0.841 | (0.40, 2), 0.920 / 0.752 / 0.827 | (0.4, 2), 0.827 | -0.014 |
| v5-rep2-latest | 0.858 | (0.45, 1), 0.936 / 0.799 / 0.862 | (0.9, 2), 0.833 | +0.004 |
| v5-rep3-latest | 0.877 | (0.40, 1), 0.910 / 0.854 / 0.881 | (1.1, 2), 0.845 | +0.004 |
| v5-rep4-latest | 0.877 | (0.45, 2), 0.927 / 0.831 / 0.876 | (0.9, 1), 0.880 | -0.001 |

- **The old val crops select too high a threshold.** The old selection is below (0.5, 2) in 10 of
  11 runs and below the cons selection in 8 (2 ties). The mean change from old to cons is +0.020.
- **The cons selection is not always a gain.** It loses in 3 of the 8 v5 replicate runs (up to
  -0.014 for v5-rep1-latest), and the gain is +0.004 on average over the 11 runs. On the 8 models
  of the stored-heatmap analysis it was +0.007 to +0.028. With 2 val crops (416 points), the
  selection can cost up to 0.014.
- **The selection widens the seed spread of the v5 recipe.** v5 and v5-rep1…4: `best.pt` SD 0.016
  at (0.5, 2), 0.023 at cons; `latest.pt` 0.017 and 0.025. The weak seeds lose, the strong seeds
  gain. Only v5-rep1 (best and latest) and v5-rep4-latest select `min_distance` 2. The other 14
  models select 1.
- **`latest.pt` and `best.pt` are equal on average at cons** (mean 0.867 and 0.868).
- **Baseline for rule 3:** v5 0.890, v5-rep1 0.835, v5-rep2 0.856, v5-rep3 0.887, v5-rep4 0.871,
  v7 0.880, v7-latest 0.896 (the last two from the stored-heatmap analysis). Mean 0.874, so rule
  4 needs a v11 mean of 0.864 or more and no seed below 0.835.

**Completion on the cluster (2026-10-27).** The job ran again for all models. The local copy
`rescore_2026-10-23/` now has its `synapses.json` and `selected_settings.txt`.

- The `cons` rows of v7 to v10 reproduce the stored-heatmap analysis exactly.
- For v7 to v10, the old val crops do not cost F1. Test v11, F1:

| Run | (0.5, 2) | `cons`: setting, F1 | `old`: setting, F1 |
|---|---:|---|---|
| v7 | 0.869 | (0.20, 1), 0.880 | (0.50, 2), 0.869 |
| v7-latest | 0.881 | (0.35, 1), 0.896 | (0.75, 1), 0.880 |
| v9 | 0.875 | (0.55, 1), 0.882 | (0.50, 1), 0.883 |
| v9-latest | 0.874 | (0.40, 1), 0.885 | (0.40, 2), 0.876 |
| v10 | 0.873 | (0.15, 1), 0.901 | (0.20, 2), 0.891 |
| v10-latest | 0.873 | (0.10, 1), 0.888 | (0.45, 2), 0.875 |

  Against (0.5, 2), the old selection changes F1 by -0.001 to +0.018. For v3 and v5 it lost in
  10 of 11 runs. So the too-high old selection of v3 and v5 does not apply to the
  `DetectionLoss` recipe. The cons selection is better than the old selection in 5 of 6 runs
  (mean +0.010).
- The `v5-fold1…4` rows are not the fold models. Their heatmaps are byte-identical to those of
  `v5-rep1…4`, so all counts are identical. Locally, the export
  `synapse_detection_model_v5-f{1..4}.pt` has the weights of `v5-fold{1..4}/best.pt` (epochs 31,
  698, 681 and 64), not those of `v5-f{1..4}/best.pt`. Compare the cluster exports in
  `trained_models/Synapses` with the local copies before you score a fold model again. Until
  then, it is not known whether the historical `v5-f1…f4` entries of `synapses.json` score the
  folds or the seed replicates. `v5-fold3-fixed` is not in the new `synapses.json`.
- The duplicates give the `old` arm of the v5 replicates at HEAD. `v5-fold1-old` confirms the true
  value of `v5-rep1-old` (0.835). v5-rep2 and v5-rep3 select the same setting as on 2026-10-21.
  v5-rep4 selects (0.85, 2) with F1 0.836, instead of (0.9, 1) with 0.843.

### v11 result (2026-10-27)

The three seeds of the refined recipe (`train_synapse_v11.sbatch`, array `v11-1…3`) are trained.
The production pipeline scored `best.pt` (`v11-<i>`) and `latest.pt` (`v11-<i>-latest`) at (0.5, 2)
on the six consensus crops. `evaluate_synapse_v11.sbatch` (rules 1 and 2) has not run yet.

Test v11 at (0.5, 2), P / R / F1 and F1 per crop:

| Run | P | R | F1 | apex2 | base2 | mid1 | mid2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v11-1-latest | 0.980 | 0.802 | 0.882 | 0.863 | 0.890 | 0.910 | 0.858 |
| v11-2-latest | 0.972 | 0.819 | 0.889 | 0.907 | 0.893 | 0.911 | 0.860 |
| v11-3-latest | 0.983 | 0.809 | 0.888 | 0.878 | 0.899 | 0.915 | 0.858 |
| v11-1, `best.pt` | 0.905 | 0.849 | 0.876 | 0.811 | 0.887 | 0.906 | 0.865 |
| v11-2, `best.pt` | 0.965 | 0.834 | 0.895 | 0.886 | 0.906 | 0.917 | 0.869 |
| v11-3, `best.pt` | 0.960 | 0.819 | 0.884 | 0.870 | 0.895 | 0.909 | 0.857 |

Spread of the test v11 F1 at (0.5, 2):

| Group | Runs | Mean | SD | Range | Worst |
|---|---:|---:|---:|---:|---:|
| v5 recipe, `best.pt` (v5, v5-rep1…4) | 5 | 0.861 | 0.016 | 0.038 | 0.840 |
| v5 recipe, `latest.pt` | 5 | 0.867 | 0.017 | 0.040 | 0.841 |
| v7, v9, v10 and their `-latest`, one seed each | 6 | 0.874 | 0.004 | 0.012 | 0.869 |
| v11, `latest.pt` | 3 | 0.886 | 0.004 | 0.007 | 0.882 |
| v11, `best.pt` | 3 | 0.885 | 0.009 | 0.019 | 0.876 |

- Every v11 `latest.pt` is better than every earlier model at (0.5, 2). The best earlier models
  are v5-latest and v7-latest (0.881). v11-1-latest is only 0.001 above them, and the paired
  bootstrap interval of a model difference is about +-0.01. The gain is recall (0.80 to 0.82,
  v7 to v10: 0.78 to 0.81) at the same precision (0.97 to 0.98).
- Rules 4.1 and 4.2 pass without a selection: the mean (0.886) is above 0.864, and the worst seed
  (0.882) is above 0.835. The baseline is at its cons setting, so this comparison favours the
  baseline.
- The spread is small, but three seeds do not prove that it is smaller. Against the v5 seeds,
  the F-test of the variances gives p 0.04 to 0.05 and the Levene test p 0.10 to 0.34. The six
  v7-family variants also have SD 0.004. So the large spread is probably a property of the v5
  recipe (flow channels, plateau schedule, `best.pt`), and v11 keeps the small spread.
- apex2 was the weak crop of the v5 seeds (0.68 to 0.85). v11 `latest.pt` gets 0.86 to 0.91.
  mid2 limits every model (0.85 to 0.87), so the recall limit of "v9 and the recall limit" is
  unchanged.
- `best.pt` is less stable: v11-1 `best.pt` is at another operating point (P 0.905, R 0.849). Use
  `latest.pt`, as planned.
- On the val crops (apex1, base1) at (0.5, 2), `latest.pt` gets F1 0.901 (v11-1), 0.912 (v11-2)
  and 0.895 (v11-3). Rule 5 uses the val F1 of the gridsearch, so this order is preliminary.

**Training cost.** One seed takes 5 h 38 min to 5 h 42 min on an A100. The plateau runs with 1e5
iterations took about 32 h. That is 5.6 times less: 3.3 times from the iterations, and an
iteration takes 0.68 s instead of about 1.15 s. `reportseff` of job 16845185: 35 to 36 % of the
16 h limit, CPU 48 to 51 %, memory 11 % of 64 GB (about 7 GB). `-t 08:00:00` and `--mem 16G` are
enough for the next run.

**Status.** v11 meets the condition for a switch. The performance is at least equal, the training
is cheaper, and the known defects (target, learning rate schedule, val split) are removed. Open:

- rules 4.3 and 5: run `evaluate_synapse_v11.sbatch`;
- the production setting: v11 is already the best model at (0.5, 2). For the v5 replicates, the
  cons selection gained only +0.004 on average and lost in 3 of 8 runs. Keep (0.5, 2), unless the
  three seeds select the same setting and gain clearly;
- the training default and the released model do not change yet.

### Settings of every model version

Common to every run: patch shape 40 × 112 × 112, batch size 32, learning rate 1e-4 with the
torch_em `ReduceLROnPlateau` (factor 0.5, patience 5 epochs, see "Training robustness"), 3,200
training and 160 validation samples per epoch, no augmentation, a Gaussian target of peak 4 and
sigma 1 voxel (v10: 0.7) divided by its patch maximum (see "Target defect and v11"), and three
validation crops. The `synapses.json` entries of the production
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
| v5-rep1 … v5-rep4 | `v5`, the v5 split; checkpoint folders `v5-f1..4` | heatmap + flow | as v5 | as v5 | none | redrawn | 1e5 | `best.pt` (epochs 278, 13, 109, 28) and `latest.pt` | not scored, see "Training robustness" |
| v6-1 | `v6` = v5 data | heatmap | MSE / MSE | none | none | redrawn | 1e5 | not recorded | not scored |
| v7 | `v7` | heatmap | `DetectionLoss` / the same | `MinPointSampler(1)` | none | fixed, seed 42 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v8 | `v8` | heatmap | masked `DetectionLoss` / the same | `MinPointSampler(0)` | cubes of half width 16 around the annotations; last 33 voxels per axis not trained; commit `4da9e70` | fixed, seed 42 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v9 | `v7` | heatmap | masked `DetectionLoss` / the same | `MinPointSampler(1)` | cubes of half width 3 around the unannotated candidates, percentile 10; commit `4da9e70` | fixed, seed 42, other draws than v7 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v10 | `v7` | heatmap, target sigma 0.7 | `DetectionLoss` / the same | `MinPointSampler(1)` | none | fixed, seed 42, other draws than v7 | 1e5 | `best.pt` and `latest.pt` | production, IHC v11 |
| v11-1 … v11-3 | `v11` = `v7` + 2 consensus val crops; split file, the 3 old val crops train | heatmap, fixed target | `DetectionLoss` / the same | `MinPointSampler(1)` | none | fixed, seed 42 | 3e4, cosine learning rate | `best.pt` and `latest.pt` | production, IHC v11, see "v11 result (2026-10-27)"; gridsearch on the consensus val crops not run yet |

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

