# Synapses

## Network training

Since PR `#133` the synapse detection network is trained without flow per default. The flow option has to be selected by supplying a `--use_flow` flag.
The training data is located in separate `images` and `labels` directories, so the training/validation split has to be reproducible. The random state `--random_state 42` has to be passed for training the networks to test the variation to ensure the same split.
An example script for training `synapses_v3-1` is `train_synapse_v3-1.sbatch`.

The training commands of every run so far are in `scripts_synapses/`. The runs before v7 need
`--legacy_recipe`, and their scripts pass it. The `v6` training data in
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

#### Critical independent bug: raw input is not normalized

**The current v3-style training path does not normalize the raw input and must be fixed before
the next training run.** `supervised_training` sets `raw_transform=None`. The local
`DetectionDataset` stores this value unchanged and only applies a transform when it is not
`None`; `ensure_tensor_with_channels` then casts the original values to floating point without
normalizing them. Consequently, the network receives the native intensity scales of a mixture of
`uint8` and `uint16` volumes. For example, `4.1L_mid_IHCribboncount_Z` is `uint8`, whereas the
other v3 volumes are `uint16` with substantially different ranges.

The upstream detection dataset normally falls back to standardization when no raw transform is
provided, but the local dataset does not implement this fallback. This is separate from the
missing-label problem and is another credible source of domain sensitivity and seed instability.
Training should define and test an explicit normalization that is also applied identically during
inference.

#### Further confounder and recommended tests

Validation patches are not fixed: `DetectionDataset.__getitem__` samples a new random bounding
box on every access, and the validation loader is shuffled. Validation loss therefore measures a
different set of patches at each evaluation. The off-target distribution makes this noise more
consequential, but random validation sampling is an independent problem.

Before interpreting another seed comparison, the training setup should:

1. add an explicit and identical training/inference raw normalization; **done in training on
   2026-09-21. The validation inference included the zero padding until 2026-09-25, see
   "v7 against v8" below. Production still uses different statistics.**
2. use a fixed validation patch set and a detection-level validation metric; **the fixed patch
   set is done. The detection-level metric is still open.**
3. mark regions outside the exhaustively annotated/IHC-valid region as *ignore*, rather than as
   background in the loss; **done as the cube mask of v8, which failed. Replaced by the
   candidate-exclusion mask of v9, see "v7 against v8" below.**
4. run a quick ablation without the three `m78l_*_cr-ctbp2` volumes, followed by a proper masked
   training run that retains their valid annotations; **the ablation is optional: v7 against v8
   shows that the unannotated spots do not limit the recall. v9 is the masked run.**
5. ~~apply the existing checkpoints at the confirmed candidate coordinates and test whether the
   lowest-loss seeds suppress both off-target candidates and annotated synapses.~~ **Obsolete:
   v7 against v8 tests the mechanism more directly.**

An IHC-derived mask can be used only for the training loss, so this test does not require an IHC
channel as network input at inference time. If the IHC channel is intended to be a model input
instead, its availability and stain-dependent behavior must first be resolved for all training and
production volumes.

#### Implemented: masked loss and raw normalization (2026-09-21)

Recommendations 1 and 3 are implemented. The training no longer imports
czii-protein-challenge; the model, the loaders, the loss and the trainer live in
`flamingo_tools/synapse_detection/training.py`.

**Loss mask.** `train_synapse_detection.py --mask_radius R` marks a cube of half width `R`
voxels around every annotation. `CsvHeatmapTransform` appends the mask as the last target
channel, and `DetectionLoss` restricts both the training loss and the validation metric to it.
Use `--mask_radius 16`, which is about 12 um at 0.38 um/voxel and stays well below the 40 voxel
radius that defines a spatially distant candidate above. **v8 is this run, and it failed. Do not
use the cube mask for new runs, see "v7 against v8" below.**

`mask_radius` also raises the label transform halo to at least `R`, so `DetectionDataset` loads
the annotations up to `R` voxels beyond the patch. Target and mask are then built from the same
points. Without that, an annotation just in front of the patch would contribute its cube to the
mask but no Gaussian to the target, and the loss would be told to drive a real synapse to zero.

The mask covers 5 % of a crop at the median for `R = 16`, but training patches are not sampled
uniformly. Measured over 48 patches from eight v5 crops with `MinPointSampler`, the mask covers
**30 % of a patch at the median** and 8 % of the patches have an empty mask. Each annotation
therefore keeps a large amount of genuine negative context, and only the distant unannotated
CTBP2 spots lose their supervision.

`MinPointSampler` becomes the default whenever a mask radius is given. Only 31 % of uniformly
sampled patches contain an annotation, and a patch without one contributes no gradient at all
once the loss is masked.

**Raw normalization.** Each crop is standardized with the mean and standard deviation of its
full `raw` array, which is what `prediction_impl` does for a volume. One difference remains: at
inference the statistics are computed inside the IHC mask, during training over the whole crop.

**Retraining v3 or v5.** Pass `--legacy_recipe`. It restores the three deltas that otherwise make
the old recipes unreachable: the raw input stays unnormalized, the validation metric goes back to
an unweighted mean squared error over every output channel, which is what selected `best.pt` for
v5, and the validation patches are redrawn on every epoch. It cannot be combined with
`--mask_radius`, because that metric cannot read a masked target. The archived scripts in
`scripts_synapses/` pass the flag.

**Comparing the next runs.** The logged loss and metric change scale twice, through the masked
mean and through the normalization. They are not comparable to the v3, v5 or v6-1 numbers. Score
models with `scripts/validation/synapses/run_evaluation.py`, and train a fresh unmasked baseline
with the same code rather than comparing against the shipped models.

**Recommendation 2, first half: the validation patch set is fixed.** `DetectionDataset` takes a
`patch_seed`, and the bounding box for a given index is a pure function of `(patch_seed, index)`,
so the same index gives the same patch in every epoch and in every data loader worker. The
training script derives the seed from `--random_state`, the same number that already fixes the
train and validation split, so the validation data is reproducible across runs as well. The
validation loader also stops shuffling: the trainer averages the metric over batches and the
masked loss normalizes within a batch, so a fixed patch set alone would not give a fixed metric.
Measured on a synthetic stand-in at the production validation size, 160 patches in batches of
32, scored against a fixed prediction so that only the patch draw moves: the metric spread over
six passes was **15 % of its mean** before the change and exactly zero after it. The noise does
not average away with `n_samples_val`, because the reported value is a mean over only five
batches and each batch is normalized by its own mask size.

**Still open: the detection-level metric.** The other half of recommendation 2 is not done. The
metric remains a voxelwise masked mean squared error, so `best.pt` is now stable but still a
proxy for detection F1, and the seed study already showed that validation loss does not rank
detectors (Spearman rho 0.39, p = 0.38). Keep scoring candidates with
`scripts/validation/synapses/run_evaluation.py`, and keep comparing `best.pt` against
`latest.pt`.

**Not used: the IHC channel.** 21 of the 29 v5/v6 crops do carry a `raw_ihc` array, so a real
IHC-derived loss mask is possible for most of them. The label-derived mask is used instead
because it is uniform over all crops and independent of the stain.

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

`v7` and `v8` are also not a single step from `v6-1`: it ran with no sampler, no raw
normalization and redrawn validation patches. And the pair is not a clean ablation of the mask
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

**Fixed: the validation normalization.** `prediction.py` pads every crop with zeros to a
multiple of the production block shape, and `prediction_impl` took the mean and standard
deviation over the padded volume. The median tissue voxel became +1.0 to +2.1 standard deviations, against
−0.2 to +1.1 with the statistics of the unpadded crop that training uses, and the standard
deviation was 1.2–2.1× too small. This affected every version, not only v8. `prediction.py`
now computes the statistics on the unpadded crop. All 2026-09-24 entries must be predicted
again. Production still differs from training: it takes the statistics inside the IHC mask of
the cochlea, while training takes them over the whole crop, zero regions included.

**Fixed: the patch sampling.** `DetectionDataset._sample_bounding_box` drew the patch start
from `[0, shape - patch_shape - 2 * halo)`, although the halo is clamped when the patch is
loaded. The last `2 * halo + 1` voxels of every axis never reached training: 33 for v8, 21 for
the flow models such as v5, and 1 for v7.

**v9: the candidate-exclusion mask.** `--ignore_percentile P` keeps the loss on every voxel
except a cube of half width 3 voxels around each unannotated CTBP2 candidate. A candidate is a
local maximum of the smoothed crop that is more than 4 voxels from every annotation and at
least as bright as the `P`th percentile of the maxima at the annotations
(`find_unannotated_candidates`). The cube around an annotation always stays supervised. On the
six test crops, `P = 10` gives 11 to 55 candidates per crop, which ignore at most 0.3 % of the
tissue. The `m78l_*_cr-ctbp2` training crops are expected to give far more, so check the counts
that the training prints. v9 keeps the data, the split, the validation patches and the sampler
of v7, so the mask is the only difference:

```bash
python $SCRIPT_DIR/train_synapse_detection.py -v v7 -m v9 --random_state 42 -s $SAVE_ROOT \
    --sampler minpoint --ignore_percentile 10
```

`scripts_synapses/train_synapse_v9.sbatch` runs it. Register the exports as
`synapse_detection_v9.pt` and `synapse_detection_v9-latest.pt`, and add `v9 v9-latest` to the
version loop of `synapse_detect_ihc11_F1val.sbatch`. v9 is a success if it keeps the precision
of v7 and gains recall.

The ignore mask cannot suppress an unannotated spot close to the IHC either (point 2). If v9
shows the same false-positive class, the IHC filter distance is the next place to look.
