# Synapses

## Network training

Since PR `#133` the synapse detection network is trained without flow per default. The flow option has to be selected by supplying a `--use_flow` flag.
The training data is located in separate `images` and `labels` directories, so the training/validation split has to be reproducible. The random state `--random_state 42` has to be passed for training the networks to test the variation to ensure the same split.
An example script for training `synapses_v3-1` is `train_synapse_v3-1.sbatch`.

A new synapse network v6 or its variation could be trained by substituting the function call with
```bash
# train synapse network v6
python $SCRIPT_DIR/train_synapse_detection.py -v v6 --random_state 42
# train synapse network v6-1
python $SCRIPT_DIR/train_synapse_detection.py -v v6 -m v6-1 --random_state 42
```
The training data is already prepared in `/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v6` and is identical to the `v5` one.

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

1. add an explicit and identical training/inference raw normalization;
2. use a fixed validation patch set and a detection-level validation metric;
3. mark regions outside the exhaustively annotated/IHC-valid region as *ignore*, rather than as
   background in the loss;
4. run a quick ablation without the three `m78l_*_cr-ctbp2` volumes, followed by a proper masked
   training run that retains their valid annotations;
5. apply the existing checkpoints at the confirmed candidate coordinates and test whether the
   lowest-loss seeds suppress both off-target candidates and annotated synapses.

An IHC-derived mask can be used only for the training loss, so this test does not require an IHC
channel as network input at inference time. If the IHC channel is intended to be a model input
instead, its availability and stain-dependent behavior must first be resolved for all training and
production volumes.

## Network application
Potentially relevant for a new synapse network.
An example script for cochlea `G_LR_000302_R` is `synapse_process_GLR000302R.sbatch`.
The volume can be masked based on an IHC segmentation, which can be local or on the S3 bucket.
The mask may cut off potential synapses because its size is currently limited to the extension of the IHC segmentation.
Future updates may improve this by dilating the mask before applying the network.

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
* check content of `flamingo_tools/postprocessing./synapse_per_ihc_utils.py` (probably already updated)

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

The script `synapse_detect_v5-variation_F1val.sbatch` was used to apply the synapse network `v5` for the validation.
The script has to be adapted once the variation scripts for v3 have been trained.
Afterwards, the accuracy can be calculated using

```bash
python scripts/validation/synapses/run_evaluation.py -v v3-1 -o ~/flamingo-tools/reproducibility/model_accuracy/
python scripts/validation/synapses/run_evaluation.py -v v3-2 -o ~/flamingo-tools/reproducibility/model_accuracy/
python scripts/validation/synapses/run_evaluation.py -v v3-3 -o ~/flamingo-tools/reproducibility/model_accuracy/
python scripts/validation/synapses/run_evaluation.py -v v3-4 -o ~/flamingo-tools/reproducibility/model_accuracy/
```
The accuracy values will be written into `reproducibility/model_accuracy/synapses.json`.
From there they can be read by `plot_fig2.py`.
