# Synapse v3 training dynamics and generalization

Date: 2026-08-24

## Scope

The v3-style runs use the heatmap-only synapse model, the same v3 training data, and the same
train/validation split. The seed variants differ through stochastic weight initialization and
patch sampling/order. This makes them suitable for measuring training-seed sensitivity, although
the original v3 run has no recoverable validation-loss record.

Test scores were measured on the same six consensus-annotated crops, with detection threshold
0.5, the same block shape and halo, and the same IHC-assignment filter. All scores below use the
same per-crop normalization, so they are internally comparable. Per-crop normalization is not the
normalization used for whole-cochlea production inference.

| Model | Selected checkpoint | Best validation loss | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| v3 | unknown | unavailable | 0.927 | 0.799 | 0.858 |
| v3-1 | epoch 750 | 0.000124 | 0.950 | 0.664 | 0.782 |
| v3-2 | epoch 380 | 0.000112 | 0.979 | 0.665 | 0.792 |
| v3-3 | epoch 848 | 0.000136 | 0.974 | 0.761 | 0.854 |
| v3-4 | epoch 134 | 0.000146 | 0.973 | 0.789 | 0.872 |
| v3-5 | epoch 71 snapshot | 0.000161 | 0.969 | 0.719 | 0.826 |
| v3-6 | epoch 99 | 0.000143 | 0.967 | 0.746 | 0.842 |
| v3-7 | epoch 43 | 0.000151 | 0.914 | 0.715 | 0.802 |

The v3-5 score is tied to the immutable epoch-71 checkpoint that was exported while its 10k run
was still active. No better checkpoint had appeared through epoch 83 when this report was written.

## Training dynamics

Validation loss fell rapidly at the start of every short run and then entered a noisy plateau.
The plateau was real in the sense that later improvements were small, but its first occurrence
did not reliably identify the final best checkpoint. For example, v3-5 improved from 0.000171 at
epoch 46 to 0.000161 at epoch 71 after 25 epochs without a new best. v3-6 reached its best at epoch
99, the last validation point of its 10k run, whereas v3-7 reached its best at epoch 43.

This supports using 10k iterations to sample seed-to-seed behavior economically, but not the
stronger claim that the first plateau always supplies the final minimum-loss checkpoint. A short
run captures the broad behavior; the exact selected checkpoint can still move late because the
validation curve fluctuates around a shallow minimum.

The generalization spread was large. Across original v3 and the seven variants, test F1 ranged
from 0.782 to 0.872, a difference of 0.090. Most weak runs were conservative: v3-1 and v3-2 had
precision of 0.950 and 0.979 but recall of only 0.664 and 0.665. Thus apparently plausible
heatmaps and low voxelwise validation loss can still yield too few peaks after thresholding and
IHC filtering. v3-7 shows that the failure mode is not exclusively low recall: it also produced
substantially more false positives than the other variants.

## Association between validation loss and test performance

For the seven variants with both quantities available, the association between numerical
validation loss and test F1 was positive rather than negative: higher (nominally worse)
validation loss tended to accompany higher test F1. However, the relationship was weak and
uncertain:

- Pearson correlation: r = 0.47, p = 0.29
- Spearman rank correlation: rho = 0.39, p = 0.38

With only seven observations, neither estimate supports a reliable monotonic relationship.
Individual results make the limitation clear. v3-2 attained the lowest validation loss
(0.000112) but only 0.792 test F1. v3-4 had a higher loss (0.000146) and the best F1 (0.872).
Conversely, v3-7 and v3-5 had the two highest losses without being the two best test models. The
data therefore support a weak tendency, not a checkpoint-selection rule.

## Likely interpretation and implications

The most likely explanation is metric mismatch amplified by a small, noisy validation set. The
training objective is a voxelwise heatmap loss, whereas the reported endpoint is point-detection
F1 after peak thresholding and IHC-based filtering. A model can reduce average heatmap error by
changing background or peak amplitudes in ways that do not improve, and can even reduce, the
number of correctly detected synapses. Three validation crops also provide a noisy estimate of
this loss, so stochastic differences in initialization and sampled patches can dominate small
differences between checkpoint minima.

Consequently:

1. Minimum validation loss should not be treated as a dependable proxy for the best detection
   model or as evidence that one seed generalizes better than another.
2. Checkpoint selection should include a detection-level validation metric using the deployed
   threshold and post-processing, ideally on a larger or more stratified validation set.
3. Multiple seeds should be trained and evaluated. Reported model uncertainty should reflect the
   seed distribution rather than a single selected run.
4. More iterations alone are unlikely to remove the problem. They may find a slightly lower loss,
   but the central issue is alignment between the selection metric and the downstream endpoint.
5. Test data should not be used to select the released seed. The present test ranking is useful
   for diagnosing sensitivity; future selection should be performed on an independent
   detection-level validation set.

## Figure 2c standard-deviation set

For Figure 2c, the requested selected set consists of the five highest-test-F1 v3-style models:
v3-4, original v3, v3-3 best, v3-6, and the v3-5 epoch-71 snapshot. From the rounded reproducibility
scores, their means and population standard deviations are:

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Precision | 0.9620 | 0.0177 |
| Recall | 0.7628 | 0.0290 |
| F1 | 0.8504 | 0.0155 |

This top-five standard deviation is selection-conditioned and is appropriate for the requested
figure display, but it is not an unbiased seed-variability estimate. Across all eight evaluated
v3-style models, the exact F1 population standard deviation is 0.0313, roughly twice the
top-five value.
