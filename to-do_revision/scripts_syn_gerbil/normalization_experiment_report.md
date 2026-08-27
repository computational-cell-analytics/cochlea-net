# Gerbil v3 synapse normalization experiment

Date: 2026-08-25

## Question

Can weak-but-recoverable CtBP2 signal in poorly stained GLR regions be recovered by replacing
the production global input mean/std with mean/std measured in each padded inference block?
Is the deficit instead mainly caused by the output threshold?

## Design

- Model: `synapse_detection_model_v3.pt` (v5 was deliberately not tested).
- Four cochleae: `G_EK_000233_L`, `G_LR_000301_L`, `G_LR_000301_R`, and
  `G_LR_000302_R`.
- Eight blocks per cochlea: four poor/lower-count and four matched good/typical blocks.
- Inference geometry matches production: block `(64, 256, 256)`, halo `(16, 64, 64)`.
- Global mode uses mean/std over the full raw volume under the resized dilated IHC mask.
- Local mode uses mean/std over masked voxels in the same padded inference block presented
  to v3.
- Absolute output thresholds: `0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9`; production is `0.5`.
- Counts below are local maxima within 3 um of a segmented IHC, summed over four selected
  blocks per regime. They are diagnostic region totals, not manuscript synapses-per-IHC.

## Contrast check

The poor GLR blocks really do have lower local contrast:

| cochlea | regime | mean local std | range |
|---|---|---:|---:|
| GEK233L | high-count good | 37.4 | 30.4-48.8 |
| GEK233L | lower-count basal | 44.8 | 21.1-78.5 |
| GLR301L | good | 38.8 | 28.0-52.9 |
| GLR301L | poor | **10.6** | 7.5-12.9 |
| GLR301R | good | 42.5 | 29.7-56.0 |
| GLR301R | poor | **15.5** | 13.2-17.1 |
| GLR302R | typical middle | 30.5 | 23.5-41.7 |
| GLR302R | lower-count basal | 37.1 | 22.4-48.0 |

GLR301L's independent low-resolution raw selection also separated poor-block p99.9 values
of 159.5-162.7 from good-block values of 335.4-827.4.

## Counts

Global-normalization counts at the production threshold and the permissive threshold:

| cochlea | regime | threshold 0.5 | threshold 0.3 | change |
|---|---|---:|---:|---:|
| GEK233L | high-count good | 356 | 360 | +4 |
| GEK233L | lower-count basal | 160 | 165 | +5 |
| GLR301L | good | 81 | 91 | +10 |
| GLR301L | poor | **5** | **7** | +2 |
| GLR301R | good | 453 | 462 | +9 |
| GLR301R | poor | **12** | **18** | +6 |
| GLR302R | typical middle | 191 | 205 | +14 |
| GLR302R | lower-count basal | 65 | 71 | +6 |

Global and block-local normalization gave identical counts at every threshold for GEK233L,
GLR301L, and GLR301R. GLR302R differed by one peak in its typical blocks at thresholds
0.3-0.5 and was identical otherwise. Maximum absolute heatmap differences were only
0.0027-0.0051 across cochleae.

## Interpretation

The poor-staining diagnosis is supported: GLR301L and GLR301R poor blocks have 2.8-3.7 times
smaller local standard deviations than their good blocks and orders-of-magnitude fewer
detections.

However, external affine local normalization is not an effective knob for v3. The first
operation in v3's first encoder block is `InstanceNorm3d(1)`, before the first convolution.
For positive standard deviations, global and local mean/std preprocessing differ only by an
affine transformation, which this instance normalization removes. In effect, v3 already
normalizes every padded inference patch internally. The tiny heatmap differences arise from
floating-point arithmetic and the normalization epsilon, not a meaningful contrast rescue.

Lowering the output threshold finds a few additional poor-region peaks but leaves the large
gap intact: GLR301L remains 7 versus 91 and GLR301R 18 versus 462 at threshold 0.3. Threshold
choice is therefore not the main explanation.

The updated hypothesis is: the GLR deficit is caused by staining/input-quality differences,
but it cannot be repaired by changing an external affine mean/std while using the current v3
architecture. A future prediction experiment would need a transformation that changes local
spatial contrast rather than only global affine scale (for example, carefully controlled
local contrast enhancement on the already selected blocks), or a retrained architecture
whose first operation does not cancel the intended normalization. These experiments were not
run here.

## Outputs

Experiment root:

`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/synapse-normalization-v3`

Combined outputs:

- `aggregate_counts.tsv`
- `combined_threshold_counts.tsv`
- `combined_normalization_stats.tsv`
- `combined_prediction_differences.tsv`

Each cochlea folder also contains both sparse heatmaps, block-level statistics, detections at
the lowest threshold, threshold counts, global-stat provenance, and a completion receipt.

## Scheduling record

- `15494608`: completed GLR302R and GLR301L.
- `15495715`: completed GLR301R.
- `15494578`: completed GEK233L, then was canceled while scanning GLR301R with the original
  repeatedly-decompressed mask path.
- `15494579`: initial GLR302R attempt stopped on a fully blank terminal block. That block had
  zero raw variance and cannot test recoverable weak signal, so it was replaced by a weak but
  nonzero basal block before the successful run.

The final runner materializes the small binary low-resolution mask in RAM before the global
statistics scan; this increased GLR301R scan throughput from roughly 15-20 to 280 blocks/s.
