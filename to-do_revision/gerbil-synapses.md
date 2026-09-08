# Gerbil ribbon-synapse diagnosis and manuscript decision

Date: 2026-08-25

## Scope and conclusion

This report covers the v3 ribbon-synapse results for the four wild-type gerbil cochleae used
for Figure 5: `G_EK_000233_L`, `G_LR_000301_L`, `G_LR_000301_R`, and
`G_LR_000302_R`.

The high value in GEK233L is not explained by an obvious prediction or postprocessing fault.
The reproducible difference is instead that parts of the GLR cochleae, most strongly
GLR301L and also GLR301R, have much poorer CtBP2 contrast. The v3 model consequently finds
very few peaks in those regions. The prediction blocks were processed, so this is distinct
from the earlier incomplete GLR302R prediction. Neither block-local affine input
normalization nor a lower output threshold materially recovers the missing detections.

For the manuscript we therefore keep production v3 settings and report the Figure 5
synapse-per-IHC distribution after excluding IHCs with zero mapped synapses. GLR301L remains
distinctly low even after this filter, so it is excluded from the synapse panel while remaining
in the SGN and IHC panels. Its result is retained for provenance. The plotted synapse estimate
is conditional among IHCs with at least one detectable/mapped ribbon, and this restriction must
be stated in the figure legend or Methods.

GLR301L was rerun once at production settings against the finalized dilated-mask IHC_v11
segmentation used for its IHC count (946 selected objects, components 1-13). The rerun has a
separate, versioned output and did not overwrite the earlier result.

## Starting point and pipeline checks

The final production-result summary is:

| cochlea | selected IHCs | mapped synapses at 3 um | mean over all selected IHCs | zero-count IHCs | mean after excluding zeros |
|---|---:|---:|---:|---:|---:|
| GEK233L | 1,018 | 16,547 | 16.254 | 68 (6.7%) | 17.418 |
| GLR301L | 946 | 3,629 | 3.836 | 524 (55.4%) | 8.600 |
| GLR301R | 1,074 | 13,543 | 12.610 | 296 (27.6%) | 17.407 |
| GLR302R | 930 | 10,204 | 10.972 | 155 (16.7%) | 13.166 |

The values above use the finalized component selections. For GLR301L the former prediction
was not treated as final because its prediction mask and IHC matching predated the finalized
dilated-mask IHC segmentation. Its earlier diagnostic result contained 4,298 detections
within the production 8 um matching radius and averaged about 4.4 synapses per IHC, but that
number is not used for Figure 5. The corrected run produced the GLR301L row above.

Several potential technical explanations were checked before interpreting the difference:

- **Incomplete prediction:** each of the five production array tasks writes a completion
  receipt. The problematic GLR301L run had all receipts, and its IHC prediction mask covered
  967 of 973 IHCs in the then-current table, including 290 of 295 IHCs without a nearby
  detection. The weak region was therefore predicted rather than skipped.
- **The known GLR302R truncation:** the original GLR302R table was genuinely incomplete, with
  583 of 930 IHCs at zero and a 13-bin contiguous helix gap whose nearest detections were
  105-682 um away. A complete v3 rerun fixed this and replaced the old result; the current
  10.972 value is from the coverage-validated result.
- **IHC component selection:** the manuscript selections are GEK233L components
  2,1,6,4,3,5; GLR301L components 1-13; GLR301R components 8,9,7,6,4,3,11,1,5,2; and
  GLR302R components 3,1,2. Including the accepted components fixes the IHC denominators but
  does not account for the spatially localized lack of synapse detections.
- **Mapping cutoff:** production detection retains assignments out to 8 um so the peak table
  remains reusable, while manuscript per-IHC counts are re-filtered to 3 um. The large gaps
  in the affected GLR regions are tens to hundreds of micrometers, so modestly changing this
  cutoff cannot bridge them.
- **GEK233L false-positive excess:** the GEK result covers its full IHC helix, its selected
  components are coherent, and its distribution after zero exclusion (17.418) is essentially
  the same as GLR301R (17.407). The evidence is more consistent with GEK233L being a
  well-stained reference than with a systematic GEK-specific over-detection.

## Targeted normalization and threshold experiment

### Design

The same production v3 model, `synapse_detection_model_v3.pt`, was evaluated on eight blocks
per cochlea: four poor/lower-count blocks and four matched good/typical blocks. Inference used
the production block shape `(64, 256, 256)` and halo `(16, 64, 64)`.

Two external normalization modes were compared:

1. the production mean and standard deviation calculated globally under the resized,
   dilated IHC mask; and
2. mean and standard deviation calculated from masked voxels in each padded inference block.

Local maxima were evaluated at absolute heatmap thresholds 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
and 0.9. Production uses 0.5. Diagnostic counts were restricted to peaks within 3 um of a
segmented IHC. These block totals are hypothesis tests, not manuscript synapses-per-IHC.
The proposed v5 comparison and further image transformations were deliberately not run.

### The input contrast difference is real

| cochlea | block regime | mean local standard deviation | range |
|---|---|---:|---:|
| GEK233L | high-count good | 37.4 | 30.4-48.8 |
| GEK233L | lower-count basal | 44.8 | 21.1-78.5 |
| GLR301L | good | 38.8 | 28.0-52.9 |
| GLR301L | poor | **10.6** | 7.5-12.9 |
| GLR301R | good | 42.5 | 29.7-56.0 |
| GLR301R | poor | **15.5** | 13.2-17.1 |
| GLR302R | typical middle | 30.5 | 23.5-41.7 |
| GLR302R | lower-count basal | 37.1 | 22.4-48.0 |

Thus, the selected poor GLR blocks had 2.8-3.7-fold lower local standard deviation than the
matched good blocks. An independent low-resolution selection for GLR301L gave raw p99.9
values of only 159.5-162.7 in poor blocks, versus 335.4-827.4 in good blocks.

### Local affine normalization does not change v3 predictions

Global and block-local normalization produced identical counts at every tested threshold for
GEK233L, GLR301L, and GLR301R. GLR302R differed by one peak in its typical blocks at
thresholds 0.3-0.5 and was otherwise identical. The maximum absolute heatmap differences
were only 0.0027-0.0051.

This negative result follows from the architecture: the first operation in the first v3
encoder block is `InstanceNorm3d(1)`, before the first convolution. Global and local
mean/std preprocessing differ only by a positive affine transform, and the internal instance
normalization removes that transform. In practical terms, v3 already normalizes every padded
patch internally. The remaining numerical differences are compatible with floating-point
rounding and the normalization epsilon.

### Lowering the output threshold is insufficient

Counts at the production threshold and the most permissive tested threshold were:

| cochlea | block regime | threshold 0.5 | threshold 0.3 | change |
|---|---|---:|---:|---:|
| GEK233L | high-count good | 356 | 360 | +4 |
| GEK233L | lower-count basal | 160 | 165 | +5 |
| GLR301L | good | 81 | 91 | +10 |
| GLR301L | poor | **5** | **7** | +2 |
| GLR301R | good | 453 | 462 | +9 |
| GLR301R | poor | **12** | **18** | +6 |
| GLR302R | typical middle | 191 | 205 | +14 |
| GLR302R | lower-count basal | 65 | 71 | +6 |

At threshold 0.3, GLR301L still had 7 poor-region peaks versus 91 in good blocks, and
GLR301R had 18 versus 462. The output threshold is therefore not the main cause of the
deficit and lowering it globally would mainly increase the false-positive burden elsewhere.

## Updated interpretation

The experiments support the staining/input-quality diagnosis and reject the two cheap rescue
mechanisms tested here. In the weak GLR regions, the recoverable punctate structure is either
absent or too degraded for the current v3 representation. Poor input cannot be assumed to
produce only a uniformly smaller output that can be restored by a lower threshold.

If this question is revisited, the next informative prediction experiment would need to alter
local spatial contrast rather than apply another affine normalization. Examples are a
carefully controlled local-contrast transformation on the already selected diagnostic blocks,
or a retrained architecture whose first operation does not cancel the intended external
normalization. These would constitute a new method and are outside the current manuscript
decision.

## Production rerun and Figure 5 rule

The GLR301L rerun uses unchanged production settings:

- v3 model and threshold 0.5;
- prediction block `(64, 256, 256)` with halo `(16, 64, 64)`;
- five prediction tasks;
- `grete:preemptible`, one `1g.20gb` MIG slice per task;
- preemption-safe task receipts and `afterok` dependency before peak detection;
- prediction assignments retained to 8 um, followed by the 3 um manuscript cutoff; and
- finalized mask
  `/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/prediction/G301L/IHC_v11_dilated_mask1/segmentation.ome.zarr`.

Submission script:

`scripts_syn_gerbil/rerun_G301L_corrected_ihc.sh`

Submitted jobs:

- preprocessing: `15496304`;
- five-task prediction array: `15496305`; and
- dependent peak detection/IHC matching: `15496306`.

Both CPU jobs use `large96s:test` with a one-hour limit. The preprocessing and detection
jobs completed in 3:51 and 9:12, respectively. All five `grete:preemptible` prediction tasks
completed in 14.1-15.4 minutes.

Versioned output:

`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/synapses-v3/G_LR_000301_L_IHC_v11_dilated_mask1`

The corrected run wrote 4,448 thresholded detections, of which 4,157 were matched within the
production 8 um radius. At the 3 um manuscript cutoff, 3,629 detections map to the 946 selected
IHCs. Of these IHCs, 422 have at least one mapped detection and 524 have zero. The resulting
mean is 3.836 over all selected IHCs and **8.600 after the requested zero exclusion**.
The corrected-mask production normalization was mean 129.662 and standard deviation 31.158.

Mechanical validation found all five task receipts and 1,583 written prediction chunks. The
coverage profile still contains 12 consecutive poor-signal bins in component 1, with median
nearest-detection distances of 53-433 um. Because every prediction task completed, this is the
expected staining-related biological coverage limitation rather than an incomplete inference
run.

Figure 5 uses all four manuscript cochleae for SGN and IHC counts, but only GEK233L, GLR301R,
and GLR302R for the synapse panel. It refuses to plot if any table for those three is missing
and drops rows with `synapse_count == 0` only for its synapse panel. Other figures retain their
previous behavior. The G301L configuration in `SYNAPSE_DICT` records components 1-13 so its
result remains reproducible even though it is not plotted.

The complete source table, including its 524 zero rows, is installed at:

`/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/synapses/ihc_counts_v11/ihc_count_G_LR_000301_L.tsv`

Across the three plotted source tables, Figure 5 retains 2,503 nonzero IHCs. Their pooled mean
is 16.098 synapses/IHC. The plotted per-cochlea zero-excluded means are 17.418 (GEK233L),
17.407 (GLR301R), and 13.166 (GLR302R). GLR301L's excluded value is 8.600.

## Reproducibility artifacts

The normalization experiment is under:

`/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/synapse-normalization-v3`

Important combined outputs are `aggregate_counts.tsv`, `combined_threshold_counts.tsv`,
`combined_normalization_stats.tsv`, and `combined_prediction_differences.tsv`. The executable
experiment definition and original focused report are in
`scripts_syn_gerbil/normalization_experiment.py` and
`scripts_syn_gerbil/normalization_experiment_report.md`.

The reproducible local table generator is
`scripts_syn_gerbil/make_G301L_figure5_table.py`.
