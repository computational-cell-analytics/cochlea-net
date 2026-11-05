# Errata of the published CochleaNet figures

Status: 2026-11-05. Branch `fix_figure_statistics`. This branch is not merged into master.

The published panels keep their appearance. The code in this folder renders them as published:
the 2D and 5C renders are pixel-identical to master. The only addition is a warning in
`literature_comparison.plot_counts_vs_literature` for values outside the y range.

This note lists the inaccuracies of the published panels, so that the figure legends can describe
the data correctly. The commits `0795119` to `45b95b2` on this branch hold a corrected rendering of
the panels (sample SD, points for n = 4, n text, higher axes, axis break). Panel ids follow the
published numbering. s is the sample SD (divisor n - 1).

## 1. Values outside the y range (2D, 5C)

Cluster run 2026-11-04. The counts come from the warning. The approximate values come from a
render with a higher axis.

| Panel | Axis | y range | Hidden values | Approximate values |
|---|---|---|---|---|
| 2D | Synapses per IHC | -1 to 41 | 10 of 2,592 IHCs | 45 to 83 |
| 5C | Synapses per IHC | -1 to 80 | 6 of 2,503 IHCs | 86 to 122, and one at 237 |

The box statistics (median, quartiles, whiskers) use all values, also the hidden ones. The
whiskers end inside the y range.

Legend text:
- 2D: "Ten of 2,592 IHCs (up to 83 synapses) lie above the plotted range."
- 5C: "Six of 2,503 IHCs (up to 237 synapses) lie above the plotted range."

## 2. Sample size and pooling (2D, 5C)

The panels show no n. The legend must give it.

| Sub-panel | n | Notes |
|---|---|---|
| 2D SGN, IHC | 4 iDISCO mouse cochleae (M_LR_000226_L/R, M_LR_000227_L/R) | one count per cochlea |
| 2D synapses | 2,592 IHCs pooled from the same 4 cochleae | IHCs without synapses are included |
| 5C SGN, IHC | 4 wild-type gerbil cochleae (G_EK_000233_L, G_LR_000301_L/R, G_LR_000302_R) | one count per cochlea |
| 5C synapses | 2,503 IHCs pooled from 3 cochleae | IHCs without synapses are removed; G_LR_000301_L is left out |

Inconsistencies:
- 2D keeps IHCs without synapses, but 5C removes them.
- 5C uses G_LR_000301_L for SGN and IHC, but not for synapses. Its CTBP2 staining has no punctate
  signal over about a third of the helix: about 4.4 detections per IHC, against 13 to 16 in the
  other cochleae (commit `8fc9f11`).

## 3. Box plots of four values (2D, 5C, SGN and IHC)

Each box summarizes 4 values. The quartiles are linear interpolations between these values. No
value lies outside 1.5 x IQR, so the whiskers show the minimum and the maximum.

Values (`util.VALUE_DICT`): 2D SGN 10333, 11153, 11398, 11820; 2D IHC 617, 640, 648, 687;
5C SGN 18541, 21801, 22284, 23717; 5C IHC 930, 946, 1018, 1074.

Legend text: "Box plots of four cochleae. The box shows the interpolated quartiles, the whiskers
show the minimum and the maximum."

## 4. SD definitions

The published code uses two SD definitions:
- Population SD (divisor n, numpy default): 2C error bars, 3C gray band, the 4C, 5D and 6C band,
  5D error bars.
- Sample SD (divisor n - 1, pandas default): the 3F, S4B and S4G bands.

The population SD is s x sqrt((n - 1) / n): 18.4% smaller than s for n = 3, 13.4% for n = 4 and
10.6% for n = 5.

2C error bars as published (population SD) and as sample SD:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| SGN (4 seeds) | 0.0084 (s 0.0097) | 0.0019 (s 0.0022) | 0.0052 (s 0.0060) |
| IHC (4 seeds) | 0.0079 (s 0.0091) | 0.0121 (s 0.0139) | 0.0048 (s 0.0056) |
| Synapse (5 models) | 0.0177 (s 0.0198) | 0.0290 (s 0.0324) | 0.0155 (s 0.0173) |

Legend text: "SD with divisor n" for the population-SD panels.

5D error bars: the density positions inside one region are not fully independent.
`sgn_density_at_position` selects SGNs by bounding-box intersection with the slice window. Thus
one SGN can count in two positions 0.05 apart. On 4 of the 8 f-Chrimson gerbil cochleae, one
position pair shares more than 10% of the SGN ids (2026-07-31,
`analysis.density_utils.report_density_overlap`). The error bars can be too small for this reason.

## 5. "untreated cochleae 95% CI" band (4C, 5D, 6C)

The band is mean +/- 1.96 x population SD of 4 untreated cochleae:
- 4C: SGN counts of the 4 iDISCO cochleae, 10114 to 12238 SGNs.
- 6C: IHC counts of the 4 iDISCO cochleae, 599 to 697 IHCs.
- 5D: per cochlea, the mean density of each region of the 4 wild-type gerbils.

The label says "95% CI", but the band is not a confidence interval by construction. For n = 4:

| Interval | Half-width |
|---|---|
| Drawn band: 1.96 x population SD | 1.697 s |
| 95% CI of the mean, t distribution, 3 df: 3.182 x s / sqrt(4) | 1.591 s |
| 95% CI of the mean, normal approximation: 1.96 x s / sqrt(4) | 0.980 s |
| 95% prediction interval for one more cochlea: 3.182 x s x sqrt(1 + 1/4) | 3.558 s |

- As a 95% CI of the untreated mean (t-based), the band is 6.7% too wide. This match is a
  coincidence of n = 4 and the population SD. It does not hold for other n.
- The panels compare single treated cochleae with the band. For this comparison the prediction
  interval applies. The drawn band holds a new untreated cochlea with a probability of only
  about 77%, not 95%.

Legend text (choose one):
- "The band shows the mean +/- 1.96 SD of four untreated cochleae."
- "The band shows the 95% confidence interval of the mean of four untreated cochleae
  (approximated as mean +/- 1.96 SD, 6.7% wider than the t-based interval)."

## 6. 2C error bars around the released model

The error bars show +/- 1 SD over 4 seed replicates (SGN, IHC) or over the 5 best synapse
networks (the released network and 4 seed variants). They are drawn around the value of the
released network, not around the replicate mean. The released SGN network lies outside the
replicate spread: precision 0.887, replicate mean 0.854 +/- 0.008 (population SD; s = 0.010).

Legend text: "Error bars show +/- 1 SD over [replicates], drawn around the released network."

## 7. Literature bands (2D, 5C)

`util._LITERATURE_REFERENCE_VALUES` (set in `faceeda`, 2026-10-20):

| Animal | SGN | IHC | Synapses per IHC |
|---|---|---|---|
| Mouse | 8894 to 11153 | 662 to 693 | 13.69 to 16.48 |
| Gerbil | 25173 to 26828 | 1081 to 1130 | 21.57 to 22.68 |

These values do not match the "Average" rows of `~/Documents/cochleanet/literature_values.ods`.
Those rows hold the values from before `faceeda`. The source, and if each pair is a range or
mean +/- SD, are open.

## 8. S4A and S7C y-axis

When the SD bounds of a trendline are not shown, the code draws them with alpha 0. Matplotlib
autoscaling includes these invisible lines, so they set the y-limits. The S4A lower limit is
about -0.63, against -0.24 without them. The S7C shift is less than 1 pixel locally. Only the axis
range changes. No data is shown incorrectly.

## 9. S4F and S4G legends

The published S4F and S4G legends are swapped relative to their panels. The code pairs them by
content. `~/Documents/cochleanet/Revisions-Figures` has `suppfig04.svg` (2026-10-23) and
`suppfig04_swapped.svg` (2026-09-02). Confirm which version the proofs use.
