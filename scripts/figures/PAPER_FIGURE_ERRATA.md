# Errata of the published CochleaNet figures

Status: 2026-11-05. Branch `fix_figure_statistics`. This branch is not merged into master.

The published panels keep their appearance. The code in this folder renders them as published:
the 2D and 5C renders are pixel-identical to master. The only addition is a warning in
`literature_comparison.plot_counts_vs_literature` for values outside the y range.

This note lists the inaccuracies of the published panels, so that the figure legends can describe
the data correctly. The commits `0795119` to `45b95b2` on this branch hold a corrected rendering of
the panels (sample SD, points for n = 4, n text, higher axes, axis break). Panel ids follow the
published numbering. s is the sample SD (divisor n - 1). Section 10 lists the open points.

## 1. Values outside the y range (2D, 5C)

Cluster run 2026-11-04. The counts come from the warning. The approximate values come from a
render with a higher axis. The run was before `c623fce`, so the warning did not print the values.
The next cluster run prints the exact values (section 10).

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

- 2D: 2,592 is the sum of the IHC_v11 counts of the 4 cochleae (687 + 648 + 617 + 640). Thus the
  pool holds every IHC of the 4 cochleae.
- 5C: the IHC_v11 counts of the 3 cochleae add up to 3,022 (1018 + 1074 + 930). If the synapse
  tables hold all these IHCs, the zero removal leaves out 519 IHCs (17%). Section 10 has the check.

Inconsistencies:
- 2D keeps IHCs without synapses, but 5C removes them.
- 5C uses G_LR_000301_L for SGN and IHC, but not for synapses. Its CTBP2 staining has no punctate
  signal over about a third of the helix: about 4.4 detections per IHC, against 13 to 16 in the
  other cochleae (commit `8fc9f11`).

Legend text:
- 2D: "SGN and IHC: n = 4 cochleae. Synapses: 2,592 IHCs pooled from the same four cochleae,
  including IHCs without synapses."
- 5C: "SGN and IHC: n = 4 cochleae. Synapses: 2,503 IHCs with at least one synapse, pooled from
  three cochleae. [k] IHCs without synapses are not shown."

## 3. Box plots of four values (2D, 5C, SGN and IHC)

Each box summarizes 4 values. The quartiles are linear interpolations between these values. No
value lies outside 1.5 x IQR, so the whiskers show the minimum and the maximum.

Values (`util.VALUE_DICT`): 2D SGN 10333, 11153, 11398, 11820; 2D IHC 617, 640, 648, 687;
5C SGN 18541, 21801, 22284, 23717; 5C IHC 930, 946, 1018, 1074.

Legend text: "Box plots of four cochleae. The box shows the interpolated quartiles, the whiskers
show the minimum and the maximum."

## 4. SD definitions and n

The published code uses two SD definitions:
- Population SD (divisor n, numpy default): 2C error bars, 3C gray band, the 4C, 5D and 6C band,
  5D error bars.
- Sample SD (divisor n - 1, pandas default): the 3F, S4B and S4G bands.

The population SD is s x sqrt((n - 1) / n): 18.4% smaller than s for n = 3, 13.4% for n = 4 and
10.6% for n = 5.

| Panel | Element | n | SD |
|---|---|---|---|
| 2C | Error bars | 4 seed replicates (SGN, IHC), 5 synapse networks (section 6) | population |
| 3C | Gray band, per octave band | 4 iDISCO cochleae | population |
| 3F | Colored band, per octave band | 2 cochleae (M_LR_000098_L, M_LR_N152_L) | sample |
| S4B | Colored band, per octave band | 3 cochleae (M_LR_000184_L/R, M_LR_000260_L) | sample |
| S4G | Colored band, per octave band | 3 cochleae (M_LR_N110_L/R, M_LR_N152_R) | sample |
| 4C, 6C | Untreated band (section 5) | 4 iDISCO cochleae | population |
| 5D | Untreated band (section 5) | 4 wild-type gerbil cochleae, one mean per region | population |
| 5D | Error bars | 3 density positions of one region of one cochlea | population |

- 3F: the sample SD of two values a and b is |a - b| / sqrt(2). Thus the band is the mean +/-
  0.71 x |a - b|, and it extends past both values.
- 5D error bars: the 3 positions inside one region are not fully independent.
  `sgn_density_at_position` selects SGNs by bounding-box intersection with the slice window. Thus
  one SGN can count in two positions 0.05 apart. On 4 of the 8 f-Chrimson gerbil cochleae, one
  position pair shares more than 10% of the SGN ids (2026-07-31,
  `analysis.density_utils.report_density_overlap`). The error bars can be too small for this
  reason.

2C error bars as published (population SD) and as sample SD:

| Model | Precision | Recall | F1 |
|---|---|---|---|
| SGN (4 seeds) | 0.0084 (s 0.0097) | 0.0019 (s 0.0022) | 0.0052 (s 0.0060) |
| IHC (4 seeds) | 0.0079 (s 0.0091) | 0.0121 (s 0.0139) | 0.0048 (s 0.0056) |
| Synapse (5 models) | 0.0177 (s 0.0198) | 0.0290 (s 0.0324) | 0.0155 (s 0.0173) |

Legend text: give the n of the table above, and "SD with divisor n" for the population-SD panels.

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

The error bars show +/- 1 population SD over 4 seed replicates (SGN, IHC) or over 5 synapse
networks. They are drawn around the value of the released network, not around the replicate mean.
For 5 of 9 bars, the replicate mean lies outside the bar:

| Model | Metric | Released | Replicate mean +/- SD | Distance in SD |
|---|---|---|---|---|
| SGN | Precision | 0.887 | 0.8545 +/- 0.0084 | 3.9 |
| SGN | Recall | 0.880 | 0.9180 +/- 0.0019 | 20.3 |
| SGN | F1 | 0.884 | 0.8852 +/- 0.0052 | 0.2 |
| IHC | Precision | 0.933 | 0.9368 +/- 0.0079 | 0.5 |
| IHC | Recall | 0.822 | 0.8328 +/- 0.0121 | 0.9 |
| IHC | F1 | 0.874 | 0.8818 +/- 0.0048 | 1.6 |
| Synapse | Precision | 0.927 | 0.9620 +/- 0.0177 | 2.0 |
| Synapse | Recall | 0.799 | 0.7628 +/- 0.0290 | 1.2 |
| Synapse | F1 | 0.858 | 0.8504 +/- 0.0155 | 0.5 |

Synapse networks: `ACCURACY_MODEL_SETS` in `paper_cochleanet_figures.py` uses `v3-4`, `v3`,
`v3-3-best`, `v3-6` and `v3-5-epoch71`. `reproducibility/model_accuracy/synapses.json` holds 7
seeds: `v3` (released), `v3-1` to `v3-4` (`v3-3` and `v3-3-best` for seed 3), `v3-5-epoch71` and
`v3-6`. The set is the 5 seeds with the highest F1 in `synapses.json`, which are the scores that 2C
shows. It includes the released network. Commit `ed40261` (2026-08-27) replaced the set `v3-1` to
`v3-4` with it. The selection makes the bars smaller:

| Synapse network set | Precision SD | Recall SD | F1 SD |
|---|---|---|---|
| Published: the 5 highest F1 | 0.018 | 0.029 | 0.016 |
| All 7 seeds (`v3-3` for seed 3) | 0.017 | 0.050 | 0.031 |
| `v3-1` to `v3-4` (before `ed40261`) | 0.011 | 0.055 | 0.038 |

Legend text: "Error bars show +/- 1 SD (divisor n) over four networks trained with different
random seeds (SGN, IHC), or over the five of seven synapse networks with the highest F1, including
the released network. The bars are drawn around the released network."

## 7. Literature bands (2D, 5C)

`util._LITERATURE_REFERENCE_VALUES` (set in `faceeda`, 2026-10-20):

| Animal | SGN | IHC | Synapses per IHC |
|---|---|---|---|
| Mouse | 8894 to 11153 | 662 to 693 | 13.69 to 16.48 |
| Gerbil | 25173 to 26828 | 1081 to 1130 | 21.57 to 22.68 |

`literature_values.md` in this folder lists every entry in a band, with checked DOIs, and the
errors of the source spreadsheet. The values before `faceeda` came from an older version of the
spreadsheet.

Each band is (mean of the lower bounds, mean of the upper bounds) over its entries. The bound pair
of one entry is a reported range, mean -/+ SD, mean -/+ SEM, or one reported value. Thus the band
is not a range, a confidence interval or an SD of the literature values.

Entries per band: mouse SGN 6, IHC 4, synapses 13; gerbil SGN 4, IHC 1, synapses 6.

Shortcomings of the entries in the bands:
- Ages before hearing onset: gerbil SGN P0 and P7 (Richter et al., 2011; 2 of 4 entries), mouse
  synapses P4-6 (Ismail Mohamad et al., 2024; 28.8 to 39.8, the highest entries) and P7 (Guo et
  al., 2022).
- Juvenile ages (P14 to P28) in the mouse SGN, IHC and synapse bands.
- Mouse SGN: the entry of Richter et al., 2011 (10402) comes from Table 1 of Schettino and Lauer,
  2013. The paper title names only gerbils. The mouse count was not checked in the paper.
- Gerbil IHC: the band is one reference (Hutson et al., 2020).
- Guo et al., 2022: two mouse synapse entries have the age P14, with different values.

Citation errors of the spreadsheet in entries of the bands (`literature_values.md` has the
corrections): Richter et al. "2001" (2011), "Keppler" (Keppeler), "Libermann" (Liberman),
"Mohammad et al., 2014" (Ismail Mohamad et al., 2024), and the gerbil SGN P7 entry with the DOI of
another paper (`10.1016/j.heares.2011.02.004`, correct `10.1016/j.heares.2011.02.003`).

Legend text: "The shaded band spans the mean lower bound to the mean upper bound of the values
reported in the literature (entries: mouse 6 SGN, 4 IHC and 13 synapse; gerbil 4, 1 and 6)."

## 8. S4A and S7C y-axis

When the SD bounds of a trendline are not shown, the code draws them with alpha 0. Matplotlib
autoscaling includes these invisible lines, so they set the y-limits. The S4A lower limit is
about -0.63, against -0.24 without them. Only the axis range changes. No data is shown incorrectly.

The S7C shift was less than 1 pixel in a local render. That render used a copy of G_LR_000301_R in
place of G_LR_000301_L, because G_LR_000301_L has no tables outside the cluster. The S7C shift on
the real data is open.

## 9. S4F and S4G legends

The published S4F and S4G legends are swapped relative to their panels. The code pairs them by
content. Two versions of the Supplementary Figure 4 file exist: one of 2026-09-02, named
"swapped", and one of 2026-10-23. Confirm which version the proofs use.

## 10. Open points

1. Exact hidden values (section 1): run 2D and 5C on the cluster. The warning prints the values.
   Update the table and the legend text.
2. IHCs without synapses in 5C (section 2): count the rows of the 3 synapse tables in
   `ihc_counts_v11`, and the rows with `synapse_count` 0. Put the number into the 5C legend text.
3. S7C y-axis on the real data (section 8).
4. S4F and S4G: the file version in the proofs (section 9).
5. Synapse networks of 2C (section 6): the reason for `v3-3-best` in place of `v3-3` and for
   `v3-5-epoch71`, and the data that selected these checkpoints.
6. The literature table of the paper: compare it with `literature_values.md` (citation errors, the
   age of the first Guo et al., 2022 P14 entry).
