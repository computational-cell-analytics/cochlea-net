# IHC count analysis of the IHC row correction

This is the full record of the analysis behind `flamingo_tools.postprocessing.ihc_morphology`: the recognition of
IHCs by their morphology, and the merge and split of IHC row instances with it. The work was done on the branch
`ihc_segmentation_mwfls` from 2026-10-29 to 2026-11-10. The branch `ihc_row_correction` (2026-11-16) separates
it from that branch and adds the count analysis on the training crops. `doc/ihc_postprocessing.md` is the
summary.

Run the scripts of this folder from this folder. Every number in this record comes from a file in `results/` or
from the output folders in section 11. The only exception is the earlier runs that section 6 names for the
rejected variants.

## 1. Question

IHC_v11 merges neighbouring IHCs and splits single IHCs into an upper and a lower part. Then the number of IHCs
in a row is wrong. The questions were:
- Q1: Which features describe a correct IHC instance, and how much do they vary between crops, species and
  positions along the cochlea?
- Q2: Can these features find the cells in a merged or split row, without a neural network?
- Q3: Does a correction with these features make the IHC count of the IHC_v11 output more accurate, on validation
  data and on the test data of `scripts/validation/IHCs/run_evaluation.py`?

## 2. Decision and implementation

Decision (2026-11-16): the IHC tips and the side-by-side order of the IHCs along the row path find the cells.
`correct_ihc_row` corrects a window of the IHC_v11 output with them. On the mouse crops, it reduces the absolute
count error per crop by about two thirds, and it makes the count of the test planes a little more accurate
(sections 7 and 8). On the gerbil crops (one animal with IHC_v11 output), the count does not improve. The
correction cannot add cells that the model misses.

Implementation in `flamingo_tools/postprocessing/ihc_morphology.py`:
- `row_frame`: smooths the row path (moving average, half width `PATH_SMOOTHING` 10 um), measures the instances
  along it and estimates the local long axis at each path point from the typical row instances within
  `AXIS_WINDOW` (15 um).
- `foot_points`: moves each voxel along the local axis onto the central line.
- `find_ihc_tips_path`: finds the tips as protrusions of the height profile along the path.
  `find_ihc_tips_height` is the baseline with one axis per crop.
- `ihc_row_reference`: estimates the typical IHC of a crop (size, length along the axis, width along the path,
  axis).
- `correct_ihc_row`: merges, splits and trims the row instances. It returns the corrected labels, a report with
  the class and the action of each input instance, and the tips.

`flamingo_tools.measurements` has the feature set 'ellipsoid' (semi-axes a >= b >= c and the main axis of the
moment ellipsoid). The shape analysis uses it for the standard IHC.

The correction is not part of a CLI or of the S3 tables. It works on a window of a segmentation. The evaluation
scripts cut the windows around the row crossings of a region (`common.correct_region`). The row path is the path
of `refine_ihc_row` (`common.row_path`): one path per row component, linked in the order of the component list,
as in the tonotopic mapping. One path through the union of the components can miss a part of a row with gaps: in
M_AMD_N142_R, it passed 435 um from the cells of crop `0818-1353-0331`, and the correction failed there.

## 3. Data

The reference is the refined annotations of the IHC_v11 training crops:
`~/Documents/lightsheet-cochlea/IHC_v11_training_data_refined` (state of 2026-11-02).
- `scripts/training/2026-11_refine_ihc_training_data.py` made them from `IHC_v11_training_data`, on branch
  `ihc_segmentation_mwfls` (pushed state `8a1aa3e`). The folder has its own `README.md`, and `changes.tsv`
  lists every change.
- The changes: 18 fixes from Martin's comments, 17 label ids that marked two cells split into connected
  components, and ids above uint32 replaced. The micro-sam changes of 2026-11-03 were reverted on 2026-11-05.
- The folders `train` and `val` follow the IHC_v11 split (`DATASETS["IHC_v11"]` in
  `scripts/training/export_training_data.py`).

The analysis uses the 76 crops whose name gives the crop center (`common.gt_crops`):

| species | crops (train / val) | animals | complete IHCs (train / val) |
|---|---|---|---|
| gerbil | 32 (25 / 7) | 2 (G_EK_000233, G_LR_000234) | 200 (156 / 44) |
| mouse | 44 (39 / 5) | 9 | 940 (819 / 121) |

- A cell is a connected component of an annotation label with 5000 px or more (`annotation_cells`). The count
  rule applies the same limit to the output instances. A complete IHC does not touch the crop border.
- The left and right cochlea of one animal are one group (G_LR_000234 L and R).
- Excluded (13 crops):
  - 3 crops of M_LR_000171 (HCAT): the name gives no crop center;
  - 4 empty crops of G_EK_000233_L;
  - 4 crops of M_LR_000226_L: a test cochlea (and the names give no crop center in um);
  - 2 crops of M_AMD_OTOF1_L: the names give no crop center.
- The test cochleae M_LR_000226/227 of `run_evaluation.py` are excluded by name (`TEST_COCHLEAE`).

## 4. Methods

**Ellipsoid features.** The semi-axes and the main axis come from the second moments of the voxels. The bounding
box of the table cannot give the main axis: its error is about 15 degree.

**Path frame.** The path through the instance centers zigzags at the IHC spacing, so it is smoothed first. Without
the smoothing, the local axis deviated by up to 79 degree. The local axis comes from the size-weighted axis tensor
of the typical row instances (0.6-1.4 of the reference size, axis not along the path) within 15 um.

**Foot points.** Each voxel moves along the local axis onto the central line. The two tips of a leaning cell then
share one foot point. A frame with the axis perpendicular to the path paired the top and bottom tips of
neighbours: in G-LR-234-R `0909`, 1 of 14 cells was correct, with foot points 14 of 14.

**Tips.** The height profile of a side holds the extreme height at each foot point. A tip is a peak of the profile
with a prominence of 1 um, at least 4.5 um from the next tip of the side, at most 45 degree from the local axis,
and at 0.25-1.0 of the reference length from the central line. A tip on the crop border is a cut face and is
dropped. The tips of the two sides pair if their foot points are at most 4.5 um apart. A pair or a single tip is
one tip cell.

**Typical IHC (`ihc_row_reference`).** A preliminary IHC comes from the volume-weighted median size, iterated over
0.5-1.5 times the size. If the row columns are more than 1.5 times longer than the typical instance, the instances
are upper and lower parts, and the size is scaled. Then the size and the path width are the medians of the
tip-verified single cells (one top and one bottom tip of the same cell). In a crop with few cells, the sizes alone
often gave a merge of 2-3 cells or a half cell: 74 of 228 simulated inputs before this rule, 14 of 404 after it.

**Correction (`correct_ihc_row`).**
1. Remove instances and pieces of less than 100 voxels.
2. Merge touching row instances at the same foot position (less than 0.4 of the reference path width apart), up
   to `max_cell_size`. Two short instances merge only if they are stacked along the axis.
3. Classify each instance:
   - `off_row`: 15 um or more from the path. A short one is removed first.
   - `normal`: the shape of an annotated IHC (at most one tip per side, size, width, length, elongation and axis
     angle in the ranges of section 5). It stays unchanged.
   - `border`: touches the crop border, at most one tip per side, not too large. It stays unchanged.
   - `uncertain`: all other row instances.
4. Split the union of the uncertain instances into connected components. A component with tip cells is split in
   1D along the foot points. The boundary between two cells is the neck (minimum of the column length) in the
   middle half between their tip centers. A cell below 0.4 of the reference size loses its tips, and two cells
   without a neck of 1 um merge if they fit into `max_cell_size`. A cell above `max_cell_size` is split by its
   size and its extent along the path. This adds the cells without tips.
5. Trim each normal cell to a disk of `max_radius_factor` times the reference width around its center, in the
   plane perpendicular to the local axis. This removes protrusions.
6. Each output label keeps its largest piece. The other pieces go to the label that they touch most.

## 5. Standard IHC and parameters

`results/standard_ihc.tsv` holds the percentiles of each feature per species and tonotopic third. The medians:

| species | IHCs | volume (um^3) | semi-axes a / b / c (um) | elongation a/b | length along the axis (um) | tip distance top / bottom (um) | spacing (um) |
|---|---|---|---|---|---|---|---|
| gerbil | 200 | 2100 | 14.52 / 7.23 / 5.34 | 1.98 | 29.70 | 14.74 / 14.62 | 10.50 |
| mouse | 940 | 1826 | 14.14 / 6.70 / 5.01 | 2.08 | 28.09 | 14.21 / 13.64 | 8.77 |

- Sizes and lengths are relative to the typical IHC of the crop. The relative features do not depend on the
  tonotopy. In mouse (42 crops with a known position), no feature has |Spearman rho| above 0.26 with the length
  fraction. In gerbil (9 crops of one cochlea), the absolute sizes decrease towards the base (volume -0.59, length
  -0.60), the relative ones do not (n_rel 0.02).
- The tips are symmetric in gerbil (a tip on the better side of the crop for 100% of the cells, on the other side
  for 99%). In mouse, the better side of the crop has a tip for 93% of the cells, the other side for 75%.

Parameters of `correct_ihc_row` and how they were set. "All" means all 76 crops, "out of fold" that the
validation in section 6 uses the value of the other animals.

| parameter | default | source |
|---|---|---|
| `max_cell_size` | 1.37 | p99 of the size relative to the crop reference; out of fold |
| `max_cell_width` | 1.39 | p98.5 of the path width; out of fold |
| `max_length_axis` | 1.18 | p99 of the length along the axis; out of fold |
| `elongation` | 1.19-3.02 | p1-p99 of a/b; out of fold |
| `max_radius_factor` | 1.0 | p99 of the trim radius (p98 of the voxel distances); out of fold |
| `min_cell_width` | 0.7 | set by hand: below the p1 of 0.78 (mouse) and 0.85 (gerbil), above a half cell |
| `max_axis_angle` | 30 deg | set by hand at a gap: 97.8% of the IHCs have 20 deg or less, 1.6% more than 30 deg |
| `min_cell_size` | 0.4 | set by hand; 0.5 and 0.6 changed nothing |
| `min_neck_depth` | 1.0 um | about p5 of the necks between annotated neighbours (15 crops) |
| `max_part_distance` | 0.4 | set on the MWfLS crops: neighbours 0.6 or more apart, parts of one cell 0.35 or less |
| tip prominence / spacing | 1.0 / 4.5 um | grid on all crops (section 6) |
| `TIP_DISTANCE` | 0.25-1.0 | p1-p99 is 0.28-0.81; the upper limit has a margin for paths off the cell centers (MWfLS) |
| `PATH_SMOOTHING` / `AXIS_WINDOW` | 10 / 15 um | set by hand |

The defaults of commit `6da9236` are the all-animal values of the refined annotations, rounded to 2 decimals.
Before, `max_cell_size` was 1.38 and `max_radius_factor` 1.01 (annotations of 2026-10-31). The change does not
change any result of the test planes.

## 6. Validation on the training crops: synthetic inputs, out of fold

`shape_analysis.py` (steps `tips` and `partition`) corrects inputs that are made from the annotation, so every
error is known. The settings of `correct_ihc_row` for the crops of one animal come from the annotated IHCs of the
other animals (`correction_settings`). The scenarios:
- `identity`: the annotation itself. The correction must keep it.
- `simulated`: the annotation with simulated errors: runs of 2-3 neighbours merged (probability 0.3) and single
  cells cut in half (probability 0.2), seeds 0-2, with the reference IHC estimated from the input.
- `union`: the connected components of all annotated cells, so touching cells are one instance. The reference IHC
  comes from the annotation (oracle).
- `union_selfref`: the same input, with the reference IHC estimated from the simulated input of seed 0.
- Path `gt`: the path through the annotated cells. Path `s3`: the S3 row path of the cochlea (`common.row_path`),
  as in production. The `s3` scenarios have fewer crops: G_LR_000234 and M_AMD_OTOF2_L have no S3 row.

Scores: F1 at IoU 0.5, the mean over the crops of the mean over the seeds. 'count error' is the summed output
count minus the summed annotated count, relative to the annotated count. The count rule of section 3 applies to
both. 'MAE' is the mean absolute count error per crop, in cells. The 95% CIs come from resampling the animals.
Gerbil has 2 animals, so it has no CI. All 76 crops, out of fold (`results/summary.tsv`, split `all`):

| scenario | path | species | crops | F1 input -> corrected | count error input -> corrected | MAE input -> corrected |
|---|---|---|---|---|---|---|
| identity | gt | gerbil | 32 | 1.000 -> 0.999 | +0.000 -> +0.003 | 0.00 -> 0.03 |
| identity | gt | mouse | 44 | 1.000 -> 0.992 [0.985, 0.998] | +0.000 -> +0.001 [-0.008, 0.010] | 0.00 -> 0.25 |
| simulated | gt | gerbil | 32 | 0.689 -> 0.904 | -0.103 -> +0.037 | 1.11 -> 0.67 |
| simulated | gt | mouse | 44 | 0.769 -> 0.935 [0.918, 0.952] | -0.066 -> -0.015 [-0.023, -0.006] | 1.64 -> 0.62 |
| simulated | s3 | gerbil | 9 | 0.699 -> 0.831 | -0.156 -> +0.090 | 2.11 -> 1.44 |
| simulated | s3 | mouse | 42 | 0.785 -> 0.938 [0.928, 0.952] | -0.128 -> -0.015 [-0.027, -0.001] | 3.17 -> 0.81 |
| union | gt | gerbil | 32 | 0.108 -> 0.922 | -0.813 -> -0.081 | 8.81 -> 0.94 |
| union | gt | mouse | 44 | 0.027 -> 0.813 [0.773, 0.857] | -0.950 -> -0.096 [-0.130, -0.066] | 23.80 -> 2.55 |
| union_selfref | gt | gerbil | 32 | 0.108 -> 0.867 | -0.813 -> -0.040 | 8.81 -> 1.25 |
| union_selfref | gt | mouse | 44 | 0.027 -> 0.800 [0.754, 0.853] | -0.950 -> -0.094 [-0.128, -0.069] | 23.80 -> 2.59 |
| union_selfref | s3 | gerbil | 9 | 0.168 -> 0.770 | -0.787 -> +0.000 | 10.67 -> 2.00 |
| union_selfref | s3 | mouse | 42 | 0.028 -> 0.765 [0.720, 0.826] | -0.949 -> -0.090 [-0.132, -0.058] | 23.57 -> 2.33 |

Findings:
- The correction keeps a correct input. On the annotation itself, the count changes by 0.3% or less. In mouse,
  18% of the crops get a different count (0.25 cells per crop).
- Simulated merges and splits: the absolute count error per crop goes down by 40% (gerbil) and 62% (mouse).
- The union of all annotated cells is the hardest input: every cell must come from its tips. The correction finds
  most cells (F1 0.77-0.92), and the count stays up to 10% too low.
- The S3 row path costs up to 6 F1 points against the path through the annotated cells. Paired on the same crops:
  `union_selfref` mouse -0.037 [-0.078, -0.005] (42 crops), gerbil -0.061 (9 crops); `simulated` mouse +0.002
  [-0.009, 0.013], gerbil -0.050.
- The val crops of IHC_v11 give the same picture (split `val` in `results/summary.tsv`), for example
  `union_selfref`/`gt` mouse 0.866 [0.839, 0.907] (5 crops).
- At IoU 0.75, the F1 is much lower (column `f1_75`, `union` mouse 0.49): the cells are found, but the boundaries
  between touching cells are not exact.
- The estimated reference IHC is outside 0.7-1.5 of the annotated reference in 14 of 406 inputs (log of the
  `partition` step).

Tip detection on all crops (`tips/grid.tsv`, fraction of the 1140 complete annotated IHCs):

| detector | prominence / spacing (um) | one tip cell | no tip | more than one | one paired tip cell |
|---|---|---|---|---|---|
| path (default) | 1.0 / 4.5 | 0.905 | 0.022 | 0.034 | 0.721 |
| height (best) | 2.0 / 4.5 | 0.689 | 0.052 | 0.116 | 0.496 |

Decisions and rejected variants. These runs used the unrefined annotations of 2026-10-29 to 2026-10-31 and were
not repeated. The record is the archived handoff
`~/Documents/MWfLS/handoff_archive/handoff_ihc_segmentation_mwfls_2026-11-05.md` (sections 7, 9 and 11), with the
outputs in `~/Documents/lightsheet-cochlea/ihc_quality/tips/point4` and `ihc_quality/point5_2026-10-31`.
- Path tips against height tips and k-means (the size-based partition of 2026-10-29), `union`, val crops: F1 0.85,
  0.83 and 0.77. On simulated mouse merges, k-means was better at first (val 0.96 against 0.91), because the
  simulated merges are exact multiples of real cells. After the fixes of point 5, the tips reached 0.946 against
  0.957.
- The partition is 1D along the foot points, with the boundary at the neck. A partition by the nearest tip segment
  (F1 0.44-0.71 on the mouse union) and a seeded watershed on the distance transform (about 0.45) were worse.
  Touching mouse IHCs have no clear necks in the distance transform.
- Side weighting: in mouse, one side of a crop has tips for more cells (section 5). The weighted variants (pair
  centers from the better side, dropping unpaired tips of the weaker side, protecting cells with a tip on the
  better side) lowered the `union` F1 of the val crops: mouse 0.838 against 0.859, gerbil 0.900 against 0.914. Both
  sides keep the same weight.
- A guard on the size fallback (fallback only with 5 or more confident instances) fixed 2 gerbil crops with a half
  cell as the estimated reference, but it lowered the `union` F1 on crops with a correct reference (G_EK `1257`,
  `0593`). It was dropped.
- `min_cell_size` 0.5 or 0.6 instead of 0.4 changed nothing on the simulated errors.
- Tip growth from the signal, to restore cut tips of the annotation, is not used: it grew a rim on intact cells.

## 7. Validation on the training crops: real IHC_v11 output

`evaluate_crops.py` corrects the S3 IHC_v11 segmentation in the box of each annotated crop and scores it against
the annotation, with the count rule and the F1 of section 6. The variants are those of the test planes
(section 8). The row of a cochlea comes from its processing file in `reproducibility/processing`; for the files
with `refine_row`, `refined_5000` is the archived row. The correction uses the module defaults.

48 crops of 8 animals have IHC_v11 output with a row: the 7 mouse training cochleae and G_EK_000233_L. 28 crops have
no IHC_v11 row and are skipped (23 of G_LR_000234, 3 of M_LR_000249_R, 2 of M_AMD_OTOF2_L). IHC_v11 trained on the
train crops, so only the 6 val crops are fair for the model output. The correction itself did not see these crops.

`results/evaluation_crops_summary.tsv`, `refined_5000` without and with the correction:

| split | species | crops (animals) | TP | FP | FN | F1 | count error | MAE | exact crops | change of the absolute count error per crop |
|---|---|---|---|---|---|---|---|---|---|---|
| val | mouse | 5 (4) | 116 -> 123 | 21 -> 9 | 19 -> 12 | 0.853 -> 0.921 | +2 -> -3 (135 cells) | 2.00 -> 0.60 | 0.20 -> 0.60 | -1.40 [-2.00, -0.50] |
| val | gerbil | 1 (1) | 8 -> 8 | 3 -> 4 | 3 -> 3 | 0.727 -> 0.696 | +0 -> +1 (11 cells) | 0.00 -> 1.00 | 1.00 -> 0.00 | +1.00 |
| train | mouse | 34 (7) | 710 -> 789 | 155 -> 59 | 146 -> 67 | 0.825 -> 0.926 | +9 -> -8 (856 cells) | 1.97 -> 0.65 | 0.29 -> 0.65 | -1.32 [-1.78, -0.97] |
| train | gerbil | 8 (1) | 106 -> 106 | 5 -> 2 | 5 -> 5 | 0.955 -> 0.968 | +0 -> -3 (111 cells) | 0.25 -> 0.38 | 0.75 -> 0.62 | +0.12 |

Findings:
- Mouse: the correction raises the F1 from 0.83-0.85 to 0.92-0.93, and the absolute count error per crop goes down
  by two thirds. The gain is the same on the val crops and on the train crops.
- The summed count error is small before and after the correction (mouse +1.1% and -1.1%), because too many and too
  few cells cancel over the crops. Only the error per crop shows the gain.
- Gerbil (one cochlea): the F1 goes up a little, but the count does not improve. Without the correction, the count
  is already exact in 7 of 9 crops. In the val crop G_EK `1257`, the correction gives 12 cells for 11. The
  annotation of this crop has known defects (cut tips, the boundary between cells 3 and 4).
- The crops are on the row, so the row filter removes few objects: 6 FP over all crops (`objects` against
  `components`). `components` and `refined_5000` give the same scores in mouse. In gerbil, `refined_5000` removes
  1 FP more.
- No correction window failed.

## 8. Verification on the test planes

`evaluate_planes.py` scores the 12 consensus planes of `run_evaluation.py` (cochleae M_LR_000226/227, 320 points,
matching tolerance 5 px). The `components` variant replays the published IHC_v11 rows (`PUBLISHED_ROWS`) and
gives the scores of `reproducibility/model_accuracy/IHC_3D.json`; the script checks this. `results/evaluation_planes.tsv`:

| variant | TP | FP | FN | precision | recall | F1 | objects - points | FN: model miss / merge |
|---|---|---|---|---|---|---|---|---|
| components (published) | 263 | 19 | 57 | 0.933 | 0.822 | 0.874 | -38 | 31 / 26 |
| components + correction | 268 | 17 | 52 | 0.940 | 0.838 | 0.886 | -35 | 31 / 21 |
| refined_1000 | 263 | 19 | 57 | 0.933 | 0.822 | 0.874 | -38 | 31 / 26 |
| refined_5000 | 263 | 16 | 57 | 0.943 | 0.822 | 0.878 | -41 | 31 / 26 |
| refined_5000 + correction | 268 | 16 | 52 | 0.944 | 0.838 | 0.887 | -36 | 31 / 21 |
| refined_10000 | 262 | 14 | 58 | 0.949 | 0.819 | 0.879 | -44 | 31 / 27 |

- A missed point (FN) is a model miss if no row object is within the matching tolerance, else it is part of a
  merge. In the published rows, every plane has fewer objects than points.
- The correction splits 5 merges (226R z859: 2, 227L z1346, 227L z772, 226L z1200), so TP goes up by 5. The FP
  stay about the same (`components` 19 -> 17, `refined_5000` 16 -> 16). The correction cannot add the 31 model
  misses (13 in 226L z485).
- A larger `min_size` of `refine_ihc_row` removes FP and TP. 5000 is the IHC_v11 default since 2026-11-11.
- Records: the branch record of 2026-11-05 has FP 17 (`refined_5000` + correction) and 18 (`components` +
  correction). The differences come from the `refine_ihc_row` of master (226_L: 654 instead of 648 row
  instances) and from the row path of `common.row_path` (227R z1178). The defaults of `6da9236` change nothing.
- The S3 tables of the test cochleae got new rows on 2026-11-11. The script replays the published rows from the S3
  instances, so the S3 component labels do not matter.

Overlap of the IHC_v11 training data with the test planes. The M226L crop names give the crop center in pixels,
in (z, y, x) order (`doc/data/IHC_v11.tsv`).
- The train crop `M226L_Vglut3_0516-1274-1078` (z 484-548) holds 13 of the 35 points of 226L z485. The model misses
  10 of these 13 cells. The annotation of the crop labels only 3 of the 13 cells (z 485-495). The overlap did not
  make the plane easier. The missing annotations are a probable cause of the misses.
- The val crop `M226L_Vglut3_2216-0850-1675` (z 2168-2264) holds 3 of the 7 points of 226L z2214.
- This is a problem of the model evaluation, not of the post-processing. The analyses of sections 5-7 do not use
  M226L (`TEST_COCHLEAE`).

## 9. Limits and open items

- The correction cannot add cells that the model misses: 31 of the 52 FN of the test planes.
- The boundaries between touching cells are not exact (F1 at IoU 0.75, section 6).
- The S3 row path costs up to 6 F1 points (section 6). The path is not at the cell centers.
- The results of single crops are sensitive to small path changes. A new sampling of the same S3 path (points moved
  by 0.5 um or less) changed the F1 of single crops in the `s3` scenarios by up to 0.10, and the mean F1 of mouse by
  0.003 or less. Compare variants with the paired deltas over many crops (`--baseline`), not on single crops.
- The module defaults come from all 76 crops. The real-output evaluation (section 7) uses them on crops of the
  same animals, so it is in sample for the settings. Section 6 is out of fold.
- Gerbil: 2 animals in section 6, and 1 animal with IHC_v11 output (section 7). The gerbil numbers have no CI.
- A window with only 1-3 complete row instances gives no reliable typical IHC.
- The reference annotations have known defects: missing cells (N97-L `0632`, N97-L `0216`, N142-R `0875`, MOTOF1L
  `0580`), cut tips (`0797`, `1157`, `0868`, `0478`, `0758`, `1257`, `0572`), nucleus holes (249-R `0800`, M226L
  `2021`), the boundary between cells 3 and 4 of G_EK `1257`, and the missing cells of M226L `0516` (section 8). A
  napari tool to fix them is the next step. After the fix, rerun all three scripts.
- Open: `filter_ihc_by_path_deviation` measures the distance to the sparse nodes of the graph path, not to the path
  sampled at 1 um. A fix changes the replays of G_LR_000233_R and G_LR_000234_R.
- The correction is not in a CLI and not in the S3 tables.

## 10. Cases from the MWfLS test crops

The MWfLS crops of `~/Documents/MWfLS/ihc_crops` are no reference. They were a testing ground for rules that the
training crops did not show. 55 of 56 comment cases of Martin pass (`2026-10_ihc_mwfls_crops.py -p` on branch
`ihc_segmentation_mwfls`). The cases behind the rules:
- 127_L `0826`: most objects are upper or lower parts of a cell, so the size-only reference was too small
  (21.7k px instead of about 30-37k). 1385 and 1541 are the parts of one cell and sit at one position along the
  path. A trim with the too small reference removed up to 94% of a cell, so only normal cells are trimmed.
- 126_R `0448`: the IHC_v9 path is not at the cell centers. An upper tip distance of 0.85 misses Martin's tips,
  so the limit is 1.0. At the end of a component, the column length is a slope without a neck. 1748 and 1751 are
  two parts of one cell.
- 126_R `1086`: short fragments of two neighbouring cells sit at about the same position. They must not merge, so
  two short instances merge only if they are stacked along the axis. 1140 gives 3 cells instead of 2 (not tuned).
- 126_R `0751`, label 191: a short fragment with the size of a cell, but not its axis.
- The neighbour spacing of the MWfLS cells (0.6 or more of the reference width) against the parts of one cell
  (0.35 or less) sets `max_part_distance` 0.4.

## 11. Rerun

Environment: the `cochlea-net` micromamba env. Set `PYTHONPATH` to the checkout if the editable install points
to another clone. The scripts read S3 (the tables and the IHC_v11 segmentations).

```bash
cd scripts/ihc_postprocessing
# Shape of the annotated IHCs and the out-of-fold validation (about 2 h, 3 worker processes of a few GB).
# The steps: reference (3 min), tips (40 min), partition (about 1 h, needs tips/gt_cells.tsv), standard.
python shape_analysis.py -o ~/Documents/lightsheet-cochlea/ihc_quality/count_analysis_2026-11
# Real IHC_v11 output at the training crops (about 1 h, 35 min with the cache).
python evaluate_crops.py -o ~/Documents/lightsheet-cochlea/ihc_postprocessing/count_analysis_2026-11 \
    --cache ~/Documents/lightsheet-cochlea/ihc_postprocessing/cache_crops
# Test planes (about 10 min with the cache).
python evaluate_planes.py -o ~/Documents/lightsheet-cochlea/ihc_postprocessing/count_analysis_2026-11 \
    --cache ~/Documents/lightsheet-cochlea/ihc_postprocessing/cache
```

Do not run the shape analysis and the crop evaluation at the same time on a 32 GB machine.

The run of 2026-11-16 reproduces `tips/grid.tsv`, `tips/gt_cells.tsv`, `tips/gt_references.tsv`,
`tips/partition.tsv` (path `gt`) and `standard_ihc.tsv` of the branch run in
`~/Documents/lightsheet-cochlea/ihc_quality/refined_2026-11-02`. In `gt_objects.tsv`, only the column
`path_distance` changed, because of the row path of `common.row_path`.

| script | output | in `results/` |
|---|---|---|
| `shape_analysis.py` | `gt_objects.tsv`, `tips/{grid,gt_cells,gt_references,partition}.tsv`, `standard_ihc.tsv`, `summary.tsv` | `standard_ihc.tsv`, `summary.tsv` |
| `evaluate_crops.py` | `evaluation_crops.tsv`, `evaluation_crops_summary.tsv`, cached S3 crops | both |
| `evaluate_planes.py` | `evaluation_planes.tsv`, `planes_IHC_v11/*+correct.tif` | `evaluation_planes.tsv` |

Inputs:
- the refined training crops (section 3);
- the consensus annotations of the test planes:
  `~/Documents/lightsheet-cochlea/final_submission/evaluation_data_standardized/IHC_evaluation/consensus`;
- the processing files of the training cochleae in `reproducibility/processing` (IHC_v11 rows) and the IHC_v4b
  rows in `reproducibility/training_crops` (S3 row path and tonotopy of the shape analysis);
- `reproducibility/model_accuracy/IHC_3D.json` (check of the published test rows).
