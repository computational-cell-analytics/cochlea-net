# IHC row correction

IHC_v11 merges neighbouring IHCs and splits single IHCs into an upper and a lower part. Then the IHC count of a
row is wrong. `correct_ihc_row` in `flamingo_tools/postprocessing/ihc_morphology.py` corrects the instances of an
IHC row with the morphology of the IHCs. This page summarizes the analysis behind it: the data, the validation on
the training crops and the verification on the IHC test planes.

The full record, the scripts and the result tables are in
[`scripts/ihc_postprocessing`](../scripts/ihc_postprocessing/README.md). The
[analysis documentation](analysis.md#ihc-row-correction) shows how to call the correction.

## The method

- Each IHC has a tip at both ends of its long axis, and the IHCs sit side by side along the row.
  `find_ihc_tips_path` finds the tips as peaks of the height profile along the smoothed row path. Each voxel
  moves along the local axis onto the central line (its foot point), so a leaning cell keeps both tips.
- `ihc_row_reference` estimates the typical IHC of a window from the cells with one top and one bottom tip.
- `correct_ihc_row` merges the parts of one cell, splits the other instances at the neck between their tip cells,
  and trims the protrusions of normal cells. It returns the corrected labels and a report with the class and the
  action of each input instance.
- The row path must follow the row. Use the path of `refine_ihc_row`: one path per row component, linked in the
  order of the component list. One path through the union of the components can miss a part of a row with gaps.

The correction works on a window of a segmentation. It is not part of a CLI or of the S3 tables.

## The data

The reference is the refined annotation of the IHC_v11 training crops (2026-11-02): 76 crops of 11 animals with
1140 complete IHCs. The test cochleae M_LR_000226/227 are excluded. The standard IHC (`results/standard_ihc.tsv`,
medians):

| species | crops | IHCs | volume (um^3) | semi-axes a / b / c (um) | length along the axis (um) | spacing (um) |
|---|---|---|---|---|---|---|
| gerbil | 32 | 200 | 2100 | 14.52 / 7.23 / 5.34 | 29.70 | 10.50 |
| mouse | 44 | 940 | 1826 | 14.14 / 6.70 / 5.01 | 28.09 | 8.77 |

The correction uses sizes and lengths relative to the typical IHC of the window. These relative features do not
depend on the position along the cochlea. The limits of `correct_ihc_row` are percentiles of these features.

## Results

Training crops (F1 at IoU 0.5; MAE is the mean absolute count error per crop, in cells; 95% CI from resampling
the animals). The synthetic inputs are made from the annotation, and the settings come from the other animals (out
of fold). The real inputs are the IHC_v11 output with the archived IHC row (`refined_5000`).

| input | species | crops | F1 before -> after | MAE before -> after |
|---|---|---|---|---|
| synthetic merges and splits, annotated path | gerbil | 32 | 0.689 -> 0.904 | 1.11 -> 0.67 |
| synthetic merges and splits, annotated path | mouse | 44 | 0.769 -> 0.935 | 1.64 -> 0.62 |
| all touching cells merged, S3 row path | mouse | 42 | 0.028 -> 0.765 | 23.57 -> 2.33 |
| real IHC_v11 output, IHC_v11 val crops | mouse | 5 | 0.853 -> 0.921 | 2.00 -> 0.60 |
| real IHC_v11 output, IHC_v11 train crops | mouse | 34 | 0.825 -> 0.926 | 1.97 -> 0.65 |
| real IHC_v11 output, all crops | gerbil | 9 | 0.934 -> 0.942 | 0.22 -> 0.44 |

IHC test planes of `scripts/validation/IHCs/run_evaluation.py` (12 planes, 320 points, matching tolerance 5 px):

| variant | TP | FP | FN | F1 | objects - points |
|---|---|---|---|---|---|
| published IHC_v11 rows | 263 | 19 | 57 | 0.874 | -38 |
| refined rows (`refined_5000`) | 263 | 16 | 57 | 0.878 | -41 |
| refined rows + `correct_ihc_row` | 268 | 16 | 52 | 0.887 | -36 |

## Findings

- On mouse crops with real IHC_v11 output, the correction reduces the absolute count error per crop by two thirds.
  On the IHC_v11 val crops, the change is -1.40 cells per crop [-2.00, -0.50].
- The summed count error is near zero before and after the correction, because too many and too few cells cancel
  over the crops. Report the count error per crop.
- Gerbil (one animal with IHC_v11 output): the F1 goes up a little, but the count does not improve. Without the
  correction, 7 of 9 gerbil crops already have the exact count.
- On the test planes, the correction splits 5 merges. 31 of the 52 missed points are model misses: no row object
  is near them. Post-processing cannot add these cells.
- The IHC_v11 train crop `M226L_Vglut3_0516-1274-1078` overlaps the test plane 226L z485. Its annotation labels
  only 3 of the 13 cells in the overlap, and the model misses 10 of them. This is a problem of the model
  evaluation, not of the post-processing.

## Limits and open items

- The boundaries between touching cells are not exact: at IoU 0.75, the F1 is much lower.
- The S3 row path is not at the cell centers. It costs up to 6 F1 points against the path through the annotated
  cells.
- The results of single crops are sensitive to small changes of the path. Compare variants with paired deltas over
  many crops.
- The module defaults come from all 76 crops, so the real-output evaluation is in sample for the settings. The
  synthetic validation is out of fold.
- The reference annotations have known defects (missing cells, cut tips, nucleus holes). A napari tool to fix them
  is the next step. After the fix, rerun the analysis.
- Open: `filter_ihc_by_path_deviation` measures the distance to the sparse nodes of the graph path, not to the
  path sampled at 1 um.
