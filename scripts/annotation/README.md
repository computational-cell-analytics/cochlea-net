# Annotation correction: validation

The scripts validate the correction tool `flamingo_tools.correct_annotation` (`flamingo_tools/annotation`,
usage in [doc/annotation.md](../../doc/annotation.md)) on the refined IHC annotations of the IHC_v11 training
crops (state of 2026-11-02). The tool is work in progress and ready for tests on IHC crops. The scripts read the
crops from `CROP_ROOT` (in the script, or in `scripts/ihc_postprocessing/common.py`).

- `validate_fill.py`: the completion of a cell from sparse z slices, with a simulated user.
- `detect_issues.py`: the issue detectors, and their recall on the known open issues.
- `validate_region.py`: the region proposal (`--region A B`), with stretches of the annotation left out.
- `validate_count.py`: the cell count of a stretch from the row spacing, with stretches of the annotation left out.
- `compare_annotations.py`: compares corrected annotations of a crop (changed cells, pieces, holes, interleaved
  borders).
- `clean_annotation.py`: the cleanup of the tool as a post-processing step on a finished annotation.
- `replay_session.py`: computes a finished session again with the current fill and cleanup, from the stored edits.
- `ihc_v11_refined_open_issues.tsv`: the known open issues of the refined annotations (missing cells, cut tips,
  nucleus holes, a wrong boundary) as an `--issues` table.

## Completion from sparse slices (`validate_fill.py`)

Two complete cells of each crop without an open issue: 142 cells of 71 crops; in 74 the long axis lies in the xy
plane, in 68 it runs along z. The scenarios:

- `cut2`, `cut4`: one tip is cut by 2 or 4 um along the long axis.
- `cut4_noisy`: as `cut4`, but the user paints the cell 1 pixel too large or too small on each painted slice.
- `removed`: the cell is missing. The first slices come from the cell eroded by 2 um, as from an imperfect proposal.
- `boundary`: the voxels of the cell within 1.5 um of a touching neighbour belong to the neighbour.

The simulated user works like the correction loop. Each visit takes the slices of `select_slices` for the current
cell and repaints a slice with the annotation if it differs from it by more than 10% of the cell area on the slice
(at least 20 pixels). The visits repeat until no slice is repainted. Dice is between the corrected cell and the
annotation, the neighbour error is the fraction of the neighbour voxels within 5 um that change their label, and
"changed slices" counts the repainted slices over all visits (the work of the user).

```bash
python validate_fill.py -o OUT --stage params -j 5
python validate_fill.py -o OUT --stage axis -j 6
python validate_fill.py -o OUT --stage cleanup -j 4
```

The stages `modes` and `params` ran with the first visit region (the cell box + 5 um, without complete
neighbours); the stage `axis` runs with `visit_region`. The region only adds context: z with spacing 2 um, cap 2 um
and vanish 0.7 gives Dice 0.983 in both stages.

### Interpolation or watershed (stage `modes`)

Mean Dice over the 5 scenarios (cap 2 um, vanish 0.5):

| fill | 1 um | 1.5 um | 2 um | 3 um | 4 um |
|---|---|---|---|---|---|
| interpolation | 0.988 | 0.985 | 0.983 | 0.979 | 0.976 |
| watershed, no elevation | 0.986 | 0.983 | 0.980 | 0.973 | 0.963 |
| watershed, gradient magnitude | 0.982 | 0.975 | 0.970 | 0.962 | 0.955 |
| watershed, inverted intensity | 0.979 | 0.971 | 0.967 | 0.962 | 0.959 |

Changed slices per cell: 11.2 / 9.0 / 7.1 / 5.8 / 5.3 for the interpolation at 1 / 1.5 / 2 / 3 / 4 um.

At 2 um, per scenario:

| fill | boundary | cut2 | cut4 | cut4_noisy | removed | neighbour error (boundary) |
|---|---|---|---|---|---|---|
| interpolation | 0.985 | 0.994 | 0.989 | 0.979 | 0.968 | 0.009 |
| watershed, gradient magnitude | 0.974 | 0.992 | 0.983 | 0.970 | 0.932 | 0.024 |
| watershed, inverted intensity | 0.976 | 0.989 | 0.978 | 0.964 | 0.926 | 0.011 |

- The interpolation is the best fill in all scenarios and for both orientations. The image does not help: the
  Vglut3 signal does not mark the cell boundaries reliably (faint cells, nucleus holes), as with `grow_tips` and
  micro-sam before. So the watershed mode and the stage `modes` were removed; the fill is the interpolation.
- A fill only changes the voxels where the edit changes the interpolation. A first version filled the whole
  interval next to a changed slice again, so it replaced annotated detail with the interpolation also where the
  edit did not reach.

### Cap and vanish (stage `params`, interpolation)

| spacing | cap | vanish | Dice | Dice p10 | visits | changed slices |
|---|---|---|---|---|---|---|
| 2 um | 3 um | 0.3 | 0.984 | 0.967 | 3.08 | 8.3 |
| 2 um | 2 um | 0.7 | 0.983 | 0.967 | 2.38 | 6.7 |
| 2 um | 3 um | 0.5 | 0.983 | 0.967 | 2.65 | 7.5 |
| 2 um | 2 um | 0.5 | 0.983 | 0.967 | 2.69 | 7.1 |
| 2 um | 3 um | 0.7 | 0.982 | 0.964 | 2.04 | 5.5 |
| 2 um | 2 um | 0.3 | 0.982 | 0.962 | 1.49 | 4.7 |
| 3 um | 3 um | 0.3 | 0.980 | 0.964 | 3.01 | 6.7 |

The settings change the Dice by less than 0.003. The defaults are spacing 2 um, cap 2 um and vanish 0.7: the Dice
of the best setting with fewer visits. A cut tip along z needs about one visit per cap distance.

### Slice axis (stage `axis`, interpolation, cap 2 um, vanish 0.7)

`--slice_axis long` places `n_slices` slices on the cell along the array axis nearest to its long axis; z places
slices 2 um apart in z. The region is `visit_region` (complete neighbours) in all settings.

| slices | Dice | Dice p10 | Dice xy / z cells | visits | changed slices xy / z cells |
|---|---|---|---|---|---|
| z, 2 um | 0.983 | 0.967 | 0.981 / 0.985 | 2.36 | 5.7 / 7.8 |
| long, 8 | 0.983 | 0.966 | 0.983 / 0.982 | 2.48 | 5.7 / 5.9 |
| long, 10 | 0.985 | 0.972 | 0.985 / 0.984 | 2.61 | 6.6 / 6.9 |
| long, 12 | 0.986 | 0.974 | 0.987 / 0.986 | 2.69 | 7.6 / 7.9 |

Per scenario (z / long 10): boundary 0.985 / 0.981, cut2 0.994 / 0.996, cut4 0.989 / 0.994, cut4_noisy 0.980 /
0.984, removed 0.967 / 0.969.

- With 10 slices, the long axis is a little more accurate than z with the same work. The gain is in the cells that
  lie in the xy plane, where z slices are long sections, and at cut tips. A moved boundary is a little worse.
- The default stays z (decision of 2026-11-17); the long axis is an option.

### Cleanup and lock (stage `cleanup`, interpolation, z, spacing 2 um, cap 2 um, vanish 0.7)

The scenarios `cut4`, `removed` and `boundary`, each also with brush errors on the painted slices: gaps of
2 x 2 pixels in the cell, stray pixels next to it, and left-over pixels of the label that the cell covers (one
of each per 150 pixels of the cell). The processing:
- off: `fill_sparse` alone, every label on the slices editable (round 1);
- focus: only the cell and the cells that touch it change; the result loses its small pieces and holes
  (`clean_labels` without smoothing);
- clean_<sigma>: focus, the cleanup of the edited slices (`clean_slice`) and the border smoothing with sigma in
  pixels.

Pieces are the small pieces per run, holes the background voxels enclosed on z slices, and interleaved the voxels
whose 3x3x3 neighbourhood holds more of a touching cell than of their own (the cell and its neighbours within
5 um). The annotated cells themselves have 0 pieces, 8.9 hole voxels and 147 interleaved voxels.

| processing | Dice (brush) | Dice | pieces (brush) | holes (brush) | interleaved (brush) | neighbour error (brush) | visits (brush) |
|---|---|---|---|---|---|---|---|
| off | 0.9738 | 0.9803 | 16.0 | 325 | 206 | 0.0020 | 2.57 |
| focus | 0.9797 | 0.9813 | 0.03 | 11.2 | 178 | 0.0018 | 2.46 |
| clean_0 | 0.9806 | 0.9809 | 0.03 | 9.8 | 156 | 0.0027 | 2.52 |
| clean_1 | 0.9805 | 0.9808 | 0.03 | 9.5 | 119 | 0.0031 | 2.53 |
| clean_2 | 0.9803 | 0.9807 | 0.03 | 9.1 | 116 | 0.0034 | 2.55 |

- The cleanup removes the effect of the brush errors on the Dice (0.9805 with them, 0.9808 without) and brings
  the pieces and holes to the level of the annotation. Locked cells never changed.
- The border smoothing with 1 pixel takes a quarter of the interleaved voxels away, below the level of the
  annotation. 2 pixels add little. The default is 1 pixel.
- The slice cleanup and the smoothing move the borders to the neighbours a little: the neighbour error rises
  from 0.0018 (focus) to 0.0027 (clean_0) and 0.0031 (clean_1), that is 0.1% of the neighbour voxels.
- A first version smoothed the cell surface against the background too. With 1 pixel, it eroded thin tip slices
  (Dice 0.979, 3.05 visits instead of 2.53, because the simulated user repainted them). The smoothing now moves
  only the borders between the cells of the visit.

### Round 3: union interpolation and holes on all planes (stages `axis` and `cleanup` again)

Round 3 (2026-11-18) changed the fill and the cleanup after the 0216 correction (see the round-3 section below):
- The fill interpolates the union of the labels too; it gives the foreground, and in it the label with the
  largest distance wins. Each label on its own left a background gap where a border moved between two slices.
- The cleanup fills the holes of a changed cell on the planes of all three axes (before: z only), and gives a
  small piece with an edge or corner contact only (near the change) to a neighbour or the background.
- Visits: cells within 1 um of the target are in the focus (`focus_gap`; before: touching cells).

Stage `axis` (fill only): the mean Dice is the same (z 0.983,
long 10 0.985), and the neighbour error is half (z 0.0016 -> 0.0008, long 10 0.0010 -> 0.0004). Stage
`cleanup`, the default `clean_1`:

| version | Dice (brush) | Dice | Dice p10 (brush) | neighbour error (brush) | pieces | holes | interleaved (brush) | visits (brush) |
|---|---|---|---|---|---|---|---|---|
| round 2 | 0.9805 | 0.9808 | 0.9635 | 0.0031 | 0.03 | 9.5 | 119 | 2.53 |
| round 3 | 0.9808 | 0.9811 | 0.9650 | 0.0023 | 0.05 | 11.5 | 122 | 2.49 |

- A first version of round 3 also gave background that the focus cells enclose together (a gap between two cells)
  to the nearest cell, on the planes of all axes. It cost 0.25 visits per cell (2.69 against 2.43) and raised the
  neighbour error: next to a cell that the user has painted on a part of its slices only, it gave the space of the
  cell to its neighbour. An ablation on the variant `focus`:

| hole step | Dice | neighbour error | visits |
|---|---|---|---|
| union of the focus cells, all planes | 0.9805 | 0.0020 | 2.69 |
| union, 3D and z planes | 0.9806 | 0.0015 | 2.47 |
| each cell, z planes (round 2) | 0.9808 | 0.0013 | 2.43 |
| each cell, all planes (round 3) | 0.9810 | 0.0013 | 2.45 |

  The union interpolation prevents the gaps at the source, so the cleanup fills the holes of each cell only. The
  seams of 0216 were holes of one cell in the y and x planes.

### The round-1 correction of 0216 (`compare_annotations.py`, `clean_annotation.py`)

Martin corrected `M-AMD-N97-L_crop_0216-0315-0680_Vglut3` with the first version (S3 base IHC_v11).
`clean_annotation.py` runs the cleanup on the cells that differ
from IHC_v11:

| output | changed cells | pieces (voxels) | duplicate ids | holes 3D / 2D | interleaved |
|---|---|---|---|---|---|
| round 1 | 9 | 25 (93) | 1 (408: 43.8k and 33.9k voxels) | 1214 / 3266 | 1520 |
| round 1 + cleanup, sigma 0 | 10 | 0 | 0 | 0 / 77 | 1470 |
| round 1 + cleanup, sigma 1 | 10 | 0 | 0 | 0 / 56 | 498 |
| round 1 + cleanup, sigma 2 | 10 | 0 | 0 | 0 / 27 | 354 |

The tenth cell is the second piece of 408, which gets the new id 434. Two cells without an item (382, 383,
Dice 0.993 with IHC_v11) changed in round 1 by the fills of the gap items. The left 2D holes enclose another
label. The next correction of 0216 with the new version is compared with the same script.

### The round-3 correction of 0216 (`replay_session.py`, `compare_annotations.py`)

Martin corrected 0216 with `--region 101 399` (S3 base IHC_v11b; 4 region cells 442-445) and then painted a missing
cell 459 by hand between 444 and 397. The output had seams and holes at the borders 442/443 and 443/444: the fill
interpolated each label on its own, so a gap opened where a border moved between two slices. `replay_session.py`
computes the session again from the stored slice edits with the union interpolation and the new cleanup:

| cell | face-connected extra pieces (stored / replayed) | 2D hole voxels, all axes | interleaved |
|---|---|---|---|
| 442 | 0 / 0 | 2 / 0 | 28 / 42 |
| 443 | 17 / 0 | 152 / 0 | 73 / 32 |
| 444 | 16 / 0 | 125 / 0 | 13 / 25 |
| 371 | 8 / 0 | 9 / 0 | 39 / 26 |
| 370, 397 | 0 / 0 | 78 / 3 | 11 / 17 |

The final annotation is the replay with Martin's 459 on top,
459 smoothed (multi-label Gaussian, sigma 2 px, within 4 px of its surface), the protrusion of 398 along 416
removed (z >= 380, 508 voxels) and the holes of the changed cells filled. Against the base IHC_v11b
(`compare_annotations.py`):

| output | changed cells | face-connected extra pieces | 2D hole voxels, z / all axes | interleaved |
|---|---|---|---|---|
| tool output | 6 | 41 | 0 / 366 | 164 |
| Martin's manual edit | 7 | 43 | 0 / 417 | 253 |
| final | 8 | 0 | 5 / 9 | 200 |

The interleaved count of the final includes 459 and 398. The left 9 hole voxels enclose another label. 459 has
28.0k voxels, axis lengths (SD) 6.6 / 2.8 / 1.9 um and a surface roughness (surface voxels / volume^(2/3)) of 5.3,
in the range of its neighbours (31-49k voxels, 6.1-7.1 / 2.8-3.4 / 2.0-2.6 um, roughness 4.8-5.4; the painted
cell had 6.8). An image snap of the 459 borders (seeded watershed on the smoothed intensity) was dropped: the
z profile of the image has an intensity dip at only 9% of the 459/397 border columns and 22% of the 444/459 ones,
against 49-62% at the borders 442/443 and 397/398, so the snap followed noise.

## Issue detectors (`detect_issues.py`)

```bash
python detect_issues.py -o OUT --stage indent -j 2
python detect_issues.py -o OUT --stage detect --reference -j 5
```

### Indent limit (stage `indent`)

The indent fraction is the closing of a cell minus the cell and the other labels, relative to the cell size.
1310 complete cells:

| closing radius | normal cells p50 / p90 / p95 / p99 | cut-tip cells above the normal p95 | hole cells |
|---|---|---|---|
| 1.5 um | 0.011 / 0.055 / 0.073 / 0.109 | 9 of 21 | 0.002-0.037 |
| 2.5 um | 0.017 / 0.070 / 0.093 / 0.135 | 9 of 21 | 0.004-0.051 |
| 3.5 um | 0.021 / 0.079 / 0.105 / 0.150 | 9 of 21 | 0.005-0.068 |

- The radius does not change the separation, so the detector uses 1.5 um (the fastest) with the limit 0.07, the
  95th percentile of the normal cells.
- The cells with a "nucleus hole" comment have no indent: their annotation is not hollow, it lacks a part of the
  cell. The unannotated part of their IHC_v11 object is also normal (5-13% of the cell, normal p90 18%), because
  IHC_v11 misses the same part. No detector finds them; the issue table and the manual items cover them.

### Items and recall (stage `detect`, with the IHC_v11 reference)

84 annotated crops, 81 with an S3 reference (the 171R HCAT crops have no position):

- Items per crop: median 2, mean 5.1, maximum 39 (171R HCAT). Items for 25% of the labels.
- Reasons: `row_*` 210, `short` 145, `indent` 121, `extent_s3` 46, `size` 10, `gap` 10, `missing_s3` 3.

Recall on the open issues, from the detectors alone (the `--issues` table adds the rest):

| issue | found | details |
|---|---|---|
| missing cell | 2 of 4 | 0632 and 0216 by the row gap; the twisted row of N142-R 0875 and the cell at the border of MOTOF1L 0580 are missed |
| cut tip | 13 of 25 | mostly `indent`, `short` and `extent_s3` |
| nucleus hole | 1 of 6 | see above |
| wrong boundary | 2 of 2 | G_EK 1257 3/4 |

Since 2026-11-18, the issue table has 4 nucleus-hole rows instead of 6: 23 and 18 of M-LR-000249-R 0800 are complete
cells that show the nucleus hole inside (Martin). The recall above counts the 6 rows of the run.

## Region proposal (`validate_region.py`)

```bash
python validate_region.py -o OUT -j 3
```

Leave-stretch-out on the refined annotations (`gt_crops` of the IHC post-processing without the open-issue crops):
in each crop, the middle stretch of 2 and of 3 neighbouring typical row cells between two typical row cells is
removed. The proposal fills it from all IHC model segmentations of the cochlea on S3 and their row correction,
with a row path, typical IHC and row correction from the annotation without the stretch. A proposed cell matches
an annotated cell with IoU 0.5 or more. 96 stretches (239 cells) in 50 crops; 14 stretches of IHC_v11 val crops.

| crops | rule | F1 | Dice of the matches | count error per stretch |
|---|---|---|---|---|
| train | newest eligible (default) | 0.971 | 0.923 | 0.15 |
| train | best score, any model | 0.956 | 0.934 | 0.15 |
| train | IHC_v11 only | 0.961 | 0.923 | 0.15 |
| train | typical IHC shape | 0.647 | 0.745 | 0.15 |
| val | newest eligible (default) | 1.000 | 0.921 | 0 |
| val | best score, any model | 1.000 | 0.940 | 0 |
| val | IHC_v11 only | 1.000 | 0.922 | 0 |
| val | typical IHC shape | 0.657 | 0.752 | 0 |

- Winners of the default rule: IHC_v11 row-corrected 187, IHC_v11b row-corrected 43, typical IHC shape 6,
  IHC_v4b row-corrected 2, a union 1. The row correction of a model wins over the model itself in all but a few
  positions, because it ranks above the model and its cells are typical more often.
- The rule 'score' takes the best-shaped candidate of any model (unions 68, IHC_v4b 50, IHC_v10 25 of its
  winners): the matched cells fit better (Dice +0.011 on train, +0.019 on val), but on the train crops it finds
  fewer cells (F1 0.956). IHC_v11 trained on the train crops, so its advantage there is optimistic; the 14 val
  stretches cannot separate the rules by F1.
- The number of cells comes from the row spacing and is wrong in 12 of 96 stretches (all train): one cell too few
  in 6 (mostly gerbil, wide cells) and one too many in 6 (all M-AMD-N89-L). The user then adds or deletes a cell.

Round 3 (2026-11-18): a candidate loses its voxels on the end cells,
also those of the base, and the applied region is cleaned. Default rule: train F1 0.968, Dice of the matches
0.928; val F1 1.000, Dice 0.922. The comparison with the table above is not clean: more cochleae have IHC_v11b on S3
now (IHC_v11b row-corrected wins 127 positions instead of 43), and the difference in F1 is less than one cell of
204. In the validation, the base is the annotation, so the end cells of the base are the annotated ones; the
rule matters for an S3 base (0216: the candidate of 445 had 78% of its voxels on the S3 end cell 397).

## Cell count from the row spacing (`validate_count.py`)

Crop N97-L 0632 (round 4) has a gap of about 51 um between the cells 140
and 160 that the image cannot fill reliably. So the count of a stretch without the cell volumes was checked: the
middle run of 2-5 typical row cells of each crop without an open issue is left out (187 stretches in 65 crops),
and k = round(gap / spacing) - 1.

| Spacing | 2 cells | 3 cells | 4 cells | 5 cells | All: correct | Mean abs. error |
|---|---|---|---|---|---|---|
| crop median (`region._row_spacing`, the `--region` rule) | 0.830 | 0.854 | 0.750 | 0.690 | 0.786 | 0.246 |
| local, 2 gaps on each side, typical cells | 0.868 | 0.771 | 0.659 | 0.690 | 0.754 | 0.283 |
| local, 2 gaps on each side, all row cells | 0.849 | 0.708 | 0.682 | 0.667 | 0.733 | 0.299 |

- The crop median is the best rule, so `--region` keeps it.
- The error is mostly one cell, and it grows with the length of the stretch.
- In crop 0632, the image cues (seams, tips, intensity maxima) counted 4 of 8 left-out stretches correctly, the
  spacing 8 of 8.
- For 140-160 of 0632, the crop median gives 5 cells and the local spacing 6.

## Tip completion

Round 5 (2026-11-18) tested two completions of cut cell tips on simulated cuts: the extension by the matched object
of a model segmentation (4 um cuts: recall 0.81, excess 0.04), and a shape extrapolation from the neighbour cells
(recall 0.81, excess 0.38). The tool does not extend tips from a segmentation for now, so the code was removed.
