# Correction of segmentation crops

**Status: work in progress.** The tool is ready for tests on IHC crops. For SGN crops, it uses the generic
detectors only, and it is not validated on them. The options, the defaults and the format of the session folder
can change.

`flamingo_tools.correct_annotation` corrects the instance segmentation of one IHC or SGN crop in napari. The
start is the S3 segmentation of the crop. The tool finds the cells to correct and shows them one after the other.
For each cell, you correct a few slices, and the tool completes the cell between them. Only the cells with an
expected error and the cells near them change. With `--region A B`, the tool segments the IHC row
between two cells again from the S3 segmentations, and you check the new cells (see [the region
mode](#the-region-mode)).

```bash
flamingo_tools.correct_annotation \
    -i M-AMD-N97-L_crop_0216-0315-0680_Vglut3.tif \
    -a M-AMD-N97-L_crop_0216-0315-0680_Vglut3_annotations.tif \
    -o <output folder> \
    --issues scripts/annotation/ihc_v11_refined_open_issues.tsv
```

- The image name gives the crop name, the cochlea, the position and the stain. The stain gives the cell type
  (Vglut3 or MYO7A: IHC, PV or CR: SGN). Give `--cell_type`, `--cochlea` or `--center X Y Z` (um) if the name
  does not give them.
- At the end, the tool writes `<crop>_annotations.tif` (the original label ids) and `<crop>_changes.tsv` (one row
  per commit) to the output folder.
- The label ids in napari are the ids of the files: the base ids, which the output keeps, and the ids of the
  comparison file. Press `m` for a new label: the tool gives an id that no cell of the crop has. The napari key
  for a new label (the largest id of the layer + 1) can give the id of a cell outside the visit, so the tool
  replaces it.

## The base and the comparison

The tool corrects one segmentation, the base, and shows a second one, the comparison, as a hidden layer.

| Command | Base | Comparison |
|---|---|---|
| without `-a` | S3 segmentation | none |
| `-a <annotation>` | S3 segmentation | the annotation |
| `-a <annotation> --edit_annotation` | the annotation | S3 segmentation |

Without an S3 segmentation (for example a crop without a position), only `--edit_annotation` works.

The S3 segmentation is the first of these sources that the cochlea has:
- IHC: `IHC_v11b`, `IHC_v11`, `IHC_v10`, `IHC_v9`;
- SGN: `SGN_v3`, `SGN_v2`, and `SGN_resized_v2` for the crops of a resized volume.

Give other sources with `--reference IHC_v9,IHC_v4b` (tried in order), or `--reference none` to work offline.
The tool checks the crop position: the center patch of the crop image must equal the S3 image of the stain. The
crop names do not all use the same convention (`M226L_*` crops give the center in pixels), so the tool tries the
known conventions. The S3 credentials are the ones of the other S3 commands.

## The items

Each item is a cell or a missing cell. The reasons of an item:

| Reason | Meaning |
|---|---|
| `pieces` | The label has several connected pieces. |
| `indent` | A closing of the cell adds a cavity or an indent. The added voxels are proposed. |
| `size` | The cell is much smaller or larger than the median cell of the crop. |
| `missing_<tag>` | An object of the comparison has no counterpart in the base. Its voxels are proposed. The tag is `annotation` or `s3`. |
| `extent_<tag>` | The comparison object of the cell reaches past the cell ends. These voxels are proposed. |
| `row_<class>_<action>` | IHC: `correct_ihc_row` does not classify the cell as normal, or changes it. |
| `short` | IHC: the cell is short along its long axis, compared with the standard IHC or with the other cells of the crop. |
| `gap` | IHC: the row has a gap. A typical IHC is proposed in the gap. |
| `table_<reason>` | A row of the `--issues` table. |
| `neighbour_<reason>` | A visit changed this focus cell, and the detector `<reason>` (`pieces`, `indent` or `size`) flags it now. |
| `split` | A separate piece of a cell became a new label after a visit (see the cleanup). |
| `region` | A cell of the region proposal (`--region`). The comment names its source. |
| `region_border` | An end cell of a region. |
| `manual` | You added the cell with "New cell at cursor". |

The detectors do not find every defect. On the refined IHC annotations they found 13 of 25 known cut tips and 2 of
4 missing cells, and they cannot find a cell that lacks a part of its volume at a nucleus with low signal. Give the
known defects in an `--issues` table, and add a missed cell with "New cell at cursor".

The `--issues` table is tab-separated with the columns `crop`, `label_id`, `z`, `y`, `x`, `reason` and `comment`.
`label_id` is an id of the annotation, `all`, or empty for a missing cell at (z, y, x). The positions are napari
positions in the crop. If the base is the S3 segmentation, a table label becomes the S3 label that holds most of
its annotated voxels, or a missing cell if no S3 label holds 30% of them. Without `-a`, a table label needs its
position.

## One visit

napari shows the cell with its complete neighbours: the bounding box of the cell, 5 um larger, grows to hold each
label in it, but at most 15 um beyond the cell (`--margin`, `--max_margin`). The layer positions are the positions
in the crop.

- `image`: the crop image, with the contrast limits 0-3000 at the start (`--contrast_limits LOW HIGH`).
- `segmentation`: the current labels of the base (read-only).
- `edit (sparse slices)`: the target cell and the cells near it on the edit slices, with the proposal of the
  item. These cells are the focus of the visit. A cell is near the target if 20 of its voxels or more
  (`--min_contact`) are at most 1 um from the target (`--focus_gap`; 0 takes the touching cells only).
- The comparison (hidden).

Only the focus cells can change. The dock widget lists them. All other cells are locked, and the tool undoes:
- paint off the edit slices (off the slices, the layer changes to pan and zoom);
- paint on a locked cell;
- paint with the label of a locked cell.

To change a locked cell too, point at it and press `o` (or "Unlock cell at cursor"). The cell becomes part of the
focus, and the edit slices show it.

The edit slices:
- `--slice_axis z` (default): z planes, one slice 0.5 um inside each z end of the cell, slices at most 2 um apart in
  between (`--spacing`), and a cap slice 2 um beyond each z end (`--cap`).
- `--slice_axis long`: planes perpendicular to the array axis (z, y or x) that is nearest to the long axis of the
  cell, with 10 slices on the cell (`--n_slices`) and the two caps. The viewer shows that axis as the slider.

The cell never grows past the cap slices. To extend a cut tip along the slice axis, paint the tip on the cap slice;
the next visit of the cell has new caps further out.

The slices also cover the fragments in the focus: the cells smaller than half the median cell size
(`--fragment_size`). So a fragment that reaches past the end of the target can be removed or merged on the slices.
For any other cell part off the slices, go to its plane and use "Add slice".

| Key | Action |
|---|---|
| `j` / `k` | Previous / next edit slice. |
| `m` | A new label, for example to split a cell. |
| `o` | Unlock the cell under the cursor. |
| `n` | Next: complete the cell from the slices and commit. |
| `Shift-N` | Accept: as Next, and the cell is done. |
| `g` | Preview the completed cell in the segmentation layer. |
| `u` | Reset the edit. |

The buttons also add or remove an edit slice at the current position, defer an item, reject a proposal, delete a
cell, add a missing cell at the cursor, and undo the last visit.

## The loop

- Next on a cell without a change marks the cell as done.
- Next on a changed cell commits the change, and the cell comes back once more at the end of the queue.
- Accept marks the cell as done at once.
- A focus cell that the visit changes comes back into the queue only if the detectors `pieces`, `indent` or
  `size` flag it.
- When the queue is empty, the detectors check the cells that changed after they were done. When they find
  nothing more, the tool writes the output.
- "Undo last visit" goes back one action: Next, Accept (also without a change), Reject, Defer, Delete or a new
  cell. The loop is as it was before the action, and the visit shows your edit of that visit again. Press it
  again to go further back. The commits of the region mode at the start stay.

The session folder `<crop>_session` in the output folder keeps every commit: the labels of the region before and
after, and your raw edit. Close napari at any time and start the same command again to continue. `--restart`
discards the session. The folder stays after the output is written, as the record of the edits;
`--remove_session` removes it.

## The completion between the slices

The tool interpolates the 2D signed distance of each label linearly between two edit slices, and the signed
distance of all labels together. The labels together give the cell voxels; each of them takes the label with the
largest distance. So two cells that share a border leave no gap where the border moves from one slice to the
next. A label on one slice only vanishes at 70% of the way to the other slice. The current labels are
interpolated in the same way, and only the voxels where the two interpolations differ change. So the cell keeps
its detail where your edit does not reach. Labels that you did not change on a slice stay fixed.

The fill does not use the image: on the refined IHC annotations, a seeded watershed on the image was less accurate
than the interpolation alone. The validation is in [`scripts/annotation`](../scripts/annotation/README.md).

## The cleanup after each fill

A brush leaves gaps, stray pixels and pixels of the label that it paints over. The tool cleans each fill. Only
the focus cells change, and only near the voxels that the edit changed:
1. On each changed edit slice: a piece of a cell with less than 2 um2 (`--min_area`) that is not its largest
   piece goes to the touching focus cell with the most contacts, or to the background. A hole of a cell goes to
   the cell. A hole is a region that the cell encloses and that holds only background; a cell body has no lumen.
2. The fill between the slices.
3. Border smoothing: within 1 um (`--boundary_band`) of the changed voxels, a voxel of a focus cell takes the
   focus cell with the largest Gaussian-smoothed mask (`--boundary_sigma`, in pixels; 0 switches it off). The
   locked cells and the background take part, but a voxel keeps its label if one of them wins. So only the
   borders between focus cells move: the outer surface, for example an extrapolated tip, stays.
4. Pieces: each changed cell keeps its largest piece. A piece with 1000 voxels or more (`--split_size`) and no
   contact to the cell is a second cell with the same id: it gets a new label and the reason `split`. A smaller
   piece without any contact to the cell, or with an edge or corner contact only near the change, goes to the
   touching focus cell with the most contacts, or to the background.
5. Holes: the holes of a changed cell near the change, in 3D and on the planes along each axis.

The preview (`g`) shows the cleaned result. `scripts/annotation/clean_annotation.py` runs steps 3-5 on a
finished annotation, for the cells that differ from the base. `scripts/annotation/replay_session.py` computes a
finished session again with the current fill and cleanup, from your stored slice edits.

## The region mode

`--region A B` segments the stretch of the IHC row between the cells A and B again. A and B are ids of the
annotation (`-a`), or of the base without an annotation. Repeat the option for more stretches. Example for the
refined annotation of `M-AMD-N97-L_crop_0216-0315-0680_Vglut3`:

```bash
flamingo_tools.correct_annotation -i <crop>.tif -a <crop>_annotations.tif --edit_annotation \
    --region 101 399 -o <output folder>
```

1. The row spacing of the crop gives the number of cells in the stretch: the distance between A and B along the
   row, divided by the spacing, minus 1. The positions are evenly between A and B.
2. The candidates come from all IHC model segmentations of the cochlea on S3 (`IHC_v<n><suffix>`, for example
   `IHC_v11` and `IHC_v4b` for N97), each as it is and with the IHC row correction (`correct_ihc_row`, the
   post-processing of IHC_v11b). The union of two overlapping candidates of two models and the annotated cells
   of the stretch are candidates too. `--region_sources IHC_v11,IHC_v10` restricts the S3 sources.
3. A candidate loses its voxels on A and B, in the annotation and in the base: the S3 cell of an end can be
   larger than the annotated one. Without half of its voxels, the candidate is dropped. Else it is eligible if
   the rest has the shape of a typical IHC of the crop: 0.6-1.4 of its size, 0.7-1.18 of its length along the
   axis, and 0.7-1.39 of its width along the row.
4. At each position, the eligible candidate of the newest model wins (`IHC_v11b` > `IHC_v11` row-corrected >
   `IHC_v11` > ... > unions > annotation). A tie goes to the candidate with the typical shape, the smallest
   offset and the largest overlap with the other models. Without an eligible candidate, the best candidate with
   0.4 of the typical size wins, else a typical IHC shape.
5. The cells of the base in the stretch are replaced, and the borders between the new cells are cleaned (steps
   3-5 of the cleanup). The items are the proposed cells (reason `region`) and A and B (`region_border`). The detectors do not run; an `--issues` table still adds its rows.
