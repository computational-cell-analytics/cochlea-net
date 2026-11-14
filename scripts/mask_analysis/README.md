# Input mask analysis

These scripts evaluate the rule that `find_mask` uses to skip empty blocks before the U-Net
prediction (`flamingo_tools/segmentation/unet_prediction.py`). The reference objects come from
the SGN and IHC segmentations on S3, restricted to their relevant components.

All outputs go to `mask_threshold_analysis/` in the repository root. Run the scripts from the
repository root with the `cochlea-net` environment.

| Step | Script | Output | Runtime |
|---|---|---|---|
| 1 | `parse_mask_jobs.py` | `mask_jobs.csv`: the mask rule of every archived mask job | seconds |
| 2 | `mask_datasets.py` | `mask_datasets.json`: datasets and the frozen train/val split | ~15 min |
| 3 | `audit_block_cuts.py` | `audit/`: objects that end on the 128-voxel block grid | ~15 min |
| 4 | `collect_lowres_features.py` | `lowres_s3/`: s3 histograms, block scores, object signal | ~4 h over S3 |
| 5 | `collect_s0_block_counts.sbatch` | `s0_counts/`: exact s0 statistics, harvested masks (HPC) | HPC array job |
| 6 | `evaluate_mask_rules.py` | `evaluation/`: fitted rules, validation, calibration | minutes |
| 7 | `make_report_figures.py` | `figures/` | seconds |
| - | `block_seam_test.py` | cell splits at block faces for one crop and several halos (CPU) | ~10 min per prediction halo |

The results and the proposal are in `mask_threshold_analysis/report.md`.

## Notes

- `mask_datasets.json` is the frozen split. The script refuses to overwrite it without `--force`.
- Step 1 reads the job archives in `~/Documents/job_archive*` and the slurm logs in
  `~/Documents/output_slurm_NHR`.
- Step 5 runs on the cluster. Copy `mask_datasets.json` and `mask_jobs.csv` next to
  `collect_s0_block_counts.py`, set the paths in the sbatch file, and copy the output folder
  back to `mask_threshold_analysis/s0_counts/`. Step 6 uses these files when they exist.
- Step 6 evaluates every rule as a block score `s_b` and a threshold `T` (a block is predicted
  if `s_b > T`). An object is complete if all blocks that its bounding box touches are
  predicted.
- Step 6 options: `--no_s0` ignores the HPC counts, `--signal_floor` sets the minimum relative
  object signal of a target object (default 1.2), and `--output_dir` writes a variant run to
  another folder (for example `evaluation_floor1.0`).
