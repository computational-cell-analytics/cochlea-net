# Processing - Segmentation and detection

Run the segmentation and the detection as Slurm jobs on the GWDG cluster, or on a local machine.

### Using Slurm

The Slurm templates and pipelines for the GWDG cluster are in the repository [slurm-workflows](https://github.com/computational-cell-analytics/slurm-workflows).
Its pipelines run the segmentation of SGNs and IHCs and the detection of ribbon synapses as chains of Slurm jobs.
See its [documentation of the processing pipelines](https://github.com/computational-cell-analytics/slurm-workflows/blob/main/doc/hpc_03_processing_pipelines.md) for the usage.

Each job runs one stage of `flamingo_tools.run_segmentation` or `flamingo_tools.run_detection` with the option `--stage`.
The stages share the output folder and run in this order:
1. `--stage mask` writes `mask.zarr`, the blocks that the network predicts.
2. `--stage predict` runs one task of a Slurm array and writes its part of `predictions.zarr`. `--prediction_instances` must match the size of the array.
3. `--stage segment` writes `segmentation.zarr`. For the detection, `--stage detect` writes `synapse_detection.tsv`.

Without `--stage`, the command runs all stages in one process.

### Locally

Use the command line interface to process data on a local machine.
The examples use short paths. Replace them with the paths of your data.

#### SGN and IHC

```bash
flamingo_tools.run_segmentation -i M_AMD_N162_L.n5 -k setup0/timepoint0/s0 -o SGN_v2 -m SGN --min_size 1000
flamingo_tools.run_segmentation -i M_AMD_N162_L.n5 -k setup1/timepoint0/s0 -o IHC_v11 -m IHC --min_size 1000
```

The command writes the segmentation to `segmentation.zarr` in the output folder.
`--min_size` removes instances with fewer voxels.
The pipelines of slurm-workflows use 1000. The default is 250.

##### Input mask

By default, the U-Net predicts only the blocks with signal above the background level.
The command writes the decision to `mask_statistics.json` in the output folder.
- `--disable_masking` predicts all blocks. Use it for a small crop that cells fill for the most part, because the mask rule needs mostly background to find the background level.
- `--background_factor` sets the threshold as a multiple of the background level (default 1.08 for SGN and 1.75 for IHC). Use a lower value for a very dim stain.
- `--legacy_mask` uses the mask rule of flamingo-tools before 2026-11.

The low-resolution models always use the legacy rule.
See the [input mask documentation](input_mask.md) for the rules and the analysis behind them.

#### Synapses

The standard case matches the detections of a CTBP2 stain to an IHC segmentation:

```bash
flamingo_tools.run_detection -i M_AMD_N162_L.n5 -k setup2/timepoint0/s0 -o synapses --mask_path IHC_v11.ome.zarr
```

The command writes two tables to the output folder:
- `synapse_detection.tsv` contains all detections.
- `synapse_detection_filtered.tsv` contains the detections that are closer to an IHC than `--max_distance` (default 3 micrometer).

The command uses two resolution levels of the IHC segmentation:
- `--mask_input_key` (default `s4`) is the downscaled level. The command dilates it into a mask and runs the prediction only on the blocks inside the mask. On a full cochlea, these blocks are approximately 3 % of the volume.
- `--mask_key` (default `s0`) is the full-resolution level. The command uses it to match the detections to the IHCs.

`--dilation_iterations` (default 4) sets the dilation in voxels of `--mask_input_key`. At `s4`, 4 voxels are 24 micrometer.
Keep the dilation larger than `--max_distance`. Otherwise, the command does not predict synapses near the border of an IHC.
The command shows a warning if the dilation is too small.

Without `--mask_path`, the command predicts on the full volume and does not match the detections.
This prediction is much slower. The command shows a warning.

The command reads the image data and the segmentation independently:
- `--s3_input` reads the image data from the S3 bucket.
- `--s3_mask` reads the IHC segmentation from the S3 bucket.

```bash
flamingo_tools.run_detection -i M_AMD_N162_L.n5 -k setup2/timepoint0/s0 -o synapses \
    --mask_path M_AMD_N162_L/images/ome-zarr/IHC_v11.ome.zarr --s3_mask
```

#### Object measures

See [Calculation of object measures](analysis.md#2---calculation-of-object-measures) in the analysis documentation.
