# Export of data at a lower resolution

`flamingo_tools.export_data` exports image data, segmentations and analysis results at a lower resolution, for example for the overview images of a figure.
The command has one sub-command for each export:

| Command | Output |
| --- | --- |
| `lower_resolution` | Image data or a segmentation, optionally filtered by components or masked to the cochlea |
| `marker` | The marker-positive (1) and the marker-negative (2) objects of a segmentation |
| `subtypes` | One binary mask for each SGN subtype, or one label volume with all subtypes |
| `synapses` | The synapse detections as a label volume or as a napari point table |
| `frequency` | The tonotopic frequency of each object of a segmentation |

Run `flamingo_tools.export_data <command> -h` to show all options of a command.

## Input

Each command reads its input in one of two modes:
- `--input` (`-i`) reads a TIF file or an n5/zarr/OME-Zarr container. `--input_key` (`-k`) selects the dataset in a container. The default is `s0`. `--name` (`-n`) sets the name of the output. The default is the file name of the input without its extension, see [synapses](#synapses) for the exception.
- `--cochlea` (`-c`) reads a cochlea in a MoBIE project. The image data comes from `<cochlea>/images/ome-zarr/<source>.ome.zarr` and the tables come from `<cochlea>/tables/<source>/default.tsv`. The output has the name of the source.

With `--s3`, the command reads the data from the S3 bucket, and a path of `--input` is relative to the bucket.
Without `--s3`, `--cochlea` reads the MoBIE project in `--mobie_dir`. The default is the MoBIE project on the cluster.
The command stops if the cochlea is not in this project.
A table that you give as a path (`--table`, `--sgn_table`, `--ihc_table`, `--synapse_table`) is always a local TSV file.

## Scale levels

`--scale` (`-s`) takes one or more scale levels. Scale N has 2^N times the voxel size of the input.
- If the container has the pyramid level N levels below the input key, the command reads this level. For example, `-k s0 -s 3` reads `s3`, and `-k setup0/timepoint0/s0 -s 3` reads `setup0/timepoint0/s3` of the n5 file of `flamingo_tools.convert_data`.
- Otherwise, the command downsamples the input: it takes the mean for image data and the nearest voxel for a segmentation. This applies to a TIF file and to a container without a pyramid, for example the `segmentation.zarr` of `flamingo_tools.run_segmentation`. The commands for a segmentation (`marker`, `subtypes`, `frequency`) always take the nearest voxel. `lower_resolution` takes the nearest voxel for 32 or 64 bit integer data, with `--filter_by_components`, and with `--labels`.

`--voxel_size` (`-v`) is the voxel size of the input key in micrometer, in (x, y, z) order. The default is `0.38 0.38 0.38`.
Give `-v 1.887779 1.887779 3.0` for the LaVision data.

## Output

The output of each scale level goes into the sub-folder `scale<N>` of `--output_folder` (`-o`).
With `--cochlea`, the sub-folders are in `<output_folder>/<cochlea>`.
A TIF file holds the voxel size of its scale level as metadata.
The commands keep an existing output. Give `-f` to replace it.

## Crops

Without a crop, the commands export the full volume.
- `--crop_center X Y Z` is the center of the crop in micrometer.
- `--roi_halo X Y Z` is the halo around the center in voxels of the target scale.
- `--axis 0|1|2` exports a 2D slice through the crop center, perpendicular to x, y or z. Without `--roi_halo`, the slice covers the full plane.
- `--suffix` adds a label to the file name, for example the position `apex`.

The file name of a crop is `<name>_crop_<x>-<y>-<z>[_axis-<axis>][_<suffix>].tif`.
`flamingo_tools.table_info` finds a crop center, for example the SGN at the length fraction 0.15 (apex):
```bash
flamingo_tools.table_info -i M_LR_000227_R/tables/SGN_v2/default.tsv --value 0.15 -c 1 --s3
```

## lower_resolution

```bash
# The PV, Vglut3 and CTBP2 channels of a cochlea on the S3 bucket at the scale levels 2 and 3
flamingo_tools.export_data lower_resolution -c M_LR_000227_R --channels PV Vglut3 CTBP2 -o <output_folder> -s 2 3 --s3

# The IHC segmentation, with only the objects of component 1
flamingo_tools.export_data lower_resolution -c M_LR_000227_R --channels IHC_v11 --filter_by_components 1 \
    -o <output_folder> -s 3 --s3

# A 2D slice of the PV channel at the apex
flamingo_tools.export_data lower_resolution -c M_LR_000227_R --channels PV --crop_center 800.8 1005.2 669.7 --axis 2 \
    --suffix apex -o <output_folder> -s 2 --s3

# The Otof channel and the IHC segmentation, masked to the region around the IHCs
flamingo_tools.export_data lower_resolution -c M_AMD_OTOF27_L --channels Otof IHC_v11 --filter_cochlea_channels IHC_v11 \
    --filter_dilation_iterations 3 -o <output_folder> -s 3 --s3

# A channel of the n5 file of flamingo_tools.convert_data
flamingo_tools.export_data lower_resolution -i <converted_data>.n5 -k setup1/timepoint0/s0 -n PV -o <output_folder> -s 3

# The segmentation of flamingo_tools.run_segmentation, filtered by the components of its table
flamingo_tools.export_data lower_resolution -i <segmentation_folder>/segmentation.zarr -k segmentation \
    --table <segmentation_table>.tsv --filter_by_components 1 -o <output_folder> -s 2

# A TIF file of LaVision data
flamingo_tools.export_data lower_resolution -i <lavision_data>.tif -v 1.887779 1.887779 3.0 -o <output_folder> -s 1

# A segmentation with 16 bit integers in a TIF file
flamingo_tools.export_data lower_resolution -i <segmentation>.tif --labels -o <output_folder> -s 1 2
```

- `--filter_by_components` keeps only the objects of these components of a segmentation. With `--input`, give the segmentation table with `--table`. With `--cochlea`, the command reads the table of each channel.
- A mask of the cochlea sets everything outside the cochlea to 0. Give the segmentation tables with `--sgn_table` and / or `--ihc_table` for `--input`, or the segmentation names with `--filter_cochlea_channels` for `--cochlea`. With both tables, the mask also covers the region between the SGNs and the IHCs. `--filter_sgn_components`, `--filter_ihc_components` and `--filter_dilation_iterations` control the mask. The output folder is then `scale<N>_dilation<iterations>`.
- `--max_intensity` clips the intensities of image data. A segmentation is not clipped. With `--cochlea`, the default clips PV (except for LaVision cochleae) and CTBP2 at 1400.
- `--labels` marks the input as a segmentation, for example a segmentation with 16 bit integers. Without it, `lower_resolution` downsamples 8 and 16 bit integer data as image data.
- `--binarize` writes 1 for every non-zero voxel.
- `--ome_zarr` writes a zarr container instead of a TIF file.

## marker

The segmentation table needs the column `marker_labels` (1 = positive, 2 = negative), see [Intensity annotation](analysis.md#intensity-annotation).
The command writes `<name>_marker_positive.tif` and `<name>_marker_negative.tif`.
```bash
# The marker labels of the SGN segmentation of a cochlea
flamingo_tools.export_data marker -c G_EK_000049_R --channels SGN_v2 -o <output_folder> -s 3 --s3

# The marker labels of an edited local table
flamingo_tools.export_data marker -c M_AMD_OTOF27_L --channels IHC_v11 --table <edited_table>.tsv -o <output_folder> -s 1 --s3

# A local segmentation and its table
flamingo_tools.export_data marker -i <segmentation_folder>/segmentation.zarr -k segmentation --table <marker_table>.tsv \
    -o <output_folder> -s 2
```

## subtypes

The segmentation table needs one column `marker_<stain>` for each subtype stain, see [Subtype analysis](analysis.md#4a---subtype-analysis).
The command writes one binary mask for each subtype, for example `SGN_v2_TypeIb.tif`.
If the stains define Type Ib and Type Ic, it also writes one mask for both types together.
With `--merge_masks`, it writes one label volume `<name>_subtypes.tif` with consecutive IDs for the subtypes.
```bash
# The subtypes of a cochlea in COCHLEAE of flamingo_tools/postprocessing/sgn_subtype_utils.py,
# with the segmentation and the stains of the cochlea
flamingo_tools.export_data subtypes -c M_LR_N110_L -o <output_folder> -s 3 --s3

# One label volume with all subtypes
flamingo_tools.export_data subtypes -c M_LR_000098_L --merge_masks -o <output_folder> -s 3 --s3

# A local segmentation, its table and the stains
flamingo_tools.export_data subtypes -i <segmentation_folder>/segmentation.zarr -k segmentation \
    --table <subtype_table>.tsv --stains CR Ntng1 -o <output_folder> -s 2
```

## synapses

The command writes the synapses as spheres into a volume with the shape of the IHC segmentation.
The value of a sphere is the ID of the matched IHC, or the synapse ID with `--use_syn_ids`.
`--id_offset` adds an offset to the values, for other colors in a viewer.
With `--points`, the command writes a CSV point table instead, which napari opens as a points layer.
Only the synapses within `--max_dist` (default 3 µm) of an IHC of the components `--filter_ihc_components` are exported.
With `--input`, the default name of the output is the name of the synapse table: the source name of a MoBIE table `<cochlea>/tables/<source>/default.tsv`, otherwise the file name of the table.
```bash
# The synapses of synapse_v3_ihc_v11, matched to IHC_v11 (the defaults)
flamingo_tools.export_data synapses -c M_LR_000227_R -o <output_folder> -s 3 --s3

# Explicit sources, as a point table
flamingo_tools.export_data synapses -c M_LR_000227_R --synapse_name synapse_v3_ihc_v11 --reference_ihcs IHC_v11 \
    --points -o <output_folder> -s 3 --s3

# A local synapse table. The input sets the shape of the output, and the IHC table filters the synapses.
flamingo_tools.export_data synapses -i <ihc_segmentation>.zarr -k segmentation --synapse_table <synapse_table>.tsv \
    --ihc_table <ihc_table>.tsv -o <output_folder> -s 2
```

## frequency

The segmentation table needs the column `frequency[kHz]`, see [Tonotopic mapping](analysis.md#tonotopic-mapping).
The command writes `frequencies_<name>.tif`, with the frequency in kHz as the value of each object.
With `--colormap`, it writes an RGBA image and a color bar `frequencies_<name>_<colormap>.png`.
```bash
# The frequencies of the SGNs of component 1
flamingo_tools.export_data frequency -c M_LR_000227_R --segmentation SGN_v2 -o <output_folder> -s 3 --s3

# An RGBA image with the viridis colormap
flamingo_tools.export_data frequency -c M_LR_000227_R --segmentation SGN_v2 --colormap viridis -o <output_folder> -s 3 --s3

# A local segmentation and its table
flamingo_tools.export_data frequency -i <segmentation_folder>/segmentation.zarr -k segmentation \
    --table <segmentation_table>.tsv -o <output_folder> -s 2
```

## Crops at tonotopic positions

`scripts/export_data/export_by_position.py` exports crops at the apex, mid and base of a cochlea with the export functions above.
It finds the crop centers in a reference segmentation table and reads the data from the S3 bucket.
The files in `reproducibility/export_lower_resolution` give the channels and the crop size of the overview images:
```bash
python scripts/export_data/export_by_position.py -c M_LR_000227_R --reference_seg SGN -s 3 -o <output_folder> \
    --json_info reproducibility/export_lower_resolution/M_LR_000227_R.json
```
`--groups` exports named groups of channels, segmentations and synapses, each centered on its own reference segmentation:
```bash
python scripts/export_data/export_by_position.py -c M_LR_000227_R --groups sgn ihc --axis 2 -s 2 -o <output_folder> --dry_run
```
`--dry_run` only prints the crop centers and the exports. `--view` opens the crops in napari after the export.

## Python

Each export has a function for a path and a function for a cochlea in a MoBIE project, in `flamingo_tools.export_data`.
The functions take the tables as pandas DataFrames:
```python
import pandas as pd
from flamingo_tools.export_data import export_lower_resolution, export_synapses_cochlea

table = pd.read_csv("segmentation_table.tsv", sep="\t")
export_lower_resolution(
    "segmentation.zarr", "exports", scale=[2, 3], input_key="segmentation", table=table, filter_by_components=[1],
)
export_synapses_cochlea("M_LR_000227_R", "exports", scale=3, points=True, s3=True)
```
