# Analysing SGN density

The SGN density is an important parameter for the analysis of SGN survival.
It is a cell density per square millimeter or per cubic millimeter.

## Calculate SGN density

`flamingo_tools.sgn_density` calculates the density of an SGN segmentation.
It needs the segmentation table, usually at `<cochlea-name>/tables/<SGN-version>/default.tsv` in a MoBIE project.
The table must have the columns of the [tonotopic mapping](analysis.md#tonotopic-mapping), because the reference points come from the run length of Rosenthal's canal.

Examples for the cochlea `M_LR_000155_L`:
```bash
# 2D slices with 10 micrometer thickness, at apex, mid and base
flamingo_tools.sgn_density -j MLR155L_metadata.json -o MLR155L_density_2d.json --json_output MLR155L_crop_2d.json \
    --mode 2d --slice_thickness 10 --min_overlap_fraction 0.2 --s3

# 3D volumes with 120 micrometer thickness
flamingo_tools.sgn_density -j MLR155L_metadata.json -o MLR155L_density_3d.json --json_output MLR155L_crop_3d.json \
    --mode 3d --slice_thickness 120 --min_overlap_fraction 0.2 --s3
```

The parameters:
- `-j, --json_input` is a JSON file with the metadata of the cochlea: the image channels, the segmentation channel, and the component list. For example, `MLR155L_metadata.json`:
```JSON
{
    "dataset_name": "M_LR_000155_L",
    "image_channel": [
        "PV",
        "GFP",
        "SGN_v2"
    ],
    "segmentation_channel": "SGN_v2",
    "component_list": [
        1
    ]
}
```
- The command finds the segmentation table and the segmentation from `dataset_name` and `segmentation_channel` in the JSON file. Use `-i` and `--seg_path` to give other paths.
- `--slice_thickness` is the thickness of the slice in micrometer, along `--axis` (default `z`).
- `--mode` is `2d` or `3d`. It selects the density per area or per volume.
- `--min_overlap_fraction` is the minimum fraction of an SGN instance that must be inside the crop. The calculation does not include an SGN with a smaller fraction. In the examples, the calculation does not include SGNs with less than 20 % of their volume inside the crop.
- `--min_overlap_volume` is the same limit as `--min_overlap_fraction`, given as a volume in cubic micrometer.
- `--json_output` is a JSON file with the crop centers and the ROI halos. Use it as the `--json_info` argument of `flamingo_tools.extract_block` to extract the subvolumes.
- `--positions` sets the positions for the density as length fractions of Rosenthal's canal. The defaults are apex (0.15), mid (0.5) and base (0.85).
- `--run_length_tolerance` is the maximum distance of an instance from the requested position along Rosenthal's canal, as a fraction of the total run length. Default: 0.1.
- By default, the command limits the slice to one turn of Rosenthal's canal.
  A slice can cut two turns where the canal is oblique to the slice axis.
  The convex hull then includes both turns and the empty space between them, and the density becomes too low.
  Thus the command clusters the instances and keeps only the cluster with the run length of the requested position.
    - `--no_cluster_filter` disables this filter. The command then measures all instances in the slice.
    - `--max_edge_distance` is the maximum distance in micrometer between two instances of one cluster. Default: 40.0.
    - `--min_cluster_size` is the minimum number of instances in a cluster that the command can select. The command treats smaller clusters as fragments. Default: 10.
- `--crop_output` and `--img_path` extract the image crops of each density position directly. Use them instead of the block-extraction JSON for `flamingo_tools.extract_block`.
- `flamingo_tools.sgn_density -h` explains the other parameters.


## Crop slices/volumes used for density calculation

Use `flamingo_tools.extract_block` to extract the crops of the density calculation:

```bash
# crop data on S3 bucket
flamingo_tools.extract_block --s3 --json_info MLR155L_crop_2d.json -o /path/to/output_crops
```
You can then also give a crop as the `--seg_path` argument to calculate the SGN density.
