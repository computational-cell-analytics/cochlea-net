# Analysing SGN density

An important parameter for analyzing SGN survivability is the SGN density. Usually it is either given as a cell density per area [mm²] or per volume [mm³].
The following examples show how

## Calculate SGN density

The density of an SGN segmentation can be calculated using `flamingo_tools.sgn_density`
The process requires the segmentation table, which is usually located at `<cochlea-name>/tables/<SGN-version>/default.tsv` in a MoBIE project. It requires tonotopic mapping because reference points are derived from the run length of the Rosenthal's canal.

Some examples for cochlea `M_LR_000155_L`:
```bash
# 2D slices with 10 µm thickness, at apex, mid and base
flamingo_tools.sgn_density -i M_LR_000155_L/tables/SGN_v2/default.tsv -o SGN_density.json --slice_thickness 10 \
    --json_input MLR155L_metadata.json --json_output MLR155L_crop_2d.json \
    --mode 2d --min_overlap_fraction 0.2 --seg_path M_LR_000155_L/images/ome-zarr/SGN_v2.ome.zarr --s3

# 3D volume with 120 µm thickness
flamingo_tools.sgn_density -i M_LR_000155_L/tables/SGN_v2/default.tsv -o SGN_density.json --slice_thickness 120 \
    --json_input MLR155L_metadata.json --json_output MLR155L_crop_3d.json \
    --mode 3d --min_overlap_fraction 0.2 --seg_path M_LR_000155_L/images/ome-zarr/SGN_v2.ome.zarr --s3
```

where the parameters:
- `--slice_thickness` refer to the thickness in z-dimension in µm
- `MLR155L_metadata.json`, the argument for ``--json_input` refers to a JSON file containing metadata, e.g.
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
specifying the image channels, the relevant segmentation channel, and the component list
- `--mode` can either be 2d or 3d, depending on the desired density output
- `--min_overlap_fraction` signifies the limit for the fraction of an SGN instance which is inside the crop used for density estimation. If the fraction is lower than this value, the SGN is omitted from the calculation. In the examples, SGNs which have less than 20% of their entire volume inside the crop will be excluded from the calculation
- `--json_output` is a JSON dictionary which contains crop centers and ROI halos which can be used as input for `flamingo_tools.extract_block` as the `--json_info` argument to crop the subvolumes
- `--positions` can be used to evaluate the SGN density at specific length fractions of Rosenthal's canal, default values are apex (0.15), mid (0.5) and base (0.85)
- additional parameters are explained in the function documentation (`flamingo_tools.sgn_density -h`)


## Crop slices/volumes used for density calculation

The crops used for density calculation can be cropped using `flamingo_tools.extract_block`:

```bash
# crop data on S3 bucket
flamingo_tools.extract_block --s3 --json_info MLR155L_crop_2d.json -o /path/to/output_crops
```
They can then also be used for the calculation of the SGN density by providing them as the `--seg_path` argument.
