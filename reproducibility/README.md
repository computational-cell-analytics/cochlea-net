# Reproducibility of the pre-/post-processing

The folders here document the steps of the pre-/post-processing and the analysis of the data.
They hold the parameter dictionaries in JSON format that those steps consume. The steps run
through the console scripts of the package; see Usage below.

| Folder | Content |
|---|---|
| `processing` | Parameters for component labeling, tonotopic mapping and object measures |
| `block_extraction` | Parameters for the extraction of blocks from a 3D volume |
| `training_crops` | Crop centers for the creation of training data, written by `scripts/training/` |
| `export_lower_resolution` | Positions and channels for the export of overview images |
| `model_accuracy` | Segmentation and detection accuracy per model, read by the figure scripts |
| `templates_processing` | Slurm templates for segmentation and detection |
| `templates_transfer` | Templates for the transfer of data into MoBIE and to S3 |

## The processing parameter files

`processing` holds one file per cochlea and segmentation, named

```
<cochlea>_<segmentation>.json
```

where `<cochlea>` is the `dataset_name` copied without change, for example `G_LR_000234_L` or
`LaVision-M03`, and `<segmentation>` is the structure, `SGN` or `IHC`. The file describes the
cochlea once at the top level and holds one section per processing step:

```json
{
	"dataset_name": "M_AMD_000126_L",
	"segmentation_channel": "IHC_v9",
	"cell_type": "ihc",
	"image_channel": ["Vglut3"],
	"component_list": [1],
	"label_components": {
		"component_list_path": [1, 2, 6],
		"min_component_length": 10,
		"min_size": 20000
	},
	"tonotopic_mapping": {},
	"object_measures": {"use_bg_mask": "yes"}
}
```

Each step reads the common keys plus its own section, and never a key that belongs to another
step. An **empty section** means the step ran with its defaults. An **absent section** means the
step was not run for this cochlea, and the command reports that and does nothing.

Common keys, passed to every step:

| Key | Meaning |
|---|---|
| `dataset_name` | The cochlea. Required, and must match the file name. |
| `segmentation_channel` | The segmentation the parameters belong to. Required. |
| `cell_type` | `sgn` or `ihc`. Must match the segmentation part of the file name. |
| `image_channel` | The stains to measure, as a list. Needed for object measures. |
| `component_list` | The components the steps select. |
| `voxel_size` | Voxel size in µm, `(x, y, z)`. Only needed where it is not 0.38 isotropic. |

Section keys:

| Section | Keys |
|---|---|
| `label_components` | `component_list_path`, `max_edge_distance`, `min_component_length`, `min_size`, `custom_dic`, `max_path_deviation` |
| `tonotopic_mapping` | `apex_position`, `component_mapping`, `include_gap`, `animal`, `otof` |
| `object_measures` | `use_bg_mask` |

`component_list_path` is the one key that renames on the way in. `label_components` receives it
as `component_list`, because there the list defines which components form the central path and
which instance counts are reported, and for six cochleae that is not the list the later steps
select. Where the two agree, only the common `component_list` is given.

`custom_dic` replaces the three plain labeling parameters: it runs the segmentation once per
entry of its `label_params` and merges the results, so `max_edge_distance`, `min_size` and
`min_component_length` are ignored next to it.

The reader is strict. An unknown key, a key in the wrong section, an unknown section name or a
missing `dataset_name` or `segmentation_channel` fails with a message naming the valid keys,
rather than being silently ignored as it was before. `block_extraction` files hold no sections
and are still read as flat dictionaries, unvalidated.

Only one segmentation version is recorded per cochlea, the one that is current. `tonotopic_mapping`
is the reference for which that is, and the IHC main line runs
`v2 < v4 < v4b < v4c < v6 < v9 < v10 < v11`; `IHC_LOWRES-v3` and `IHC_resized_v4b` are separate
lineages and do not compare against it.

To add a cochlea, write a new file with the name above, then add the cochlea to `cohorts.json`.
Edit these files by hand: `update_json` in `flamingo_tools/json_util.py` merges only top-level
keys and would replace a whole section.

## Cohorts

`cohorts.json` records which cochlea belongs to which cohort. It replaces the cohort names that
the parameter filenames used to carry.

Cohorts sit on two axes, because two different properties were both called a cohort before:

- `protocol` is the sample preparation and clearing protocol, for example `fHC`, `fDISCO` or
  `PELCOfHC2`.
- `group` is the experimental grouping, for example `chreef_mouse`, `otof_mouse` or
  `fchrimson_gerbil`.

A cochlea has at most one cohort per axis, and it appears on at least one axis. A cochlea that
neither axis can classify belongs to the `undefined` group, with a note that says what the
cochlea is used for.

Each cohort carries a `label` for plots, an `animal`, a `description`, and the member list in
`cochleae`. `aliases` holds the retired cohort names, so that an old filename stays findable.
The file records membership only. Plot colors stay in `scripts/figures/util.py`, and component
lists stay with the per-cochlea parameters.

`flamingo_tools/postprocessing/synapse_per_ihc_utils.py` and `scripts/figures/util.py` hold the
same membership in Python. No code reads `cohorts.json` yet. `test/test_cohorts.py` fails if the
registries disagree.

## Usage

Each step runs through a console script of the package. All of them take the parameter file with
`--json_info`, and an output directory or file with `-o`. Add `--s3` to read the data from the S3
bucket, or `--mobie_dir` to point at a local MoBIE project. Pass `-i` instead of `--json_info` to
process a single table or image and set the parameters on the command line; run a command with
`--help` for those options.

One file drives all three steps, and a step that the file does not describe is skipped, so a whole
folder can be processed in a loop:

```bash
for f in reproducibility/processing/*.json; do
    flamingo_tools.label_components  --json_info "$f" -o <out-dir> --s3
    flamingo_tools.tonotopic_mapping --json_info "$f" -o <out-dir> --s3
    flamingo_tools.object_measures   --json_info "$f" -o <out-dir> --s3
done
```

### Labeling of components in the segmentation

The labeling can erode the segmentation to exclude artifacts. It can also vary the minimal number
of nodes in a component, and the minimal distance between two nodes of the same component. For an
IHC segmentation, `--path_file` and `--max_path_deviation` add the central-path deviation filter.

```bash
flamingo_tools.label_components --json_info <JSON-file> -o <out-dir> --s3
```

### Tonotopic mapping

The mapping assigns a tonotopic frequency to each object. The command derives the species and the
OTOF frequency mapping from `dataset_name`; a file can override both with `animal` and `otof`.

```bash
flamingo_tools.tonotopic_mapping --json_info <JSON-file> -o <out-dir> --s3
```

### Object measures

The measurement computes the morphology and the intensity per object, for the image channels of
`image_channel`.

```bash
flamingo_tools.object_measures --json_info <JSON-file> -o <out-dir> --s3
```

### Extraction of blocks from a 3D volume

Blocks are needed for annotations, for empty crops, and for other regions of interest. These
parameters live in `block_extraction` and keep the older flat format.

```bash
flamingo_tools.extract_block --json_info <JSON-file> -o <out-dir> --s3
```

`flamingo_tools.equidistant_centers --json_info <JSON-file>` recomputes evenly spaced crop centers
along the cochlea and writes them back into the file. `flamingo_tools.extract_central_blocks` does
both at once, and takes its parameter file with `-i` rather than `--json_info`.
