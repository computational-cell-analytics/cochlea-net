# Reproducibility of the pre-/post-processing

The folders here document the steps of the pre-/post-processing and the analysis of the data.
Each folder holds the parameter dictionaries in JSON format that one processing step consumes.
The steps run through the console scripts of the package; see Usage below.

| Folder | Content |
|---|---|
| `block_extraction` | Parameters for the extraction of blocks from a 3D volume |
| `label_components` | Parameters for the labeling of connected components in a segmentation |
| `object_measures` | Parameters for the morphology and intensity measurements per object |
| `tonotopic_mapping` | Parameters for the assignment of tonotopic frequencies |
| `training_crops` | Crop centers for the creation of training data, written by `scripts/training/` |
| `export_lower_resolution` | Positions and channels for the export of overview images |
| `model_accuracy` | Segmentation and detection accuracy per model, read by the figure scripts |
| `templates_processing` | Slurm templates for segmentation and detection |
| `templates_transfer` | Templates for the transfer of data into MoBIE and to S3 |

## Naming of the parameter files

The files in `block_extraction`, `label_components`, `object_measures` and `tonotopic_mapping`
follow one scheme:

```
<cochlea>_<segmentation>[_<purpose>].json
```

- `<cochlea>` is the `dataset_name` of the entry, copied without change, for example
  `G_LR_000234_L` or `LaVision-M03`.
- `<segmentation>` is the anatomical structure, either `SGN` or `IHC`. It is never the
  segmentation version.
- `<purpose>` distinguishes two different jobs for one cochlea and one structure in one folder.
  Use one of `annotation`, `domain`, `empty`, `rbOtof`, `torn-components`, `train` or `variance`.
  Add a new term only if none of these fits.

Each file holds exactly one cochlea, and normally one entry: the segmentation version that is
current for that cochlea. `tonotopic_mapping` is the reference for which version that is. The IHC
main line runs `v2 < v4 < v4b < v4c < v6 < v9 < v10 < v11`; `IHC_LOWRES-v3` and
`IHC_resized_v4b` are separate lineages and do not compare against it.

A file keeps a second entry only while an older version is still in use. `M_LR_000226_L` is the
example: `IHC_v4c` stays next to `IHC_v11` because `scripts/figures/plot_fig3.py` still reads the
`IHC_v4c` table for a Figure 1 panel. The `*_SGN_variance.json` files are the other case, where
`SGN_v2-1` to `SGN_v2-4` are four replicas of one version rather than a version history.

```json
[
	{
		"dataset_name": "M_LR_000226_L",
		"segmentation_channel": "IHC_v4c",
		"cell_type": "ihc"
	},
	{
		"dataset_name": "M_LR_000226_L",
		"segmentation_channel": "IHC_v11",
		"cell_type": "ihc",
		"component_list": [1, 3],
		"min_component_length": 20,
		"max_edge_distance": 30
	}
]
```

Every entry needs `dataset_name`. The commands read `segmentation_channel` to build the table
path, and `image_channel` to build the image paths. All other keys pass through to the processing
function, so an entry can record any parameter that the function accepts.

To add a cochlea, write a new file with the name above, then add the cochlea to `cohorts.json`.

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

Each step runs through a console script of the package. All four take the parameter file with
`--json_info`, and an output directory or file with `-o`. Add `--s3` to read the data from the S3
bucket, or `--mobie_dir` to point at a local MoBIE project. Pass `-i` instead of `--json_info` to
process a single table or image and set the parameters on the command line; run a command with
`--help` for those options.

A file with several entries needs an output *directory*, because each entry writes its own table.

### Extraction of blocks from a 3D volume

Blocks are needed for annotations, for empty crops, and for other regions of interest.

```bash
flamingo_tools.extract_block --json_info <JSON-file> -o <out-dir> --s3
```

`flamingo_tools.equidistant_centers --json_info <JSON-file>` recomputes evenly spaced crop
centers along the cochlea and writes them back into the file. `flamingo_tools.extract_central_blocks`
does both at once, and takes its parameter file with `-i` rather than `--json_info`.

### Labeling of components in the segmentation

The labeling can erode the segmentation to exclude artifacts. It can also vary the minimal number
of nodes in a component, and the minimal distance between two nodes of the same component. For an
IHC segmentation, `--path_file` and `--max_path_deviation` add the central-path deviation filter.

```bash
flamingo_tools.label_components --json_info <JSON-file> -o <out-dir> --s3
```

### Object measures

The measurement computes the morphology and the intensity per object, for one or more image
channels.

```bash
flamingo_tools.object_measures --json_info <JSON-file> -o <out-dir> --s3
```

### Tonotopic mapping

The mapping assigns a tonotopic frequency to each object. The command selects the species and the
OTOF frequency mapping from `dataset_name`.

```bash
flamingo_tools.tonotopic_mapping --json_info <JSON-file> -o <out-dir> --s3
```

With `--json_info`, this command takes every parameter from the file: `--cell_type`,
`-c/--components`, `--apex_position` and the S3 options given on the command line are ignored,
because only the entry is forwarded. Put them in the file instead. It also has no `--mobie_dir`,
so local runs need the working directory to be the MoBIE project root.
