# Reproducibility of the pre-/post-processing

The folders here document the steps of the pre-/post-processing and the analysis of the data.
Each folder holds a script and the parameter dictionaries in JSON format that the script consumes.

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
  Use one of `train`, `domain`, `empty`, `annotation`, `variance`, `torn-components`, `rbOtof`,
  `PV-GFP` or `resized`. Add a new term only if none of these fits.

Each file holds exactly one cochlea. Several segmentation versions of that cochlea are several
entries in the same file. The example below records two runs of `M_LR_000226_L`, on `IHC_v4c` and
on `IHC_v11`:

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

Every entry needs `dataset_name`. The scripts read `segmentation_channel` to build the table
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

Each script takes one JSON file with `-j`, and an output directory or file with `-o`. Add `--s3`
to read the data from the S3 bucket. Use `-i` instead of `-j` to pass a single input path and set
the parameters on the command line; run a script with `--help` for those options.

### Extraction of blocks from a 3D volume

Blocks are needed for annotations, for empty crops, and for other regions of interest.

```bash
python block_extraction/repro_block_extraction.py -j <JSON-file> -o <out-dir>
```

`block_extraction/repro_equidistant_centers.py` computes evenly spaced crop centers along the
cochlea and writes them back into a JSON file.

### Labeling of components in the segmentation

The labeling can erode the segmentation to exclude artifacts. It can also vary the minimal number
of nodes in a component, and the minimal distance between two nodes of the same component.

```bash
python label_components/repro_label_components.py -j <JSON-file> -o <out-dir>
```

### Object measures

The measurement computes the morphology and the intensity per object, for one or more image
channels.

```bash
python object_measures/repro_object_measures.py -j <JSON-file> -o <out-dir>
```

### Tonotopic mapping

The mapping assigns a tonotopic frequency to each object. The script selects the species and the
OTOF frequency mapping from `dataset_name`.

```bash
python tonotopic_mapping/repro_tonotopic_mapping.py -j <JSON-file> -o <out-dir>
```
