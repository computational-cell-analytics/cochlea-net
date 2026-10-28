# Training data for SGN, IHC, and synapse networks

The files in this folder list the image crops for the training, validation, and testing of the SGN, IHC, and synapse detection networks of the CochleaNet paper.
Each file is a table of image crops in TSV format.
Each row contains the original name, a standardized version of the name, the crop center, the number of samples, and other parameters.

## Creation process

First, collect the file names of the crops manually in the column `Original`.
The column `Dataset` gives the subdirectory.
Then the script `scripts/analysis/analyze_training_data.py` adds the other columns.
The script uses the functions `add_metadata_to_crop_table` and `add_metadata_to_crop_table_synapses` located at `flamingo_tools/analysis/training_data_utils.py`.
It selects the function from the name of the table, which must contain `IHC`, `SGN`, or `synapse`.

```bash
# analyze the predefined tables listed in DEFAULT_TABLES
python scripts/analysis/analyze_training_data.py

# analyze a single table
python scripts/analysis/analyze_training_data.py -i doc/data/IHC_v11.tsv -d /path/to/training_data/IHC
```

The data directory must contain the sub-directories `train` and `val`, or `images` and `labels`.
Synapse crops need `--test_dir` in addition, because their tables also contain a `test` dataset.

The table is edited in place.
Rows with complete and consistent measures are skipped and their crops are not read.
Use `--recompute` to measure all crops again.

A crop that cannot be measured does not stop the run.
The script reports the crop, keeps its row empty, writes the measures of all other crops, and exits with a non-zero status.
The next run then measures only the crops that failed.

### Overview

Each analyzed table adds one entry to `doc/data/overview.json`, which reports the number of crops and instances per dataset.
The entries are sorted alphabetically by table name.
Existing entries of other tables are kept.
Use `-o` to write the overview to a different location.

A table with crops that have no instance count is not added to the overview, because its totals would be too low.

#### Species

Each entry also reports the same crop and instance counts per species, under the key `species`.
Only the species present in the table are listed.

The species comes from the first component of the `Standardized` name.
The components are separated by `-`.
The dictionary `SPECIES_PREFIXES` in the script maps this component to the species, for example `M` to `mouse` and `G` to `gerbil`.
Extend the dictionary to add a species.

A crop whose first component is not in the dictionary is counted under `unknown`.
The script reports these crops, so that a new prefix does not pass unnoticed.

## Export for publication

The script `scripts/training/export_training_data.py` copies the training data of `SGN_v2`, `IHC_v11`, `synapses_v3`, `SGN-lowres_v5` and `IHC-lowres_v3` to a new folder with publication names.

```bash
python scripts/training/export_training_data.py -i /path/to/training_data -o /path/to/export
```

The publication name is the `Standardized` name, with the cochlea ID replaced by the sample alias of the CochleaNet paper.
Cochleae without a paper alias get a new alias that continues after the highest alias in use.
The marker `-resized` is not part of the publication name, because it only documents an early processing error.
The output folders `train` and `val` contain the split of the published checkpoints, which differs from the original folders.
The synapse `test` crops are not exported.

### Published model versions

The published files use the model versions of BioImage.IO and not the internal versions.
All five models are version 1 on BioImage.IO, so each dataset folder and README refers to version 1:

| Internal version | Published version | BioImage.IO model |
|---|---|---|
| `SGN_v2` | `SGN_v1` | CochleaNet SGN Segmentation |
| `IHC_v11` | `IHC_v1` | CochleaNet IHC Segmentation |
| `synapses_v3` | `synapses_v1` | CochleaNet Ribbon Synapse Detection |
| `SGN-lowres_v5` | `SGN-lowres_v1` | CochleaNet Low-Resolution SGN Segmentation |
| `IHC-lowres_v3` | `IHC-lowres_v1` | CochleaNet Low-Resolution IHC Segmentation |

The key `published` of `DATASETS` in the export script sets the published version.
The internal versions stay in this folder, in the export script, and in `overview.tsv`.
On BioImage.IO, the internal checkpoint name is in `config.cochlea_net.source_checkpoint` of `rdf.yaml`.

### Low-resolution data

The original low-resolution crops are in the folders `images` and `labels`, and the label names differ from the image names.
The `split.json` next to each checkpoint lists the image and label pairs of the training.
Before the export and the analysis, link these pairs into one folder per split, with the name of the image:

```python
split = json.load(open("cochlea_distance_unet_ihc-lowres-v3/split.json"))
for name, files in split.items():
    os.makedirs(f"IHC-lowres_v3/{name}")
    for image, label in zip(files["images"], files["labels"]):
        stem = os.path.splitext(os.path.basename(image))[0]
        os.link(image, f"IHC-lowres_v3/{name}/{stem}.tif")
        os.link(label, f"IHC-lowres_v3/{name}/{stem}_annotations.tif")
```

The column `Dataset` of `SGN-lowres_v5.tsv` and `IHC-lowres_v3.tsv` is the split of the checkpoint.
These tables are not part of `overview.json`, because most low-resolution objects are smaller than the size threshold of its counts (1000 pixels).

### Output

The script writes `crops.tsv` into each dataset folder and `overview.tsv` into the output folder.
`overview.tsv` maps each original name to its `Standardized` name, its paper alias, and its publication name.
The columns `Dataset` and `Published_dataset` give the internal and the published version.
It is for internal use and is not part of the published data.

## IHC split for single sample

The first trainings had little annotated data.
Thus one volume was split into a subvolume for training and a subvolume for validation:
```
arr_img = imageio.imread("171R_Vglut3_apexIHC_HCAT_C1.tif")
arr_seg = imageio.imread("171R_Vglut3_apexIHC_HCAT_C1_annotations.tif")

img_train = arr_img[:,:600,:]
seg_train = arr_seg[:,:600,:]

img_val = arr_img[:,600:,:]
seg_val = arr_seg[:,600:,:]
```
