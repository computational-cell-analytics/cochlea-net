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
The components are separated by `-` or `_`, because some crops keep the older underscore form.
The dictionary `SPECIES_PREFIXES` in the script maps this component to the species, for example `M` to `mouse` and `G` to `gerbil`.
Extend the dictionary to add a species.

A crop whose first component is not in the dictionary is counted under `unknown`.
The script reports these crops, so that a new prefix does not pass unnoticed.

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
