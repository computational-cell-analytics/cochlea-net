# Training data for SGN, IHC, and synapse networks

The files in this folder give an overview over the image crops used for the training, validation, and testing of the SGN, IHC, and synapse detection networks which were used in the CochleaNet paper.
Each file features a list of image crops in the format of a dataframe in CSV format.
Each row contains the original name, a standardized version of the name, its crop center, the number of samples and other parameters.

## Creation process

First, the file names of the crops are manually collected and stored in the `Original` column.
The subdirectory is specified in form of the `Dataset` entry.
Once these values are given, the table is expanded with the script `scripts/analysis/analyze_training_data.py`.
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

### Overview

Each analyzed table adds one entry to `doc/data/overview.json`, which reports the number of crops and instances per dataset.
The entries are sorted alphabetically by table name.
Existing entries of other tables are kept.
Use `-o` to write the overview to a different location.

## IHC split for single sample

Due to the lack of annotation data, one volume was split into a subvolume for training and validation for the initial trainings.
```
arr_img = imageio.imread("171R_Vglut3_apexIHC_HCAT_C1.tif")
arr_seg = imageio.imread("171R_Vglut3_apexIHC_HCAT_C1_annotations.tif")

img_train = arr_img[:,:600,:]
seg_train = arr_seg[:,:600,:]

img_val = arr_img[:,600:,:]
seg_val = arr_seg[:,600:,:]
```
