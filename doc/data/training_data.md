# Training data for SGN, IHC, and synapse networks

The files in this folder give an overview over the image crops used for the training, validation, and testing of the SGN, IHC, and synapse detection networks which were used in the CochleaNet paper.
Each file features a list of image crops in the format of a dataframe in CSV format.
Each row contains the original name, a standardized version of the name, its crop center, the number of samples and other parameters.

## Creation process

First, the file names of the crops are manually collected and stored in the `Original` column.
The subdirectory is specified in form of the `Dataset` entry.
Once these values are given, the table can be expanded using the functions `add_metadata_to_crop_table` and `add_metadata_to_crop_table_synapses` located at `flamingo_tools/analysis/training_data_utils.py`.


### SGN and IHC

```python
from flamingo_tools.analysis.training_data_utils import add_metadata_to_crop_table

# data divided into "train" and "val" directories
data_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/IHC/2025-07-IHC_supervised"
input_path = "/path/to/flamingo-tools/doc/data/IHC_v4.tsv"
add_metadata_to_crop_table(input_path, data_dir)

# data divided into "images" and "labels"
data_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/IHC/2025-09-IHC_v7_supervised"
label_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/IHC/2025-09-IHC_v7_supervised/labels"
input_path = "/path/to/flamingo-tools/doc/data/IHC_v7.tsv"
add_metadata_to_crop_table(input_path, data_dir, label_dir=label_dir)
```

### Synapses
```python
from flamingo_tools.analysis.training_data_utils import add_metadata_to_crop_table_synapses

train_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v4"
test_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v4"
input_path = "/path/to/flamingo-tools/doc/data/synapses_v4.tsv"

add_metadata_to_crop_table_synapses(input_path, train_dir, test_dir)
```

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
