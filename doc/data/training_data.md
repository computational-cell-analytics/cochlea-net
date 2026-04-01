## Overview

The metadata table for a training dataset can be expanded using:
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

### IHC

Due to the lack of annotation data, one volume was split into a subvolume for training and validation for the initial trainings.
```
arr_img = imageio.imread("171R_Vglut3_apexIHC_HCAT_C1.tif")
arr_seg = imageio.imread("171R_Vglut3_apexIHC_HCAT_C1_annotations.tif")

img_train = arr_img[:,:600,:]
seg_train = arr_seg[:,:600,:]

img_val = arr_img[:,600:,:]
seg_val = arr_seg[:,600:,:]
```
