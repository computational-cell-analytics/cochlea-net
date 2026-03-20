## Overview

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
