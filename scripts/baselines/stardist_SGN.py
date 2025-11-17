# This code is from schmit et al 2018 and wiegert et al 2020; to work for our datasets
# addendum 2025-07: Code edited for the application on SGNs for lightsheet cochlea evaluation
from __future__ import print_function, unicode_literals, absolute_import, division
import json
import time
import os.path

import numpy as np
import tifffile
from csbdeep.utils import normalize
from tifffile import imread
from stardist import random_label_cmap
from stardist.models import StarDist3D

np.random.seed(6)
lbl_cmap = random_label_cmap()

cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
model_dir = os.path.join(cochlea_dir, "/other_models/Stardist/Codes/segmentation-of-sgns-in-lsm/models")
data_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops")

# SGN specific
sgn_dir = os.path.join(data_dir,  "F1ValidationSGNs", "for_consensus_annotation")
image_files = [entry.path for entry in os.scandir(sgn_dir) if entry.is_file() and ".tif" in entry.path]

# output folder
val_sgn_dir = os.path.join(cochlea_dir, "predictions/val_sgn")
out_dir = os.path.join(val_sgn_dir, "stardist")

for img_path in image_files:
    img_arr = imread(img_path)

    start = time.perf_counter()
    basename = os.path.basename(img_path)
    basename = ".".join(basename.split(".")[:-1])
    timer_output = os.path.join(out_dir, f"{basename}_timer.json")
    out_path = os.path.join(out_dir, f"{basename}_seg.tif")

    n_channel = 1 if img_arr.ndim == 3 else img_arr.shape[-1]
    axis_norm = (0, 1, 2)   # normalize channels independently

    model = StarDist3D(None, name='Model3D', basedir=model_dir)

    # normalize input image
    img_arr = normalize(img_arr, 1.0, 99.8)
    labels, details = model.predict_instances(img_arr, axes='ZYX', n_tiles=(3, 3, 3))

    duration = time.perf_counter() - start
    time_dict = {"total_duration[s]": duration}
    with open(timer_output, "w") as f:
        json.dump(time_dict, f, indent='\t', separators=(',', ': '))

    tifffile.imwrite(out_path, labels)
