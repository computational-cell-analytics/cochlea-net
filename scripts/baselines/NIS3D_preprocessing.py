import os

import numpy as np
import tifffile

NIS3D_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/NIS3D"
TRAIN_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/2025-07_NIS3D/train"
VAL_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/2025-07_NIS3D/val"
TEST_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/2025-07_NIS3D(test"

# ---Training data---

# clear: contains only 2,3,4 as seg ids
train_dict_01 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/cross-image/train"),
    "name": "Drosophila_2",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": TRAIN_DIR,
    "output_name": "Drosophila_2_annotations.tif",
}

# contains 1, 2, 3, 4
train_dict_02 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/cross-image/train"),
    "name": "Zebrafish_2",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": TRAIN_DIR,
    "output_name": "Zebrafish_2_annotations.tif",
}

# contains 1, 3, 4
train_dict_03 = {
   "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/cross-image/train"),
   "name": "MusMusculus_2",
   "conf_file": "scoreOfConfidence.tif",
   "gt_file": "gt.tif",
   "output_dir": TRAIN_DIR,
   "output_name": "MusMusculus_2_annotations.tif",
}

# ---Validation data---

val_dict_01 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/in-image/train"),
    "name": "Drosophila_1",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": VAL_DIR,
    "output_name": "Drosophila_1_iitrain_annotations.tif",
}

val_dict_02 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/in-image/train"),
    "name": "Zebrafish_1",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": VAL_DIR,
    "output_name": "Zebrafish_1_iitrain_annotations.tif",
}

val_dict_03 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/in-image/train"),
    "name": "MusMusculus_1",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": VAL_DIR,
    "output_name": "MusMusculus_1_iitrain_annotations.tif",
}

# ---Test data---

test_dict_01 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/in-image/test"),
    "name": "Drosophila_1",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": TEST_DIR,
    "output_name": "Drosophila_1_iitest_annotations.tif",
}

test_dict_02 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/in-image/test"),
    "name": "Zebrafish_1",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": TEST_DIR,
    "output_name": "Zebrafish_1_iitest_annotations.tif",
}

test_dict_03 = {
    "data_dir": os.path.join(NIS3D_DIR, "suggestive_splitting/in-image/test"),
    "name": "MusMusculus_1",
    "conf_file": "ConfidenceScore.tif",
    "gt_file": "GroundTruth.tif",
    "output_dir": TEST_DIR,
    "output_name": "MusMusculus_1_iitest_annotations.tif",
}


def filter_unmasked_data(conf_path, in_path, out_path):
    conf = tifffile.imread(conf_path)
    gt = tifffile.imread(in_path)
    segmentation_ids = list(np.unique(conf)[1:])
    if 1 in segmentation_ids:
        instance_ids = list(np.unique(gt)[1:])
        print(f"Number of instances before filtering: {len(instance_ids)}")
        gt[conf == 1] = 0
        instance_ids = list(np.unique(gt)[1:])
        print(f"Number of instances after filtering: {len(instance_ids)}")
        tifffile.imwrite(out_path, gt)
    else:
        instance_ids = list(np.unique(gt)[1:])
        print(f"Number of instances: {len(instance_ids)}")
        tifffile.imwrite(out_path, gt)


def process_data_dicts(data_dicts):
    for data_dict in data_dicts:
        data_dir = data_dict["data_dir"]
        dataset = os.path.join(data_dir, data_dict["name"])
        conf_path = os.path.join(dataset, data_dict["conf_file"])
        gt_path = os.path.join(dataset, data_dict["gt_file"])

        out_dir = data_dict["output_dir"]
        out_name = data_dict["output_name"]
        out_path = os.path.join(out_dir, out_name)
        filter_unmasked_data(conf_path, in_path=gt_path, out_path=out_path)


def prepare_training_data():
    """Prepare training data based on NIS3D data.

    Cross-image data of half of the samples is used for training.
    The other half of the samples is divided into validation data used for training and test data.
    The in-image data is used for this, so that every remaining sample is split in half.
    """
    process_data_dicts([test_dict_01, test_dict_02, test_dict_03])
