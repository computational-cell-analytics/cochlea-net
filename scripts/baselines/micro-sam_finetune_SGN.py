import argparse
import os
from datetime import datetime

import numpy as np

from micro_sam.training import default_sam_loader, train_instance_segmentation
from micro_sam.training.util import get_raw_transform

ROOT_TRAINING_DATA = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data"
ROOT_SGN_DATA = f"{ROOT_TRAINING_DATA}/SGN/2026-04_SGN-v2-data_micro-sam"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", "-i", help="The root folder with the annotated training crops.",
        default=ROOT_SGN_DATA,
    )
    parser.add_argument(
        "--name", help="Optional name for the model to be trained. If not given the current date is used."
    )
    args = parser.parse_args()

    root = args.root
    run_name = datetime.now().strftime("%Y%m%d") if args.name is None else args.name
    name = f"cochlea_micro_sam_{run_name}"

    train_dir = os.path.join(root, "train")
    train_image_paths = [entry.path for entry in os.scandir(train_dir) if
                         "annotation" not in entry.name and
                         ".tif" in entry.name]
    train_label_paths = [entry.path for entry in os.scandir(train_dir) if
                         "annotation" in entry.name and
                         ".tif" in entry.name]

    train_image_paths.sort()
    train_label_paths.sort()

    val_dir = os.path.join(root, "val")
    val_image_paths = [entry.path for entry in os.scandir(val_dir) if
                       "annotation" not in entry.name and
                       ".tif" in entry.name]
    val_label_paths = [entry.path for entry in os.scandir(val_dir) if
                       "annotation" in entry.name and
                       ".tif" in entry.name]
    val_image_paths.sort()
    val_label_paths.sort()

    patch_shape = (256, 256)
    min_size = 1
    batch_size = 1  # the training batch size

    train_loader = default_sam_loader(
        raw_paths=train_image_paths, raw_key=None, label_paths=train_label_paths, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        raw_transform=get_raw_transform("normalize_percentile"),
        num_workers=6, batch_size=batch_size, is_train=True,
        min_size=min_size,
    )
    val_loader = default_sam_loader(
        raw_paths=val_image_paths, raw_key=None, label_paths=val_label_paths, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        raw_transform=get_raw_transform("normalize_percentile"),
        num_workers=6, batch_size=1, is_train=False,
        min_size=min_size,
    )

    train_instance_segmentation(
        name=name, model_type="vit_b_lm", train_loader=train_loader, val_loader=val_loader,
        n_epochs=50,
        save_root=".",
    )


if __name__ == "__main__":
    main()
