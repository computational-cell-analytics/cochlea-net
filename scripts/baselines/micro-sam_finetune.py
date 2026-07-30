import argparse
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np

from micro_sam.training import default_sam_loader, train_instance_segmentation
from micro_sam.training.util import get_raw_transform

ROOT_TRAINING_DATA = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data"
ROOT_SGN_DATA = f"{ROOT_TRAINING_DATA}/SGN/2026-04_SGN-v2-data_micro-sam"
ANNOTATION_SUFFIX = "_annotations"


def list_image_label_pairs(data_dir: str) -> Tuple[List[str], List[str]]:
    """List the paired 2D image and annotation TIF files of a training directory.

    The annotation of `<stem>.tif` is `<stem>_annotations.tif`. Pairing by name avoids the silent
    image / annotation shift that sorting two separate file lists can introduce.
    """
    names = sorted(entry.name for entry in os.scandir(data_dir) if entry.name.endswith(".tif"))
    annotation_names = {name for name in names if ANNOTATION_SUFFIX in name}

    image_paths, label_paths = [], []
    for name in names:
        if name in annotation_names:
            continue
        annotation_name = f"{name[:-len('.tif')]}{ANNOTATION_SUFFIX}.tif"
        if annotation_name not in annotation_names:
            raise ValueError(f"{os.path.join(data_dir, name)} has no annotation file.")
        image_paths.append(os.path.join(data_dir, name))
        label_paths.append(os.path.join(data_dir, annotation_name))

    return image_paths, label_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cell_type", "-c", default="sgn", help="The cell type to train for, either 'sgn' or 'ihc'.",
    )
    parser.add_argument(
        "--name", help="Optional name for the model to be trained. If not given the current date is used."
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000,
        help="Number of training samples per epoch. By default one sample per training image is used, "
        "which results in very long epochs for a dataset of 2D slices.",
    )
    parser.add_argument("--n_epochs", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()

    run_name = datetime.now().strftime("%Y%m%d") if args.name is None else args.name
    name = f"cochlea_micro_sam_{run_name}"

    if args.cell_type == "sgn":
        root = f"{ROOT_TRAINING_DATA}/SGN/2026-04_SGN-v2-data_micro-sam"
    elif args.cell_type == "ihc":
        root = f"{ROOT_TRAINING_DATA}/IHC/2026-07_IHC-v11-data_micro-sam"
    else:
        raise ValueError("Choose either 'sgn' or 'ihc' for --cell_type.")

    train_image_paths, train_label_paths = list_image_label_pairs(os.path.join(root, "train"))
    val_image_paths, val_label_paths = list_image_label_pairs(os.path.join(root, "val"))
    print(f"Training on {len(train_image_paths)} slices, validating on {len(val_image_paths)} slices.")

    patch_shape = (256, 256)
    min_size = 1
    batch_size = 1  # the training batch size

    n_samples_val: Optional[int] = None if args.n_samples is None else max(5, args.n_samples // 10)

    # is_seg_dataset=False selects the ImageCollectionDataset, which stores only the file paths and
    # reads each slice on demand. The SegmentationDataset that torch_em would choose for TIF input
    # instead keeps one memory map per image and per annotation open for the lifetime of the dataset.
    # With tens of thousands of 2D slices that exceeds the per-process mapping limit
    # (vm.max_map_count, 65530 by default) and mmap then fails with "OSError: [Errno 12] Cannot
    # allocate memory", although only a small amount of RAM is in use.
    train_loader = default_sam_loader(
        raw_paths=train_image_paths, raw_key=None, label_paths=train_label_paths, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        raw_transform=get_raw_transform("normalize_percentile"),
        num_workers=args.num_workers, batch_size=batch_size, is_train=True,
        min_size=min_size, is_seg_dataset=True, n_samples=args.n_samples,
    )
    val_loader = default_sam_loader(
        raw_paths=val_image_paths, raw_key=None, label_paths=val_label_paths, label_key=None,
        patch_shape=patch_shape, with_segmentation_decoder=True,
        train_instance_segmentation_only=True,
        raw_transform=get_raw_transform("normalize_percentile"),
        num_workers=args.num_workers, batch_size=1, is_train=False,
        min_size=min_size, is_seg_dataset=True, n_samples=n_samples_val,
    )

    train_instance_segmentation(
        name=name, model_type="vit_b_lm", train_loader=train_loader, val_loader=val_loader,
        n_epochs=args.n_epochs,
        save_root=".",
    )


if __name__ == "__main__":
    main()
