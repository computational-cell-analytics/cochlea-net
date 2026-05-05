import argparse
import os
import sys
from glob import glob

from sklearn.model_selection import train_test_split
from detection_dataset import DetectionDataset, MinPointSampler

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/schilling40/u15000/czii-protein-challenge/detection")

from utils.training.training import supervised_training  # noqa

ROOT_SYNAPSE_DATA = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data"  # noqa


def get_paths(image_dir, label_dir, split):
    image_paths = sorted(glob(os.path.join(image_dir, "*.zarr")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.csv")))
    assert len(image_paths) == len(label_paths)

    train_images, val_images, train_labels, val_labels = train_test_split(
        image_paths, label_paths, test_size=2, random_state=42
    )

    if split == "train":
        image_paths = train_images
        label_paths = train_labels
    else:
        image_paths = val_images
        label_paths = val_labels

    return image_paths, label_paths


def train(root_data_dir, version="v4"):
    image_dir = os.path.join(root_data_dir, version, "images")
    label_dir = os.path.join(root_data_dir, version, "labels")
    model_name = f"synapse_detection_{version}"

    train_paths, train_label_paths = get_paths(image_dir, label_dir, "train")
    val_paths, val_label_paths = get_paths(image_dir, label_dir, "val")
    # We need to give the paths for the test loader, although it's never used.
    test_paths, test_label_paths = val_paths, val_label_paths

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")

    patch_shape = [40, 112, 112]
    batch_size = 32
    check = False

    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths=train_label_paths,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        raw_key="raw",
        patch_shape=patch_shape, batch_size=batch_size,
        check=check,
        lr=1e-4,
        n_iterations=int(1e5),
        out_channels=5,
        augmentations=None,
        eps=1e-5,
        sigma=1,
        lower_bound=None,
        upper_bound=None,
        test_paths=test_paths,
        test_label_paths=test_label_paths,
        # save_root="",
        dataset_class=DetectionDataset,
        n_samples_train=3200,
        n_samples_val=160,
        sampler=MinPointSampler(min_points=1, p_reject=0.8),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a network for synapse detection."
    )

    parser.add_argument("-i", "--input_dir", type=str, default=ROOT_SYNAPSE_DATA)
    parser.add_argument("-v", "--version", type=str, default="v4")

    args = parser.parse_args()
    train(args.input_dir, args.version)


if __name__ == "__main__":
    main()
