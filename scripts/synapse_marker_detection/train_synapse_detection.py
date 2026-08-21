import argparse
import json
import os
import sys
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from flamingo_tools.synapse_detection.detection_dataset import (
    CsvHeatmapFlowTransform,
    CsvHeatmapTransform,
    DetectionDataset,
    MinPointSampler,
)

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/schilling40/u15000/czii-protein-challenge/detection")

from utils.training.training import supervised_training  # noqa

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
ROOT_SYNAPSE_DATA = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data")


def train(
    root_data_dir, version="v5", val_sample_size=3, model_suffix=None, random_state=None,
    use_flow=False, sampler_name=None,
):
    if model_suffix is None:
        model_suffix = version
        json_path = os.path.join(root_data_dir, version, "train_val_split.json")
        if random_state is None:
            random_state = 42
    else:
        json_path = os.path.join(root_data_dir, version, f"train_val_split_{model_suffix}.json")
        if random_state is None:
            random_state = sum([ord(char) for char in model_suffix.lower()])
        print(f"Using random state {random_state}.")

    # The sampler was silently dropped by the czii-protein-challenge version used for v3, so
    # training without flow reproduces v3 only when no sampler is applied.
    if sampler_name is None:
        sampler_name = "minpoint" if use_flow else "none"
    sampler = MinPointSampler(min_points=1, p_reject=0.8) if sampler_name == "minpoint" else None

    if use_flow:
        out_channels = 5
        label_transform = CsvHeatmapFlowTransform(sigma=1, eps=1e-5)
        # Keep the combined heatmap and flow loss of supervised_training.
        loss_kwargs = {}
    else:
        out_channels = 1
        label_transform = CsvHeatmapTransform(sigma=1, eps=1e-5)
        # The combined loss takes the MSE over the empty slice pred[:, 1:] for a single output
        # channel, which returns nan. Pass the loss through a dict so that the script does not
        # depend on importing CombinedLoss from the upstream repository.
        loss_kwargs = {"loss_fn": torch.nn.MSELoss(reduction="mean")}

    image_dir = os.path.join(root_data_dir, version, "images")
    label_dir = os.path.join(root_data_dir, version, "labels")
    model_name = f"synapse_detection_{model_suffix}"

    image_paths = sorted(glob(os.path.join(image_dir, "*.zarr")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.csv")))
    assert len(image_paths) == len(label_paths)

    train_paths, val_paths, train_label_paths, val_label_paths = train_test_split(
        image_paths, label_paths, test_size=val_sample_size, random_state=random_state,
    )

    # We need to give the paths for the test loader, although it's never used.
    test_paths, test_label_paths = val_paths, val_label_paths

    train_val_dic = {
        "train": [os.path.splitext(os.path.basename(f))[0] for f in train_paths],
        "val": [os.path.splitext(os.path.basename(f))[0] for f in val_paths],
        "flow": use_flow,
        "sampler": sampler_name,
    }

    with open(json_path, "w") as f:
        json.dump(train_val_dic, f, indent='\t', separators=(',', ': '))

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")
    print(f"{out_channels} output channels, flow: {use_flow}, sampler: {sampler_name}")

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
        out_channels=out_channels,
        augmentations=None,
        label_transform=label_transform,
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
        sampler=sampler,
        num_workers=8,
        **loss_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Train a network for synapse detection."
    )

    parser.add_argument("-i", "--input_dir", type=str, default=ROOT_SYNAPSE_DATA)
    parser.add_argument("-v", "--version", type=str, default="v5")
    parser.add_argument("-r", "--random_state", type=int, default=None,
                        help="Random state for train and validation split. Default: 42 for fixed versions.")
    parser.add_argument("-m", "--model_suffix", type=str, default=None,
                        help="Custom suffix for model name. Default: Same as version.")
    parser.add_argument("--use_flow", action="store_true",
                        help="Train the 4 stereographic flow channels in addition to the heatmap. "
                             "Default: train the heatmap only, as for synapse_detection_v3.")
    parser.add_argument("--sampler", type=str, default=None, choices=["none", "minpoint"],
                        help="Sampler to reject patches with too few points. "
                             "Default: minpoint with --use_flow, none without.")

    args = parser.parse_args()
    train(
        root_data_dir=args.input_dir,
        version=args.version,
        model_suffix=args.model_suffix,
        random_state=args.random_state,
        use_flow=args.use_flow,
        sampler_name=args.sampler,
    )


if __name__ == "__main__":
    main()
