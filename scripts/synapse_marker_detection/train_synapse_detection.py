import argparse
import json
import os
from glob import glob

import torch
from sklearn.model_selection import train_test_split
from flamingo_tools.synapse_detection.detection_dataset import (
    CsvHeatmapFlowTransform,
    CsvHeatmapTransform,
    MinPointSampler,
)
from flamingo_tools.synapse_detection.training import DetectionLoss, supervised_training

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
ROOT_SYNAPSE_DATA = os.path.join(COCHLEA_DIR, "training_data/synapses/training_data")
SAVE_ROOT = "/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools/networks/synapses"


def train(
    root_data_dir, version="v5", val_sample_size=3, model_suffix=None, random_state=None,
    use_flow=False, sampler_name=None, n_iterations=int(1e5), mask_radius=None, save_root=SAVE_ROOT,
    legacy_recipe=False,
):
    if mask_radius is not None and mask_radius < 1:
        raise ValueError(f"The mask radius must be at least one voxel, got {mask_radius}.")
    if legacy_recipe and mask_radius is not None:
        raise ValueError(
            "--legacy_recipe reproduces the v3 and v5 training, which had no loss mask. Its "
            "validation metric covers every target channel and cannot read a masked target."
        )

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

    # A masked loss ignores every patch without annotations, and only 31% of uniformly sampled
    # patches contain one. Without flow and without a mask the sampler is off, because the
    # czii-protein-challenge version used for v3 silently dropped it.
    if sampler_name is None:
        sampler_name = "minpoint" if (use_flow or mask_radius is not None) else "none"
    # MinPointSampler accepts a patch when n_points > min_points, so the min_points=1 of v5 needs
    # two annotations. A masked run keeps every patch that has one, because a single annotation
    # already gives a usable mask.
    min_points = 0 if mask_radius is not None else 1
    sampler = MinPointSampler(min_points=min_points, p_reject=0.8) if sampler_name == "minpoint" else None

    if use_flow:
        out_channels = 5
        label_transform = CsvHeatmapFlowTransform(sigma=1, eps=1e-5, mask_radius=mask_radius)
    else:
        out_channels = 1
        label_transform = CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=mask_radius)
    loss = DetectionLoss(flow_weight=0.1 if use_flow else 0.0, masked=mask_radius is not None)
    # v3 and v5 selected best.pt with an unweighted mean squared error over every output channel,
    # they trained on unnormalized input, and they redrew the validation patches on every epoch.
    # The legacy recipe restores all three together.
    metric = torch.nn.MSELoss(reduction="mean") if legacy_recipe else None
    # The same number that fixes the train / validation split also fixes the patches inside the
    # validation crops, so that the metric selecting best.pt is scored on the same data.
    val_patch_seed = None if legacy_recipe else random_state

    image_dir = os.path.join(root_data_dir, version, "images")
    label_dir = os.path.join(root_data_dir, version, "labels")
    model_name = f"synapse_detection_{model_suffix}"

    image_paths = sorted(glob(os.path.join(image_dir, "*.zarr")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.csv")))
    assert len(image_paths) == len(label_paths)

    train_paths, val_paths, train_label_paths, val_label_paths = train_test_split(
        image_paths, label_paths, test_size=val_sample_size, random_state=random_state,
    )

    train_val_dic = {
        "train": [os.path.splitext(os.path.basename(f))[0] for f in train_paths],
        "val": [os.path.splitext(os.path.basename(f))[0] for f in val_paths],
        "flow": use_flow,
        "sampler": sampler_name,
        "mask_radius": mask_radius,
        "legacy_recipe": legacy_recipe,
        "val_patch_seed": val_patch_seed,
    }

    with open(json_path, "w") as f:
        json.dump(train_val_dic, f, indent='\t', separators=(',', ': '))

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")
    print(f"{out_channels} output channels, flow: {use_flow}, sampler: {sampler_name}")
    print(f"Loss mask radius: {mask_radius}, legacy recipe: {legacy_recipe}")

    supervised_training(
        name=model_name,
        train_paths=train_paths,
        train_label_paths=train_label_paths,
        val_paths=val_paths,
        val_label_paths=val_label_paths,
        raw_key="raw",
        patch_shape=[40, 112, 112],
        batch_size=32,
        lr=1e-4,
        n_iterations=n_iterations,
        out_channels=out_channels,
        label_transform=label_transform,
        loss=loss,
        metric=metric,
        normalize_raw=not legacy_recipe,
        val_patch_seed=val_patch_seed,
        save_root=save_root,
        n_samples_train=3200,
        n_samples_val=160,
        sampler=sampler,
        num_workers=8,
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
                             "Default: train the heatmap only, the output layout of "
                             "synapse_detection_v3.")
    parser.add_argument("--sampler", type=str, default=None, choices=["none", "minpoint"],
                        help="Sampler to reject patches with too few points. "
                             "Default: minpoint with --use_flow or --mask_radius, none without.")
    parser.add_argument("-n", "--n_iterations", type=int, default=int(1e5),
                        help="Number of training iterations. Default: 100000.")
    parser.add_argument("--mask_radius", type=int, default=None,
                        help="Half width in voxels of the cube marked around each annotation. "
                             "The loss is restricted to these regions, so that unannotated CTBP2 "
                             "spots outside the IHCs do not count as background. Use 16. "
                             "Default: no mask, the loss covers the full patch.")
    parser.add_argument("-s", "--save_root", type=str, default=SAVE_ROOT,
                        help=f"Folder for the checkpoints. Default: {SAVE_ROOT}.")
    parser.add_argument("--legacy_recipe", action="store_true",
                        help="Reproduce the training recipe of synapse_detection_v3 and v5: no raw "
                             "normalization, and an unweighted mean squared error over all output "
                             "channels as the validation metric, and validation patches that are "
                             "redrawn on every epoch. Required to retrain those models. Cannot be "
                             "combined with --mask_radius.")

    args = parser.parse_args()
    train(
        root_data_dir=args.input_dir,
        version=args.version,
        model_suffix=args.model_suffix,
        random_state=args.random_state,
        use_flow=args.use_flow,
        sampler_name=args.sampler,
        n_iterations=args.n_iterations,
        mask_radius=args.mask_radius,
        save_root=args.save_root,
        legacy_recipe=args.legacy_recipe,
    )


if __name__ == "__main__":
    main()
