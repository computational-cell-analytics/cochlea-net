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
    use_flow=False, sampler_name=None, n_iterations=int(1e5), save_root=SAVE_ROOT,
    legacy_recipe=False, sigma=1.0, fixed_target=False, lr_schedule="plateau",
):
    if use_flow and sigma != 1:
        # The flow correction at inference assumes the target sigma (_HEATMAP_FLOW_SIGMA = 1 in
        # flamingo_tools/segmentation/synapse_detection.py) and would shift every detection.
        raise ValueError(f"The flow targets need sigma 1, got {sigma}.")
    if legacy_recipe and fixed_target:
        raise ValueError("--legacy_recipe reproduces v3 and v5, which were trained without --fixed_target.")

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

    # The heatmap-only legacy runs (v3, v6-1) had no sampler, because the czii-protein-challenge
    # version used for v3 silently dropped it. Every other recipe uses it, v7 included.
    if sampler_name is None:
        sampler_name = "none" if (legacy_recipe and not use_flow) else "minpoint"
    # MinPointSampler accepts a patch when n_points > min_points, so min_points=1 needs two annotations.
    sampler = MinPointSampler(min_points=1, p_reject=0.8) if sampler_name == "minpoint" else None

    image_dir = os.path.join(root_data_dir, version, "images")
    label_dir = os.path.join(root_data_dir, version, "labels")
    model_name = f"synapse_detection_{model_suffix}"

    image_paths = sorted(glob(os.path.join(image_dir, "*.zarr")))
    label_paths = sorted(glob(os.path.join(label_dir, "*.csv")))
    assert len(image_paths) == len(label_paths)

    transform_class = CsvHeatmapFlowTransform if use_flow else CsvHeatmapTransform
    out_channels = 5 if use_flow else 1
    label_transform = transform_class(sigma=sigma, eps=1e-5, fixed_target=fixed_target)
    loss = DetectionLoss(flow_weight=0.1 if use_flow else 0.0)
    # v3 and v5 selected best.pt with an unweighted mean squared error over every output channel,
    # and they redrew the validation patches on every epoch. The legacy recipe restores both.
    metric = torch.nn.MSELoss(reduction="mean") if legacy_recipe else None
    # The same number that fixes the train / validation split also fixes the patches inside the
    # validation crops, so that the metric selecting best.pt is scored on the same data.
    val_patch_seed = None if legacy_recipe else random_state

    train_paths, val_paths, train_label_paths, val_label_paths = train_test_split(
        image_paths, label_paths, test_size=val_sample_size, random_state=random_state,
    )

    train_val_dic = {
        "train": [os.path.splitext(os.path.basename(f))[0] for f in train_paths],
        "val": [os.path.splitext(os.path.basename(f))[0] for f in val_paths],
        "flow": use_flow,
        "sampler": sampler_name,
        "sigma": sigma,
        "fixed_target": fixed_target,
        "legacy_recipe": legacy_recipe,
        "val_patch_seed": val_patch_seed,
        "lr_schedule": lr_schedule,
    }

    with open(json_path, "w") as f:
        json.dump(train_val_dic, f, indent='\t', separators=(',', ': '))

    print("Start training with:")
    print(len(train_paths), "tomograms for training")
    print(len(val_paths), "tomograms for validation")
    print(f"{out_channels} output channels, flow: {use_flow}, sampler: {sampler_name}")
    print(f"Fixed target: {fixed_target}, legacy recipe: {legacy_recipe}, lr schedule: {lr_schedule}")

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
        val_patch_seed=val_patch_seed,
        save_root=save_root,
        n_samples_train=3200,
        n_samples_val=160,
        sampler=sampler,
        num_workers=8,
        lr_schedule=lr_schedule,
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
                             "Default: none with --legacy_recipe and without --use_flow, else minpoint.")
    parser.add_argument("-n", "--n_iterations", type=int, default=int(1e5),
                        help="Number of training iterations. Default: 100000.")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Standard deviation in voxels of the Gaussian target around each "
                             "annotation. A smaller value keeps adjacent synapses apart in the "
                             "target. Default: 1, the value of every model except v10. Only 1 "
                             "works with --use_flow.")
    parser.add_argument("--fixed_target", action="store_true",
                        help="Give every isolated annotation the same target peak, and keep the "
                             "Gaussians of the annotations just outside the patch. Every model up "
                             "to v10 was trained without it. Cannot be combined with --legacy_recipe.")
    parser.add_argument("--lr_schedule", type=str, default="plateau", choices=["plateau", "cosine"],
                        help="'plateau' halves the learning rate when the validation metric stops "
                             "improving, as for every model up to v10. It reaches its floor early. "
                             "'cosine' decays it to zero over the whole run; use latest.pt with it. "
                             "Default: plateau.")
    parser.add_argument("-s", "--save_root", type=str, default=SAVE_ROOT,
                        help=f"Folder for the checkpoints. Default: {SAVE_ROOT}.")
    parser.add_argument("--legacy_recipe", action="store_true",
                        help="Reproduce the training recipe of synapse_detection_v3 and v5: an "
                             "unweighted mean squared error over all output channels as the "
                             "validation metric, validation patches that are redrawn on every "
                             "epoch, and no sampler without --use_flow. Required to retrain those "
                             "models.")

    args = parser.parse_args()
    train(
        root_data_dir=args.input_dir,
        version=args.version,
        model_suffix=args.model_suffix,
        random_state=args.random_state,
        use_flow=args.use_flow,
        sampler_name=args.sampler,
        n_iterations=args.n_iterations,
        save_root=args.save_root,
        legacy_recipe=args.legacy_recipe,
        sigma=args.sigma,
        fixed_target=args.fixed_target,
        lr_schedule=args.lr_schedule,
    )


if __name__ == "__main__":
    main()
