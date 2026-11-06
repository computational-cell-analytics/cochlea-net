"""Supervised training of the ribbon synapse detection model.

Replaces the `supervised_training` entry point of the czii-protein-challenge repository, which
was imported through a hard-coded path into another user's home directory and could therefore
not be pinned or changed from here.
"""

import math
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch_em
from torch.utils.data import DataLoader
from torch_em.data.concat_dataset import ConcatDataset
from torch_em.model import AnisotropicUNet

from .detection_dataset import DetectionDataset

# Downscaling per U-Net level. The first level keeps the z resolution, as the patch is shallow in z.
SCALE_FACTORS = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


class DetectionLoss(nn.Module):
    """Mean squared error for the detection heatmap and the optional flow channels.

    Args:
        flow_weight: Weight of the flow channels. Keep it at zero for a heatmap-only model, which
            has a single output channel and no flow channels to compare.
    """
    def __init__(self, flow_weight: float = 0.0):
        super().__init__()
        self.flow_weight = flow_weight
        # Read by torch_em.util.get_constructor_arguments when the trainer is serialized.
        self.init_kwargs = {"flow_weight": flow_weight}

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Without this a 5 channel target trains a 1 channel model on the heatmap alone, silently.
        if target.shape[1] != prediction.shape[1]:
            raise ValueError(
                f"Expected {prediction.shape[1]} target channels, got {target.shape[1]}."
            )

        loss = ((prediction[:, :1] - target[:, :1]) ** 2).mean()
        if self.flow_weight:
            loss = loss + self.flow_weight * ((prediction[:, 1:] - target[:, 1:]) ** 2).mean()
        return loss


class EpochCosineSchedule(torch.optim.lr_scheduler.CosineAnnealingLR):
    """Cosine decay of the learning rate to zero, with one step per epoch.

    torch_em passes the validation metric to `step`. This schedule ignores it, so that the learning
    rate does not depend on a noisy validation metric.

    Args:
        optimizer: The optimizer.
        T_max: The number of epochs of the run.
    """
    def __init__(self, optimizer: torch.optim.Optimizer, T_max: int):
        super().__init__(optimizer, T_max=T_max)
        # Read by torch_em.util.get_constructor_arguments when the trainer is serialized.
        self.init_kwargs = {"T_max": T_max}

    def step(self, metric=None):
        super().step()


def get_3d_model(in_channels: int, out_channels: int, initial_features: int = 32) -> nn.Module:
    """Get the anisotropic 3D U-Net for synapse detection.

    Args:
        in_channels: The number of input channels of the network.
        out_channels: The number of output channels of the network.
        initial_features: The number of features in the first level of the U-Net.

    Returns:
        The U-Net.
    """
    return AnisotropicUNet(
        scale_factors=SCALE_FACTORS,
        in_channels=in_channels,
        out_channels=out_channels,
        initial_features=initial_features,
        gain=2,
        final_activation=None,
    )


def _samples_per_dataset(n_samples, n_datasets):
    if n_samples is None:
        return [None] * n_datasets
    per_dataset, remainder = divmod(n_samples, n_datasets)
    return [per_dataset + 1 if i < remainder else per_dataset for i in range(n_datasets)]


def _get_loader(
    raw_paths, label_paths, raw_key, patch_shape, batch_size, num_workers,
    label_transform, sampler, n_samples, patch_seed=None,
):
    datasets = [
        DetectionDataset(
            raw_path=raw_path, raw_key=raw_key, label_path=label_path, patch_shape=patch_shape,
            label_transform=label_transform, sampler=sampler, n_samples=n_samples_dataset,
            # Offset per crop, so that two crops do not draw the same patch positions.
            patch_seed=None if patch_seed is None else patch_seed + index,
        )
        for index, (raw_path, label_path, n_samples_dataset)
        in enumerate(zip(raw_paths, label_paths, _samples_per_dataset(n_samples, len(raw_paths))))
    ]
    # Fixed patches only give a reproducible metric when the batches are fixed too, because the
    # trainer averages the metric over batches.
    shuffle = patch_seed is None
    loader = DataLoader(
        ConcatDataset(*datasets), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    loader.shuffle = shuffle
    return loader


def supervised_training(
    name: str,
    train_paths: Sequence[str],
    train_label_paths: Sequence[str],
    val_paths: Sequence[str],
    val_label_paths: Sequence[str],
    raw_key: str,
    patch_shape: Sequence[int],
    label_transform: Callable,
    loss: nn.Module,
    out_channels: int,
    batch_size: int = 1,
    lr: float = 1e-4,
    n_iterations: int = int(1e5),
    save_root: Optional[str] = None,
    n_samples_train: Optional[int] = None,
    n_samples_val: Optional[int] = None,
    sampler: Optional[Callable] = None,
    num_workers: int = 8,
    metric: Optional[nn.Module] = None,
    val_patch_seed: Optional[int] = None,
    lr_schedule: str = "plateau",
) -> None:
    """Train the synapse detection model.

    Args:
        name: The name of the checkpoint to be trained.
        train_paths: Filepaths to the zarr files with the training images.
        train_label_paths: Filepaths to the CSV files with the training annotations.
        val_paths: Filepaths to the zarr files with the validation images.
        val_label_paths: Filepaths to the CSV files with the validation annotations.
        raw_key: The key of the image data within the zarr files.
        patch_shape: The patch shape used for a training example.
        label_transform: The transform that creates the target from the annotations.
        loss: The training loss. Unless `metric` is given it is also the validation metric, so
            that the criterion selecting 'best.pt' agrees with the one that is optimized.
        out_channels: The number of output channels of the U-Net.
        batch_size: The batch size for training.
        lr: The initial learning rate.
        n_iterations: The number of iterations to train for.
        save_root: The folder where the checkpoint is saved.
        n_samples_train: The number of samples per training epoch.
        n_samples_val: The number of samples per validation epoch.
        sampler: The sampler for rejecting patches with too few annotations.
        num_workers: The number of data loader workers.
        metric: The validation metric, which selects 'best.pt'. By default the loss is reused.
        val_patch_seed: Seed that fixes the validation patches, so that every epoch is scored on
            the same data. Without it the validation set is redrawn at random on every epoch and
            the metric that selects 'best.pt' carries that noise.
        lr_schedule: "plateau" is the torch_em default of every model up to v10: it halves the
            learning rate when the validation metric does not improve for 5 epochs. In v5 to v10
            the learning rate reached its floor of 1.2e-8 within 100 to 200 of 1000 epochs.
            "cosine" decays it to zero over the whole run, independent of the validation metric.
            Use 'latest.pt' with it.
    """
    if lr_schedule not in ("plateau", "cosine"):
        raise ValueError(f"Unknown lr_schedule {lr_schedule}, expected 'plateau' or 'cosine'.")
    loader_kwargs = dict(
        raw_key=raw_key, patch_shape=patch_shape, batch_size=batch_size, num_workers=num_workers,
        label_transform=label_transform, sampler=sampler,
    )
    train_loader = _get_loader(train_paths, train_label_paths, n_samples=n_samples_train, **loader_kwargs)
    val_loader = _get_loader(
        val_paths, val_label_paths, n_samples=n_samples_val, patch_seed=val_patch_seed, **loader_kwargs
    )

    model = get_3d_model(in_channels=1, out_channels=out_channels)

    trainer = torch_em.default_segmentation_trainer(
        name=name,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=lr,
        mixed_precision=True,
        log_image_interval=100,
        compile_model=False,
        save_root=save_root,
        loss=loss,
        metric=loss if metric is None else metric,
    )
    if lr_schedule == "cosine":
        # The trainer serializes its scheduler in fit, so replacing it here is recorded correctly.
        trainer.lr_scheduler = EpochCosineSchedule(
            trainer.optimizer, T_max=math.ceil(n_iterations / len(train_loader))
        )
    trainer.fit(n_iterations)
