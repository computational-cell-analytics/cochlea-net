"""Supervised training of the ribbon synapse detection model.

Replaces the `supervised_training` entry point of the czii-protein-challenge repository, which
was imported through a hard-coded path into another user's home directory and could therefore
not be pinned or changed from here.
"""

import multiprocessing as mp
from functools import partial
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn
import torch_em
import zarr
from elf import parallel
from torch.utils.data import DataLoader
from torch_em.data.concat_dataset import ConcatDataset
from torch_em.model import AnisotropicUNet
from torch_em.transform.raw import standardize

from .detection_dataset import DetectionDataset

# Downscaling per U-Net level. The first level keeps the z resolution, as the data is anisotropic.
SCALE_FACTORS = [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]


class DetectionLoss(nn.Module):
    """Mean squared error for the detection heatmap and the optional flow channels.

    Args:
        flow_weight: Weight of the flow channels. Keep it at zero for a heatmap-only model, which
            has a single output channel and no flow channels to compare.
        masked: Whether the target carries a binary loss mask in its last channel. Only voxels
            inside the mask then contribute to the loss.
    """
    def __init__(self, flow_weight: float = 0.0, masked: bool = False):
        super().__init__()
        self.flow_weight = flow_weight
        self.masked = masked
        # Read by torch_em.util.get_constructor_arguments when the trainer is serialized.
        self.init_kwargs = {"flow_weight": flow_weight, "masked": masked}

    @staticmethod
    def _mse(prediction, target, mask):
        error = (prediction - target) ** 2
        if mask is None:
            return error.mean()
        # Mean over the masked voxels of the whole batch. A patch without annotations has an
        # empty mask and contributes nothing; the clamp keeps an all-empty batch finite.
        n_values = mask.sum() * prediction.shape[1]
        return (error * mask).sum() / n_values.clamp(min=1.0)

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Without this a target that carries a mask channel while `masked` is off trains happily
        # with the mask ignored, which is silent and wrong.
        expected_channels = prediction.shape[1] + (1 if self.masked else 0)
        if target.shape[1] != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} target channels for a {prediction.shape[1]} channel "
                f"prediction with masked={self.masked}, got {target.shape[1]}."
            )

        mask = None
        if self.masked:
            mask, target = target[:, -1:], target[:, :-1]

        loss = self._mse(prediction[:, :1], target[:, :1], mask)
        if self.flow_weight:
            loss = loss + self.flow_weight * self._mse(prediction[:, 1:], target[:, 1:], mask)
        return loss


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


def _crop_standardization(raw_path, raw_key):
    """Standardize with the statistics of the full crop.

    `elf.parallel.mean_and_std` is the function that computes the statistics at inference, in
    `unet_prediction.calc_mean_and_std` and in `prediction_impl`. There they are taken inside the
    IHC mask, here over the complete crop.
    """
    raw = zarr.open(raw_path, mode="r")[raw_key]
    mean, std = parallel.mean_and_std(raw, n_threads=min(16, mp.cpu_count()))
    if std == 0:
        raise ValueError(f"The image data in {raw_path}:{raw_key} is constant and cannot be standardized.")
    return partial(standardize, mean=float(mean), std=float(std))


def _get_loader(
    raw_paths, label_paths, raw_key, patch_shape, batch_size, num_workers,
    label_transform, sampler, n_samples, normalize_raw, patch_seed=None,
):
    datasets = [
        DetectionDataset(
            raw_path=raw_path, raw_key=raw_key, label_path=label_path, patch_shape=patch_shape,
            raw_transform=_crop_standardization(raw_path, raw_key) if normalize_raw else None,
            label_transform=label_transform, sampler=sampler, n_samples=n_samples_dataset,
            # Offset per crop, so that two crops do not draw the same patch positions.
            patch_seed=None if patch_seed is None else patch_seed + index,
        )
        for index, (raw_path, label_path, n_samples_dataset)
        in enumerate(zip(raw_paths, label_paths, _samples_per_dataset(n_samples, len(raw_paths))))
    ]
    # Fixed patches only give a reproducible metric when the batches are fixed too: the trainer
    # averages the metric over batches, and the masked loss normalizes within a batch.
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
    normalize_raw: bool = True,
    val_patch_seed: Optional[int] = None,
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
        normalize_raw: Whether to standardize each crop with its own mean and standard deviation.
            Switch it off only to reproduce the training of v3 and v5, which never normalized.
        val_patch_seed: Seed that fixes the validation patches, so that every epoch is scored on
            the same data. Without it the validation set is redrawn at random on every epoch and
            the metric that selects 'best.pt' carries that noise.
    """
    loader_kwargs = dict(
        raw_key=raw_key, patch_shape=patch_shape, batch_size=batch_size, num_workers=num_workers,
        label_transform=label_transform, sampler=sampler, normalize_raw=normalize_raw,
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
    trainer.fit(n_iterations)
