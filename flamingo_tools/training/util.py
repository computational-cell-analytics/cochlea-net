from typing import List, Optional, Sequence, Tuple

import torch.nn as nn
import torch_em
from torch_em.model import UNet3d, AnisotropicUNet
from torch.utils.data import DataLoader


def get_3d_model(
    out_channels: int = 3, final_activation: Optional[str] = "Sigmoid", scale_factors: Optional[List[List[int]]] = None
) -> nn.Module:
    """Get a 3D U-Net for segmentation or detection tasks.

    Args:
        out_channels: The number of output channels of the network.
        final_activation: The activation applied to the last layer.
            Set to 'None' for no activation; by default this applies a Sigmoid activation.

    Returns:
        The 3D U-Net.
    """
    if scale_factors is None:
        return UNet3d(in_channels=1, out_channels=out_channels, initial_features=32, final_activation=final_activation)
    else:
        return AnisotropicUNet(
            in_channels=1, out_channels=out_channels, initial_features=32,
            final_activation=final_activation, scale_factors=scale_factors
        )


def get_supervised_loader(
    image_paths: Sequence[str],
    label_paths: Sequence[str],
    patch_shape: Tuple[int, int, int],
    batch_size: int,
    image_key: Optional[str] = None,
    label_key: Optional[str] = None,
    n_samples: Optional[int] = None,
    raw_transform: Optional[callable] = None,
    anisotropy: Optional[float] = None,
) -> DataLoader:
    """Get a data loader for a supervised segmentation task.

    Args:
        image_paths: The filepaths to the image data. These can be stored either in tif or in hdf5/zarr/n5.
        image_paths: The filepaths to the label masks. These can be stored either in tif or in hdf5/zarr/n5.
        patch_shape: The 3D patch shape for training.
        batch_Size: The batch size for training.
        image_key: Internal path for the image data. This is only required for hdf5/zarr/n5 data.
        image_key: Internal path for the label masks. This is only required for hdf5/zarr/n5 data.
        n_samples: The number of samples to use for training.
        raw_transform: Optional transformation for the raw data.
        anisotropy: The anisotropy factor for distance target computation.

    Returns:
        The data loader.
    """
    assert len(image_paths) == len(label_paths)
    assert len(image_paths) > 0
    sampling = None if anisotropy is None else (anisotropy, 1.0, 1.0)
    label_transform = torch_em.transform.label.PerObjectDistanceTransform(
        distances=True, boundary_distances=True, foreground=True, sampling=sampling,
    )
    sampler = torch_em.data.sampler.MinInstanceSampler(p_reject=0.8)
    loader = torch_em.default_segmentation_loader(
        raw_paths=image_paths, raw_key=image_key, label_paths=label_paths, label_key=label_key,
        batch_size=batch_size, patch_shape=patch_shape, label_transform=label_transform,
        n_samples=n_samples, num_workers=8, shuffle=True, sampler=sampler
    )
    return loader
