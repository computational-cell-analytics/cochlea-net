"""Utility functionality for downloading CochleaNet models.
"""

import argparse
import importlib
import os
import tempfile
from typing import Dict, Optional, Union

import pooch
import torch
from .file_utils import get_cache_dir

_UNET3D_KWARGS = {
    "in_channels": 1,
    "out_channels": 3,
    "depth": 4,
    "initial_features": 32,
    "gain": 2,
    "final_activation": "Sigmoid",
}

_ANISOTROPIC_UNET_KWARGS = {
    "in_channels": 1,
    "scale_factors": [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "initial_features": 32,
    "gain": 2,
    "anisotropic_kernel": False,
}

# The CochleaNet models on BioImage.IO. The sha256 is the hash of the 'pytorch_state_dict' weights.
# The model class and the model kwargs give the architecture of these weights.
MODELS = {
    "SGN": {
        "bioimageio_id": "knowledgeable-mouse",
        "version": "v0",
        "sha256": "fb25affdbf7b05d9ea02458e44125866ceb93dadcc35389ae7d2c7c368887f52",
        "model_class": "torch_em.model.unet.UNet3d",
        "model_kwargs": _UNET3D_KWARGS,
    },
    "IHC": {
        "bioimageio_id": "content-fish",
        "version": "v0",
        "sha256": "8c345996ea30256eb479553b0121a4f2ea978d11b596c19477881b67f8174a09",
        "model_class": "torch_em.model.unet.UNet3d",
        "model_kwargs": _UNET3D_KWARGS,
    },
    "Synapses": {
        "bioimageio_id": "loyal-ant",
        "version": "v0",
        "sha256": "37c435e34665df08b67673835f5fa0e69388d99fe3a46717f5d052a19a749856",
        "model_class": "torch_em.model.unet.AnisotropicUNet",
        "model_kwargs": {**_ANISOTROPIC_UNET_KWARGS, "out_channels": 1, "final_activation": None},
    },
    "SGN-lowres": {
        "bioimageio_id": "nervous-shark",
        "version": "v0",
        "sha256": "2fe193929460a5933550b087c8221950fb92fcf313bb695d94e7c40fed7c0e17",
        "model_class": "torch_em.model.unet.AnisotropicUNet",
        "model_kwargs": {**_ANISOTROPIC_UNET_KWARGS, "out_channels": 3, "final_activation": "Sigmoid"},
    },
    "IHC-lowres": {
        "bioimageio_id": "confused-cow",
        "version": "v0",
        "sha256": "77785144b839af15f95f2542946f04fab86061dd793a76798fdd8169eee60014",
        "model_class": "torch_em.model.unet.UNet3d",
        "model_kwargs": _UNET3D_KWARGS,
    },
}

BIOIMAGEIO_FILE_URL = (
    "https://hypha.aicell.io/bioimage-io/artifacts/{bioimageio_id}/files/{file_name}?version={version}"
)


def get_model_url(model_type: str, file_name: str = "weights.pt") -> str:
    """Get the BioImage.IO download URL for a file of a pretrained model.

    Args:
        model_type: The model type.
        file_name: The file of the BioImage.IO model, for example 'weights.pt' or 'rdf.yaml'.

    Returns:
        The download URL.
    """
    model = _get_model_spec(model_type)
    return BIOIMAGEIO_FILE_URL.format(
        bioimageio_id=model["bioimageio_id"], file_name=file_name, version=model["version"]
    )


def _get_model_spec(model_type):
    if model_type not in MODELS:
        raise ValueError(f"Invalid model: {model_type}. Choose one of {list(MODELS.keys())}.")
    return MODELS[model_type]


def _create_model(model_type):
    model = _get_model_spec(model_type)
    module_path, class_name = model["model_class"].rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    return model_class(**model["model_kwargs"])


def _get_default_device():
    # Check that we're in CI and use the CPU if we are.
    # Otherwise the tests may run out of memory on MAC if MPS is used.
    if os.getenv("GITHUB_ACTIONS") == "true":
        return "cpu"
    # Use cuda enabled gpu if it's available.
    if torch.cuda.is_available():
        device = "cuda"
    # As second priority use mps.
    # See https://pytorch.org/docs/stable/notes/mps.html for details
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    # Use the CPU as fallback.
    else:
        device = "cpu"
    return device


def get_device(device: Optional[Union[str, torch.device]] = None) -> Union[str, torch.device]:
    """Get the torch device.

    If no device is passed the default device for your system is used.
    Else it will be checked if the device you have passed is supported.

    Args:
        device: The input device.

    Returns:
        The device.
    """
    if device is None or device == "auto":
        device = _get_default_device()
    else:
        device_type = device if isinstance(device, str) else device.type
        if device_type.lower() == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("PyTorch CUDA backend is not available.")
        elif device_type.lower() == "mps":
            if not (torch.backends.mps.is_available() and torch.backends.mps.is_built()):
                raise RuntimeError("PyTorch MPS backend is not available or is not built correctly.")
        elif device_type.lower() == "cpu":
            pass  # cpu is always available
        else:
            raise RuntimeError(f"Unsupported device: {device}. Please choose from 'cpu', 'cuda', or 'mps'.")
    return device


def get_model_path(model_type: str) -> str:
    """Get the local path to a pretrained model.

    The first call downloads the model weights from BioImage.IO. It saves the weights together with the
    model architecture as a torch_em checkpoint in the model directory of the CochleaNet cache.

    Args:
        model_type: The model type.

    Returns:
        The local path to the model checkpoint.
    """
    model = _get_model_spec(model_type)
    model_dir = os.path.join(get_cache_dir(), "models")
    checkpoint_path = os.path.join(model_dir, f"{model_type}-{model['bioimageio_id']}-{model['version']}.pt")
    if os.path.exists(checkpoint_path):
        return checkpoint_path

    os.makedirs(model_dir, exist_ok=True)
    # Download to a separate directory for each call, so that parallel jobs do not share a partial download.
    with tempfile.TemporaryDirectory(dir=model_dir) as download_dir:
        weights_path = pooch.retrieve(
            get_model_url(model_type), known_hash=f"sha256:{model['sha256']}",
            fname="weights.pt", path=download_dir,
        )
        checkpoint = {
            "init": {"model_class": model["model_class"], "model_kwargs": model["model_kwargs"]},
            "model_state": torch.load(weights_path, map_location="cpu", weights_only=True),
        }
        tmp_checkpoint_path = os.path.join(download_dir, "checkpoint.pt")
        torch.save(checkpoint, tmp_checkpoint_path)
        os.replace(tmp_checkpoint_path, checkpoint_path)
    return checkpoint_path


def get_model(model_type: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    """Get the model for a specific segmentation type.

    Args:
        model_type: The model for one of the following segmentation or detection tasks:
            'SGN', 'IHC', 'Synapses', 'SGN-lowres', 'IHC-lowres'.
        device: The device to use.

    Returns:
        The model.
    """
    if device is None:
        device = get_device(device)
    checkpoint = torch.load(get_model_path(model_type), map_location="cpu", weights_only=True)
    model = _create_model(model_type)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    return model


def get_default_segmentation_settings(model_type: str) -> Dict[str, Union[str, float]]:
    """Get the default settings for instance segmentation post-processing for a given model.

    Args:
        model_type: The model. One of 'SGN', 'SGN-lowres', 'IHC', 'IHC-lowres'.

    Returns:
        Dictionary with the default segmentation settings.
    """
    all_default_kwargs = {
        "SGN": {
            "center_distance_threshold": 0.4,
            "boundary_distance_threshold": 0.5,
            "fg_threshold": 0.5,
            "distance_smoothing": 0.0,
            "seg_class": "sgn",
        },
        "SGN-lowres": {
            "center_distance_threshold": None,
            "boundary_distance_threshold": 0.5,
            "fg_threshold": 0.5,
            "distance_smoothing": 0.0,
            "seg_class": "sgn_low",
        },
        "IHC": {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.6,
            "fg_threshold": 0.5,
            "distance_smoothing": 0.6,
            "seg_class": "ihc",
        },
        "IHC-lowres": {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.6,
            "fg_threshold": 0.5,
            "distance_smoothing": 0.6,
            "seg_class": "ihc",
        },
    }
    if model_type not in all_default_kwargs:
        raise ValueError(f"Invalid model: {model_type}. Choose one of {list(all_default_kwargs.keys())}.")
    default_kwargs = all_default_kwargs[model_type]
    return default_kwargs


def get_default_tiling() -> Dict[str, Dict[str, int]]:
    """Determine the tile shape and halo depending on the available VRAM.

    Returns:
        The default tiling settings for the available computational resources.
    """
    if torch.cuda.is_available():
        # The default halo size.
        halo = {"x": 64, "y": 64, "z": 16}

        # Determine the GPU RAM and derive a suitable tiling.
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9

        if vram >= 80:
            tile = {"x": 640, "y": 640, "z": 80}
        elif vram >= 40:
            tile = {"x": 512, "y": 512, "z": 64}
        elif vram >= 20:
            tile = {"x": 352, "y": 352, "z": 48}
        elif vram >= 10:
            tile = {"x": 256, "y": 256, "z": 32}
            halo = {"x": 64, "y": 64, "z": 8}  # Choose a smaller halo in z.
        else:
            raise NotImplementedError(f"Infererence with a GPU with {vram} GB VRAM is not supported.")

        tiling = {"tile": tile, "halo": halo}
        print(f"Determined tile size for CUDA: {tiling}")

    elif torch.backends.mps.is_available():  # Check for Apple Silicon (MPS)
        tile = {"x": 256, "y": 256, "z": 16}
        halo = {"x": 16, "y": 16, "z": 4}
        tiling = {"tile": tile, "halo": halo}
        print(f"Determined tile size for MPS: {tiling}")

    else:
        tiling = {
            "tile": {"x": 64, "y": 64, "z": 64},
            "halo": {"x": 32, "y": 32, "z": 16},
        }
        print(f"Determining default tiling for CPU: {tiling}")

    return tiling


def download_model():
    """Download pre-trained CochleaNet models to the local cache.

    Use this in environments where internet access is not available at inference time
    (e.g. restricted HPC compute nodes). Run on a login node before submitting jobs.
    """
    all_models = list(MODELS.keys())

    parser = argparse.ArgumentParser(
        description="Download pre-trained CochleaNet models to the local cache. "
        "Run this on a machine with internet access before using 'flamingo_tools.run_segmentation' "
        "or 'flamingo_tools.run_detection' on a node without internet access."
    )
    parser.add_argument(
        "-m", "--model_type", nargs="+", choices=all_models, default=None,
        help=f"The model(s) to download. Choose one or more of {all_models}. "
        "Downloads all models if not specified.",
    )
    args = parser.parse_args()

    model_types = args.model_type if args.model_type is not None else all_models
    for model_type in model_types:
        print(f"Downloading '{model_type}'...")
        model_path = get_model_path(model_type)
        print(f"  -> cached at: {model_path}")
