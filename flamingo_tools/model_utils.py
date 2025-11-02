import os
from typing import Dict, Optional, Union

import pooch
import torch
from .file_utils import get_cache_dir


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


def get_model_registry() -> None:
    """Get the model registry for downloading pre-trained CochleaNet models.
    """
    registry = {
        "SGN": "3058690b49015d6210a8e8414eb341c34189fee660b8fac438f1fdc41bdfff98",
        "IHC": "89afbcca08ed302aa6dfbaba5bf2530fc13339c05a604b6f2551d97cf5f12774",
        "Synapses": "2a42712b056f082b4794f15cf41b15678aab0bec1acc922ff9f0dc76abe6747e",
        # TODO
        # "SGN-lowres": "",
        # "IHC-lowres": "",
    }
    urls = {
        "SGN": "https://owncloud.gwdg.de/index.php/s/NZ2vv7hxX1imITG/download",
        "IHC": "https://owncloud.gwdg.de/index.php/s/GBBJkPQFraz1ZzU/download",
        "Synapses": "https://owncloud.gwdg.de/index.php/s/A9W5NmOeBxiyZgY/download",
        # TODO
        # "SGN-lowres": "",
        # "IHC-lowres": "",
    }
    cache_dir = get_cache_dir()
    models = pooch.create(
        path=os.path.join(cache_dir, "models"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return models


def get_model_path(model_type: str) -> str:
    """Get the local path to a pretrained model.

    Args:
        The model type.

    Returns:
        The local path to the model.
    """
    model_registry = get_model_registry()
    model_path = model_registry.fetch(model_type)
    return model_path


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
    model_path = get_model_path(model_type)
    model = torch.load(model_path, weights_only=False)
    model.to(device)
    return model


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

    # I am not sure what is reasonable on a cpu. For now choosing very small tiling.
    # (This will not work well on a CPU in any case.)
    else:
        tiling = {
            "tile": {"x": 96, "y": 96, "z": 16},
            "halo": {"x": 16, "y": 16, "z": 4},
        }
        print(f"Determining default tiling for CPU: {tiling}")

    return tiling
