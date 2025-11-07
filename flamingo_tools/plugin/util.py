import re
from typing import Optional, Union

import torch
from napari.utils.notifications import show_info
from ..model_utils import get_device


def _load_custom_model(model_path: str, device: Optional[Union[str, torch.device]] = None) -> torch.nn.Module:
    model_path = _clean_filepath(model_path)
    if device is None:
        device = get_device(device)
    try:
        model = torch.load(model_path, map_location=torch.device(device), weights_only=False)
    except Exception as e:
        print(e)
        print("model path", model_path)
        return None
    return model


def _available_devices():
    available_devices = []
    for i in ["cuda", "mps", "cpu"]:
        try:
            device = get_device(i)
        except RuntimeError:
            pass
        else:
            available_devices.append(device)
    return available_devices


def _get_current_tiling(tiling: dict, default_tiling: dict, model_type: str):
    # get tiling values from qt objects
    for k, v in tiling.items():
        for k2, v2 in v.items():
            if isinstance(v2, int):
                continue
            elif hasattr(v2, "value"):  # If it's a QSpinBox, extract the value
                tiling[k][k2] = v2.value()
            else:
                raise TypeError(f"Unexpected type for tiling value: {type(v2)} at {k}/{k2}")
    show_info(f"Using tiling: {tiling}")
    return tiling


def _clean_filepath(filepath):
    # Remove 'file://' prefix if present
    if filepath.startswith("file://"):
        filepath = filepath[7:]

    # Remove escape sequences and newlines
    filepath = re.sub(r'\\.', '', filepath)
    filepath = filepath.replace('\n', '').replace('\r', '')

    return filepath
