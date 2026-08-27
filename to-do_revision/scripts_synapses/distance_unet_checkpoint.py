"""Load a torch-em distance U-Net checkpoint without rebuilding its data loaders.

The torch-em version in the ``new-stack`` environment eagerly reopens all training
TIFFs when ``load_model`` restores the saved trainer. Inference only needs the model
initialization and ``model_state``, so the dataset objects are replaced by inert stubs
during safe, weights-only deserialization.
"""

import importlib
import os

import numpy as np
import torch


_DATASET_GLOBALS = {
    "torch_em.data.concat_dataset.ConcatDataset",
    "torch_em.data.segmentation_dataset.SegmentationDataset",
}


class _DatasetStub:
    """Inert replacement for datasets embedded in torch-em trainer checkpoints."""

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __setstate__(self, state):
        # The trainer state is irrelevant for inference. In particular, do not call
        # SegmentationDataset.__setstate__, which eagerly reads all training images.
        self.state = None


def _safe_checkpoint_globals(checkpoint_path):
    safe_globals = []
    for global_name in torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint_path):
        if global_name in _DATASET_GLOBALS:
            safe_globals.append((_DatasetStub, global_name))
            continue

        module_name, attribute_name = global_name.rsplit(".", 1)
        module = importlib.import_module(module_name)
        safe_globals.append(getattr(module, attribute_name))

    # NumPy dtype subclasses are created dynamically and are not reported by
    # get_unsafe_globals_in_checkpoint, but are needed for the stored RNG state.
    safe_globals.extend(value for value in vars(np.dtypes).values() if isinstance(value, type))
    return safe_globals


def load_distance_unet(checkpoint, device="cpu", name="best"):
    """Load only the inference model from a trusted torch-em checkpoint."""
    checkpoint_path = os.path.join(checkpoint, f"{name}.pt") if os.path.isdir(checkpoint) else checkpoint
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    safe_globals = _safe_checkpoint_globals(checkpoint_path)
    with torch.serialization.safe_globals(safe_globals):
        state = torch.load(
            checkpoint_path,
            map_location="cpu",
            mmap=True,
            weights_only=True,
        )

    model_class_path = state["init"]["model_class"]
    model_kwargs = state["init"]["model_kwargs"]
    module_name, class_name = model_class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_name), class_name)
    model = model_class(**model_kwargs)
    model.load_state_dict(state["model_state"])
    if device is not None:
        model.to(device)
    return model
