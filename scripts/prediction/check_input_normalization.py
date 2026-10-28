"""Check that U-Net checkpoints ignore a global mean/std normalization of their input.

Usage: python check_input_normalization.py CHECKPOINT [CHECKPOINT ...]

CHECKPOINT is a torch_em checkpoint folder, a model file, or a trainer checkpoint file.
The exit status is 1 if a global standardization changes the prediction of a checkpoint.
"""
import argparse
import sys

import torch

import flamingo_tools.synapse_detection.detection_dataset as detection_dataset
from flamingo_tools.segmentation.unet_prediction import _load_model

# Old synapse checkpoints reference the bare module name.
sys.modules.setdefault("detection_dataset", detection_dataset)

# Relative output difference above which the normalization matters.
TOLERANCE = 1e-3


def _predict(model, x):
    """Return the first leaf module of the forward pass and the prediction."""
    calls = []
    leaves = [m for m in model.modules() if not list(m.children())]
    hooks = [m.register_forward_pre_hook(lambda mod, inp: calls.append(mod)) for m in leaves]
    with torch.no_grad():
        y = model(x)
    for hook in hooks:
        hook.remove()
    return calls[0], y


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    failed = False
    for path in args.checkpoints:
        model = _load_model(path, device="cpu").eval()
        in_channels = next(m for m in model.modules() if isinstance(m, torch.nn.Conv3d)).in_channels
        # Raw-like intensities, in a shape that a depth-4 U-Net can downsample.
        x = torch.rand(1, in_channels, 32, 64, 64, generator=torch.Generator().manual_seed(0)) * 1000 + 200
        first, y_raw = _predict(model, x)
        _, y_std = _predict(model, (x - x.mean()) / x.std())
        diff = ((y_raw - y_std).abs().max() / y_raw.abs().max()).item()
        failed |= diff >= TOLERANCE
        print(path)
        print(f"  first layer: {first}")
        print(f"  relative max difference, raw vs. standardized input: {diff:.1e}")
        print("  -> " + ("mean/std has no effect" if diff < TOLERANCE else "mean/std CHANGES the prediction"))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
