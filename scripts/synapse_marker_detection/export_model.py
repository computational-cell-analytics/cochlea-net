import argparse
import importlib
import os
import sys

import torch
from torch_em.util import load_model

sys.path.append("/home/pape/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/pape41/u12086/Work/my_projects/czii-protein-challenge")
sys.path.append("/user/schilling40/u15000/czii-protein-challenge")

# Allow loading checkpoints that were saved when detection_dataset was a local script.
import flamingo_tools.synapse_detection.detection_dataset as _dd

sys.modules["detection_dataset"] = _dd


def _model_from_checkpoint(ckpt):
    """Reconstruct a model from a torch_em-style trainer checkpoint dict."""
    model_class_path = ckpt["init"]["model_class"]
    model_kwargs = ckpt["init"]["model_kwargs"]
    module_path, class_name = model_class_path.rsplit(".", 1)
    model_class = getattr(importlib.import_module(module_path), class_name)
    model = model_class(**model_kwargs)
    model.load_state_dict(ckpt["model_state"])
    return model


def export_model(input_, output):
    if os.path.isdir(input_):
        model = load_model(input_, device="cpu")
    else:
        obj = torch.load(input_, map_location="cpu", weights_only=False)
        if isinstance(obj, dict) and "model_state" in obj:
            model = _model_from_checkpoint(obj)
        else:
            model = obj
    model.eval()
    torch.save(model, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()
    export_model(args.input, args.output)


if __name__ == "__main__":
    main()
