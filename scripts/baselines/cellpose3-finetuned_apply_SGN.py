import json
import os
import time
from typing import Optional, List, Union, Tuple

import numpy as np
import torch
from pathlib import Path
from tqdm import trange
from natsort import natsorted

try:
    from cellpose import core, io, models
    _cellpose_is_installed = True
except ImportError:
    _cellpose_is_installed = False


def segment_using_custom_cellpose(
    image: np.ndarray,
    checkpoint_path: Union[os.PathLike, str],
    channels: Optional[List[int]] = [0, 0],
    diameter: Optional[Union[float, int]] = 0,
    return_flows: bool = False,
    **kwargs
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Supports inference on images using custom trained (or finetuned) CellPose models.

    Args:
        image: The input image.
        checkpoint_path: The path where the trained model checkpoints are stored.
        channels: The channel parameters to be used for inference.
        diameter: The diameter of the objects.
        return_flows: Whether to return the predicted masks, flows and styles.

    Returns:
        masks: The instance segmentation.
    """
    assert _cellpose_is_installed, "Please install 'cellpose'."

    use_gpu = torch.cuda.is_available()

    model = models.CellposeModel(gpu=use_gpu, pretrained_model=checkpoint_path)

    if diameter and diameter == 0:
        kwargs["diameter"] = model.diam_labels

    if channels:
        kwargs["channels"] = channels

    masks, flows, styles = model.eval(image, **kwargs)

    if return_flows:
        return masks, flows, styles
    else:
        return masks


def main():
    io.logger_setup()  # run this to get printing of progress

    # Check if colab notebook instance has GPU access
    if core.use_gpu() is False:
        raise ImportError("No GPU access, change your runtime")

    evaluation_mode = "revision"
    model_path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/SGN/cellpose3-cyto3_sgn_2026-05-05" # noqa

    # *** change to your google drive folder path ***
    if evaluation_mode == "validation":
        cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
        input_dir = os.path.join(cochlea_dir, "training_data/SGN/2026-04_SGN-v2-data_cellpose3/val")
        out_dir = os.path.join(cochlea_dir, "predictions/val_sgn/cellpose3_finetuned_val")
    elif evaluation_mode == "revision":
        cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
        input_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation")
        out_dir = os.path.join(cochlea_dir, "predictions/val_sgn/cellpose3_finetuned")
    else:
        raise ValueError("Choose either 'validation' or 'revision' as evaluation mode.")

    input_dir = Path(input_dir)
    os.makedirs(out_dir, exist_ok=True)
    if not input_dir.exists():
        raise FileNotFoundError("directory does not exist")

    # *** change to your image extension ***
    image_ext = ".tif"
    # list all files
    files = natsorted([f for f in input_dir.glob("*" + image_ext) if
                       "_masks" not in f.name and "_flows" not in f.name and "_annotations" not in f.name])

    if len(files) == 0:
        raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
    else:
        print(f"{len(files)} images in folder:")

    for f in files:
        print(f.name)

    diameter = 22

    for i in trange(len(files)):
        f = files[i]
        start = time.perf_counter()
        img = io.imread(f)
        masks = segment_using_custom_cellpose(img, model_path, diameter=diameter)

        basename = "".join(f.name.split(".")[:-1])
        out_path = os.path.join(out_dir, f"{basename}_seg.tif")
        io.imsave(out_path, masks)

        if evaluation_mode == "revision":
            timer_output = os.path.join(out_dir, f"{basename}_timer.json")
            duration = time.perf_counter() - start
            time_dict = {"total_duration[s]": duration}
            with open(timer_output, "w") as f:
                json.dump(time_dict, f, indent='\t', separators=(',', ': '))


if __name__ == "__main__":
    main()
