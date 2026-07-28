import argparse
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Literal, Optional, Union, Tuple

import numpy as np
from tqdm import tqdm

try:
    from tifffile import imread as read_tif
except ImportError:
    from imageio.v3 import imread as read_tif

try:
    from cellpose import train, core, io, models
    _cellpose_is_installed = True
except ImportError:
    _cellpose_is_installed = False

ROOT_TRAINING_DATA = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data"
ROOT_SGN_DATA = f"{ROOT_TRAINING_DATA}/SGN/2026-04_SGN-v2-data_cellpose3"
ANNOTATION_SUFFIX = "_annotations"


def list_image_label_pairs(
    data_dir: str,
    annotation_suffix: str = ANNOTATION_SUFFIX,
) -> Tuple[List[str], List[str], List[str]]:
    """List the paired 2D image and annotation TIF files of a training directory.

    Files with 'empty' or 'flows' in the name are ignored. The annotation of `<stem>.tif` is
    `<stem><annotation_suffix>.tif`; pairing by name avoids a silent image / annotation shift
    that sorting two separate file lists can introduce.

    Args:
        data_dir: Directory with the 2D image and annotation TIF files.
        annotation_suffix: Suffix that marks an annotation file.

    Returns:
        The image paths, the matching annotation paths, and the images without an annotation.
    """
    names = sorted(
        entry.name for entry in os.scandir(data_dir)
        if entry.name.endswith(".tif") and "empty" not in entry.name and "flows" not in entry.name
    )
    annotation_names = {name for name in names if annotation_suffix in name}

    image_paths, label_paths, unpaired = [], [], []
    for name in names:
        if name in annotation_names:
            continue
        annotation_name = f"{name[:-len('.tif')]}{annotation_suffix}.tif"
        if annotation_name in annotation_names:
            image_paths.append(os.path.join(data_dir, name))
            label_paths.append(os.path.join(data_dir, annotation_name))
        else:
            unpaired.append(os.path.join(data_dir, name))

    return image_paths, label_paths, unpaired


def count_foreground_pixels(label_paths: List[str], n_workers: int = 8) -> List[int]:
    """Count the foreground pixels of every annotation file."""
    def count(path):
        return int(np.count_nonzero(read_tif(path)))

    with ThreadPoolExecutor(n_workers) as pool:
        counts = list(tqdm(pool.map(count, label_paths), total=len(label_paths), desc="Checking annotations"))
    return counts


def filter_degenerate_pairs(
    image_paths: List[str],
    label_paths: List[str],
    min_label_pixels: int = 2,
    n_workers: int = 8,
) -> Tuple[List[str], List[str], List[str]]:
    """Remove training pairs whose annotation has too few foreground pixels.

    Cellpose fails on an annotation with exactly one foreground pixel: in
    `cellpose.dynamics._extend_centers_gpu` the gradient array is squeezed along axis 0, collapses
    to a scalar, and `np.stack(..., axis=-2)` then raises
    "AxisError: axis -2 is out of bounds for array of dimension 1". Such slices occur where a 3D
    instance is clipped to a single pixel in its first or last z-slice.

    Args:
        image_paths: The image paths.
        label_paths: The matching annotation paths.
        min_label_pixels: Minimum number of foreground pixels an annotation must have.
        n_workers: Number of threads for reading the annotations.

    Returns:
        The kept image paths, the kept annotation paths, and the dropped annotation paths.
    """
    counts = count_foreground_pixels(label_paths, n_workers=n_workers)
    keep = [count >= min_label_pixels for count in counts]

    kept_images = [path for path, valid in zip(image_paths, keep) if valid]
    kept_labels = [path for path, valid in zip(label_paths, keep) if valid]
    dropped = [path for path, valid in zip(label_paths, keep) if not valid]
    return kept_images, kept_labels, dropped


def run_cellpose3_finetuning(
    train_image_files: List[Union[os.PathLike, str]],
    train_label_files: List[Union[os.PathLike, str]],
    val_image_files: List[Union[os.PathLike, str]],
    val_label_files: List[Union[os.PathLike, str]],
    checkpoint_name: Optional[str] = None,
    save_root: Optional[Union[os.PathLike, str]] = None,
    initial_model: str = "cyto3",
    n_epochs: int = 100,
    channels_to_use_for_training: Literal["Grayscale", "Blue", "Green", "Red"] = "Grayscale",
    second_training_channel: Literal["Blue", "Green", "Red", "None"] = "None",
    learning_rate: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 1e-5,
    batch_size: int = 8,
    optimizer_choice: Literal["AdamW", "SGD"] = "AdamW",
    **kwargs
) -> Tuple[Union[os.PathLike, str], float]:
    """Functionality for finetuning (or training) CellPose3 models.

    This script is inspired from: https://github.com/MouseLand/cellpose/blob/main/notebooks/run_cellpose_2.ipynb.

    NOTE: The current support is to finetune (or train) CellPose models according to "CellPose 2.0":
    - Pachitariu et al. - https://doi.org/10.1038/s41592-022-01663-4
    This version is adapted from https://github.com/anwai98/tukra/blob/master/tukra/training/cellpose.py

    Please cite it if you use this functionality in your research.

    Args:
        train_image_files (List[os.PathLike, str]): List of paths of the training image files.
        train_label_files (List[os.PathLike, str]): List of paths of the training image files.
        val_image_files (List[os.PathLike, str]): List of paths of the training image files.
        val_label_files (List[os.PathLike, str]): List of paths of the training image files.
        save_root (str, os.PathLike): Where to save the trained model.
        checkpoint_name (str, None): The name of model checkpoint with which it will be saved.
        initial_model (str): The pretrained model to initialize CellPose with for finetuning (or, train from scratch).
        n_epochs (int): The total number of epochs for training.
        channels_to_use_for_training (str): The first channel to be used for training.
        second_training_channel (str): The second channel to be used for training.
        learning_rate (float): The learning rate for training.
        momentum (float): The momentum for the optimizer.
        weight_decay (float): The weight decay for the optimizer.
        batch_size: The number of patches to batch together per iteration.
        optimizer_choice (str): The choice of optimizer. Either "AdamW" (default) or "SGD".
        kwargs: The additional parameters for the `cellpose.train.tran_seg` functionality.

    Returns:
        model_path: The filepath where the trained model is stored.
        diam_labels: The diameter of objects in labels in the training set.
    """
    assert _cellpose_is_installed, "Please install 'cellpose'."

    # Here we match the channel to number
    channels_to_use_for_training = channels_to_use_for_training.title()
    if channels_to_use_for_training == "Grayscale":
        chan = 0
    elif channels_to_use_for_training == "Blue":
        chan = 3
    elif channels_to_use_for_training == "Green":
        chan = 2
    elif channels_to_use_for_training == "Red":
        chan = 1
    else:
        raise ValueError(f"'{chan}' is not a valid channel to use for training.")

    second_training_channel = second_training_channel.title()
    if second_training_channel == "Blue":
        chan2 = 3
    elif second_training_channel == "Green":
        chan2 = 2
    elif second_training_channel == "Red":
        chan2 = 1
    elif second_training_channel == "None":
        chan2 = 0
    else:
        raise ValueError(f"'{chan2}' is not a valid second training channel.")

    # In case we would like to train CellPose from scratch.
    if initial_model == "scratch":
        initial_model = "None"

    # Check whether GPU is activated or not.
    use_GPU = core.use_gpu()
    yn = ['NO', 'YES']
    print(f'>>> GPU activated? {yn[use_GPU]}')

    # start logger (to see training across epochs)
    io.logger_setup()

    # Let's define the CellPose model (without size model)
    model = models.CellposeModel(gpu=use_GPU, model_type=initial_model)

    os.makedirs(save_root, exist_ok=True)

    model_path = train.train_seg(
        net=model.net,
        train_files=train_image_files,
        train_labels_files=train_label_files,
        test_files=val_image_files,
        test_labels_files=val_label_files,
        channels=[chan, chan2],
        save_path=save_root,
        n_epochs=n_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        SGD=(optimizer_choice == "SGD"),
        batch_size=batch_size,
        model_name=checkpoint_name,
        momentum=momentum,
        **kwargs
    )

    # Diameter of labels in the training images (useful for evaluation)
    diam_labels = model.net.diam_labels.item()

    return model_path, diam_labels


def _load_split(data_dir: str, split: str, min_label_pixels: int) -> Tuple[List[str], List[str]]:
    split_dir = os.path.join(data_dir, split)
    image_paths, label_paths, unpaired = list_image_label_pairs(split_dir)
    if unpaired:
        raise ValueError(f"{len(unpaired)} images in {split_dir} have no annotation, e.g. {unpaired[0]}.")

    image_paths, label_paths, dropped = filter_degenerate_pairs(
        image_paths, label_paths, min_label_pixels=min_label_pixels,
    )
    print(f"{split}: using {len(image_paths)} pairs, dropped {len(dropped)} with < {min_label_pixels} label pixels.")
    return image_paths, label_paths


def finetune_cellpose3(
    output_dir: str,
    data_dir: str = ROOT_SGN_DATA,
    checkpoint_name: str = "finetune_cyto3_sgn",
    min_label_pixels: int = 2,
):
    """Wrapper for finetuning Cellpose3 for 2D SGN or IHC segmentation.
    Explicit finetuning function is based on https://github.com/anwai98/tukra/blob/master/tukra/training/cellpose.py.

    Args:
        output_dir: Output directory for weights.
        data_dir: Data directory with train and val data.
        checkpoint_name: Checkpoint name for model.
        min_label_pixels: Minimum number of foreground pixels per annotation. See `filter_degenerate_pairs`.
    """
    train_image_paths, train_label_paths = _load_split(data_dir, "train", min_label_pixels)
    val_image_paths, val_label_paths = _load_split(data_dir, "val", min_label_pixels)

    checkpoint_path, _ = run_cellpose3_finetuning(
        train_image_files=train_image_paths,
        train_label_files=train_label_paths,
        val_image_files=val_image_paths,
        val_label_files=val_label_paths,
        save_root=output_dir,
        checkpoint_name=checkpoint_name,
        initial_model="cyto3",
        n_epochs=100,
    )

    print(f"The model has been stored at '{checkpoint_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Pre-process external training data for OCT-SAM. Output data is stored in H5 format."
    )

    parser.add_argument("-i", "--input_dir", type=str, default=ROOT_SGN_DATA)
    parser.add_argument("-o", "--output_dir", type=str, required=True)
    parser.add_argument("-n", "--name", type=str, default="finetune_cyto3_sgn",
                        help="Optional name for the model to be trained. Default: finetune_cyto3_sgn.")
    parser.add_argument("-c", "--cell_type", type=str, default="sgn",)

    args = parser.parse_args()

    ROOT_TRAINING_DATA = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data"
    if args.cell_type == "sgn":
        input_dir = f"{ROOT_TRAINING_DATA}/SGN/2026-04_SGN-v2-data_cellpose3"
    elif args.cell_type == "ihc":
        input_dir = f"{ROOT_TRAINING_DATA}/IHC/2026-07_IHC-v11-data_cellpose3"
    else:
        raise ValueError("Choose either 'sgn' or 'ihc' as --cell_type.")

    finetune_cellpose3(
        output_dir=args.output_dir,
        data_dir=input_dir,
        checkpoint_name=args.name,
    )


if __name__ == "__main__":
    main()
