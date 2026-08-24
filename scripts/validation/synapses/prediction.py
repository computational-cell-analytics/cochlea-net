"""Run the synapse detection on the validation crops using the production settings.

The point of this script is to quantify what the whole-cochlea runs actually do, so every
parameter that the production entry point (`flamingo_tools.segmentation.synapse_detection.
marker_detection`, driven by scripts/synapse_marker_detection/marker_detection.py) fixes is
mirrored here:

  * peak detection threshold 0.5, the value hard-coded in every whole-cochlea caller,
  * the production prediction block shape / halo,
  * `output_channels` taken from the model rather than assumed to be 1,
  * IHC matching at the production `max_distance`,
  * one mean/std for the whole volume, optionally supplied from a real cochlea.

Two of these need care on crops. The production block shape only tiles volumes much larger than
one block, so the crops are zero-padded (see `_padded_input`). And production derives a single
mean/std from the whole masked cochlea and applies it everywhere, whereas a per-crop mean/std
rescales each crop independently -- the crops differ by more than 4x in mean intensity, so the
two are not interchangeable when the detection threshold is absolute. Pass --mean/--std to use
global values; without them each crop is normalised on its own, which is not what production does.
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import zarr

from elf.io import open_file
from flamingo_tools.segmentation.unet_prediction import prediction_impl, run_unet_prediction
from flamingo_tools.segmentation.synapse_detection import (
    synapse_detection_from_prediction,
    _get_model_out_channels,
    _PREDICTION_BLOCK_SHAPE,
    _PREDICTION_HALO,
)

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
_IHC_MODEL = os.path.join(COCHLEA_DIR, "trained_models/IHC/v4_cochlea_distance_unet_IHC_supervised_2025-07-14")
_SYNAPSE_MODELS_DIR = os.path.join(COCHLEA_DIR, "trained_models/Synapses")
_TEST_IMAGE_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images")
_TEST_REF_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels")
VOXEL_SIZE = (0.38, 0.38, 0.38)  # µm per voxel in x, y, z order.

# Production values, kept in one place so a pipeline change is easy to mirror.
PRODUCTION_THRESHOLD = 0.5  # detect_synapse_peaks_template.sbatch and the marker_detection default.
PRODUCTION_MAX_DISTANCE = 3.0  # marker_detection() default; callers have also used 5, 8 and 20.

# Predictions produced with the production settings live under their own root. The predictions
# made with the earlier, per-version settings are still at predictions/val_synapses/<version>;
# they are what the historical entries in reproducibility/model_accuracy/synapses.json describe.
_PROD_PRED_ROOT = os.path.join(COCHLEA_DIR, "predictions/val_synapses/production")


def _entry(model_name: str, version: str) -> dict:
    return {
        "image_root": _TEST_IMAGE_ROOT,
        "ref_root": _TEST_REF_ROOT,
        "pred_root": os.path.join(_PROD_PRED_ROOT, version),
        "synapse_model": os.path.join(_SYNAPSE_MODELS_DIR, model_name),
        "ihc_model": _IHC_MODEL,
    }


PREDICTION_DICT = {
    # v3 and v3-1 are heatmap-only models trained on the 15-crop v3 data (v3-1 is a second seed).
    "v3": _entry("synapse_detection_model_v3.pt", "v3"),
    # v3-1 .. v3-4 are seed replicates: identical training data and split (random_state 42),
    # differing only in weight initialization and patch order, which are not seeded.
    "v3-1": _entry("synapse_detection_model_v3-1.pt", "v3-1"),
    "v3-2": _entry("synapse_detection_model_v3-2.pt", "v3-2"),
    "v3-3": _entry("synapse_detection_model_v3-3.pt", "v3-3"),
    "v3-4": _entry("synapse_detection_model_v3-4.pt", "v3-4"),
    "v4": _entry("synapse_detection_model_v4.pt", "v4"),
    # v5 and v6-1 share the same 29-crop training data. v5 adds the four stereographic flow
    # channels (and the MinPointSampler that comes with them); v6-1 is heatmap-only like v3.
    # v5 vs v6-1 therefore isolates the architecture/recipe, v3 vs v6-1 the training data.
    "v5": _entry("synapse_detection_model_v5.pt", "v5"),
    "v6-1": _entry("synapse_detection_model_v6-1.pt", "v6-1"),
    # v3 data with the v5 recipe (flow + MinPointSampler), the missing cell of the
    # data x architecture grid. Its validation metric (the combined heatmap+flow loss) bottoms
    # out at iteration 900 and never improves, so 'best' is an almost untrained checkpoint;
    # 'latest' is evaluated alongside it to test whether the selection metric is at fault.
    "v3-flow-1-best": _entry("synapse_detection_model_v3-flow-1-best.pt", "v3-flow-1-best"),
    "v3-flow-1-latest": _entry("synapse_detection_model_v3-flow-1-latest.pt", "v3-flow-1-latest"),
}


def _padded_input(input_path: str, input_key: str, output_folder: str) -> Tuple[str, Tuple[int, ...]]:
    """Zero-pad a validation crop up to a multiple of the production block shape.

    `prediction_impl` hands the U-Net blocks of `block_shape + 2 * halo`, whose shape has to be
    divisible by the U-Net's downsampling factors. A whole cochlea satisfies this because it is
    far larger than one block, but the validation crops are not, so the production block shape
    fails on them without padding. The padded region predicts as background and any detection
    landing in it is dropped afterwards.

    Args:
        input_path: Path to the crop in ZARR format.
        input_key: Key of the image data inside the crop.
        output_folder: Folder to write the padded copy into.

    Returns:
        The path to the padded copy and the shape of the original, unpadded data.
    """
    raw = np.asarray(zarr.open(store=input_path, mode="r")[input_key][:])
    shape = raw.shape
    target = tuple(int(np.ceil(s / b) * b) for s, b in zip(shape, _PREDICTION_BLOCK_SHAPE))
    if target == shape:
        return input_path, shape

    padded_path = os.path.join(output_folder, "padded_input.zarr")
    if not os.path.exists(padded_path):
        padded = np.zeros(target, dtype=raw.dtype)
        padded[: shape[0], : shape[1], : shape[2]] = raw
        f = zarr.open(store=padded_path, mode="w")
        f.create_array(input_key, data=padded, chunks=(64, 128, 128))
    print(f"Padded {tuple(shape)} to {target} for the production block shape.")
    return padded_path, shape


def _drop_padding_detections(detection_path: str, shape: Tuple[int, ...]) -> None:
    """Remove detections that fall inside the zero padding added by `_padded_input`."""
    limits = [s * vs for s, vs in zip(shape, VOXEL_SIZE)]
    for path in (detection_path, detection_path.replace(".tsv", "_no-flow.tsv")):
        if not os.path.isfile(path):
            continue
        det = pd.read_csv(path, sep="\t")
        keep = (det.z < limits[0]) & (det.y < limits[1]) & (det.x < limits[2])
        if not keep.all():
            print(f"Dropped {int((~keep).sum())} detections inside the padding of {os.path.basename(path)}.")
        det[keep].to_csv(path, index=False, sep="\t")


def pred_synapse_impl(
    input_path: str,
    output_folder: str,
    model_path: str,
    mean: Optional[float] = None,
    std: Optional[float] = None,
):
    """Predict synapses for a single file using the production settings.

    Args:
        input_path: Path to the image data.
        output_folder: Folder for the prediction and the detections.
        model_path: Path to the synapse detection model.
        mean: Mean used for normalization. Production derives one value for the whole masked
            cochlea, so pass it here to reproduce that. By default it is computed per crop.
        std: Standard deviation used for normalization, see `mean`.
    """
    input_key = "raw"
    os.makedirs(output_folder, exist_ok=True)

    prediction_path, shape = _padded_input(input_path, input_key, output_folder)

    prediction_impl(
        input_path=prediction_path, input_key=input_key, output_folder=output_folder,
        model_path=model_path,
        scale=None, block_shape=_PREDICTION_BLOCK_SHAPE, halo=_PREDICTION_HALO,
        apply_postprocessing=False,
        # A flow model has five output channels; assuming one silently discards the flow.
        output_channels=_get_model_out_channels(model_path),
        mean=mean, std=std,
    )

    output_path = os.path.join(output_folder, "predictions.zarr")
    detection_path = os.path.join(output_folder, "synapse_detection.tsv")

    # block_shape is left to the default so that it is derived from the prediction chunks,
    # exactly as in the whole-cochlea runs.
    synapse_detection_from_prediction(
        output_path, detection_path,
        prediction_key="prediction",
        threshold=PRODUCTION_THRESHOLD,
        save_no_flow=True,
    )
    _drop_padding_detections(detection_path, shape)


def predict_synapses(
    input_root: str,
    output_root: str,
    model_path: str,
    mean: Optional[float] = None,
    std: Optional[float] = None,
):
    """Predict synapses for multiple files in an input directory.

    `mean` and `std` are passed through to `pred_synapse_impl`; a single pair applied to every
    crop is what the whole-cochlea runs do.
    """
    files = sorted(glob(os.path.join(input_root, "*.zarr")))
    for ff in files:
        output_folder = os.path.join(output_root, Path(ff).stem)
        if os.path.exists(os.path.join(output_folder, "predictions.zarr", "prediction")):
            print("Synapse prediction in", ff, "already done")
            continue
        else:
            print("Predicting synapses in", ff)
        pred_synapse_impl(ff, output_folder, model_path, mean=mean, std=std)


def pred_ihc_impl(
    input_path: str,
    output_folder: str,
    model_path: str,
):
    """Predict IHC for a single file.
    """
    run_unet_prediction(
        input_path, input_key="raw_ihc", output_folder=output_folder, model_path=model_path, min_size=1000,
        seg_class="ihc", center_distance_threshold=0.5, boundary_distance_threshold=0.6,
        distance_smoothing=0.6, use_mask=False,
    )


def predict_ihcs(
    input_root: str,
    output_root: str,
    model_path: str,
):
    """Predict IHCs for multiple files in an input directory.
    """
    files = sorted(glob(os.path.join(input_root, "*.zarr")))
    for ff in files:
        output_folder = os.path.join(output_root, f"{Path(ff).stem}_ihc")
        if os.path.exists(os.path.join(output_folder, "predictions.zarr", "prediction")):
            print("IHC segmentation in", ff, "already done")
            continue
        else:
            print("Segmenting IHCs in", ff)
        pred_ihc_impl(ff, output_folder, model_path)


def _filter_synapse_impl(detections, ihc_file, output_path):
    from flamingo_tools.segmentation.synapse_detection import map_and_filter_detections

    with open_file(ihc_file, mode="r") as f:
        if "segmentation_filtered" in f:
            print("Using filtered segmentation!")
            segmentation = open_file(ihc_file)["segmentation_filtered"][:]
        else:
            segmentation = open_file(ihc_file)["segmentation"][:]

    filtered_detections = map_and_filter_detections(
        segmentation, detections, max_distance=PRODUCTION_MAX_DISTANCE, voxel_size=VOXEL_SIZE,
    )
    filtered_detections.to_csv(output_path, index=False, sep="\t")


def filter_synapses(
    input_root: str,
    output_root: str,
):
    """Filter detected synapse of prediction based on IHC proximity.
    """
    input_files = sorted(glob(os.path.join(input_root, "*.zarr")))
    for ff in input_files:
        ihc = os.path.join(output_root, f"{Path(ff).stem}_ihc", "segmentation.zarr")
        output_folder = os.path.join(output_root, Path(ff).stem)
        synapses = os.path.join(output_folder, "synapse_detection.tsv")
        synapses = pd.read_csv(synapses, sep="\t")
        # marker_detection() writes 'synapse_detection_filtered.tsv'; use the same name here so
        # that run_evaluation.py picks up validation and whole-cochlea output identically.
        output_path = os.path.join(output_folder, "synapse_detection_filtered.tsv")
        _filter_synapse_impl(synapses, ihc, output_path)


def filter_gt(
    input_root: str,
    gt_root: str,
    output_root: str,
):
    """Filter synapses of ground truth reference based on IHC proximity.
    """
    input_files = sorted(glob(os.path.join(input_root, "*.zarr")))
    gt_files = sorted(glob(os.path.join(gt_root, "*.csv")))
    for ff, gt in zip(input_files, gt_files):
        ihc = os.path.join(output_root, f"{Path(ff).stem}_ihc", "segmentation.zarr")
        output_folder, fname = os.path.split(gt)
        output_path = os.path.join(output_folder, fname.replace(".csv", "_filtered.tsv"))

        gt = pd.read_csv(gt)
        gt = gt.rename(columns={"axis-0": "z", "axis-1": "y", "axis-2": "x"})
        gt.insert(0, "spot_id", np.arange(1, len(gt) + 1))

        # Consensus annotations are stored in voxel coordinates, while
        # map_and_filter_detections expects physical coordinates in µm.
        gt["x"] *= VOXEL_SIZE[0]
        gt["y"] *= VOXEL_SIZE[1]
        gt["z"] *= VOXEL_SIZE[2]

        _filter_synapse_impl(gt, ihc, output_path)


def _check_prediction(input_file, ihc_file, detection_file):
    import napari

    synapses = pd.read_csv(detection_file, sep="\t")[["z", "y", "x"]].values

    vglut = open_file(input_file)["raw_ihc"][:]
    ctbp2 = open_file(input_file)["raw"][:]
    ihcs = open_file(ihc_file)["segmentation"][:]

    v = napari.Viewer()
    v.add_image(vglut)
    v.add_image(ctbp2)
    v.add_labels(ihcs)
    v.add_points(synapses)
    napari.run()


def check_predictions_multi(
    input_root: str,
    output_root: str,
):
    """Check multiple detections of synapses and the respective IHC segmentation.

    Args:
        input_root: Directory containing data of CTBP2 and IHC channel.
        output_root: Output folder where the predicted synapses, IHC segmentation and filtered synapses are saved.
    """
    input_files = sorted(glob(os.path.join(input_root, "*.zarr")))
    for ff in input_files:
        ihc = os.path.join(output_root, f"{Path(ff).stem}_ihc", "segmentation.zarr")
        synapses = os.path.join(output_root, Path(ff).stem, "filtered_synapse_detection.tsv")
        _check_prediction(ff, ihc, synapses)


def process_everything(
    input_root: str,
    gt_root: str,
    output_root: str,
    synapse_model_path: str,
    ihc_model_path: str,
    mean: Optional[float] = None,
    std: Optional[float] = None,
):
    """Process images for validation of synapse detection.

    Args:
        input_root: Directory containing data of CTBP2 and IHC channel.
        gt_root: Folder that contains the ground truth data of the synapses.
        output_root: Output path where the predicted synapses, IHC segmentation and filtered synapses are saved.
        synapse_model_path: File path to synapse detection model.
        ihc_model_path: File path to IHC segmentation model.
        mean: Mean for normalization, applied to every crop. See `pred_synapse_impl`.
        std: Standard deviation for normalization, applied to every crop.
    """
    predict_synapses(input_root, output_root, synapse_model_path, mean=mean, std=std)
    predict_ihcs(input_root, output_root, ihc_model_path)
    filter_synapses(input_root, output_root)
    filter_gt(input_root, gt_root, output_root)


def main():
    parser = argparse.ArgumentParser(
        description="Process test data for synapse detection."
    )
    parser.add_argument(
        "-v", "--version", type=str, default=None,
        help="Use pre-defined directories for a specific network version, e.g. v3, v4, ..."
    )
    parser.add_argument(
        "-i", "--input_root", type=str, default=None,
        help="Folder that contains the data of the CTBP2 [raw] and the IHC stain [raw_ihc] channel."
    )
    parser.add_argument(
        "-g", "--gt_root", type=str, default=None,
        help="Folder that contains the ground truth data of the synapses."
    )
    parser.add_argument(
        "-o", "--output_root", type=str, default=None,
        help="Output path where the predicted synapses, IHC segmentation and filtered synapses are saved."
    )
    parser.add_argument(
        "--model_synapse", type=str, default=None,
        help="File path to synapse detection model."
    )
    parser.add_argument(
        "--model_ihc", type=str, default=None,
        help="File path to model for IHC segmentation."
    )
    parser.add_argument(
        "--mean", type=float, default=None,
        help="Mean for normalization, applied to every crop. Production derives a single value "
             "for the whole masked cochlea, so pass it here to reproduce production. "
             "By default each crop is normalized on its own, which production does not do."
    )
    parser.add_argument(
        "--std", type=float, default=None,
        help="Standard deviation for normalization, applied to every crop. See --mean."
    )
    parser.add_argument(
        "--pred_root", type=str, default=None,
        help="Override the output root of --version, keeping its image and reference roots. "
             "Useful for writing a normalization variant to a separate directory."
    )
    parser.add_argument(
        "--mean_std_json", type=str, default=None,
        help="JSON file with 'mean' and 'std' entries, as written by a whole-cochlea "
             "normalization run. Takes precedence over --mean/--std."
    )

    args = parser.parse_args()

    mean, std = args.mean, args.std
    if args.mean_std_json is not None:
        with open(args.mean_std_json) as f:
            mean_std = json.load(f)
        mean, std = float(mean_std["mean"]), float(mean_std["std"])
    if (mean is None) != (std is None):
        raise ValueError("Pass both --mean and --std, or neither.")
    if mean is None:
        print("No mean/std given: normalizing each crop on its own. This is NOT the production "
              "behavior, which applies one mean/std derived from the whole masked cochlea.")
    else:
        print(f"Using the production normalization for every crop: mean={mean}, std={std}")

    if args.version is not None:
        valid_versions = list(PREDICTION_DICT.keys())
        if args.version not in valid_versions:
            raise ValueError(f"Version {args.version} is not supported. Supported versions: {valid_versions}")
        entry = PREDICTION_DICT[args.version]
        input_root = entry["image_root"]
        gt_root = entry["ref_root"]
        output_root = entry["pred_root"] if args.pred_root is None else os.path.join(
            args.pred_root, args.version
        )
        synapse_model = entry["synapse_model"]
        ihc_model = entry["ihc_model"]
    else:
        input_root = args.input_root
        gt_root = args.gt_root
        output_root = args.output_root
        synapse_model = args.model_synapse
        ihc_model = args.model_ihc

    process_everything(
        input_root=input_root,
        gt_root=gt_root,
        output_root=output_root,
        synapse_model_path=synapse_model,
        ihc_model_path=ihc_model,
        mean=mean,
        std=std,
    )


if __name__ == "__main__":
    main()
