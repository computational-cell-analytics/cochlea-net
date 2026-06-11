import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

from elf.io import open_file
from flamingo_tools.segmentation.unet_prediction import prediction_impl, run_unet_prediction
from flamingo_tools.segmentation.synapse_detection import synapse_detection_from_prediction

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
_IHC_MODEL = os.path.join(COCHLEA_DIR, "trained_models/IHC/v4_cochlea_distance_unet_IHC_supervised_2025-07-14")
_SYNAPSE_MODELS_DIR = os.path.join(COCHLEA_DIR, "trained_models/Synapses")
_TEST_IMAGE_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/images")
_TEST_REF_ROOT = os.path.join(COCHLEA_DIR, "training_data/synapses/test_data/v5/labels")

PREDICTION_DICT = {
    "v3": {
        "image_root": _TEST_IMAGE_ROOT,
        "ref_root": _TEST_REF_ROOT,
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v3"),
        "synapse_model": os.path.join(_SYNAPSE_MODELS_DIR, "synapse_detection_model_v3.pt"),
        "ihc_model": _IHC_MODEL,
    },
    "v4": {
        "image_root": _TEST_IMAGE_ROOT,
        "ref_root": _TEST_REF_ROOT,
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v4"),
        "synapse_model": os.path.join(_SYNAPSE_MODELS_DIR, "synapse_detection_model_v4.pt"),
        "ihc_model": _IHC_MODEL,
    },
    "v5": {
        "image_root": _TEST_IMAGE_ROOT,
        "ref_root": _TEST_REF_ROOT,
        "pred_root": os.path.join(COCHLEA_DIR, "predictions/val_synapses/v5"),
        "synapse_model": os.path.join(_SYNAPSE_MODELS_DIR, "synapse_detection_model_v5.pt"),
        "ihc_model": _IHC_MODEL,
    },
}


def pred_synapse_impl(
    input_path: str,
    output_folder: str,
    model_path: str,
):
    """Predict synapses for a single file.
    """
    input_key = "raw"

    block_shape = (32, 128, 128)
    halo = (16, 64, 64)

    os.makedirs(output_folder, exist_ok=True)

    prediction_impl(
        input_path, input_key, output_folder, model_path,
        scale=None, block_shape=block_shape, halo=halo,
        apply_postprocessing=False, output_channels=1,
    )

    output_path = os.path.join(output_folder, "predictions.zarr")
    detection_path = os.path.join(output_folder, "synapse_detection.tsv")

    prediction_key = "prediction"
    synapse_detection_from_prediction(
        output_path, detection_path,
        prediction_key=prediction_key,
        block_shape=block_shape,
        model_path=model_path,
    )


def predict_synapses(
    input_root: str,
    output_root: str,
    model_path: str,
):
    """Predict synapses for multiple files in an input directory.
    """
    files = sorted(glob(os.path.join(input_root, "*.zarr")))
    for ff in files:
        output_folder = os.path.join(output_root, Path(ff).stem)
        if os.path.exists(os.path.join(output_folder, "predictions.zarr", "prediction")):
            print("Synapse prediction in", ff, "already done")
            continue
        else:
            print("Predicting synapses in", ff)
        pred_synapse_impl(ff, output_folder, model_path)


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

    max_distance = 5  # 5 micrometer
    filtered_detections = map_and_filter_detections(segmentation, detections, max_distance=max_distance)
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
        output_path = os.path.join(output_folder, "filtered_synapse_detection.tsv")
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
):
    """Process images for validation of synapse detection.

    Args:
        input_root: Directory containing data of CTBP2 and IHC channel.
        gt_root: Folder that contains the ground truth data of the synapses.
        output_root: Output path where the predicted synapses, IHC segmentation and filtered synapses are saved.
        synapse_model_path: File path to synapse detection model.
        ihc_model_path: File path to IHC segmentation model.
    """
    predict_synapses(input_root, output_root, synapse_model_path)
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

    args = parser.parse_args()

    if args.version is not None:
        valid_versions = list(PREDICTION_DICT.keys())
        if args.version not in valid_versions:
            raise ValueError(f"Version {args.version} is not supported. Supported versions: {valid_versions}")
        entry = PREDICTION_DICT[args.version]
        input_root = entry["image_root"]
        gt_root = entry["ref_root"]
        output_root = entry["pred_root"]
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
    )


if __name__ == "__main__":
    main()
