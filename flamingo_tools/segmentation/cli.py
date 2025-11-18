"""private
"""
import argparse

from .unet_prediction import run_unet_prediction
from .synapse_detection import marker_detection
from ..model_utils import get_model_path


def _get_model_path(model_type, checkpoint_path=None):
    if checkpoint_path is None:
        model_path = get_model_path(model_type)
    else:
        model_path = checkpoint_path
    return model_path


def _parse_kwargs(extra_kwargs, **default_kwargs):
    def _convert_argval(value):
        # The values for the parsed arguments need to be in the expected input structure as provided.
        # i.e. integers and floats should be in their original types.
        try:
            return int(value)
        except ValueError:
            return float(value)

    extra_kwargs = {
        extra_kwargs[i].lstrip("--"): _convert_argval(extra_kwargs[i + 1]) for i in range(0, len(extra_kwargs), 2)
    }

    known_kwargs, unknown_kwargs = {}, {}
    for name, val in extra_kwargs.items():
        if name in default_kwargs:
            known_kwargs[name] = val
        else:
            unknown_kwargs[name] = val

    missing_default_kwargs = {name: val for name, val in default_kwargs.items() if name not in known_kwargs}
    if unknown_kwargs:
        raise ValueError(
            f"The following options are not supported: {list(unknown_kwargs.keys())}."
            f"Did you mean any of the following additional options: {list(missing_default_kwargs.keys())}?"
        )

    known_kwargs.update(missing_default_kwargs)
    return known_kwargs


def _parse_segmentation_kwargs(extra_kwargs, model_type):
    if model_type.startswith("SGN"):
        default_kwargs = {
            "center_distance_threshold": 0.4,
            "boundary_distance_threshold": 0.5,
            "fg_threshold": 0.5,
            "distance_smoothing": 0.0,
            "seg_class": "sgn_low" if model_type == "SGN-lowres" else "sgn",
        }
    else:
        assert model_type.startswith("IHC")
        default_kwargs = {
            "center_distance_threshold": 0.5,
            "boundary_distance_threshold": 0.6,
            "fg_threshold": 0.5,
            "distance_smoothing": 0.6,
            "seg_class": "ihc",
        }

    kwargs = _parse_kwargs(extra_kwargs, **default_kwargs)
    return kwargs


def run_segmentation():
    """private
    """
    segmentation_models = ["SGN", "IHC", "SGN-lowres", "IHC-lowres"]

    parser = argparse.ArgumentParser(
        description="Segment individual cells in volumetric light microscopy data. "
        "This function supports the segmentation of SGNs and IHCs, at high- and low-resolution. "
        "Which model to use is specified with the argument '--model_type' ('-m'). "
        "It also supports custom models via the (optional) argument '--checkpoint_path' ('-c'). "
    )
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The path to the input data. Supports .tif, .zarr, .h5 or .n5 files. "
        "Zarr, HDF5 or N5 files also require the 'input_key' argument."
    )
    parser.add_argument(
        "-k", "--input_key",
        help="The key to the input data. This refers to an internal data path for Zarr, HDF5 or N5 data."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True,
        help="The output folder where the segmentation result and intermediates are stored. "
        "The segmentation result is stored in the file 'segmentation.zarr' under the key 'segmentation'."
    )
    parser.add_argument(
        "-m", "--model_type", required=True,
        help=f"The model to use for segmentation. One of {segmentation_models}.",
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        help="Path to the checkpoint of a customly fine-tuned model (optional).",
    )
    parser.add_argument(
        "--min_size", type=int, default=250,
        help="The minimum size of cells in the resulting segmentation.",
    )
    parser.add_argument(
        "--disable_masking", action="store_true",
        help="Whether to disable intensity-based masking. By default, "
        "segmentation is only applied for parts of the data with foreground signal."
    )
    args, extra_kwargs = parser.parse_known_args()

    if args.model_type not in segmentation_models:
        raise ValueError(f"Unknown model: {args.model_type}. Choose one of {segmentation_models}.")
    segmentation_kwargs = _parse_segmentation_kwargs(extra_kwargs, args.model_type)

    model_path = _get_model_path(args.model_type, args.checkpoint_path)
    run_unet_prediction(
        input_path=args.input_path, input_key=args.input_key,
        output_folder=args.output_folder, model_path=model_path,
        min_size=args.min_size, use_mask=args.disable_masking,
        **segmentation_kwargs,
    )


def run_detection():
    """private
    """
    detection_models = ["Synapses"]

    parser = argparse.ArgumentParser(
        description="Detect spot intensity signals in volumetric light microscopy data. "
        "This function supports the detection of synapses in high-resolution light-microscopy data. "
        "It also supports custom models via the (optional) argument '--checkpoint_path' ('-c'). "
    )
    parser.add_argument(
        "-i", "--input_path", required=True,
        help="The path to the input data. Supports .tif, .zarr, .h5 or .n5 files."
        "Zarr, HDF5 or N5 files also require the 'input_key' argument."
    )
    parser.add_argument(
        "-k", "--input_key",
        help="The key to the input data. This refers to an internal data path for Zarr, HDF5 or N5 data."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True,
        help="The output folder where the detectio result and intermediates are stored. "
        "The result is stored in the file 'synapse_detection.tsv'. "
        "In case detections are assigned to segmentation masks and filtered (via '--mask_path' and '--mask_key'), "
        "the corresponding results are stored in 'synapse_detection_filtered.tsv'."

    )
    parser.add_argument(
        "-m", "--model_type", default="Synapses",
        help=f"The model to use for detection. One of {detection_models}.",
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        help="Path to the checkpoint of a customly fine-tuned model (optional).",
    )
    parser.add_argument(
        "--mask_path",
        help="Path to a segmentation mask to use for assigning and filtering the detected synapses (optional). "
        "If given, each detected synapse will be assigned to the closest object in the segmentation and "
        "synapses that are more distant than 'max_distance' will be removed from the result."
    )
    parser.add_argument(
        "--mask_key",
        help="The key to the mask data. Refers to an internal data path, see '--input_key' ('-k') for details."
    )
    parser.add_argument(
        "--max_distance", type=float, default=2.0,
        help="The maximal distance (in microns) for matching synapses to segmented cells. "
        "Synapses with a larger distance will be filtered from the result."
    )
    parser.add_argument(
        "--resolution", type=float, default=0.38, help="The resolution of the data (in microns)."
    )
    args = parser.parse_args()
    if args.model_type not in detection_models:
        raise ValueError(f"Unknown model: {args.model_type}. Choose one of {detection_models}.")

    model_path = _get_model_path(args.model_type, args.checkpoint_path)
    marker_detection(
        input_path=args.input_path, input_key=args.input_key,
        output_folder=args.output_folder, model_path=model_path,
        mask_path=args.mask_path, mask_input_key=args.mask_key,
        max_distance=args.max_distance, resolution=args.resolution,
    )
