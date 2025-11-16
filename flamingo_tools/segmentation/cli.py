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
        model_path = ...   # TODO
    return model_path


def run_segmentation():
    """private
    """
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input_path", required=True, help="The path to the input data.")
    parser.add_argument("-k", "--input_key", required=True, help="The key to the input data.")
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("-m", "--model_type", required=True)
    parser.add_argument("-c", "--checkpoint_path")
    parser.add_argument("--min_size", type=int, default=250)
    # TODO other stuff
    args = parser.parse_args()

    segmentation_models = ["SGN", "IHC", "SGN-lowres", "IHC-lowres"]
    if args.model_type not in segmentation_models:
        raise ValueError
    model_path = _get_model_path(args.model_type, args.checkpoint_path)
    run_unet_prediction(
        input_path=args.input_path, input_key=args.input_key,
        output_folder=args.output_folder, model_path=model_path,
        min_size=args.min_size,
    )


def run_detection():
    """private
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", default="Synapses")
    args = parser.parse_args()
    detection_models = ["Synapses"]
    if args.model_type not in detection_models:
        raise ValueError
    model_path = _get_model_path(args.model_type, args.checkpoint_path)
    # TODO
    marker_detection(
        input_path=args.input_path, input_key=args.input_key,
        output_folder=args.output_folder, model_path=model_path,
    )
