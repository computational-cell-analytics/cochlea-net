"""private
"""
import argparse

from . import synapse_detection, unet_prediction
from .. import s3_utils
from ..model_utils import get_model_path, get_default_segmentation_settings

WATERSHED_OPTIONS = ("center_distance_threshold", "boundary_distance_threshold", "fg_threshold", "distance_smoothing")
S3_OPTIONS = ("s3_credentials", "s3_bucket_name", "s3_service_endpoint")


def _get_model_path(model_type, checkpoint_path=None):
    if checkpoint_path is None:
        model_path = get_model_path(model_type)
    else:
        model_path = checkpoint_path
    return model_path


def _optional_float(value):
    return None if value.strip() in ("", "None") else float(value)


def _add_s3_arguments(parser, flags):
    for flag, help_text in flags.items():
        parser.add_argument(f"--{flag}", action="store_true", help=help_text)
    parser.add_argument(
        "--s3_credentials", default=None,
        help="Input file containing S3 credentials. "
        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported."
    )
    parser.add_argument(
        "--s3_bucket_name", default=None, help="S3 bucket name. Optional if BUCKET_NAME was exported."
    )
    parser.add_argument(
        "--s3_service_endpoint", default=None,
        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported."
    )


def _s3_kwargs(args):
    return {name: getattr(args, name) for name in S3_OPTIONS}


def _check_stage_options(parser, args, last_stage, applies_to):
    """Reject an option that the selected stage does not use.

    `applies_to` maps the destination of an option to the stages that use it. The stage None is
    the full run. Only the last stage reads no image data, so all others require the input path.
    """
    if args.stage != last_stage and args.input_path is None:
        parser.error("the following arguments are required: -i/--input_path")
    for dest, stages in applies_to.items():
        default = parser.get_default(dest)
        if args.stage not in stages and getattr(args, dest, default) != default:
            mode = "the full run" if args.stage is None else f"--stage {args.stage}"
            parser.error(f"--{dest} does not apply to {mode}.")


def run_segmentation():
    """private
    """
    segmentation_models = ["SGN", "IHC", "SGN-lowres", "IHC-lowres"]

    parser = argparse.ArgumentParser(
        description="Segment individual cells in volumetric light microscopy data. "
        "This function supports the segmentation of SGNs and IHCs, at high- and low-resolution. "
        "Which model to use is specified with the argument '--model_type' ('-m'). "
        "It also supports custom models via the (optional) argument '--checkpoint_path' ('-c'). "
        "By default, all stages run in one process. '--stage' runs one stage as a separate job, "
        "for example in a Slurm pipeline."
    )
    parser.add_argument(
        "-i", "--input_path",
        help="The path to the input data. Supports .tif, .zarr, .h5 or .n5 files. "
        "Zarr, HDF5 or N5 files also require the 'input_key' argument. Required except for '--stage segment'."
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
        "-m", "--model_type", required=True, choices=segmentation_models,
        help="The model to use for segmentation. It also selects the rule of the mask and the default values "
        "of the watershed options.",
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        help="Path to the checkpoint of a customly fine-tuned model (optional).",
    )
    parser.add_argument(
        "--min_size", type=int, default=250,
        help="The minimum size of cells in the resulting segmentation.",
    )
    # SUPPRESS leaves an option that is not given out of the namespace, so that 'None' stays a valid value.
    parser.add_argument(
        "--center_distance_threshold", type=_optional_float, default=argparse.SUPPRESS,
        help="The threshold applied to the center distance prediction to derive the seeds. "
        "'None' does not use the center distances. By default, the value of the model."
    )
    parser.add_argument(
        "--boundary_distance_threshold", type=_optional_float, default=argparse.SUPPRESS,
        help="The threshold applied to the boundary distance prediction to derive the seeds. "
        "'None' does not use the boundary distances. By default, the value of the model."
    )
    parser.add_argument(
        "--fg_threshold", type=float, default=argparse.SUPPRESS,
        help="The threshold applied to the foreground prediction to derive the mask of the watershed. "
        "By default, the value of the model."
    )
    parser.add_argument(
        "--distance_smoothing", type=float, default=argparse.SUPPRESS,
        help="The sigma of the gaussian smoothing of the distance predictions. 0 applies no smoothing. "
        "By default, the value of the model."
    )
    parser.add_argument(
        "--disable_masking", action="store_true",
        help="Whether to disable intensity-based masking. By default, "
        "segmentation is only applied for parts of the data with foreground signal."
    )
    parser.add_argument(
        "--legacy_mask", action="store_true",
        help="Mask with the percentile rule of flamingo-tools before 2026-11 instead of the default rule "
        "relative to the background level."
    )
    parser.add_argument(
        "--absolute_threshold", type=_optional_float,
        help="The minimal block percentile of the legacy rule ('--legacy_mask'). "
        "By default, the threshold is derived from the data."
    )
    parser.add_argument(
        "--background_factor", type=float,
        help="The threshold of the default mask rule as a multiple of the background level. By default, 1.08 "
        "for SGN and 1.75 for IHC. The low-resolution models and '--legacy_mask' do not accept it."
    )
    _add_s3_arguments(parser, {"s3_input": "Read the image data from the S3 bucket."})

    stages = parser.add_argument_group("Slurm stages")
    stages.add_argument(
        "--stage", choices=["mask", "predict", "segment"],
        help="Run one stage as a separate job. The stages share the output folder and run in this order: "
        "'mask' writes 'mask.zarr', 'predict' runs one task of a Slurm array and writes 'predictions.zarr', "
        "'segment' writes 'segmentation.zarr'. By default, all stages run in one process."
    )
    stages.add_argument(
        "--prediction_instances", type=int, default=1,
        help="The number of tasks of '--stage predict'. It must match the size of the Slurm array. "
        "Each task reads its index from SLURM_ARRAY_TASK_ID."
    )

    args = parser.parse_args()
    reading = (None, "mask", "predict")
    _check_stage_options(parser, args, "segment", {
        "input_path": reading, "input_key": reading, "s3_input": reading,
        **{name: reading for name in S3_OPTIONS},
        "checkpoint_path": (None, "predict"),
        "disable_masking": (None,),
        "legacy_mask": (None, "mask"), "absolute_threshold": (None, "mask"), "background_factor": (None, "mask"),
        "min_size": (None, "segment"),
        **{name: (None, "segment") for name in WATERSHED_OPTIONS},
        "prediction_instances": ("predict",),
    })

    settings = get_default_segmentation_settings(args.model_type)
    watershed_kwargs = {name: getattr(args, name, settings[name]) for name in WATERSHED_OPTIONS}
    # The watershed raises this error only after the prediction, which can take hours.
    seed_thresholds = (watershed_kwargs["center_distance_threshold"], watershed_kwargs["boundary_distance_threshold"])
    if seed_thresholds == (None, None):
        parser.error("--center_distance_threshold and --boundary_distance_threshold cannot both be None.")
    mask_kwargs = {
        "seg_class": settings["seg_class"], "legacy_mask": args.legacy_mask,
        "absolute_threshold": args.absolute_threshold, "background_factor": args.background_factor,
    }

    if args.stage == "mask":
        unet_prediction.run_unet_prediction_preprocess_slurm(
            input_path=args.input_path, input_key=args.input_key, output_folder=args.output_folder,
            s3=args.s3_input, **_s3_kwargs(args), **mask_kwargs,
        )
    elif args.stage == "predict":
        unet_prediction.run_unet_prediction_slurm(
            input_path=args.input_path, input_key=args.input_key, output_folder=args.output_folder,
            model_path=_get_model_path(args.model_type, args.checkpoint_path),
            prediction_instances=args.prediction_instances, seg_class=settings["seg_class"],
            s3=args.s3_input, **_s3_kwargs(args),
        )
    elif args.stage == "segment":
        unet_prediction.run_unet_segmentation_slurm(
            output_folder=args.output_folder, min_size=args.min_size, **watershed_kwargs,
        )
    else:
        input_path = s3_utils.resolve_path(
            args.input_path, args.s3_input, bucket_name=args.s3_bucket_name,
            service_endpoint=args.s3_service_endpoint, credential_file=args.s3_credentials,
        )
        unet_prediction.run_unet_prediction(
            input_path=input_path, input_key=args.input_key,
            output_folder=args.output_folder, model_path=_get_model_path(args.model_type, args.checkpoint_path),
            min_size=args.min_size, use_mask=not args.disable_masking, **mask_kwargs, **watershed_kwargs,
        )


def run_detection():
    """private
    """
    detection_models = ["Synapses"]

    parser = argparse.ArgumentParser(
        description="Detect spot intensity signals in volumetric light microscopy data. "
        "This function supports the detection of synapses in high-resolution light-microscopy data. "
        "It also supports custom models via the (optional) argument '--checkpoint_path' ('-c'). "
        "The standard case is to pass an IHC segmentation via '--mask_path': the inference then "
        "runs only on the region around the IHCs, and the detections are matched to them. "
        "Without it the inference runs on the full volume and the detections are not matched. "
        "By default, all stages run in one process. '--stage' runs one stage as a separate job, "
        "for example in a Slurm pipeline."
    )
    parser.add_argument(
        "-i", "--input_path",
        help="The path to the input data. Supports .tif, .zarr, .h5 or .n5 files. "
        "Zarr, HDF5 or N5 files also require the 'input_key' argument. Required except for '--stage detect'."
    )
    parser.add_argument(
        "-k", "--input_key",
        help="The key to the input data. This refers to an internal data path for Zarr, HDF5 or N5 data."
    )
    parser.add_argument(
        "-o", "--output_folder", required=True,
        help="The output folder where the detection result and intermediates are stored. "
        "The result is stored in the file 'synapse_detection.tsv'. "
        "In case detections are assigned to segmentation masks and filtered (via '--mask_path' and '--mask_key'), "
        "the corresponding results are stored in 'synapse_detection_filtered.tsv'."
    )
    parser.add_argument(
        "-m", "--model_type", default="Synapses", choices=detection_models,
        help="The model to use for detection.",
    )
    parser.add_argument(
        "-c", "--checkpoint_path",
        help="Path to the checkpoint of a customly fine-tuned model (optional).",
    )
    parser.add_argument(
        "--mask_path",
        help="Path to a segmentation mask to use for assigning and filtering the detected synapses. "
        "Each detected synapse will be assigned to the closest object in the segmentation and "
        "synapses that are more distant than 'max_distance' will be removed from the result. "
        "Without it the inference runs on the full volume, which is a lot more expensive."
    )
    parser.add_argument(
        "--mask_input_key", default="s4",
        help="The key to the downscaled mask data, used to restrict the inference to the region "
        "around the segmented cells. Refers to an internal data path, see '--input_key' ('-k') for details."
    )
    parser.add_argument(
        "--mask_key", default="s0",
        help="The key to the mask data at full resolution, used to match the detections to the "
        "segmented cells. Refers to an internal data path, see '--input_key' ('-k') for details."
    )
    parser.add_argument(
        "--dilation_iterations", type=int, default=4,
        help="The number of dilation steps applied to the mask, in voxels of '--mask_input_key'. "
        "The mask must extend at least 'max_distance' beyond the segmented cells."
    )
    parser.add_argument(
        "--max_distance", type=float, default=3.0,
        help="The maximal distance (in microns) for matching synapses to segmented cells. "
        "Synapses with a larger distance will be filtered from the result."
    )
    parser.add_argument(
        "-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Threshold for peak detection."
    )
    _add_s3_arguments(parser, {
        "s3_input": "Read the image data from the S3 bucket.",
        "s3_mask": "Read the segmentation mask from the S3 bucket.",
    })

    stages = parser.add_argument_group("Slurm stages")
    stages.add_argument(
        "--stage", choices=["mask", "predict", "detect"],
        help="Run one stage as a separate job. The stages share the output folder and run in this order: "
        "'mask' writes 'mask.zarr', 'predict' runs one task of a Slurm array and writes 'predictions.zarr', "
        "'detect' writes the detection tables. By default, all stages run in one process."
    )
    stages.add_argument(
        "--prediction_instances", type=int, default=1,
        help="The number of tasks of '--stage predict'. It must match the size of the Slurm array. "
        "Each task reads its index from SLURM_ARRAY_TASK_ID."
    )
    stages.add_argument(
        "--force_overwrite", action="store_true",
        help="Recompute the detections of '--stage detect' even if the table of an earlier run exists."
    )

    args = parser.parse_args()
    reading = (None, "mask", "predict")
    matching = (None, "mask", "detect")
    _check_stage_options(parser, args, "detect", {
        "input_path": reading, "input_key": reading, "s3_input": reading,
        "checkpoint_path": (None, "predict"),
        "mask_path": matching, "s3_mask": matching, "max_distance": matching, "voxel_size": matching,
        "mask_input_key": (None, "mask"), "dilation_iterations": (None, "mask"),
        "mask_key": (None, "detect"), "threshold": (None, "detect"),
        "prediction_instances": ("predict",), "force_overwrite": ("detect",),
    })

    if args.stage == "mask":
        synapse_detection.run_synapse_prediction_preprocess_slurm(
            input_path=args.input_path, input_key=args.input_key, output_folder=args.output_folder,
            mask_path=args.mask_path, mask_input_key=args.mask_input_key,
            dilation_iterations=args.dilation_iterations, voxel_size=args.voxel_size,
            max_distance=args.max_distance, s3_input=args.s3_input, s3_mask=args.s3_mask, **_s3_kwargs(args),
        )
    elif args.stage == "predict":
        synapse_detection.run_synapse_prediction_slurm(
            input_path=args.input_path, input_key=args.input_key, output_folder=args.output_folder,
            model_path=_get_model_path(args.model_type, args.checkpoint_path),
            prediction_instances=args.prediction_instances, s3_input=args.s3_input, **_s3_kwargs(args),
        )
    elif args.stage == "detect":
        synapse_detection.run_synapse_detection_slurm(
            output_folder=args.output_folder, voxel_size=args.voxel_size, threshold=args.threshold,
            mask_path=args.mask_path, mask_key=args.mask_key, max_distance=args.max_distance,
            force_overwrite=args.force_overwrite, s3_mask=args.s3_mask, **_s3_kwargs(args),
        )
    else:
        synapse_detection.marker_detection(
            input_path=args.input_path, input_key=args.input_key,
            output_folder=args.output_folder, model_path=_get_model_path(args.model_type, args.checkpoint_path),
            mask_path=args.mask_path,
            mask_input_key=args.mask_input_key, mask_key=args.mask_key,
            dilation_iterations=args.dilation_iterations,
            max_distance=args.max_distance, voxel_size=args.voxel_size,
            threshold=args.threshold,
            # The image and the segmentation are resolved independently, so the image can be read
            # from a local file while the segmentation is read from the bucket.
            s3_input=args.s3_input, s3_mask=args.s3_mask, **_s3_kwargs(args),
        )
