import argparse
import os
import sys

import importlib.util
from pathlib import Path

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"


MODEL_DIR_SGN = os.path.join(COCHLEA_DIR, "trained_models/SGN")
IMAGE_DIR_SGN = os.path.join(COCHLEA_DIR, "AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation")
VAL_DIR_SGN = os.path.join(COCHLEA_DIR, "predictions/val_sgn")
MODEL_DICT_SGN = {
    "distance_unet_f0": {
        "version": "SGN_v2", "fold": 0,
        "path": os.path.join(MODEL_DIR_SGN, "v2_cochlea_distance_unet_SGN_supervised_2025-05-27"),
    },
    "distance_unet_f1": {
        "version": "SGN_v2", "fold": 1,
        "path": os.path.join(MODEL_DIR_SGN, "v2-1_cochlea_distance_unet_SGN_supervised_2026-05-04"),
    },
    "distance_unet_f2": {
        "version": "SGN_v2", "fold": 2,
        "path": os.path.join(MODEL_DIR_SGN, "v2-2_cochlea_distance_unet_SGN_supervised_2026-05-05"),
    },
    "distance_unet_f3": {
        "version": "SGN_v2", "fold": 3,
        "path": os.path.join(MODEL_DIR_SGN, "v2-3_cochlea_distance_unet_SGN_supervised_2026-05-07"),
    },
    "distance_unet_f4": {
        "version": "SGN_v2", "fold": 4,
        "path": os.path.join(MODEL_DIR_SGN, "v2-4_cochlea_distance_unet_SGN_supervised_2026-05-07"),
    },
}

MODEL_DIR_IHC = os.path.join(COCHLEA_DIR, "trained_models/IHC")
IMAGE_DIR_IHC = os.path.join(COCHLEA_DIR, "AnnotatedImageCrops/F1ValidationIHCs")
VAL_DIR_IHC = os.path.join(COCHLEA_DIR, "predictions/val_ihc")
MODEL_DICT_IHC = {
    "distance_unet_f0": {
        "version": "IHC_v11", "fold": 0,
        "path": os.path.join(MODEL_DIR_IHC, "v11_cochlea_distance_unet_IHC_supervised_2026-07-20"),
    },
    "distance_unet_f1": {
        "version": "IHC_v11", "fold": 1,
        "path": os.path.join(MODEL_DIR_IHC, "v11-1_cochlea_distance_unet_IHC_supervised_2026-07-28"),
    },
    "distance_unet_f2": {
        "version": "IHC_v11", "fold": 2,
        "path": os.path.join(MODEL_DIR_IHC, "v11-2_cochlea_distance_unet_IHC_supervised_2026-07-28"),
    },
    "distance_unet_f3": {
        "version": "IHC_v11", "fold": 3,
        "path": os.path.join(MODEL_DIR_IHC, "v11-3_cochlea_distance_unet_IHC_supervised_2026-07-28"),
    },
    "distance_unet_f4": {
        "version": "IHC_v11", "fold": 4,
        "path": os.path.join(MODEL_DIR_IHC, "v11-4_cochlea_distance_unet_IHC_supervised_2026-07-28"),
    },
}

# load run_prediction distance unet
current_dir = Path(__file__).resolve().parent
module_path = os.path.join(current_dir.parent, "prediction", "run_prediction_distance_unet.py")

spec = importlib.util.spec_from_file_location("run_prediction_distance_unet", module_path)
run_prediction_distance_unet = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_prediction_distance_unet)


def run_prediction_wrapper(
    image_dir: str,
    output_dir: str,
    model_dir: str,
    seg_class: str = "sgn",
    block_shape: tuple[int] = (128, 128, 128),
    halo: tuple[int] = (16, 32, 32),
    force_overwrite: bool = False,
) -> None:
    images = [entry.path for entry in os.scandir(image_dir) if entry.is_file() and ".tif" in entry.path]
    for image in images:
        stem = ".".join(os.path.basename(image).split(".")[:-1])
        output_file = os.path.join(output_dir, f"{stem}_seg.tif")
        if os.path.isfile(output_file) and not force_overwrite:
            print(f"Skipping prediction of {image}. Prediction already exists.")
        else:
            sys.argv = [
                module_path,
                f"--input={image}",
                f"--output_folder={output_dir}",
                f"--model={model_dir}",
                f"--block_shape=[{block_shape[0]},{block_shape[1]},{block_shape[2]}]",
                f"--halo=[{halo[0]},{halo[1]},{halo[2]}]",
                "--memory",
                "--time",
                f"--seg_class={seg_class}"
            ]

            run_prediction_distance_unet.main()


def main():
    parser = argparse.ArgumentParser(
        description="Script to apply a CochleaNet distance U-Net model (SGN or IHC) to TIF files "
                    "in an image directory. If --model and --output are omitted, the script falls "
                    "back to evaluating five SGN folds to determine the statistical variance."
    )

    parser.add_argument("-m", "--model", type=str, default=None,
                        help="Model directory of the CochleaNet model to apply (matching --seg_class).")
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Directory containing images in TIF format to run inference on.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for predictions.")
    parser.add_argument("-s", "--seg_class", type=str, default="sgn",
                        help="Segmentation class. Either 'ihc' or 'sgn'. Default: sgn.")
    parser.add_argument("-f", "--force_overwrite", action="store_true",
                        help="Recompute predictions even if an output file already exists.")

    args = parser.parse_args()

    # Process all 5 folds of the network
    if args.model is None and args.output is None and args.input is None:
        model_paths = []
        output_dirs = []
        if args.seg_class == "sgn":
            for name, model_dic in MODEL_DICT_SGN.items():
                output_dir = os.path.join(VAL_DIR_SGN, name)
                output_dirs.append(output_dir)
                model_paths.append(model_dic["path"])
                input_dir = IMAGE_DIR_SGN
        elif args.seg_class == "ihc":
            for name, model_dic in MODEL_DICT_IHC.items():
                output_dir = os.path.join(VAL_DIR_IHC, name)
                output_dirs.append(output_dir)
                model_paths.append(model_dic["path"])
                input_dir = IMAGE_DIR_IHC
        else:
            raise ValueError("Choose either 'sgn' or 'ihc' as --seg_class.")
    # raise an error if a non-viable combination of model and output is provided
    elif args.model is None or args.output is None or args.input:
        raise ValueError("Provide all, a model, an input and an output directory, or none.")
    else:
        model_paths = [args.model]
        output_dirs = [args.output]
        input_dir = args.input

    for (model_path, output_dir) in zip(model_paths, output_dirs):
        print(f"Running prediction of {args.seg_class} with model {model_path}. Save results in {output_dir}.")
        os.makedirs(output_dir, exist_ok=True)
        run_prediction_wrapper(
            image_dir=input_dir,
            output_dir=output_dir,
            model_dir=model_path,
            seg_class=args.seg_class,
            force_overwrite=args.force_overwrite,
        )


if __name__ == "__main__":
    main()
