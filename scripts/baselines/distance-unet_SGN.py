import os
import sys

script_dir = "/user/schilling40/u15000/flamingo-tools/scripts/prediction"
sys.path.append(script_dir)

import run_prediction_distance_unet

work_dir = "/mnt/lustre-grete/usr/u15000"
checkpoint_dir = os.path.join(work_dir, "checkpoints")
model_name = "cochlea_distance_unet_SGN_supervised_2025-05-27"
model_dir = os.path.join(checkpoint_dir, model_name)
checkpoint = os.path.join(checkpoint_dir, model_name, "best.pt")

cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"

image_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation")
out_dir = os.path.join(cochlea_dir, "predictions/val_sgn")  # /distance_unet

boundary_distance_threshold = 0.5
seg_class = "sgn"

block_shape = (128, 128, 128)
halo = (16, 32, 32)

images = [entry.path for entry in os.scandir(image_dir) if entry.is_file()]

for image in images[:1]:
    sys.argv = [
        os.path.join(script_dir, "run_prediction_distance_unet.py"),
        f"--input={image}",
        f"--output_folder={out_dir}",
        f"--model={model_dir}",
        "--block_shape=[128,128,128]",
        "--halo=[16,32,32]",
        "--memory",
        "--time",
        f"--seg_class={seg_class}"
    ]

    run_prediction_distance_unet.main()
