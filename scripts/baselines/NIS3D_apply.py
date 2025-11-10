import os
import sys

script_dir = "/user/schilling40/u15000/flamingo-tools/scripts/prediction"
sys.path.append(script_dir)

import run_prediction_distance_unet

checkpoint_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/nucleus"
model_name = "NIS3D_supervised_2025-07-17"
model_dir = os.path.join(checkpoint_dir, model_name)
checkpoint = os.path.join(checkpoint_dir, model_name, "best.pt")

cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"

image_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/nucleus/2025-07_NIS3D/test"

out_dir = os.path.join(cochlea_dir, "predictions", "val_nucleus", "distance_unet_NIS3D")  # /distance_unet

boundary_distance_threshold = 0.5
seg_class = "ihc"

block_shape = (128, 128, 128)
halo = (16, 32, 32)

block_shape_str = ",".join([str(b) for b in block_shape])
halo_str = ",".join([str(h) for h in halo])

images = [entry.path for entry in os.scandir(image_dir) if entry.is_file() and "iitest.tif" in entry.path]

for image in images:
    sys.argv = [
        os.path.join(script_dir, "run_prediction_distance_unet.py"),
        f"--input={image}",
        f"--output_folder={out_dir}",
        f"--model={model_dir}",
        f"--block_shape=[{block_shape_str}]",
        f"--halo=[{halo_str}]",
        "--memory",
        "--time",
        "--no_masking",
        f"--seg_class={seg_class}",
        f"--boundary_distance_threshold={boundary_distance_threshold}"
    ]

    run_prediction_distance_unet.main()
