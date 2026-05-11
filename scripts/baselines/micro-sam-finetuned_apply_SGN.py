import json
import os
import subprocess
import time

cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"

image_dir = os.path.join(cochlea_dir, "predictions/val_sgn/images2D")
image_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation")
out_dir = os.path.join(cochlea_dir, "predictions", "val_sgn", "micro-sam_finetuned")
model_dir = os.path.join(cochlea_dir, "trained_models", "SGN")

images = [entry.path for entry in os.scandir(image_dir) if entry.is_file()]
checkpoint = os.path.join(model_dir, "cochlea_micro-sam_sgn-v2_2d_2026-05-05.pt")

os.makedirs(out_dir, exist_ok=True)

model = "vit_b_lm"

for image_file in images:

    abs_path = os.path.abspath(image_file)
    basename = os.path.splitext(os.path.basename(abs_path))[0]

    timer_output = os.path.join(out_dir, f"{basename}_timer.json")
    out_path = os.path.join(out_dir, f"{basename}_seg.tif")
    start = time.perf_counter()

    subprocess_args = [
        "micro_sam.automatic_segmentation",
        f"--input_path={image_file}",
        f"--output_path={out_path}",
        f"--model_type={model}",
        f"--checkpoint={checkpoint}",
        "--ndim=3",
        "--tile_shape", "256", "256",
        "--halo", "64", "64"
    ]
    subprocess.run(subprocess_args, check=True)

    duration = time.perf_counter() - start
    time_dict = {"total_duration[s]": duration}
    with open(timer_output, "w") as f:
        json.dump(time_dict, f, indent='\t', separators=(',', ': '))
