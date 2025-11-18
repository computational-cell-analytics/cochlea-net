import json
import os
import subprocess
import time

cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"

image_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationSGNs/for_consensus_annotation")
out_dir = os.path.join(cochlea_dir, "predictions", "val_sgn", "micro-sam")

images = [entry.path for entry in os.scandir(image_dir) if entry.is_file()]

model = "vit_b_lm"

for image_file in images:

    abs_path = os.path.abspath(image_file)
    basename = ".".join(os.path.basename(abs_path).split(".")[:-1])

    timer_output = os.path.join(out_dir, f"{basename}_timer.json")
    out_path = os.path.join(out_dir, f"{basename}.tif")
    start = time.perf_counter()

    subprocess_args = [
        "micro_sam.automatic_segmentation",
        f"--input_path={image_file}",
        f"--output_path={out_path}",
        f"--model_type={model}",
        "--ndim=3",
        "--tile_shape", "512", "512",
        "--halo", "64", "64"
    ]
    subprocess.run(subprocess_args, check=True)

    duration = time.perf_counter() - start
    time_dict = {"total_duration[s]": duration}
    with open(timer_output, "w") as f:
        json.dump(time_dict, f, indent='\t', separators=(',', ': '))
