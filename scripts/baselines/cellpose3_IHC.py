import json
import os
import time

from cellpose import core, denoise, io, models
from pathlib import Path
from tqdm import trange
from natsort import natsorted

io.logger_setup()  # run this to get printing of progress

# Check if colab notebook instance has GPU access
if core.use_gpu() is False:
    raise ImportError("No GPU access, change your runtime")

model = models.CellposeModel(gpu=True)

# *** change to your google drive folder path ***
cochlea_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
input_dir = os.path.join(cochlea_dir, "AnnotatedImageCrops/F1ValidationIHCs")
out_dir = os.path.join(cochlea_dir, "predictions/val_ihc/cellpose3")

input_dir = Path(input_dir)
if not input_dir.exists():
    raise FileNotFoundError("directory does not exist")

# *** change to your image extension ***
image_ext = ".tif"

# list all files
files = natsorted([f for f in input_dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])

if len(files) == 0:
    raise FileNotFoundError("no image files found, did you specify the correct folder and extension?")
else:
    print(f"{len(files)} images in folder:")

for f in files:
    print(f.name)

flow_threshold = 0.4
cellprob_threshold = 0.0
tile_norm_blocksize = 0

masks_ext = ".png" if image_ext == ".png" else ".tif"
for i in trange(len(files)):
    f = files[i]
    start = time.perf_counter()

    img = io.imread(f)

    basename = "".join(f.name.split(".")[:-1])
    out_path = os.path.join(out_dir, f"{basename}_seg.tif")
    timer_output = os.path.join(out_dir, f"{basename}_timer.json")

    io.logger_setup()  # run this to get printing of progress

    # DEFINE CELLPOSE MODEL
    # model_type="cyto3" or "nuclei", or other model
    # restore_type: "denoise_cyto3", "deblur_cyto3", "upsample_cyto3", "denoise_nuclei", "deblur_nuclei"
    model = denoise.CellposeDenoiseModel(gpu=True, model_type="cyto3", restore_type="denoise_cyto3")

    diameter = 20

    masks, flows, styles, imgs_dn = model.eval(img, diameter=diameter, channels=[0, 0])

    # masks, flows, styles = model.eval(img, batch_size=32, flow_threshold=flow_threshold,
    #                                   cellprob_threshold=cellprob_threshold,
    #                                   normalize={"tile_norm_blocksize": tile_norm_blocksize})

    io.imsave(out_path, masks)

    duration = time.perf_counter() - start
    time_dict = {"total_duration[s]": duration}
    with open(timer_output, "w") as f:
        json.dump(time_dict, f, indent='\t', separators=(',', ': '))
