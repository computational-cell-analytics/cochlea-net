import os
import subprocess
from shutil import copyfile

import imageio.v3 as imageio
import napari
import pandas as pd
import zarr
from flamingo_tools.test_data import _sample_registry

data_dict = {
    "SGN": "PV",
    "IHC": "VGlut3",
    "SGN-lowres": "PV-lowres",
    "IHC-lowres": "MYO-lowres",
    "Synapses": "CTBP2",
}


def check_segmentation_model(model_name):
    output_folder = f"result_{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    input_path = os.path.join(output_folder, f"{model_name}.tif")
    if not os.path.exists(input_path):
        data_path = _sample_registry().fetch(data_dict[model_name])
        copyfile(data_path, input_path)

    subprocess.run(
        ["flamingo_tools.run_segmentation", "-i", input_path, "-o", output_folder, "-m", model_name]
    )
    output_path = os.path.join(output_folder, "segmentation.zarr")
    segmentation = zarr.open(output_path)["segmentation"][:]

    image = imageio.imread(input_path)
    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation, name=f"{model_name}-segmentation")
    napari.run()


def check_detection_model():
    model_name = "Synapses"
    output_folder = f"result_{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    input_path = os.path.join(output_folder, f"{model_name}.tif")
    if not os.path.exists(input_path):
        data_path = _sample_registry().fetch(data_dict[model_name])
        copyfile(data_path, input_path)

    subprocess.run(
        ["flamingo_tools.run_detection", "-i", input_path, "-o", output_folder, "-m", model_name]
    )
    output_path = os.path.join(output_folder, "synapse_detection.tsv")
    prediction = pd.read_csv(output_path, sep="\t")[["z", "y", "x"]]

    image = imageio.imread(input_path)
    v = napari.Viewer()
    v.add_image(image)
    v.add_points(prediction)
    napari.run()


def main():
    # SGN segmentation:
    # - Prediction works well on the CPU.
    # check_segmentation_model("SGN")

    # IHC segmentation:
    # - Prediction does not work well on the CPU.
    # check_segmentation_model("IHC")

    # SGN segmentation (lowres):
    # - Prediction does not work well on the CPU.
    # check_segmentation_model("SGN-lowres")

    # IHC segmentation (lowres):
    # - The prediction seems to work (on the CPU), but a lot of merges.
    # -> Update the segmentation params?
    # check_segmentation_model("IHC-lowres")

    # Synapse detection:
    # - Prediction works well on the CPU.
    check_detection_model()


if __name__ == "__main__":
    main()
