import os
import subprocess
from shutil import copyfile

import imageio.v3 as imageio
import napari
import pandas as pd
import zarr
from flamingo_tools.test_data import _sample_registry

view = True
data_dict = {
    "SGN": "PV",
    "IHC": "VGlut3",
    "SGN-lowres": "PV-lowres",
    "IHC-lowres": "MYO-lowres",
    "Synapses": "CTBP2",
}


def check_segmentation_model(model_name, checkpoint_path=None, input_path=None, check_prediction=False, **kwargs):
    output_folder = f"result_{model_name}"
    os.makedirs(output_folder, exist_ok=True)
    if input_path is None:
        input_path = os.path.join(output_folder, f"{model_name}.tif")
        if not os.path.exists(input_path):
            data_path = _sample_registry().fetch(data_dict[model_name])
            copyfile(data_path, input_path)

    output_path = os.path.join(output_folder, "segmentation.zarr")
    if not os.path.exists(output_path):
        cmd = [
            "flamingo_tools.run_segmentation", "-i", input_path, "-o", output_folder, "-m", model_name,
            "--disable_masking", "--min_size", "5",
        ]
        for name, val in kwargs.items():
            cmd.extend([f"--{name}", str(val)])
        if checkpoint_path is not None:
            cmd.extend(["-c", checkpoint_path])
        subprocess.run(cmd)

    if view:
        segmentation = zarr.open(output_path)["segmentation"][:]
        image = imageio.imread(input_path)
        v = napari.Viewer()
        v.add_image(image)
        if check_prediction:
            prediction = zarr.open(os.path.join(output_folder, "predictions.zarr"))["prediction"][:]
            v.add_image(prediction)
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

    output_path = os.path.join(output_folder, "synapse_detection.tsv")
    if not os.path.exists(output_path):
        subprocess.run(
            ["flamingo_tools.run_detection", "-i", input_path, "-o", output_folder, "-m", model_name]
        )

    if view:
        prediction = pd.read_csv(output_path, sep="\t")[["z", "y", "x"]]
        image = imageio.imread(input_path)
        v = napari.Viewer()
        v.add_image(image)
        v.add_points(prediction)
        napari.run()


def main():
    # SGN segmentation:
    # - Prediction works well on the CPU.
    # - Prediction works well on the GPU.
    # check_segmentation_model("SGN")

    # IHC segmentation:
    # - Prediction works well on the CPU.
    # - Prediction works well on the GPU.
    # check_segmentation_model("IHC")

    # SGN segmentation (lowres):
    # - Prediction works well on the CPU.
    # - Prediction works well on the GPU.
    check_segmentation_model(
        "SGN-lowres",
        # boundary_distance_threshold=0.5, center_distance_threshold=None,
    )

    # IHC segmentation (lowres):
    # - Prediction works well on the CPU.
    # - Prediction works well on the GPU.
    # check_segmentation_model("IHC-lowres")

    # Synapse detection:
    # - Prediction works well on the CPU.
    # - Prediction works well on the GPU.
    # check_detection_model()


if __name__ == "__main__":
    main()
