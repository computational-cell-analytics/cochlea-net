import os
from subprocess import run

import pandas as pd
from flamingo_tools.segmentation import run_unet_prediction
from flamingo_tools.postprocessing.label_components import label_components_sgn
from mobie import add_segmentation
from mobie.metadata import add_remote_project_metadata

MODEL_PATH = "/mnt/vast-nhr/home/pape41/u12086/Work/my_projects/flamingo-tools/scripts/training/checkpoints/cochlea_distance_unet_low-res-sgn-v2"  # noqa
RESOLUTION = (3.0, 1.887779, 1.887779)
SEG_NAME = "SGN_LOWRES-v2"


def segment_sgns(input_path, input_key, output_folder):
    run_unet_prediction(
        input_path, input_key, output_folder,
        model_path=MODEL_PATH, min_size=50,
        block_shape=(64, 128, 128), halo=(8, 32, 32),
        center_distance_threshold=0.5, boundary_distance_threshold=0.5,
    )


def add_to_mobie(output_folder, mobie_dir, dataset_name):
    segmentation_path = os.path.join(output_folder, "segmentation.zarr")
    segmentation_key = "segmentation"

    scale_factors = 4 * [[2, 2, 2]]
    chunks = (96, 96, 96)

    add_segmentation(
        segmentation_path, segmentation_key,
        mobie_dir, dataset_name=dataset_name,
        segmentation_name=SEG_NAME,
        resolution=RESOLUTION,
        scale_factors=scale_factors, chunks=chunks,
        unit="micrometer",
    )


def compute_components(mobie_dir, dataset_name):
    table_path = os.path.join(mobie_dir, dataset_name, "tables", SEG_NAME, "default.tsv")
    table = pd.read_csv(table_path, sep="\t")
    # This may need to be further adapted
    if "M04" in dataset_name:
        max_edge_distance = 70
    else:
        max_edge_distance = 30

    table = label_components_sgn(table, min_size=100,
                                 threshold_erode=None,
                                 min_component_length=1000,
                                 max_edge_distance=max_edge_distance,
                                 iterations_erode=0)
    table.to_csv(table_path, sep="\t", index=False)


def upload_to_s3(mobie_dir, dataset_name):
    service_endpoint = "https://s3.fs.gwdg.de"
    bucket_name = "cochlea-lightsheet"

    add_remote_project_metadata(mobie_dir, bucket_name, service_endpoint)

    # run(["module", "load", "rclone"])
    run(["rclone", "--progress", "copyto",
         f"{mobie_dir}/{dataset_name}/dataset.json",
         f"cochlea-lightsheet:cochlea-lightsheet/{dataset_name}/dataset.json"])
    run(["rclone", "--progress", "copyto",
         f"{mobie_dir}/{dataset_name}/images/ome-zarr",
         f"cochlea-lightsheet:cochlea-lightsheet/{dataset_name}/images/ome-zarr"])
    run(["rclone", "--progress", "copyto",
         f"{mobie_dir}/{dataset_name}/tables/{SEG_NAME}",
         f"cochlea-lightsheet:cochlea-lightsheet/{dataset_name}/tables/{SEG_NAME}"])


def segmentation_workflow(mobie_dir, output_folder, dataset_name):
    input_path = os.path.join(mobie_dir, dataset_name, "images/ome-zarr/PV.ome.zarr")
    input_key = "s0"

    segment_sgns(input_path, input_key, output_folder)
    add_to_mobie(output_folder, mobie_dir, dataset_name)
    compute_components(mobie_dir, dataset_name)
    upload_to_s3(mobie_dir, dataset_name)


def segment_M04():
    mobie_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"  # noqa
    output_folder = "./segmentation/M04"
    dataset_name = "LaVision-M04"
    segmentation_workflow(mobie_dir, output_folder, dataset_name)


def segment_Mar05():
    mobie_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"  # noqa
    output_folder = "./segmentation/Mar05"
    dataset_name = "LaVision-Mar05"
    segmentation_workflow(mobie_dir, output_folder, dataset_name)


def main():
    segment_M04()
    segment_Mar05()


if __name__ == "__main__":
    main()
