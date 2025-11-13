import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.measurements import compute_object_measures


def compute_volume(
    cochlea, output_path, voxel_spacing, seg_name="SGN_v2", component_list=None,
):
    img_path = f"{cochlea}/images/ome-zarr/PV.ome.zarr"
    seg_path = f"{cochlea}/images/ome-zarr/{seg_name}.ome.zarr"

    img_path, _ = get_s3_path(img_path)
    seg_path, _ = get_s3_path(seg_path)

    if component_list is None:
        component_list = [1]

    segmentation_table_path = f"{cochlea}/tables/{seg_name}/default.tsv"
    feature_set = "morphology"
    compute_object_measures(
        image_path=img_path,
        segmentation_path=seg_path,
        segmentation_table_path=segmentation_table_path,
        output_table_path=output_path,
        resolution=voxel_spacing,
        feature_set=feature_set,
        component_list=component_list,
        s3_flag=True,
        image_key="s0",
        segmentation_key="s0",
    )


def compute_sgn_volumes_flamingo():
    output_folder = "./data/volumes_flamingo"
    os.makedirs(output_folder, exist_ok=True)
    cochleae = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]
    voxel_spacing = (0.38, 0.38, 0.38)
    for cochlea in cochleae:
        output_path = os.path.join(output_folder, f"{cochlea}.tsv")
        if os.path.exists(output_path):
            continue
        compute_volume(cochlea, output_path, voxel_spacing)


def compute_ihc_volumes_flamingo():
    output_folder = "./data/ihc_volumes_flamingo"
    os.makedirs(output_folder, exist_ok=True)
    cochleae = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]
    voxel_spacing = (0.38, 0.38, 0.38)
    for cochlea in cochleae:
        output_path = os.path.join(output_folder, f"{cochlea}.tsv")
        if os.path.exists(output_path):
            continue
        compute_volume(cochlea, output_path, voxel_spacing, seg_name="IHC_v4c")


def compute_sgn_volumes_lavision():
    output_folder = "./data/volumes_lavision"
    os.makedirs(output_folder, exist_ok=True)
    cochleae = ["LaVision-M02", "LaVision-M03"]
    # Note: we used a wrong voxel spacing in MoBIE (1.9 micron in-plane instead of 0.76)
    # We solved this here in a hacky fashion by hard-coding the resolution for the volume
    # calculation temporariliy in the measurement function.
    voxel_spacing = (3.0, 1.887779, 1.887779)
    # voxel_spacing = (3.0, 0.76, 0.76)
    for cochlea in cochleae:
        output_path = os.path.join(output_folder, f"{cochlea}.tsv")
        if os.path.exists(output_path):
            continue
        compute_volume(cochlea, output_path, voxel_spacing, seg_name="SGN_LOWRES-v5c")


def compute_ihc_volumes_lavision():
    output_folder = "./data/ihc_volumes_lavision"
    os.makedirs(output_folder, exist_ok=True)
    cochleae = ["LaVision-M02", "LaVision-M03"]
    # Note: we used a wrong voxel spacing in MoBIE (1.9 micron in-plane instead of 0.76)
    # We solved this here in a hacky fashion by hard-coding the resolution for the volume
    # calculation temporariliy in the measurement function.
    voxel_spacing = (3.0, 1.887779, 1.887779)
    # voxel_spacing = (3.0, 0.76, 0.76)
    component_lists = [[1, 2], [1]]
    for cochlea, component_list in zip(cochleae, component_lists):
        output_path = os.path.join(output_folder, f"{cochlea}.tsv")
        if os.path.exists(output_path):
            continue
        compute_volume(
            cochlea, output_path, voxel_spacing, seg_name="IHC_LOWRES-v3", component_list=component_list,
        )


def compare_sgn_volumes():
    cochleae_flamingo = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]
    alias_flamingo = ["M_01L", "M_01LR", "M_02L", "M_02R"]
    cochleae_lavision = ["LaVision-M02", "LaVision-M03"]
    alias_lavision = ["LV_M_01", "LV_M_02"]

    folder_flamingo = "./data/volumes_flamingo"
    folder_lavision = "./data/volumes_lavision"

    data_flamingo = []
    for cochlea in cochleae_flamingo:
        x = pd.read_csv(os.path.join(folder_flamingo, f"{cochlea}.tsv"), sep="\t")
        volumes = x["volume"].values
        volumes = volumes[~np.isnan(volumes)]
        data_flamingo.append(volumes)

    data_lavision = []
    for cochlea in cochleae_lavision:
        x = pd.read_csv(os.path.join(folder_lavision, f"{cochlea}.tsv"), sep="\t")
        volumes = x["volume"].values
        volumes = volumes[~np.isnan(volumes)]
        data_lavision.append(volumes)

    fig, axes = plt.subplots(2, sharey=True)

    ax = axes[0]
    ax.boxplot(data_flamingo, tick_labels=alias_flamingo)
    ax.set_ylabel("SGN Volume [µm^3]")

    ax = axes[1]
    ax.boxplot(data_lavision, tick_labels=alias_lavision)
    ax.set_ylabel("SGN Volume [µm^3]")

    plt.show()


def compare_ihc_volumes():
    cochleae_flamingo = ["M_LR_000226_L", "M_LR_000226_R", "M_LR_000227_L", "M_LR_000227_R"]
    alias_flamingo = ["M_01L", "M_01LR", "M_02L", "M_02R"]
    cochleae_lavision = ["LaVision-M02", "LaVision-M03"]
    alias_lavision = ["LV_M_01", "LV_M_02"]

    folder_flamingo = "./data/ihc_volumes_flamingo"
    folder_lavision = "./data/ihc_volumes_lavision"

    data_flamingo = []
    size_threshold = 30000
    for cochlea in cochleae_flamingo:
        x = pd.read_csv(os.path.join(folder_flamingo, f"{cochlea}.tsv"), sep="\t")
        volumes = x["volume"].values
        volumes = volumes[~np.isnan(volumes)]
        volumes = volumes[volumes < size_threshold]
        data_flamingo.append(volumes)

    data_lavision = []
    for cochlea in cochleae_lavision:
        x = pd.read_csv(os.path.join(folder_lavision, f"{cochlea}.tsv"), sep="\t")
        volumes = x["volume"].values
        volumes = volumes[~np.isnan(volumes)]
        volumes = volumes[volumes < size_threshold]
        data_lavision.append(volumes)

    fig, axes = plt.subplots(2, sharey=True)

    ax = axes[0]
    ax.boxplot(data_flamingo, tick_labels=alias_flamingo)
    ax.set_ylabel("IHC Volume [µm^3]")

    ax = axes[1]
    ax.boxplot(data_lavision, tick_labels=alias_lavision)
    ax.set_ylabel("IHC Volume [µm^3]")

    plt.show()


def main():
    # compute_sgn_volumes_flamingo()
    # compute_sgn_volumes_lavision()
    compare_sgn_volumes()

    # compute_ihc_volumes_flamingo()
    # compute_ihc_volumes_lavision()
    compare_ihc_volumes()


if __name__ == "__main__":
    main()
