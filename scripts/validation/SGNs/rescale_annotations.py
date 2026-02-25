import json
import os
import shutil
from glob import glob

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT, create_s3_target


def get_scale_factor():
    original_path = "/mnt/ceph-hdd/cold/nim00007/cochlea-lightsheet/M_LR_000169_R/MLR000169R_PV.tif"
    original_shape = tifffile.memmap(original_path).shape

    cochlea = "M_LR_000169_R"
    internal_path = os.path.join(cochlea, "images", "ome-zarr", "SGN_v2.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

    input_key = "s0"
    with zarr.open(s3_store, mode="r") as f:
        new_shape = f[input_key].shape

    scale_factor = tuple(
        float(nsh) / float(osh) for nsh, osh in zip(new_shape, original_shape)
    )
    return scale_factor


def get_shape():
    cochlea = "M_LR_000169_R_fused"
    s3 = create_s3_target()
    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    print("Available sources:")
    for source in info["sources"].keys():
        print(source)

    internal_path = os.path.join(cochlea, "images", "ome-zarr", "PV.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

    input_key = "s0"
    with zarr.open(s3_store, mode="r") as f:
        new_shape = f[input_key].shape
    return new_shape


def rescale_annotations(annotation_file, output_folder, new_shape, original_shape):
    # 0.) Split the name into its parts.
    fname = os.path.basename(annotation_file)
    name_components = fname.split("_")
    z = int(name_components[2][1:])

    # 1.) Find the matching raw file and get its shape.
    root = os.path.split(os.path.split(annotation_file)[0])[0]
    tif_name = "_".join(name_components[:-1])
    image_file = os.path.join(root, f"{tif_name}.tif")
    assert os.path.exists(image_file), image_file
    this_shape = tifffile.memmap(image_file).shape

    # 2.) Determine if the annotations have to be reshaped,
    if this_shape[1:] == new_shape[1:]:  # No, they don't have to be reshaped.
        # In this case we copy the annotations and that's it.
        print(annotation_file, "does not need to be rescaled")
        output_path = os.path.join(output_folder, fname)
        shutil.copyfile(annotation_file, output_path)
        return
    elif this_shape[1:] == original_shape[1:]:  # Yes, they have to be reshaped
        pass
    else:
        raise ValueError(f"Unexpected shape: {this_shape}")

    # 3.) Rescale the annotations.
    scale_factor = [float(ns) / float(os) for ns, os in zip(new_shape, original_shape)]

    annotations = pd.read_csv(annotation_file)
    annotations_rescaled = annotations.copy()
    annotations_rescaled["axis-1"] = annotations["axis-1"] * scale_factor[1]
    annotations_rescaled["axis-2"] = annotations["axis-2"] * scale_factor[2]

    new_z = int(np.round(z * scale_factor[0]))
    name_components[2] = f"z{new_z}"
    name_components = name_components[:-1] + ["rescaled"] + name_components[-1:]
    new_fname = "_".join(name_components)

    output_path = os.path.join(output_folder, new_fname)
    annotations_rescaled.to_csv(output_path, index=False)


def rescale_all_annotations():
    prefix = "MLR169R_PV"
    # shape = get_shape()
    original_shape = (1921, 1479, 2157)
    new_shape = (5089, 3915, 5665)

    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
    annotation_folders = ["AnnotationsEK", "AnnotationsAMD", "AnnotationsLR"]
    for folder in annotation_folders:
        output_folder = os.path.join(root, folder, "rescaled")
        os.makedirs(output_folder, exist_ok=True)

        files = glob(os.path.join(root, folder, "*.csv"))
        for annotation_file in files:
            fname = os.path.basename(annotation_file)
            if not fname.startswith(prefix):
                continue
            print("Rescaling", annotation_file)
            rescale_annotations(annotation_file, output_folder, new_shape, original_shape)


# Download the two new slices.
def download_new_data():
    from flamingo_tools.validation import fetch_data_for_evaluation

    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
    output_folder = os.path.join(root, "for_consensus_annotation")
    os.makedirs(output_folder, exist_ok=True)

    cochlea = "M_LR_000169_R_fused"
    files = [
        "AnnotationsEK/rescaled/MLR169R_PV_z1913_base_full_rescaled_annotationsEK.csv",
        "AnnotationsEK/rescaled/MLR169R_PV_z2594_mid_full_rescaled_annotationsEK.csv"
    ]
    for ff in files:
        annotation_path = os.path.join(root, ff)

        fname = os.path.basename(annotation_path)
        name_components = fname.split("_")

        tif_name = "_".join(name_components[:-1])
        image_file = os.path.join(output_folder, f"{tif_name}.tif")

        _, _, image = fetch_data_for_evaluation(
            annotation_path, cache_path=None, cochlea=cochlea, extra_data="PV", z_extent=10
        )
        print(image.shape)
        print("Writing to:", image_file)
        imageio.imwrite(image_file, image)


def check_rescaled_annotations():
    import napari
    from flamingo_tools.validation import fetch_data_for_evaluation

    root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/F1ValidationSGNs"
    annotation_folders = ["AnnotationsEK/rescaled", "AnnotationsAMD/rescaled", "AnnotationsLR/rescaled"]
    cochlea = "M_LR_000169_R_fused"

    for folder in annotation_folders:
        annotation_paths = sorted(glob(os.path.join(root, folder, "*.csv")))
        for annotation_path in annotation_paths:
            segmentation, annotations, image = fetch_data_for_evaluation(
                annotation_path, cache_path=None, components_for_postprocessing=[1], cochlea=cochlea, extra_data="PV",
            )
            v = napari.Viewer()
            v.add_image(image)
            v.add_labels(segmentation)
            v.add_points(annotations)
            v.title = annotation_path
            napari.run()


def main():
    # rescale_all_annotations()
    # check_rescaled_annotations()

    # MLR169R_PV_z1913_base_full_rescaled.tif
    # MLR169R_PV_z2594_mid_full_rescaled.tif
    download_new_data()


# Rescale the point annotations for the cochlea MLR169R, which was
# annotated at the original scale, but then rescaled for segmentation.
if __name__ == "__main__":
    main()
