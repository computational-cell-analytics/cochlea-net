import os
from pathlib import Path

import h5py
import imageio.v3 as imageio
import numpy as np
import pandas as pd
import zarr


def get_voxel_size(imaris_file):
    with h5py.File(imaris_file, "r") as f:
        info = f["/DataSetInfo/Image"]
        ext = [[float(b"".join(info.attrs[f"ExtMin{i}"]).decode()),
                float(b"".join(info.attrs[f"ExtMax{i}"]).decode())] for i in range(3)]
        size = [int(b"".join(info.attrs[dim]).decode()) for dim in ["X", "Y", "Z"]]
        vsize = np.array([(max_ - min_) / s for (min_, max_), s in zip(ext, size)])
    return vsize


def get_transformation(imaris_file):
    with h5py.File(imaris_file) as f:
        info = f["DataSetInfo"]["Image"].attrs
        ext_min = np.array([float(b"".join(info[f"ExtMin{i}"]).decode()) for i in range(3)])
        ext_max = np.array([float(b"".join(info[f"ExtMax{i}"]).decode()) for i in range(3)])
        size = [int(b"".join(info[dim]).decode()) for dim in ["X", "Y", "Z"]]
        spacing = (ext_max - ext_min) / size                              # µm / voxel

    # build 4×4 affine: world → index
    T = np.eye(4)
    T[:3, :3] = np.diag(1 / spacing)            # scale
    T[:3, 3] = -ext_min / spacing              # translate

    return T


def get_group_structure(file_path):
    group_names = []

    with h5py.File(file_path, 'r') as f:
        def visit_func(name):
            if isinstance(f[name], h5py.Group):
                # Extract the group name (without the full path)
                group_names.append(name)
        f.visit(visit_func)

    return group_names


def extract_training_data(imaris_file, output_folder, tif_file=None, crop=True, annotation_key=None):
    group_structure = get_group_structure(imaris_file)
    if annotation_key is None:
        # take second set of annotations if available
        if "Scene/Content/Points1" in group_structure:
            annotation_key = "Scene/Content/Points1"
        else:
            annotation_key = "Scene/Content/Points0"

    if annotation_key not in group_structure:
        raise ValueError(f"Imaris group {annotation_key} could not be found.")
    point_key = f"/{annotation_key}/CoordsXYZR"
    with h5py.File(imaris_file, "r") as f:
        if point_key not in f:
            print("Skipping", imaris_file, "due to missing annotations")
            return
        points = f[point_key][:]
        points = points[:, :-1]

        g = f["/DataSet/ResolutionLevel 0/TimePoint 0"]
        # The first channel is ctbp2 / the synapse marker channel.
        data = g["Channel 0/Data"][:]
        # The second channel is vglut / the ihc channel.
        if "Channel 1" in g:
            ihc_data = g["Channel 1/Data"][:]
        else:
            ihc_data = None

    T = get_transformation(imaris_file)
    points = (T @ np.c_[points, np.ones(len(points))].T).T[:, :3]
    points = points[:, ::-1]

    if crop:
        crop_box = np.where(data != 0)
        crop_box = tuple(slice(0, int(cb.max() + 1)) for cb in crop_box)
        data = data[crop_box]

    if tif_file is None:
        original_data = None
    else:
        original_data = imageio.imread(tif_file)

    if output_folder is None:
        import napari
        v = napari.Viewer()
        v.add_image(data)
        if ihc_data is not None:
            v.add_image(ihc_data)
        if original_data is not None:
            v.add_image(original_data, visible=False)
        v.add_points(points)
        v.title = os.path.basename(imaris_file)
        napari.run()
    else:
        image_folder = os.path.join(output_folder, "images")
        os.makedirs(image_folder, exist_ok=True)

        label_folder = os.path.join(output_folder, "labels")
        os.makedirs(label_folder, exist_ok=True)

        fname = Path(imaris_file).stem
        image_file = os.path.join(image_folder, f"{fname}.zarr")
        label_file = os.path.join(label_folder, f"{fname}.csv")

        coords = pd.DataFrame(points, columns=["axis-0", "axis-1", "axis-2"])
        coords.to_csv(label_file, index=False)

        f = zarr.open(image_file, "a")
        # Avoid zarr.errors.ContainsArrayError
        if not os.path.isdir(os.path.join(image_file, "raw")):
            f.create_dataset("raw", data=data)

        if ihc_data is not None:
            if not os.path.isdir(os.path.join(image_file, "raw_ihc")):
                f.create_dataset("raw_ihc", data=ihc_data)
