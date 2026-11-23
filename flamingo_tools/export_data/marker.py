"""Export the marker-positive and the marker-negative objects of a segmentation.
"""
import os
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from flamingo_tools.export_data_utils import (
    cochlea_table, export_bounding_box, export_name, export_output_path, normalize_scales, normalize_voxel_size,
    write_volume,
)
from flamingo_tools.file_utils import open_scale_level
from flamingo_tools.s3_utils import MOBIE_FOLDER, default_image_path

MARKER_GROUPS = ("positive", "negative")


def apply_marker_labels(segmentation, label_ids_positive, label_ids_negative, group=None):
    """Relabel a segmentation by marker class.
    Positive segmentation instances are set to 1, negative to 2, everything else to 0.

    Both masks are derived from the original label IDs before anything is written, because the
    class values 1 and 2 are themselves valid label IDs.

    Args:
        segmentation: Segmentation to relabel. Modified in place.
        label_ids_positive: Label IDs of the marker-positive instances.
        label_ids_negative: Label IDs of the marker-negative instances.
        group: Restrict the output to one class, either 'positive', 'negative', or None for both.

    Returns:
        The relabeled segmentation.
    """
    if group == "positive":
        label_ids_negative = []
    elif group == "negative":
        label_ids_positive = []
    elif group is not None:
        raise ValueError("Choose either 'positive' or 'negative' as group value.")

    positive = np.isin(segmentation, label_ids_positive)
    negative = np.isin(segmentation, label_ids_negative)

    segmentation[~(positive | negative)] = 0
    segmentation[positive] = 1
    segmentation[negative] = 2

    return segmentation.astype("float32")


def marker_label_ids(table: pd.DataFrame) -> Tuple[List[int], List[int]]:
    """Get the label IDs of the marker-positive and the marker-negative objects of a table.

    Args:
        table: Segmentation table with the columns "label_id" and "marker_labels" (1 = positive, 2 = negative).

    Returns:
        The label IDs of the positive and of the negative objects.
    """
    missing_columns = sorted({"label_id", "marker_labels"}.difference(table.columns))
    if missing_columns:
        raise ValueError(f"The marker table is missing required columns: {missing_columns}.")
    positive = table.loc[table["marker_labels"] == 1, "label_id"].tolist()
    negative = table.loc[table["marker_labels"] == 2, "label_id"].tolist()
    return positive, negative


def export_marker(
    input_path: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    table: pd.DataFrame,
    input_key: Optional[str] = "s0",
    name: Optional[str] = None,
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    voxel_size: Union[float, Sequence[float]] = (0.38, 0.38, 0.38),
    force: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> None:
    """Export the marker-positive and the marker-negative objects of a segmentation at lower resolutions.

    The output for each scale s is <output_folder>/scale<s>/<name>_marker_<group><crop suffix>.tif, with the
    group "positive" (objects set to 1) and "negative" (objects set to 2).

    Args:
        input_path: Path to the segmentation: a tif file, or a zarr/n5 container.
        output_folder: Output folder.
        scale: Scale level(s), see `flamingo_tools.file_utils.open_scale_level`.
        table: Segmentation table with the marker labels, see `marker_label_ids`.
        input_key: Key of the segmentation in the container. Ignored for a tif file.
        name: Name of the export. Default: the file name of the input without its extension.
        crop_center: Optional crop center as [x, y, z] in µm, see
            `flamingo_tools.export_data.lower_resolution.export_lower_resolution`.
        roi_halo: Halo around the crop center in pixels at the target scale.
        axis: Optional axis index into (x, y, z) to crop as a single-pixel 2D slice at the crop center.
        suffix: Optional label appended to the name of a crop.
        voxel_size: Voxel size of the input in µm, scalar or (x, y, z).
        force: Replace an existing output. Otherwise, an existing output is kept.
        s3: Read the input from the S3 bucket.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    voxel_size = normalize_voxel_size(voxel_size)
    name = export_name(input_path) if name is None else name
    label_ids_positive, label_ids_negative = marker_label_ids(table)

    for s in normalize_scales(scale):
        out_folder = os.path.join(output_folder, f"scale{s}")

        out_paths = {}
        for group in MARKER_GROUPS:
            out_path = export_output_path(out_folder, f"{name}_marker_{group}", False, crop_center, axis, suffix)
            if os.path.exists(out_path) and not force:
                print(f"Skipping {out_path}. File already exists.")
            else:
                out_paths[group] = out_path
        if not out_paths:
            continue

        level = open_scale_level(
            input_path, input_key, s, from_s3=s3, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint,
            credential_file=s3_credentials,
        )
        start, stop = export_bounding_box(level.shape, crop_center, roi_halo, voxel_size, s, axis)
        segmentation = level.read(start, stop, labels=True)
        print("Data shape", segmentation.shape)

        os.makedirs(out_folder, exist_ok=True)
        for group, out_path in out_paths.items():
            print(f"Filtering {group} marker instances.")
            data = apply_marker_labels(segmentation.copy(), label_ids_positive, label_ids_negative, group=group)
            write_volume(out_path, data, voxel_size, s)


def export_marker_cochlea(
    cochlea: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    channels: Sequence[str],
    table_path: Optional[str] = None,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **kwargs,
) -> None:
    """Export the marker-positive and the marker-negative objects of segmentations of a cochlea in a MoBIE project.

    The output is written to <output_folder>/<cochlea>, see `export_marker`.

    Args:
        cochlea: Name of the cochlea.
        output_folder: Output folder.
        scale: Scale level(s).
        channels: Names of the segmentations.
        table_path: Optional local table with the marker labels, e.g. an edited table. It replaces the default
            table <cochlea>/tables/<channel>/default.tsv of every channel.
        mobie_dir: Local MoBIE project directory. Ignored when s3 is set.
        s3: Read the data from the S3 bucket instead of the local MoBIE project.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        kwargs: Further arguments of `export_marker`, e.g. crop_center, roi_halo, axis, suffix, voxel_size or force.
    """
    s3_kwargs = dict(
        s3=s3, s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name, s3_service_endpoint=s3_service_endpoint,
    )
    local_table = None if table_path is None else pd.read_csv(table_path, sep="\t")
    for channel in channels:
        table = local_table
        if table is None:
            table = cochlea_table(cochlea, channel, mobie_dir=mobie_dir, **s3_kwargs)
        export_marker(
            default_image_path(cochlea, channel, s3=s3, mobie_dir=mobie_dir),
            os.path.join(output_folder, cochlea),
            scale,
            table,
            input_key="s0",
            name=channel,
            **s3_kwargs,
            **kwargs,
        )
