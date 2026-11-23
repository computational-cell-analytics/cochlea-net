"""Export the tonotopic frequency of each object of a segmentation as an image.
"""
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import tifffile

from flamingo_tools.export_data_utils import (
    cochlea_table, export_bounding_box, export_name, export_output_path, normalize_scales, normalize_voxel_size,
    write_volume,
)
from flamingo_tools.file_utils import open_scale_level
from flamingo_tools.s3_utils import MOBIE_FOLDER, default_image_path


def _write_colorbar(out_path, norm, cmap, name):
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=(6, 1.3))
    fig.subplots_adjust(bottom=0.5)
    colorbar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
    colorbar.set_label("Frequency [kHz]")
    plt.title(f"Tonotopic Mapping: {name}")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


def export_frequency(
    input_path: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    table: pd.DataFrame,
    input_key: Optional[str] = "s0",
    name: Optional[str] = None,
    components: List[int] = [1],
    colormap: Optional[str] = None,
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
    """Export the tonotopic frequency of the objects of a segmentation at lower resolutions.

    The output for each scale s is <output_folder>/scale<s>/frequencies_<name>[_<colormap>]<crop suffix>.tif.
    Without a colormap, each object has its frequency in kHz as value. With a colormap, the output is an RGBA
    image, and a color bar is written to frequencies_<name>_<colormap>.png.

    Args:
        input_path: Path to the segmentation: a tif file, or a zarr/n5 container.
        output_folder: Output folder.
        scale: Scale level(s), see `flamingo_tools.file_utils.open_scale_level`.
        table: Segmentation table with the columns "label_id", "component_labels" and "frequency[kHz]".
        input_key: Key of the segmentation in the container. Ignored for a tif file.
        name: Name of the export. Default: the file name of the input without its extension.
        components: Export only the objects of these components.
        colormap: Optional name of a matplotlib colormap for an RGBA output.
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

    max_id = int(table.label_id.values.max())
    table = table[table.component_labels.isin(components)]
    if len(table) == 0:
        raise ValueError(f"The table has no objects of the components {list(components)}.")
    frequencies = table["frequency[kHz]"].values
    freq_range = (float(frequencies.min()), float(frequencies.max()))
    print(f"Frequency range: {freq_range[0]} - {freq_range[1]} kHz")

    out_name = f"frequencies_{name}" if colormap is None else f"frequencies_{name}_{colormap}"
    for s in normalize_scales(scale):
        out_folder = os.path.join(output_folder, f"scale{s}")
        out_path = export_output_path(out_folder, out_name, False, crop_center, axis, suffix)
        if os.path.exists(out_path) and not force:
            print(f"Skipping {out_path}. File already exists.")
            continue

        level = open_scale_level(
            input_path, input_key, s, from_s3=s3, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint,
            credential_file=s3_credentials,
        )
        start, stop = export_bounding_box(level.shape, crop_center, roi_halo, voxel_size, s, axis)
        # A segmentation in a TIF file of `export_lower_resolution` has the data type float32.
        segmentation = level.read(start, stop, labels=True).astype("int64", copy=False)

        lut = np.zeros(max(max_id, int(segmentation.max(initial=0))) + 1, dtype="float32")
        lut[table.label_id.values.astype("int64")] = frequencies
        output = lut[segmentation]

        os.makedirs(out_folder, exist_ok=True)
        if colormap is None:
            write_volume(out_path, output, voxel_size, s)
            continue

        import matplotlib.pyplot as plt
        from matplotlib import colors

        norm = colors.Normalize(vmin=freq_range[0], vmax=freq_range[1], clip=True)
        cmap = plt.get_cmap(colormap)
        background = output == 0
        output = cmap(norm(output))
        output[background] = (0, 0, 0, 0)
        print("Writing", out_path)
        tifffile.imwrite(out_path, output, bigtiff=True, compression="zlib")
        _write_colorbar(os.path.join(out_folder, f"{out_name}.png"), norm, cmap, name)


def export_frequency_cochlea(
    cochlea: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    segmentation: str,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **kwargs,
) -> None:
    """Export the tonotopic frequency of the objects of a segmentation of a cochlea in a MoBIE project.

    The output is written to <output_folder>/<cochlea>, see `export_frequency`.

    Args:
        cochlea: Name of the cochlea.
        output_folder: Output folder.
        scale: Scale level(s).
        segmentation: Name of the segmentation with the tonotopic mapping, e.g. "SGN_v2".
        mobie_dir: Local MoBIE project directory. Ignored when s3 is set.
        s3: Read the data from the S3 bucket instead of the local MoBIE project.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        kwargs: Further arguments of `export_frequency`, e.g. components, colormap, crop_center, roi_halo, axis,
            suffix, voxel_size or force.
    """
    s3_kwargs = dict(
        s3=s3, s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name, s3_service_endpoint=s3_service_endpoint,
    )
    export_frequency(
        default_image_path(cochlea, segmentation, s3=s3, mobie_dir=mobie_dir),
        os.path.join(output_folder, cochlea),
        scale,
        cochlea_table(cochlea, segmentation, mobie_dir=mobie_dir, **s3_kwargs),
        input_key="s0",
        name=segmentation,
        **s3_kwargs,
        **kwargs,
    )
