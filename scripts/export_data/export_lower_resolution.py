import argparse
import os
from typing import List, Optional, Sequence, Tuple
import warnings

import numpy as np
import pandas as pd
import tifffile
import zarr
from elf.parallel import isin

from flamingo_tools.export_data_utils import (
    compute_crop_bb, crop_filter_volume, crop_suffix, filter_volume_downscale_factors, normalize_voxel_size,
)
from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
from flamingo_tools.postprocessing.label_components import filter_cochlea_volume, filter_cochlea_volume_single
# from skimage.segmentation import relabel_sequential


def filter_component(fs, segmentation, cochlea, seg_name, components):
    # First, we download the MoBIE table for this segmentation.
    internal_path = os.path.join(BUCKET_NAME, cochlea, "tables", seg_name, "default.tsv")
    with fs.open(internal_path, "r") as f:
        table = pd.read_csv(f, sep="\t")

    # Then we get the ids for the components and us them to filter the segmentation.
    component_mask = np.isin(table.component_labels.values, components)
    keep_label_ids = table.label_id.values[component_mask].astype("int64")
    if max(keep_label_ids) > np.iinfo("uint16").max:
        warnings.warn(f"Label ID exceeds maximum of data type 'uint16': {np.iinfo('uint16').max}.")

    filter_mask = np.zeros(segmentation.shape, dtype="bool")
    filter_mask = ~isin(segmentation, keep_label_ids, out=filter_mask, verbose=True, block_shape=(128, 128, 128))
    segmentation[filter_mask] = 0
    segmentation = segmentation.astype("float32")
    return segmentation


def filter_cochlea(
    cochlea: str,
    filter_cochlea_channels: str,
    ds_factor: Sequence[int],
    voxel_size: Tuple[float, float, float],
    sgn_components: Optional[List[int]] = None,
    ihc_components: Optional[List[int]] = None,
    dilation_iterations: int = 8,
) -> np.ndarray:
    """Pre-process information for filtering cochlea volume based on segmentation table.
    Differentiates between the input of a single channel of either IHC or SGN or if both are supplied.
    If a single channel is given, the filtered volume contains
    a down-sampled segmentation area, which has been dilated.
    If both IHC and SGN segmentation are supplied, a more specialized dilation
    is applied to ensure that the connecting volume is not filtered.

    Args:
        cochlea: Name of cochlea.
        filter_cochlea_channels: Segmentation table(s) used for filtering.
        ds_factor: Down-sampling factor for filtering, in pixel per axis, in (x, y, z) order.
        voxel_size: Voxel size of the data in micrometer, in (x, y, z) order.
        sgn_components: Component labels for filtering SGN segmentation table.
        ihc_components: Component labels for filtering IHC segmentation table.
        dilation_iterations: Iterations for dilating binary segmentation mask.

    Returns:
        Binary 3D array of filtered cochlea
    """
    # we check if the supplied channels contain an SGN and IHC channel
    sgn_channels = [ch for ch in filter_cochlea_channels if "SGN" in ch]
    sgn_channel = None if len(sgn_channels) == 0 else sgn_channels[0]

    ihc_channels = [ch for ch in filter_cochlea_channels if "IHC" in ch]
    ihc_channel = None if len(ihc_channels) == 0 else ihc_channels[0]

    if ihc_channel is None and sgn_channel is None:
        raise ValueError("Channels supplied for filtering cochlea volume do not contain an IHC or SGN segmentation.")

    if sgn_channel is not None:
        internal_path = os.path.join(cochlea, "tables", sgn_channel, "default.tsv")
        tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        with fs.open(tsv_path, "r") as f:
            table_sgn = pd.read_csv(f, sep="\t")

    if ihc_channel is not None:
        internal_path = os.path.join(cochlea, "tables", ihc_channel, "default.tsv")
        tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        with fs.open(tsv_path, "r") as f:
            table_ihc = pd.read_csv(f, sep="\t")

    if sgn_channel is None:
        # filter based in IHC segmentation
        return filter_cochlea_volume_single(table_ihc, components=ihc_components, voxel_size=voxel_size,
                                            scale_factor=ds_factor, dilation_iterations=dilation_iterations)
    elif ihc_channel is None:
        # filter based on SGN segmentation
        return filter_cochlea_volume_single(table_sgn, components=sgn_components, voxel_size=voxel_size,
                                            scale_factor=ds_factor, dilation_iterations=dilation_iterations)
    else:
        # filter based on SGN and IHC segmentation with a specialized function
        return filter_cochlea_volume(table_sgn, table_ihc,
                                     sgn_components=sgn_components,
                                     ihc_components=ihc_components,
                                     scale_factor=ds_factor,
                                     voxel_size=voxel_size,
                                     dilation_iterations=dilation_iterations)


def export_lower_resolution(
    cochlea: str,
    scale: List[int],
    output_folder: str,
    channels: List[str] = ["PV", "VGlut3", "CTBP2"],
    filter_by_components: Optional[List[int]] = None,
    filter_sgn_components: List[int] = [1],
    filter_ihc_components: List[int] = [1],
    binarize: bool = False,
    filter_cochlea_channels: Optional[List[str]] = None,
    filter_dilation_iterations: int = 8,
    ome_zarr: bool = False,
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
):
    crop = crop_center is not None
    voxel_size = normalize_voxel_size(voxel_size)

    # calculate single filter mask for all lower resolutions
    if filter_cochlea_channels is not None:
        ds_factor = filter_volume_downscale_factors(voxel_size)
        filter_volume = filter_cochlea(cochlea, filter_cochlea_channels,
                                       ds_factor=ds_factor, voxel_size=voxel_size,
                                       sgn_components=filter_sgn_components,
                                       ihc_components=filter_ihc_components,
                                       dilation_iterations=filter_dilation_iterations)
        filter_volume = np.transpose(filter_volume, (2, 1, 0))
        ds_factor_zyx = np.array(ds_factor[::-1], dtype="float64")

    # iterate through exporting lower resolutions
    for s in scale:
        if filter_cochlea_channels is not None:
            out_folder = os.path.join(output_folder, cochlea, f"scale{s}_dilation{filter_dilation_iterations}")
        else:
            out_folder = os.path.join(output_folder, cochlea, f"scale{s}")
        os.makedirs(out_folder, exist_ok=True)

        input_key = f"s{s}"
        for channel in channels:
            if crop:
                out_path = os.path.join(out_folder, f"{channel}{crop_suffix(crop_center, axis, suffix)}.tif")
            else:
                out_path = os.path.join(out_folder, f"{channel}.tif")
            if os.path.exists(out_path):
                print(f"Skipping {out_path}. File already exists.")
                continue

            print("Exporting channel", channel)
            internal_path = os.path.join(cochlea, "images", "ome-zarr", f"{channel}.ome.zarr")
            s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
            f = zarr.open(s3_store, mode="r")
            if crop:
                start, stop = compute_crop_bb(
                    crop_center, roi_halo, voxel_size=voxel_size, scale=s, shape=f[input_key].shape, axis=axis,
                )
                data = f[input_key][start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]].astype("float32")
            else:
                data = f[input_key][:].astype("float32")
                start, stop = np.zeros(3, dtype=int), np.array(data.shape, dtype=int)
            print("Data shape", data.shape)
            if filter_by_components is not None:
                print(f"Filtering channel {channel} by components {filter_by_components}.")
                data = filter_component(fs, data, cochlea, channel, filter_by_components)
            if filter_cochlea_channels is not None:
                us_factor = ds_factor_zyx / (2 ** s)
                applied_filter = crop_filter_volume(filter_volume, start, stop, us_factor)
                data[applied_filter == 0] = 0

            # filtering of bright outliers
            if "PV" in channel and "LaVision" not in cochlea:
                max_intensity = 1400
                data[data > max_intensity] = max_intensity
            if "CTBP2" in channel:
                max_intensity = 1400
                data[data > max_intensity] = max_intensity

            if binarize:
                data = (data > 0).astype("uint16")

            if ome_zarr:
                out_path = os.path.join(out_folder, f"{channel}.ome.zarr")
                output_key = "image"
                f_out = zarr.open(out_path, mode="w")
                f_out.create_array(output_key, data=data, compressors=zarr.codecs.GzipCodec())
            else:
                tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--channels", nargs="+", type=str, default=["PV", "VGlut3", "CTBP2"])
    parser.add_argument("--filter_by_components", nargs="+", type=int, default=None)
    parser.add_argument("--filter_sgn_components", nargs="+", type=int, default=[1])
    parser.add_argument("--filter_ihc_components", nargs="+", type=int, default=[1])
    parser.add_argument("--binarize", action="store_true")
    parser.add_argument("--filter_cochlea_channels", nargs="+", type=str, default=None)
    parser.add_argument("--filter_dilation_iterations", type=int, default=8)
    parser.add_argument("--ome_zarr", action="store_true")
    parser.add_argument("--crop_center", nargs=3, type=float, default=None,
                        help="Crop center as x y z in µm. Requires --roi_halo.")
    parser.add_argument("--roi_halo", nargs=3, type=int, default=None,
                        help="Halo around the crop center as halo_x halo_y halo_z in pixels at the target scale. "
                        "Optional when --axis is given: the whole plane is cropped in that case.")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None,
                        help="Axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at the crop center. "
                        "Requires --crop_center.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Extra label appended to the output filename after the crop/axis suffix, "
                        "e.g. a position name such as 'apex'.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    args = parser.parse_args()
    if args.crop_center is not None:
        if args.roi_halo is None and args.axis is None:
            parser.error("--crop_center requires --roi_halo, unless --axis is also given "
                         "(the whole plane is cropped in that case).")
    elif args.axis is not None:
        parser.error("--axis requires --crop_center.")

    export_lower_resolution(**vars(args))


if __name__ == "__main__":
    main()
