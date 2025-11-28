import argparse
import os
from typing import List, Optional
import warnings

import numpy as np
import pandas as pd
import tifffile
import zarr
from elf.parallel import isin

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
from flamingo_tools.postprocessing.label_components import filter_cochlea_volume, filter_cochlea_volume_single
# from skimage.segmentation import relabel_sequential


def filter_component(fs, segmentation, cochlea, seg_name, components):
    # First, we download the MoBIE table for this segmentation.
    internal_path = os.path.join(BUCKET_NAME, cochlea, "tables",  seg_name, "default.tsv")
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
    sgn_components: Optional[List[int]] = None,
    ihc_components: Optional[List[int]] = None,
    ds_factor: int = 24,
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
        sgn_components: Component labels for filtering SGN segmentation table.
        ihc_components: Component labels for filtering IHC segmentation table.
        ds_factor: Down-sampling factor for filtering.
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
        internal_path = os.path.join(cochlea, "tables",  sgn_channel, "default.tsv")
        tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        with fs.open(tsv_path, "r") as f:
            table_sgn = pd.read_csv(f, sep="\t")

    if ihc_channel is not None:
        internal_path = os.path.join(cochlea, "tables",  ihc_channel, "default.tsv")
        tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        with fs.open(tsv_path, "r") as f:
            table_ihc = pd.read_csv(f, sep="\t")

    if sgn_channel is None:
        # filter based in IHC segmentation
        return filter_cochlea_volume_single(table_ihc, components=ihc_components,
                                            scale_factor=ds_factor, dilation_iterations=dilation_iterations)
    elif ihc_channel is None:
        # filter based on SGN segmentation
        return filter_cochlea_volume_single(table_sgn, components=sgn_components,
                                            scale_factor=ds_factor, dilation_iterations=dilation_iterations)
    else:
        # filter based on SGN and IHC segmentation with a specialized function
        return filter_cochlea_volume(table_sgn, table_ihc,
                                     sgn_components=sgn_components,
                                     ihc_components=ihc_components,
                                     scale_factor=ds_factor,
                                     dilation_iterations=dilation_iterations)


def upscale_volume(
    target_data: np.ndarray,
    downscaled_volume: np.ndarray,
    upscale_factor: int,
) -> np.ndarray:
    """Up-scale binary 3D mask to dimensions of target data.
    After an initial up-scaling, the dimensions are cropped or zero-padded to fit the target shape.

    Args:
        target_data: Reference data for up-scaling.
        downscaled_volume: Down-scaled binary 3D array.
        upscale_factor: Initial factor for up-scaling binary array.

    Returns:
        Resized binary array.
    """
    target_shape = target_data.shape
    upscaled_filter = np.repeat(
        np.repeat(
            np.repeat(downscaled_volume, upscale_factor, axis=0),
            upscale_factor, axis=1),
        upscale_factor, axis=2)
    resized = np.zeros(target_shape, dtype=target_data.dtype)
    min_x, min_y, min_z = tuple(min(upscaled_filter.shape[i], target_shape[i]) for i in range(3))
    resized[:min_x, :min_y, :min_z] = upscaled_filter[:min_x, :min_y, :min_z]
    return resized


def export_lower_resolution(args):
    # calculate single filter mask for all lower resolutions
    if args.filter_cochlea_channels is not None:
        ds_factor = 48
        filter_volume = filter_cochlea(args.cochlea, args.filter_cochlea_channels,
                                       sgn_components=args.filter_sgn_components,
                                       ihc_components=args.filter_ihc_components,
                                       dilation_iterations=args.filter_dilation_iterations, ds_factor=ds_factor)
        filter_volume = np.transpose(filter_volume, (2, 1, 0))

    # iterate through exporting lower resolutions
    for scale in args.scale:
        if args.filter_cochlea_channels is not None:
            output_folder = os.path.join(args.output_folder, args.cochlea,
                                         f"scale{scale}_dilation{args.filter_dilation_iterations}")
        else:
            output_folder = os.path.join(args.output_folder, args.cochlea, f"scale{scale}")
        os.makedirs(output_folder, exist_ok=True)

        input_key = f"s{scale}"
        for channel in args.channels:
            out_path = os.path.join(output_folder, f"{channel}.tif")
            if os.path.exists(out_path):
                continue

            print("Exporting channel", channel)
            internal_path = os.path.join(args.cochlea, "images",  "ome-zarr", f"{channel}.ome.zarr")
            s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
            with zarr.open(s3_store, mode="r") as f:
                data = f[input_key][:].astype("float32")
            print("Data shape", data.shape)
            if args.filter_by_components is not None:
                print(f"Filtering channel {channel} by components {args.filter_by_components}.")
                data = filter_component(fs, data, args.cochlea, channel, args.filter_by_components)
            if args.filter_cochlea_channels is not None:
                us_factor = ds_factor // (2 ** scale)
                upscaled_filter = upscale_volume(data, filter_volume, upscale_factor=us_factor)
                data[upscaled_filter == 0] = 0
                if "PV" in channel:
                    max_intensity = 1400
                    data[data > max_intensity] = 0
                if "CTBP2" in channel:
                    max_intensity = 1400
                    data[data > max_intensity] = 300

            if args.binarize:
                data = (data > 0).astype("uint16")

            if args.ome_zarr:
                out_path = os.path.join(output_folder, f"{channel}.ome.zarr")
                output_key = "image"
                f_out = zarr.open(out_path, mode="w")
                f_out.create_dataset(output_key, data=data, compression="gzip")
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
    args = parser.parse_args()

    export_lower_resolution(args)


if __name__ == "__main__":
    main()
