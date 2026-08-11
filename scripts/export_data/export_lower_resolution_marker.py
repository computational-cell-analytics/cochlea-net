import argparse
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.export_data_utils import compute_crop_bb, crop_suffix
from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
# from skimage.segmentation import relabel_sequential


def filter_marker_instances(cochlea, segmentation, seg_name, group=None):
    """Filter segmentation with marker labels.
    Positive segmentation instances are set to 1, negative to 2.
    """
    internal_path = os.path.join(cochlea, "tables", seg_name, "default.tsv")
    tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    with fs.open(tsv_path, "r") as f:
        table_seg = pd.read_csv(f, sep="\t")

    label_ids_positive = list(table_seg.loc[table_seg["marker_labels"] == 1, "label_id"])
    label_ids_negative = list(table_seg.loc[table_seg["marker_labels"] == 2, "label_id"])

    if group is None:
        label_ids_marker = label_ids_positive + label_ids_negative
        filter_mask = ~np.isin(segmentation, label_ids_marker)
        segmentation[filter_mask] = 0

        filter_mask = np.isin(segmentation, label_ids_positive)
        segmentation[filter_mask] = 1
        filter_mask = np.isin(segmentation, label_ids_negative)
        segmentation[filter_mask] = 2
    elif group == "positive":
        filter_mask = ~np.isin(segmentation, label_ids_positive)
        segmentation[filter_mask] = 0
        filter_mask = np.isin(segmentation, label_ids_positive)
        segmentation[filter_mask] = 1
    elif group == "negative":
        filter_mask = ~np.isin(segmentation, label_ids_negative)
        segmentation[filter_mask] = 0
        filter_mask = np.isin(segmentation, label_ids_negative)
        segmentation[filter_mask] = 2
    else:
        raise ValueError("Choose either 'positive' or 'negative' as group value.")

    segmentation = segmentation.astype("float32")
    return segmentation


def export_lower_resolution(
    cochlea: str,
    scale: List[int],
    output_folder: str,
    channels: List[str] = ["PV", "VGlut3", "CTBP2"],
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
):
    crop = crop_center is not None

    # iterate through exporting lower resolutions
    for s in scale:
        out_folder = os.path.join(output_folder, cochlea, f"scale{s}")
        os.makedirs(out_folder, exist_ok=True)

        for group in ["positive", "negative"]:

            input_key = f"s{s}"
            for channel in channels:

                if crop:
                    out_path = os.path.join(
                        out_folder, f"{channel}_marker_{group}{crop_suffix(crop_center, axis, suffix)}.tif"
                    )
                else:
                    out_path = os.path.join(out_folder, f"{channel}_marker_{group}.tif")
                if os.path.exists(out_path):
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
                print("Data shape", data.shape)

                print(f"Filtering {group} marker instances.")
                data = filter_marker_instances(cochlea, data, channel, group=group)
                tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--channels", nargs="+", type=str, default=["PV", "VGlut3", "CTBP2"])
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
