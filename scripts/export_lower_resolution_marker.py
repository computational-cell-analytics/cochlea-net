import argparse
import os

import numpy as np
import pandas as pd
import tifffile
import zarr

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


def export_lower_resolution(args):

    # iterate through exporting lower resolutions
    for scale in args.scale:
        output_folder = os.path.join(args.output_folder, args.cochlea, f"scale{scale}")
        os.makedirs(output_folder, exist_ok=True)

        for group in ["positive", "negative"]:

            input_key = f"s{scale}"
            for channel in args.channels:

                out_path = os.path.join(output_folder, f"{channel}_marker_{group}.tif")
                if os.path.exists(out_path):
                    continue

                print("Exporting channel", channel)
                internal_path = os.path.join(args.cochlea, "images", "ome-zarr", f"{channel}.ome.zarr")
                s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
                with zarr.open(s3_store, mode="r") as f:
                    data = f[input_key][:].astype("float32")
                print("Data shape", data.shape)

                print(f"Filtering {group} marker instances.")
                data = filter_marker_instances(args.cochlea, data, channel, group=group)
                tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--channels", nargs="+", type=str, default=["PV", "VGlut3", "CTBP2"])
    args = parser.parse_args()

    export_lower_resolution(args)


if __name__ == "__main__":
    main()
