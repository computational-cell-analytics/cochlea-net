import os
import json

import numpy as np
import pandas as pd
import tifffile
import zarr
from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT


def extract_block_from_s3(args):
    os.makedirs(args.output_folder, exist_ok=True)

    resolution = 0.38 * (2 ** args.scale)
    center = json.loads(args.position)
    center = [int(ce / resolution) for ce in center[::-1]]

    for source in args.sources:
        print("Extracting source:", source, "from", args.cochlea)
        internal_path = os.path.join(args.cochlea, "images",  "ome-zarr", f"{source}.ome.zarr")
        s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

        input_key = f"s{args.scale}"
        with zarr.open(s3_store, mode="r") as f:
            ds = f[input_key]
            roi = tuple(slice(max(0, ce - ha), min(sh, ce + ha)) for ce, ha, sh in zip(center, args.halo, ds.shape))
            data = ds[roi]

        if args.component_ids is not None:
            table_path = os.path.join(BUCKET_NAME, args.cochlea, "tables",  source, "default.tsv")
            with fs.open(table_path, "r") as f:
                table = pd.read_csv(f, sep="\t")
            keep_ids = table[table.component_labels.isin(args.component_ids)].label_id.values
            mask = np.isin(data, keep_ids)
            data[~mask] = 0

        coord_string = "-".join([str(c).zfill(4) for c in center])
        out_path = os.path.join(args.output_folder, f"{args.cochlea}_{source}_scale{args.scale}_{coord_string}.tif")

        if args.as_float:
            data = data.astype("float32")
        tifffile.imwrite(out_path, data, compression="zlib")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--sources", "-s", required=True, nargs="+")
    parser.add_argument("--position", "-p", required=True)
    parser.add_argument("--halo", nargs="+", type=int, default=[32, 128, 128])
    parser.add_argument("--scale", type=int, default=0)
    parser.add_argument("--as_float", action="store_true")
    parser.add_argument("--component_ids", type=int, nargs="+")
    args = parser.parse_args()

    extract_block_from_s3(args)


if __name__ == "__main__":
    main()
