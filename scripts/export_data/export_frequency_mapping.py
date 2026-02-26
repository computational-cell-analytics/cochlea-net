import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
import zarr
from matplotlib import cm, colors

from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, create_s3_target, get_s3_path
# from tqdm import tqdm


def export_frequency_mapping(cochlea, scale, output_folder, source_name, colormap=None):
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]

    # Load the seg table and filter the compartments.
    source = sources[source_name]["segmentation"]
    rel_path = source["tableData"]["tsv"]["relativePath"]
    table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
    table = pd.read_csv(table_content, sep="\t")
    max_id = int(table.label_id.values.max())
    table = table[table.component_labels == 1]

    # Determine the frequency range.
    frequencies = table["frequency[kHz]"].values
    freq_range = (float(frequencies.min()), float(frequencies.max()))

    # Load the segmentation.
    seg_path = os.path.join(cochlea, source["imageData"]["ome.zarr"]["relativePath"])
    s3_store, _ = get_s3_path(seg_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    input_key = f"s{scale}"
    f = zarr.open(s3_store, mode="r")
    seg = f[input_key][:]

    mapping = {int(seg_id): freq for seg_id, freq in zip(table.label_id, frequencies)}
    lut = np.zeros(max_id + 1, dtype="float32")
    for k, v in mapping.items():
        lut[k] = v

    print("Creating output ...")
    output = lut[seg]
    if colormap is not None:
        norm = colors.Normalize(vmin=freq_range[0], vmax=freq_range[1], clip=True)
        cmap = plt.get_cmap(colormap)
        mask = output == 0
        output = cmap(norm(output))
        output[mask] = (0, 0, 0, 0)

    # Write the output.
    out_folder = os.path.join(output_folder, cochlea, f"scale{scale}")
    os.makedirs(out_folder, exist_ok=True)

    if colormap is None:
        out_path = os.path.join(out_folder, f"frequencies_{source_name}.tif")
    else:
        out_path = os.path.join(out_folder, f"frequencies_{source_name}_{colormap}.tif")

    print("Writing output to", out_path)
    tifffile.imwrite(out_path, output, bigtiff=True, compression="zlib")

    print("Frequency range:")
    print(freq_range[0], "-", freq_range[1], "kHz")

    if colormap is not None:
        fig, ax = plt.subplots(figsize=(6, 1.3))
        fig.subplots_adjust(bottom=0.5)
        cb = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal")
        cb.set_label("Frequency [kHz]")
        plt.title(f"Tonotopic Mapping: {source_name}")
        plt.tight_layout()
        out_path = os.path.join(out_folder, f"frequencies_{source_name}_{colormap}.png")
        plt.savefig(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--source_name", "-n", required=True)
    parser.add_argument("--colormap")
    args = parser.parse_args()

    export_frequency_mapping(args.cochlea, args.scale, args.output_folder, args.source_name, args.colormap)


if __name__ == "__main__":
    main()
