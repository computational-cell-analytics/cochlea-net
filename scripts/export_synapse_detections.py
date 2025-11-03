import argparse
import json
import os
from typing import List

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, create_s3_target, get_s3_path
from skimage.morphology import ball
from tqdm import tqdm


def export_synapse_detections(
    cochlea: str,
    scales: List[int],
    output_folder: str,
    synapse_name: str,
    reference_ihcs: str,
    max_dist: float,
    radius: float,
    id_offset: int,
    filter_ihc_components: List[int],
    position: str,
    halo: List[int],
    as_float: bool = False,
    use_syn_ids: bool = False,
):
    """Export synapse detections from S3..

    Args:
        cochlea: Cochlea name on S3 bucket.
        scales: Scale for export of lower resolution.
        output_folder: The output folder for saving the exported data.
        synapse_name: The name of the synapse detection source.
        reference_ihcs: Name of IHC segmentation.
        max_dist: Maximal distance of synapse to IHC segmentation.
        radius: The radius for writing the synapse points to the output volume.
        id_offset: Offset of label id of synapse output to have different colours for visualization.
        filter_ihc_components: Component label(s) for filtering IHC segmentation.
        position: Optional position for extracting a crop from the data. Requires to also pass halo.
        halo: Halo for extracting a crop from the data.
        as_float: Whether to save the exported data as floating point values.
        use_syn_ids: Whether to write the synapse IDs or the matched IHC IDs to the output volume.
    """
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]

    # Load the synapse table.
    syn = sources[synapse_name]["spots"]
    rel_path = syn["tableData"]["tsv"]["relativePath"]
    table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")

    syn_table = pd.read_csv(table_content, sep="\t")
    syn_table = syn_table[syn_table.distance_to_ihc <= max_dist]

    # Get the reference segmentation info.
    reference_seg_info = sources[reference_ihcs]["segmentation"]

    # Get the segmentation table.
    rel_path = reference_seg_info["tableData"]["tsv"]["relativePath"]
    seg_table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
    seg_table = pd.read_csv(seg_table_content, sep="\t")

    # Only keep synapses that match to segmented IHCs of the main component.
    valid_ihcs = seg_table[seg_table.component_labels.isin(filter_ihc_components)].label_id
    syn_table = syn_table[syn_table.matched_ihc.isin(valid_ihcs)]

    for scale in scales:
        # Get the reference shape at the given scale level.
        seg_path = os.path.join(cochlea, reference_seg_info["imageData"]["ome.zarr"]["relativePath"])
        s3_store, _ = get_s3_path(seg_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        input_key = f"s{scale}"
        with zarr.open(s3_store, mode="r") as f:
            shape = f[input_key].shape

        # Scale the coordinates according to the scale level.
        resolution = 0.38
        coordinates = syn_table[["z", "y", "x"]].values
        coordinates /= resolution
        coordinates /= (2 ** scale)
        coordinates = np.round(coordinates, 0).astype("int")

        ihc_ids = syn_table["matched_ihc"].values
        syn_ids = syn_table["spot_id"].values

        if position is not None:
            assert halo is not None
            center = json.loads(position)
            assert len(halo) == len(center)
            center = [int(ce / (resolution * (2 ** scale))) for ce in center[::-1]]
            start = np.array([max(0, ce - ha) for ce, ha in zip(center, halo)])[None]
            stop = np.array([min(sh, ce + ha) for ce, ha, sh in zip(center, halo, shape)])[None]

            mask = ((coordinates >= start) & (coordinates < stop)).all(axis=1)
            coordinates = coordinates[mask]
            coordinates -= start

            ihc_ids = ihc_ids[mask]
            syn_ids = syn_ids[mask]

            shape = tuple(int(sto - sta) for sta, sto in zip(start.squeeze(), stop.squeeze()))

        # Create the output.
        output = np.zeros(shape, dtype="uint16")
        mask = ball(radius).astype(bool)

        ids = syn_ids if use_syn_ids else ihc_ids

        for coord, syn_id in tqdm(
            zip(coordinates, ids), total=len(coordinates), desc="Writing synapses to volume"
        ):
            bb = tuple(slice(c - radius, c + radius + 1) for c in coord)
            try:
                output[bb][mask] = syn_id + id_offset
            except IndexError:
                print("Index error for", coord)
                continue

        # Write the output.
        out_folder = os.path.join(output_folder, cochlea, f"scale{scale}")
        os.makedirs(out_folder, exist_ok=True)
        if id_offset != 0:
            out_path = os.path.join(out_folder, f"{synapse_name}_offset{id_offset}.tif")
        else:
            out_path = os.path.join(out_folder, f"{synapse_name}.tif")

        if as_float:
            output = output.astype("float32")

        print("Writing synapses to", out_path)
        tifffile.imwrite(out_path, output, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--scale", "-s", nargs="+", type=int, required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    parser.add_argument("--synapse_name", default="synapse_v3_ihc_v4b")
    parser.add_argument("--reference_ihcs", default="IHC_v4b")
    parser.add_argument("--max_dist", type=float, default=3.0)
    parser.add_argument("--radius", type=int, default=3)
    parser.add_argument("--id_offset", type=int, default=0)
    parser.add_argument("--filter_ihc_components", nargs="+", type=int, default=[1])
    parser.add_argument("--position", default=None)
    parser.add_argument("--halo", default=None, nargs="+", type=int)
    parser.add_argument("--as_float", action="store_true")
    parser.add_argument("--use_syn_ids", action="store_true")
    args = parser.parse_args()

    export_synapse_detections(
        args.cochlea, args.scale, args.output_folder,
        args.synapse_name, args.reference_ihcs,
        args.max_dist, args.radius,
        args.id_offset, args.filter_ihc_components,
        position=args.position, halo=args.halo,
        as_float=args.as_float, use_syn_ids=args.use_syn_ids,
    )


if __name__ == "__main__":
    main()
