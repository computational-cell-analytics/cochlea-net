import argparse
import json
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.export_data_utils import compute_crop_bb, crop_suffix, normalize_voxel_size
from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, create_s3_target, get_s3_path
from skimage.morphology import ball
from tqdm import tqdm


def export_synapse_detections(
    cochlea: str,
    scales: List[int],
    output_folder: str,
    synapse_name: str = "synapse_v3_ihc_v4b",
    reference_ihcs: str = "IHC_v4b",
    max_dist: float = 3.0,
    radius: float = 3,
    id_offset: int = 0,
    filter_ihc_components: List[int] = [1],
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    use_syn_ids: bool = False,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    points: bool = False,
):
    """Export synapse detections from S3.

    Args:
        cochlea: Cochlea name on S3 bucket.
        scales: Scale for export of lower resolution.
        output_folder: The output folder for saving the exported data.
        synapse_name: The name of the synapse detection source.
        reference_ihcs: Name of IHC segmentation.
        max_dist: Maximal distance of synapse to IHC segmentation.
        radius: The radius in pixel for writing the synapse points to the output volume. The
            footprint is a sphere in pixel space, so it is anisotropic for anisotropic data.
        id_offset: Offset of label id of synapse output to have different colours for visualization.
        filter_ihc_components: Component label(s) for filtering IHC segmentation.
        crop_center: Optional crop center as [x, y, z] in µm. Requires roi_halo, unless axis is given.
        roi_halo: Halo around the crop center as [halo_x, halo_y, halo_z] in pixels at the target scale.
            Optional when axis is given: the whole plane is cropped in that case.
        axis: Optional axis index into (x, y, z) to crop as a single-pixel 2D slice at the
            crop center. Requires crop_center.
        suffix: Optional extra label appended to the output filename after the crop/axis suffix,
            e.g. a position name such as "apex".
        use_syn_ids: Whether to write the synapse IDs or the matched IHC IDs to the output volume.
        voxel_size: Voxel size of the data in micrometer, in (x, y, z) order.
        points: Whether to write a CSV point table per scale instead of the TIFF volume.
            The table has the columns axis-0, axis-1, axis-2 (z, y, x in pixels at the target scale,
            relative to the crop), spot_id and matched_ihc.
    """
    voxel_size = normalize_voxel_size(voxel_size)
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
        f = zarr.open(s3_store, mode="r")
        shape = f[input_key].shape

        # Scale the coordinates according to the scale level.
        coords_px = syn_table[["z", "y", "x"]].values / np.array(voxel_size[::-1]) / (2 ** scale)
        coordinates = np.round(coords_px, 0).astype("int")

        ihc_ids = syn_table["matched_ihc"].values
        syn_ids = syn_table["spot_id"].values

        if crop_center is not None:
            out_suffix = crop_suffix(crop_center, axis, suffix)
            start, stop = compute_crop_bb(
                crop_center, roi_halo, voxel_size=voxel_size, scale=scale, shape=shape, axis=axis
            )

            mask = ((coordinates >= start) & (coordinates < stop)).all(axis=1)
            coordinates = coordinates[mask] - start
            coords_px = coords_px[mask] - start

            ihc_ids = ihc_ids[mask]
            syn_ids = syn_ids[mask]

            shape = tuple(int(sto - sta) for sta, sto in zip(start, stop))
        else:
            out_suffix = ""

        out_folder = os.path.join(output_folder, cochlea, f"scale{scale}")
        os.makedirs(out_folder, exist_ok=True)
        offset_suffix = f"_offset{id_offset}" if id_offset != 0 else ""
        out_stem = os.path.join(out_folder, f"{synapse_name}{offset_suffix}{out_suffix}")

        if points:
            # The axis-N column names make napari's built-in CSV reader open the file as a points layer.
            point_table = pd.DataFrame(coords_px, columns=["axis-0", "axis-1", "axis-2"])
            point_table["spot_id"] = syn_ids
            point_table["matched_ihc"] = ihc_ids
            print("Writing synapse points to", f"{out_stem}.csv")
            point_table.to_csv(f"{out_stem}.csv", index=False)
            continue

        # Create the output.
        output = np.zeros(shape, dtype="float32")
        radius = int(radius)
        mask = ball(radius).astype(bool)

        ids = syn_ids if use_syn_ids else ihc_ids

        for coord, syn_id in tqdm(
            zip(coordinates, ids), total=len(coordinates), desc="Writing synapses to volume"
        ):
            # The footprint is clipped to the volume, so that synapses close to the border and
            # synapses in a 2D slice (where one axis is a single pixel) are written as well.
            starts = [max(0, c - radius) for c in coord]
            stops = [min(sh, c + radius + 1) for c, sh in zip(coord, shape)]
            if any(sto <= sta for sta, sto in zip(starts, stops)):
                continue
            bb = tuple(slice(sta, sto) for sta, sto in zip(starts, stops))
            footprint = tuple(
                slice(sta - c + radius, sto - c + radius) for sta, sto, c in zip(starts, stops, coord)
            )
            output[bb][mask[footprint]] = syn_id + id_offset

        print("Writing synapses to", f"{out_stem}.tif")
        tifffile.imwrite(f"{out_stem}.tif", output, bigtiff=True, compression="zlib")


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
    parser.add_argument("--use_syn_ids", action="store_true")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    parser.add_argument("--points", action="store_true",
                        help="Write a CSV point table (z, y, x pixels at each scale) instead of the TIFF volume. "
                        "napari opens it as a points layer.")
    args = parser.parse_args()
    if args.crop_center is not None:
        if args.roi_halo is None and args.axis is None:
            parser.error("--crop_center requires --roi_halo, unless --axis is also given "
                         "(the whole plane is cropped in that case).")
    elif args.axis is not None:
        parser.error("--axis requires --crop_center.")

    export_synapse_detections(
        args.cochlea, args.scale, args.output_folder,
        args.synapse_name, args.reference_ihcs,
        args.max_dist, args.radius,
        args.id_offset, args.filter_ihc_components,
        crop_center=args.crop_center, roi_halo=args.roi_halo, axis=args.axis, suffix=args.suffix,
        use_syn_ids=args.use_syn_ids, voxel_size=args.voxel_size, points=args.points,
    )


if __name__ == "__main__":
    main()
