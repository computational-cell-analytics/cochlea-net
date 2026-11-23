"""Export synapse detections as a label volume or as a napari point table.
"""
import os
from typing import List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from skimage.morphology import ball
from tqdm import tqdm

from flamingo_tools.export_data_utils import (
    cochlea_table, crop_suffix, export_bounding_box, normalize_scales, normalize_voxel_size, write_volume,
)
from flamingo_tools.file_utils import open_scale_level
from flamingo_tools.s3_utils import MOBIE_FOLDER, default_image_path


def export_synapses(
    input_path: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    synapse_table: pd.DataFrame,
    input_key: Optional[str] = "s0",
    name: str = "synapses",
    ihc_table: Optional[pd.DataFrame] = None,
    max_dist: float = 3.0,
    radius: int = 3,
    id_offset: int = 0,
    filter_ihc_components: List[int] = [1],
    use_syn_ids: bool = False,
    points: bool = False,
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
    """Export synapse detections at lower resolutions.

    The output for each scale s is <output_folder>/scale<s>/<name>[_offset<id_offset>]<crop suffix>.tif, or
    .csv with `points`.

    Args:
        input_path: Path to a volume that sets the output shape, usually the IHC segmentation: a tif file, or a
            zarr/n5 container. Only its shape is read.
        output_folder: Output folder.
        scale: Scale level(s), see `flamingo_tools.file_utils.open_scale_level`.
        synapse_table: Synapse table with the columns "x", "y", "z" (in µm), "spot_id", "matched_ihc" and
            "distance_to_ihc".
        input_key: Key of the volume in the container. Ignored for a tif file.
        name: Name of the export.
        ihc_table: Optional IHC segmentation table. If given, only the synapses that are matched to an IHC of
            the components `filter_ihc_components` are exported.
        max_dist: Maximal distance of a synapse to its IHC, in µm.
        radius: Radius of a synapse in the output volume, in pixels. The footprint is a sphere in pixel space,
            so it is anisotropic for anisotropic data.
        id_offset: Offset added to the IDs in the output volume, for different colors in a visualization.
        filter_ihc_components: Component labels of the IHCs, see `ihc_table`.
        use_syn_ids: Write the synapse IDs to the output volume instead of the IDs of the matched IHCs.
        points: Write a CSV point table instead of the volume. The table has the columns axis-0, axis-1, axis-2
            (z, y, x in pixels at the target scale, relative to the crop), spot_id and matched_ihc.
        crop_center: Optional crop center as [x, y, z] in µm, see
            `flamingo_tools.export_data.lower_resolution.export_lower_resolution`.
        roi_halo: Halo around the crop center in pixels at the target scale.
        axis: Optional axis index into (x, y, z) to crop as a single-pixel 2D slice at the crop center.
        suffix: Optional label appended to the name of a crop.
        voxel_size: Voxel size of the volume in µm, scalar or (x, y, z).
        force: Replace an existing output. Otherwise, an existing output is kept.
        s3: Read the volume from the S3 bucket.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    voxel_size = normalize_voxel_size(voxel_size)
    syn_table = synapse_table[synapse_table.distance_to_ihc <= max_dist]
    if ihc_table is not None:
        valid_ihcs = ihc_table[ihc_table.component_labels.isin(filter_ihc_components)].label_id
        syn_table = syn_table[syn_table.matched_ihc.isin(valid_ihcs)]

    offset_suffix = f"_offset{id_offset}" if id_offset != 0 else ""
    out_suffix = "" if crop_center is None else crop_suffix(crop_center, axis, suffix)
    extension = ".csv" if points else ".tif"

    for s in normalize_scales(scale):
        out_folder = os.path.join(output_folder, f"scale{s}")
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, f"{name}{offset_suffix}{out_suffix}{extension}")
        if os.path.exists(out_path) and not force:
            print(f"Skipping {out_path}. File already exists.")
            continue

        shape = open_scale_level(
            input_path, input_key, s, from_s3=s3, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint,
            credential_file=s3_credentials,
        ).shape
        coords_px = syn_table[["z", "y", "x"]].values / np.array(voxel_size[::-1]) / (2 ** s)
        coordinates = np.round(coords_px, 0).astype("int")
        ihc_ids = syn_table["matched_ihc"].values
        syn_ids = syn_table["spot_id"].values

        if crop_center is not None:
            start, stop = export_bounding_box(shape, crop_center, roi_halo, voxel_size, s, axis)
            inside = ((coordinates >= start) & (coordinates < stop)).all(axis=1)
            coordinates = coordinates[inside] - start
            coords_px = coords_px[inside] - start
            ihc_ids = ihc_ids[inside]
            syn_ids = syn_ids[inside]
            shape = tuple(int(sto - sta) for sta, sto in zip(start, stop))

        if points:
            # The axis-N column names make napari's built-in CSV reader open the file as a points layer.
            point_table = pd.DataFrame(coords_px, columns=["axis-0", "axis-1", "axis-2"])
            point_table["spot_id"] = syn_ids
            point_table["matched_ihc"] = ihc_ids
            print("Writing synapse points to", out_path)
            point_table.to_csv(out_path, index=False)
            continue

        output = np.zeros(shape, dtype="float32")
        radius = int(radius)
        footprint_mask = ball(radius).astype(bool)
        ids = syn_ids if use_syn_ids else ihc_ids
        for coord, syn_id in tqdm(zip(coordinates, ids), total=len(coordinates), desc="Writing synapses to volume"):
            # The footprint is clipped to the volume, so that synapses close to the border and
            # synapses in a 2D slice (where one axis is a single pixel) are written as well.
            starts = [max(0, c - radius) for c in coord]
            stops = [min(sh, c + radius + 1) for c, sh in zip(coord, shape)]
            if any(sto <= sta for sta, sto in zip(starts, stops)):
                continue
            bb = tuple(slice(sta, sto) for sta, sto in zip(starts, stops))
            footprint = tuple(slice(sta - c + radius, sto - c + radius) for sta, sto, c in zip(starts, stops, coord))
            output[bb][footprint_mask[footprint]] = syn_id + id_offset

        write_volume(out_path, output, voxel_size, s)


def export_synapses_cochlea(
    cochlea: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    synapse_name: str = "synapse_v3_ihc_v11",
    reference_ihcs: str = "IHC_v11",
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **kwargs,
) -> None:
    """Export the synapse detections of a cochlea in a MoBIE project.

    The output is written to <output_folder>/<cochlea>, see `export_synapses`. The IHC segmentation sets the
    output shape, and its table filters the synapses by the IHC components.

    Args:
        cochlea: Name of the cochlea.
        output_folder: Output folder.
        scale: Scale level(s).
        synapse_name: Name of the synapse detection source.
        reference_ihcs: Name of the IHC segmentation that the synapses are matched to.
        mobie_dir: Local MoBIE project directory. Ignored when s3 is set.
        s3: Read the data from the S3 bucket instead of the local MoBIE project.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        kwargs: Further arguments of `export_synapses`, e.g. max_dist, radius, id_offset, filter_ihc_components,
            use_syn_ids, points, crop_center, roi_halo, axis, suffix, voxel_size or force.
    """
    s3_kwargs = dict(
        s3=s3, s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name, s3_service_endpoint=s3_service_endpoint,
    )
    export_synapses(
        default_image_path(cochlea, reference_ihcs, s3=s3, mobie_dir=mobie_dir),
        os.path.join(output_folder, cochlea),
        scale,
        cochlea_table(cochlea, synapse_name, mobie_dir=mobie_dir, **s3_kwargs),
        input_key="s0",
        name=synapse_name,
        ihc_table=cochlea_table(cochlea, reference_ihcs, mobie_dir=mobie_dir, **s3_kwargs),
        **s3_kwargs,
        **kwargs,
    )
