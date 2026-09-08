"""Edit OTOF positive and negative IHC assignments with data from S3."""

import argparse
import os
from typing import Optional

import numpy as np
import pandas as pd
import zarr

import otof_label_editor as _shared


BASE_VOXEL_SIZE_XYZ = _shared.BASE_VOXEL_SIZE_XYZ
MARKER_COLUMN = _shared.MARKER_COLUMN
NEGATIVE = _shared.NEGATIVE
OTOF_NAME = _shared.OTOF_NAME
POSITIVE = _shared.POSITIVE
SEGMENTATION_NAME = _shared.SEGMENTATION_NAME
EditorData = _shared.EditorData
_label_roi = _shared._label_roi
_update_marker_mask = _shared._update_marker_mask
assignment_for_label = _shared.assignment_for_label
build_editable_layers = _shared.build_editable_layers
export_table = _shared.export_table
mask_signal_to_ihcs = _shared.mask_signal_to_ihcs
prepare_editor_data = _shared.prepare_editor_data
run_editor = _shared.run_editor
switch_assignment = _shared.switch_assignment
validate_table = _shared.validate_table
validate_volumes = _shared.validate_volumes


DEFAULT_COCHLEA = "M_AMD_OTOF27_L"
DEFAULT_SCALE = 3


def _read_scale(group, scale_key: str, source_name: str) -> np.ndarray:
    if scale_key not in group:
        available = sorted(group.keys())
        raise ValueError(f"Source '{source_name}' has no scale '{scale_key}'. Available scales: {available}.")
    return group[scale_key][:]


def load_editor_data(cochlea: str, scale: int, masking_radius: Optional[float] = None) -> EditorData:
    """Load the downsampled volumes and the source table from S3."""
    if scale < 0:
        raise ValueError("The pyramid scale must be zero or greater.")

    from flamingo_tools.s3_utils import BUCKET_NAME, SERVICE_ENDPOINT, get_s3_path

    scale_key = f"s{scale}"
    print(f"Loading {cochlea}/{SEGMENTATION_NAME} at {scale_key} ...")
    segmentation_path = os.path.join(cochlea, "images", "ome-zarr", f"{SEGMENTATION_NAME}.ome.zarr")
    segmentation_store, fs = get_s3_path(
        segmentation_path,
        bucket_name=BUCKET_NAME,
        service_endpoint=SERVICE_ENDPOINT,
    )
    segmentation_group = zarr.open(segmentation_store, mode="r")
    segmentation = _read_scale(segmentation_group, scale_key, SEGMENTATION_NAME)

    print(f"Loading {cochlea}/{OTOF_NAME} at {scale_key} ...")
    otof_path = os.path.join(cochlea, "images", "ome-zarr", f"{OTOF_NAME}.ome.zarr")
    otof_store, _ = get_s3_path(otof_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    otof_group = zarr.open(otof_store, mode="r")
    otof = _read_scale(otof_group, scale_key, OTOF_NAME)

    table_path = os.path.join(cochlea, "tables", SEGMENTATION_NAME, "default.tsv")
    resolved_table_path, _ = get_s3_path(table_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    print(f"Loading {table_path} ...")
    with fs.open(resolved_table_path, "r") as table_file:
        table = pd.read_csv(table_file, sep="\t")

    scale_zyx = tuple(value * (2 ** scale) for value in BASE_VOXEL_SIZE_XYZ[::-1])
    return prepare_editor_data(
        table,
        segmentation,
        otof,
        scale_zyx,
        scale_key,
        masking_radius=masking_radius,
    )


def default_output_path(cochlea: str) -> str:
    return f"{cochlea}_{SEGMENTATION_NAME}_default.tsv"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit positive and negative OTOF assignments in a downsampled IHC segmentation."
    )
    parser.add_argument(
        "--cochlea",
        "-c",
        default=DEFAULT_COCHLEA,
        help=f"Cochlea to load from S3. Default: {DEFAULT_COCHLEA}",
    )
    parser.add_argument(
        "--scale",
        "-s",
        type=int,
        default=DEFAULT_SCALE,
        help=f"OME-Zarr pyramid level to load. Default: {DEFAULT_SCALE}",
    )
    parser.add_argument(
        "--output-table",
        "-o",
        default=None,
        help="Local TSV path for the edited table. Default: <cochlea>_IHC_v11_default.tsv",
    )
    parser.add_argument(
        "--masking",
        dest="masking_radius",
        type=float,
        default=None,
        metavar="RADIUS_UM",
        help="Keep OTOF signal within this radius of assigned IHCs, in micrometers. Default: no masking.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_table = args.output_table or default_output_path(args.cochlea)
    data = load_editor_data(args.cochlea, args.scale, masking_radius=args.masking_radius)
    run_editor(args.cochlea, output_table, data)


if __name__ == "__main__":
    main()
