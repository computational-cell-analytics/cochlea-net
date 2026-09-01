"""Scale an old synapse detection table from pixel coordinates to micrometer.

Tables written before commit e666ab3 ("Synapse coordinates in physical units", 2026-01-07) store
x, y and z as pixel indices. The same commit taught `map_and_filter_detections` to divide by the
voxel size, so an old table is scaled down a second time. Its coordinates then land about 2.6x
outside the volume, no IHC is found nearby, and the distance filter drops almost every detection.

This script applies the missing multiplication. It writes a new file and never modifies the input.

Example:
    python scripts/rescale_synapse_detections.py -i M_LR_000226_L_synapses_v3.tsv
"""
import argparse
import os
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

COORDINATE_COLUMNS = ("x", "y", "z")

# Columns that only a filtered table carries. Their unit depends on when the file was written,
# so this script leaves them alone instead of guessing.
FILTER_COLUMNS = ("matched_ihc", "distance_to_ihc")


def _normalize_voxel_size(voxel_size: Union[float, Sequence[float]]) -> Tuple[float, float, float]:
    """Expand a scalar or single-element voxel size to an (x, y, z) triple."""
    if isinstance(voxel_size, (int, float)):
        return (float(voxel_size),) * 3
    voxel_size = tuple(float(vs) for vs in voxel_size)
    if len(voxel_size) == 1:
        return voxel_size * 3
    if len(voxel_size) != 3:
        raise ValueError(f"Expected 1 or 3 values for the voxel size, got {len(voxel_size)}.")
    return voxel_size


def _bounding_box(table: pd.DataFrame) -> str:
    return ", ".join(
        f"{col}: {table[col].min():.2f} - {table[col].max():.2f}" for col in COORDINATE_COLUMNS
    )


def main(
    input_path: str,
    output_path: Optional[str] = None,
    voxel_size: Union[float, Sequence[float]] = (0.38, 0.38, 0.38),
    force: bool = False,
) -> str:
    """Scale the coordinates of an old synapse detection table to micrometer.

    Args:
        input_path: Input path to a synapse detection table in TSV format.
        output_path: Output path for the scaled table. Defaults to '<basename>_um.tsv'
            next to the input.
        voxel_size: The voxel size of the data in micrometer, in (x, y, z) order.
        force: Scale the table even if its coordinates already look like micrometer.

    Returns:
        The path of the scaled table.
    """
    voxel_size = _normalize_voxel_size(voxel_size)
    table = pd.read_csv(input_path, sep="\t")

    missing = [col for col in COORDINATE_COLUMNS if col not in table.columns]
    if missing:
        raise ValueError(f"{input_path} has no {', '.join(missing)} column(s). Not a detection table?")

    # Old tables hold integer pixel indices. Current tables hold sub-voxel floats, because the
    # peak detection applies a flow correction. Integer coordinates are the reliable marker of
    # the old format.
    is_pixel = all(
        np.allclose(table[col].to_numpy(), np.round(table[col].to_numpy())) for col in COORDINATE_COLUMNS
    )
    if not is_pixel and not force:
        raise ValueError(
            f"The coordinates in {input_path} are not integers, so the table is already in "
            "micrometer. Scaling it again would be wrong. Use --force to override."
        )

    present_filter_columns = [col for col in FILTER_COLUMNS if col in table.columns]
    if present_filter_columns:
        print(
            f"WARNING: {input_path} contains {', '.join(present_filter_columns)}. These columns are "
            "left unchanged, because their unit depends on the version that wrote them. Re-run "
            "flamingo_tools.segmentation.synapse_detection.map_and_filter_detections on the scaled "
            "table to recompute them."
        )

    print(f"Read {len(table)} detections from {input_path}")
    print(f"  bounding box in pixel:      {_bounding_box(table)}")

    for col, vs in zip(COORDINATE_COLUMNS, voxel_size):
        table[col] = table[col] * vs

    print(f"  bounding box in micrometer: {_bounding_box(table)}")

    if output_path is None:
        base, ext = os.path.splitext(os.path.abspath(input_path))
        output_path = f"{base}_um{ext}"
    table.to_csv(output_path, index=False, sep="\t")
    print(f"Wrote the scaled table to {output_path}")

    return output_path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for scaling an old synapse detection table from pixel to micrometer.")

    parser.add_argument("-i", "--input", required=True, type=str, help="Input synapse detection TSV file.")
    parser.add_argument("-o", "--output", type=str,
                        help="Output TSV file. Default is '<basename>_um.tsv' next to the input.")

    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer, in (x, y, z) order. Default: 0.38 0.38 0.38")
    parser.add_argument("--force", action="store_true",
                        help="Scale the table even if its coordinates already look like micrometer.")

    args = parser.parse_args()

    main(args.input, args.output, args.voxel_size, args.force)
