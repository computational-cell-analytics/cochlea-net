import warnings
from typing import Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull

from flamingo_tools.analysis.seg_table_utils import filter_table

REFERENCE_PRESETS = {
    "apex": 0.15,
    "mid": 0.5,
    "base": 0.85,
}

# Maps the slice axis to the two axes of the projection plane used for area computation.
_AXIS_PROJECTION = {
    "z": ("x", "y"),
    "y": ("x", "z"),
    "x": ("y", "z"),
}


def sgn_density_at_position(
    table: pd.DataFrame,
    reference_position: Union[float, str] = "mid",
    slice_thickness: float = 40.0,
    run_length_tolerance: float = 0.1,
    component_label: int = 1,
    axis: str = "z",
    length_fraction_column: str = "length_fraction",
) -> dict:
    """Calculate SGN density at a specific cochlear position using a planar slice.

    Selects all SGN instances whose bounding boxes overlap a slice of given thickness
    centered on the reference position, then removes instances whose run-length fraction
    differs too much from the reference (handles oblique cochlear orientation). Density
    is computed as count / convex-hull area of the selected anchor points projected onto
    the plane perpendicular to `axis`.

    Args:
        table: SGN segmentation table with columns anchor_{x,y,z}, bb_min/max_{x,y,z},
               component_labels, label_id, and the length_fraction column. Coordinates
               must be in physical units (µm). The table is expected to have been through
               tonotopic mapping so that the length_fraction column is present.
        reference_position: Anatomical preset ('apex' → 0.15, 'mid' → 0.5, 'base' → 0.85)
                            or a custom float in [0, 1].
        slice_thickness: Total slice thickness in µm (default 40 µm).
        run_length_tolerance: Max abs difference in length_fraction allowed for inclusion.
                              Instances outside this window are excluded to filter out
                              other cochlear turns passing through the same z-range.
        component_label: Component label of the main RC component (default 1).
        axis: Volume axis perpendicular to the slice plane ('x', 'y', or 'z'; default 'z').
        length_fraction_column: Column name for the run-length fraction (default 'length_fraction').

    Returns:
        dict with keys:
            reference_fraction  - resolved fraction used as the target position
            reference_label_id  - label_id of the SGN instance nearest to that fraction
            slice_center        - anchor coordinate along `axis` for the reference (µm)
            slice_min           - lower bound of the slice window (µm)
            slice_max           - upper bound of the slice window (µm)
            n_sgns              - number of SGNs counted in the slice
            area_µm²            - cross-sectional area from convex hull (µm²)
            density_µm⁻²       - n_sgns / area_µm²
            axis                - axis used for slicing
    """
    if axis not in _AXIS_PROJECTION:
        raise ValueError(f"axis must be one of {list(_AXIS_PROJECTION)}, got '{axis}'")

    # Validate required columns.
    required_cols = (
        ["label_id", "component_labels", length_fraction_column]
        + [f"anchor_{a}" for a in ("x", "y", "z")]
        + [f"bb_min_{a}" for a in ("x", "y", "z")]
        + [f"bb_max_{a}" for a in ("x", "y", "z")]
    )
    missing = [c for c in required_cols if c not in table.columns]
    if missing:
        raise ValueError(f"Table is missing required columns: {missing}")

    # Resolve reference fraction.
    if isinstance(reference_position, str):
        if reference_position not in REFERENCE_PRESETS:
            raise ValueError(
                f"reference_position string must be one of {list(REFERENCE_PRESETS)}, "
                f"got '{reference_position}'"
            )
        ref_frac = REFERENCE_PRESETS[reference_position]
    else:
        ref_frac = float(reference_position)
        if not (0.0 <= ref_frac <= 1.0):
            raise ValueError(f"reference_position float must be in [0, 1], got {ref_frac}")

    # Filter to main component.
    comp_table = filter_table(table, column_subset=[component_label], column="component_labels")
    if comp_table.empty:
        raise ValueError(f"No rows found for component_label={component_label}")

    # Find the SGN instance closest to the reference fraction.
    frac_diff = (comp_table[length_fraction_column] - ref_frac).abs()
    ref_row = comp_table.iloc[frac_diff.argmin()]
    ref_label_id = int(ref_row["label_id"])
    slice_center = float(ref_row[f"anchor_{axis}"])
    slice_min = slice_center - slice_thickness / 2.0
    slice_max = slice_center + slice_thickness / 2.0

    # Select by bounding-box overlap with the slice.
    bb_min_col = f"bb_min_{axis}"
    bb_max_col = f"bb_max_{axis}"
    in_slice = comp_table[
        (comp_table[bb_min_col] <= slice_max) & (comp_table[bb_max_col] >= slice_min)
    ]

    # Filter by run-length proximity to exclude other cochlear turns.
    in_slice = in_slice[
        (in_slice[length_fraction_column] - ref_frac).abs() <= run_length_tolerance
    ]

    n_sgns = len(in_slice)

    # Compute cross-sectional area via convex hull.
    proj_axes = _AXIS_PROJECTION[axis]
    coords_2d = in_slice[[f"anchor_{a}" for a in proj_axes]].values

    area = _compute_area(coords_2d)
    density = n_sgns / area if area > 0 else float("nan")

    return {
        "reference_fraction": ref_frac,
        "reference_label_id": ref_label_id,
        "slice_center": slice_center,
        "slice_min": slice_min,
        "slice_max": slice_max,
        "n_sgns": n_sgns,
        "area_µm²": area,
        "density_µm⁻²": density,
        "axis": axis,
    }


def _compute_area(coords_2d: np.ndarray) -> float:
    """Return convex hull area of 2D points; falls back to bounding rectangle for < 3 points."""
    if len(coords_2d) == 0:
        warnings.warn("No SGN instances in slice; area is 0.", stacklevel=3)
        return 0.0

    if len(coords_2d) < 3:
        warnings.warn(
            f"Only {len(coords_2d)} SGN instance(s) in slice; falling back to bounding-rectangle area.",
            stacklevel=3,
        )
        return _bbox_area(coords_2d)

    try:
        hull = ConvexHull(coords_2d)
        return float(hull.volume)  # For 2D input, hull.volume is the area.
    except Exception:
        warnings.warn("ConvexHull failed; falling back to bounding-rectangle area.", stacklevel=3)
        return _bbox_area(coords_2d)


def _bbox_area(coords_2d: np.ndarray) -> float:
    if len(coords_2d) == 0:
        return 0.0
    ranges = coords_2d.max(axis=0) - coords_2d.min(axis=0)
    return float(ranges[0] * ranges[1])
