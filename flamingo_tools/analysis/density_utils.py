import math
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import regionprops

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

# Maps the physical axis name to the corresponding dimension index in a (Z, Y, X) array.
_AXIS_ZYX_DIM = {"z": 0, "y": 1, "x": 2}


def sgn_density_at_position(
    table: pd.DataFrame,
    reference_position: Union[float, str] = "mid",
    slice_thickness: float = 10.0,
    run_length_tolerance: float = 0.1,
    component_label: int = 1,
    axis: str = "z",
    length_fraction_column: str = "length_fraction",
    mode: str = "2d",
    segmentation=None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    min_overlap_fraction: Optional[float] = None,
    seg_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Calculate SGN density at a specific cochlear position using a planar slice.

    Selects all SGN instances whose bounding boxes overlap a slice of given thickness
    centered on the reference position, then removes instances whose run-length fraction
    differs too much from the reference (handles oblique cochlear orientation). Density
    is computed as count / convex-hull area (2D mode) or convex-hull volume (3D mode).

    Args:
        table: SGN segmentation table with columns anchor_{x,y,z}, bb_min/max_{x,y,z},
               component_labels, label_id, and the length_fraction column. Coordinates
               must be in physical units (µm). The table is expected to have been through
               tonotopic mapping so that the length_fraction column is present.
        reference_position: Anatomical preset ('apex' → 0.15, 'mid' → 0.5, 'base' → 0.85)
                            or a custom float in [0, 1].
        slice_thickness: Total slice thickness in µm (default 10 µm).
        run_length_tolerance: Max abs difference in length_fraction allowed for inclusion.
                              Instances outside this window are excluded to filter out
                              other cochlear turns passing through the same z-range.
        component_label: Component label of the main RC component (default 1).
        axis: Volume axis perpendicular to the slice plane ('x', 'y', or 'z'; default 'z').
        length_fraction_column: Column name for the run-length fraction (default 'length_fraction').
        mode: Density mode. '2d' (default) projects anchor points onto the plane perpendicular
              to `axis` and computes density per area (SGN/µm²). '3d' uses the full 3D convex
              hull of all anchor points and computes density per volume (SGN/µm³); a larger
              slice_thickness is recommended in this mode.
        segmentation: Optional zarr array or numpy ndarray shaped (Z, Y, X) containing the
                      SGN instance segmentation. When provided together with
                      `min_overlap_fraction`, each candidate SGN is kept only if the fraction
                      of its voxels that fall within the slice window meets the threshold.
        voxel_size: Voxel size in µm per axis (x, y, z order; default 0.38 µm isotropic).
                    Used to convert the physical slice window to pixel indices when
                    `segmentation` is given.
        min_overlap_fraction: Minimum fraction in (0, 1] of an SGN's voxels (`n_pixels`) that
                              must lie within the slice for the instance to be counted. Only
                              active when `segmentation` is also provided. Default None
                              (no segmentation-based filtering).
        seg_offset: Physical coordinates (µm) of pixel (0, 0, 0) of the segmentation array,
                    in (x, y, z) order. Use (0, 0, 0) for a full OME-ZARR volume; set to
                    the crop origin when `segmentation` is a pre-extracted sub-volume.

    Returns:
        dict with keys:
            reference_fraction  - resolved fraction used as the target position
            reference_label_id  - label_id of the SGN instance nearest to that fraction
            slice_center        - anchor coordinate along `axis` for the reference (µm)
            slice_min           - lower bound of the slice window (µm)
            slice_max           - upper bound of the slice window (µm)
            slice_thickness     - slice thickness used (µm)
            n_sgns              - number of SGNs counted in the slice
            area                - convex-hull cross-sectional area (µm²); only in mode='2d'
            volume              - convex-hull volume (µm³); only in mode='3d'
            density             - n_sgns / area (mode='2d') or n_sgns / volume (mode='3d')
            mode                - density mode used ('2d' or '3d')
            axis                - axis used for slicing
            bb_min              - [x, y, z] lower corner of 3D bounding box (µm)
            bb_max              - [x, y, z] upper corner of 3D bounding box (µm)
            bb_center           - [x, y, z] center of 3D bounding box (µm); pass as `coords`
                                  to flamingo_tools.extract_block for block visualisation
            min_overlap_fraction - value passed in (or None when not used)
    """
    if axis not in _AXIS_PROJECTION:
        raise ValueError(f"axis must be one of {list(_AXIS_PROJECTION)}, got '{axis}'")
    if mode not in ("2d", "3d"):
        raise ValueError(f"mode must be '2d' or '3d', got '{mode}'")

    # Validate required columns.
    required_cols = (
        ["label_id", "component_labels", length_fraction_column] +
        [f"anchor_{a}" for a in ("x", "y", "z")] +
        [f"bb_min_{a}" for a in ("x", "y", "z")] +
        [f"bb_max_{a}" for a in ("x", "y", "z")]
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

    # Optional: filter by actual voxel overlap with the slice sub-volume.
    if segmentation is not None and min_overlap_fraction is not None:
        in_slice = _filter_by_segmentation_overlap(
            in_slice, segmentation, slice_min, slice_max,
            axis, voxel_size, seg_offset, min_overlap_fraction,
        )

    n_sgns = len(in_slice)

    # Compute extent (area in 2D, volume in 3D) via convex hull.
    if mode == "2d":
        proj_axes = _AXIS_PROJECTION[axis]
        hull_coords = in_slice[[f"anchor_{a}" for a in proj_axes]].values
        extent_key = "area"
    else:
        hull_coords = in_slice[["anchor_x", "anchor_y", "anchor_z"]].values
        extent_key = "volume"

    extent = _compute_hull_extent(hull_coords)
    density = n_sgns / extent if extent > 0 else float("nan")

    # 3D bounding box covering all selected SGN instance bounding boxes.
    if n_sgns > 0:
        bb_min = [
            float(in_slice["bb_min_x"].min()),
            float(in_slice["bb_min_y"].min()),
            float(in_slice["bb_min_z"].min()),
        ]
        bb_max = [
            float(in_slice["bb_max_x"].max()),
            float(in_slice["bb_max_y"].max()),
            float(in_slice["bb_max_z"].max()),
        ]
        bb_center = [(lo + hi) / 2.0 for lo, hi in zip(bb_min, bb_max)]
    else:
        bb_min = bb_max = bb_center = [float("nan"), float("nan"), float("nan")]

    return {
        "reference_fraction": ref_frac,
        "reference_label_id": ref_label_id,
        "slice_center": slice_center,
        "slice_min": slice_min,
        "slice_max": slice_max,
        "slice_thickness": slice_thickness,
        "n_sgns": n_sgns,
        extent_key: extent,
        "density": density,
        "mode": mode,
        "axis": axis,
        "bb_min": bb_min,
        "bb_max": bb_max,
        "bb_center": bb_center,
        "min_overlap_fraction": min_overlap_fraction,
    }


def _compute_hull_extent(coords: np.ndarray) -> float:
    """Return convex hull area (2D input) or volume (3D input); falls back to bounding box."""
    if len(coords) == 0:
        warnings.warn("No SGN instances in slice; extent is 0.", stacklevel=3)
        return 0.0

    ndim = coords.shape[1]
    min_pts = ndim + 1  # 3 for 2D, 4 for 3D
    if len(coords) < min_pts:
        warnings.warn(
            f"Only {len(coords)} SGN instance(s) in slice; falling back to bounding-box extent.",
            stacklevel=3,
        )
        return _bbox_extent(coords)

    try:
        hull = ConvexHull(coords)
        return float(hull.volume)  # area for 2D input, volume for 3D input
    except Exception:
        warnings.warn("ConvexHull failed; falling back to bounding-box extent.", stacklevel=3)
        return _bbox_extent(coords)


def _bbox_extent(coords: np.ndarray) -> float:
    """Product of axis ranges — area for 2D, volume for 3D."""
    if len(coords) == 0:
        return 0.0
    ranges = coords.max(axis=0) - coords.min(axis=0)
    result = 1.0
    for r in ranges:
        result *= float(r)
    return result


def _filter_by_segmentation_overlap(
    table: pd.DataFrame,
    segmentation,
    slice_min: float,
    slice_max: float,
    axis: str,
    voxel_size: Tuple[float, float, float],
    seg_offset: Tuple[float, float, float],
    min_overlap_fraction: float,
) -> pd.DataFrame:
    """Filter SGN instances by their voxel overlap with the slice sub-volume.

    Loads the slice slab from `segmentation` (a zarr array or numpy array in (Z, Y, X)
    order), counts how many voxels of each label_id fall inside, and keeps only rows
    where `voxels_in_slice / n_pixels >= min_overlap_fraction`.

    Args:
        table: SGN table (already reduced to candidate instances).
        segmentation: Segmentation array shaped (Z, Y, X).
        slice_min: Lower bound of the slice in µm along `axis`.
        slice_max: Upper bound of the slice in µm along `axis`.
        axis: Physical axis name ('x', 'y', or 'z') used for slicing.
        voxel_size: Voxel size in µm per axis (x, y, z order).
        seg_offset: Physical origin of seg[0,0,0] in µm (x, y, z order).
        min_overlap_fraction: Minimum overlap fraction for inclusion.

    Returns:
        Filtered DataFrame.
    """
    # Map physical axis (x/y/z) to its index in the (x,y,z) tuple and to zarr dim (Z,Y,X).
    phys_index = {"x": 0, "y": 1, "z": 2}[axis]
    dim = _AXIS_ZYX_DIM[axis]

    offset_a = seg_offset[phys_index]
    vs_a = voxel_size[phys_index]

    px_start = max(0, math.floor((slice_min - offset_a) / vs_a))
    px_end = min(segmentation.shape[dim], math.ceil((slice_max - offset_a) / vs_a) + 1)

    idx = [slice(None)] * 3
    idx[dim] = slice(px_start, px_end)
    sub_vol = np.asarray(segmentation[tuple(idx)])

    # Count voxels per label in the sub-volume using regionprops (efficient label scan).
    voxels_in_slice = {prop.label: prop.area for prop in regionprops(sub_vol)}

    label_ids = table["label_id"].astype(int).values
    n_pixels = table["n_pixels"].astype(int).values
    vox_in_slice = np.array([voxels_in_slice.get(lid, 0) for lid in label_ids])
    overlap = np.where(n_pixels > 0, vox_in_slice / n_pixels, 0.0)
    filtered = table[overlap >= min_overlap_fraction]
    if filtered.empty:
        warnings.warn(
            "All SGN instances were removed by the segmentation overlap filter.",
            stacklevel=4,
        )
    return filtered


def sgn_density_profile(
    table: pd.DataFrame,
    positions: Optional[List[Union[float, str]]] = None,
    slice_thickness: float = 10.0,
    run_length_tolerance: float = 0.1,
    component_label: int = 1,
    axis: str = "z",
    length_fraction_column: str = "length_fraction",
    mode: str = "2d",
    segmentation=None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    min_overlap_fraction: Optional[float] = None,
    seg_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> dict:
    """Compute SGN density at multiple cochlear positions.

    Args:
        table: SGN segmentation table (see sgn_density_at_position for required columns).
        positions: List of positions to evaluate. Each entry is either a preset string
                   ('apex', 'mid', 'base') or a float in [0, 1]. Default: ['apex', 'mid', 'base'].
        slice_thickness: Total slice thickness in µm (default 10 µm).
        run_length_tolerance: Max abs difference in length_fraction for inclusion (default 0.1).
        component_label: Component label of the main RC component (default 1).
        axis: Volume axis perpendicular to the slice plane (default 'z').
        length_fraction_column: Column name for the run-length fraction (default 'length_fraction').
        mode: '2d' (default) or '3d' — see sgn_density_at_position.
        segmentation: Optional segmentation array (Z, Y, X) — see sgn_density_at_position.
        voxel_size: Voxel size in µm per axis (x, y, z order; default 0.38 µm isotropic).
        min_overlap_fraction: Minimum voxel overlap fraction — see sgn_density_at_position.
        seg_offset: Physical origin of seg[0,0,0] in µm (x, y, z order; default (0,0,0)).

    Returns:
        dict keyed by position label (preset name or string-formatted float), each value
        being the result dict from sgn_density_at_position.
    """
    if positions is None:
        positions = ["apex", "mid", "base"]

    results = {}
    for pos in positions:
        key = pos if isinstance(pos, str) else str(pos)
        results[key] = sgn_density_at_position(
            table,
            reference_position=pos,
            slice_thickness=slice_thickness,
            run_length_tolerance=run_length_tolerance,
            component_label=component_label,
            axis=axis,
            length_fraction_column=length_fraction_column,
            mode=mode,
            segmentation=segmentation,
            voxel_size=voxel_size,
            min_overlap_fraction=min_overlap_fraction,
            seg_offset=seg_offset,
        )
    return results


def _auto_roi_halo(
    bb_min: List[float],
    bb_max: List[float],
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    axis: str = "z",
    slice_thickness: Optional[float] = None,
    min_halo: int = 10,
) -> List[int]:
    """Compute roi_halo in pixels sized to match the slice exactly.

    Along the slice axis the halo is derived from slice_thickness / 2 so the
    extracted block matches the slice used for the density calculation.
    Along the two projection axes the bounding box half-extents are used.

    Args:
        bb_min: [x, y, z] lower corner of the SGN bounding box in µm.
        bb_max: [x, y, z] upper corner of the SGN bounding box in µm.
        voxel_size: Voxel size in µm per axis (default 0.38 µm isotropic).
        axis: Axis that was perpendicular to the slice plane ('x', 'y', or 'z').
        slice_thickness: Slice thickness in µm used for the density calculation.
                         When provided, the halo along ``axis`` is set to
                         ceil(slice_thickness / 2 / voxel_size); otherwise the
                         bounding box extent is used for all axes.
        min_halo: Minimum halo size in pixels per axis (default 10).

    Returns:
        List of 3 ints [halo_x, halo_y, halo_z] in pixels.
    """
    import math
    if any(np.isnan(v) for v in list(bb_min) + list(bb_max)):
        return [128, 128, 64]
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    halo = []
    for i, (lo, hi, vs) in enumerate(zip(bb_min, bb_max, voxel_size)):
        if i == axis_index and slice_thickness is not None:
            half_px = math.ceil(slice_thickness / 2.0 / vs)
        else:
            half_px = math.ceil((hi - lo) / 2.0 / vs)
        halo.append(max(min_halo, half_px))
    return halo


def _build_block_extraction_dict(
    density_results: dict,
    input_json_params: Optional[dict] = None,
    roi_halo: Optional[List[int]] = None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> List[dict]:
    """Build a block-extraction JSON list compatible with flamingo_tools.extract_block --json_info.

    Returns a list of dicts, one per evaluated position. Each dict contains a single
    crop_center (the bb_center of that position, rounded to ints) and its own roi_halo
    so that differently-sized ROIs can be extracted per position in one pass.

    roi_halo priority per entry:
        1. Explicit ``roi_halo`` argument (same value applied to all positions).
        2. ``roi_halo`` key from ``input_json_params`` (same for all positions).
        3. Auto-computed: slice_thickness / 2 along the slice axis; bounding box
           half-extents along the two projection axes. Both are converted to pixels
           using ``voxel_size``. The axis and slice_thickness are read from each
           position's density result.

    Args:
        density_results: Output of sgn_density_profile — dict keyed by position label.
        input_json_params: Optional dict from a ChReef-format input JSON, providing
                           dataset_name, image_channel, segmentation_channel, roi_halo, etc.
        roi_halo: Explicit ROI halo in pixels [x, y, z] applied to all positions.
        voxel_size: Voxel size in µm per axis used for auto roi_halo computation.

    Returns:
        List of dicts, each compatible with flamingo_tools.extract_block --json_info.
        The list can be used directly as the JSON input for extract_block_json_wrapper.
    """
    # Metadata shared by all positions.
    common: dict = {}
    if input_json_params is not None:
        for key in ("dataset_name", "image_channel", "segmentation_channel", "cell_type", "component_list"):
            if key in input_json_params:
                common[key] = input_json_params[key]

    # Explicit or JSON-level fallback halo (None means auto-compute per position).
    global_halo: Optional[List[int]] = None
    if roi_halo is not None:
        global_halo = list(roi_halo)
    elif input_json_params is not None and "roi_halo" in input_json_params:
        global_halo = list(input_json_params["roi_halo"])

    result_list: List[dict] = []
    for position_label, pos_result in density_results.items():
        entry = dict(common)
        entry["position_label"] = position_label

        center = pos_result.get("bb_center", [float("nan")] * 3)
        entry["crop_centers"] = [[round(c) for c in center]]

        if global_halo is not None:
            entry["roi_halo"] = global_halo
        else:
            bb_min = pos_result.get("bb_min", [float("nan")] * 3)
            bb_max = pos_result.get("bb_max", [float("nan")] * 3)
            entry["roi_halo"] = _auto_roi_halo(
                bb_min, bb_max, voxel_size,
                axis=pos_result.get("axis", "z"),
                slice_thickness=pos_result.get("slice_thickness"),
            )

        result_list.append(entry)

    return result_list
