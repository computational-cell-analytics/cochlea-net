import json
import math
import os
import warnings
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from skimage.measure import regionprops

from flamingo_tools.analysis.seg_table_utils import filter_table
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.s3_utils import MOBIE_FOLDER, get_s3_path
from flamingo_tools.json_util import export_dictionary_as_json

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
    component_list: List[int] = [1],
    axis: str = "z",
    length_fraction_column: str = "length_fraction",
    mode: str = "2d",
    segmentation=None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    min_overlap_fraction: Optional[float] = None,
    min_overlap_volume: Optional[float] = None,
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
        component_list: Component label(s) of the main RC component (default 1).
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
        min_overlap_volume: Alternative to `min_overlap_fraction`: minimum overlap expressed
                            as an absolute voxel volume in µm³. Exactly one of the two
                            overlap parameters may be set. Default None.

    Returns:
        dict with keys:
            reference_fraction   - resolved fraction used as the target position
            reference_label_id   - label_id of the SGN instance nearest to that fraction
            slice_center         - anchor coordinate along `axis` for the reference (µm)
            slice_min            - lower bound of the slice window (µm)
            slice_max            - upper bound of the slice window (µm)
            slice_thickness      - slice thickness used (µm)
            n_sgns               - number of SGNs counted in the slice
            area                 - convex-hull cross-sectional area (µm²); only in mode='2d'
            volume               - convex-hull volume (µm³); only in mode='3d'
            density              - n_sgns / area (mode='2d') or n_sgns / volume (mode='3d')
            mode                 - density mode used ('2d' or '3d')
            axis                 - axis used for slicing
            bb_min               - [x, y, z] lower corner of 3D bounding box (µm)
            bb_max               - [x, y, z] upper corner of 3D bounding box (µm)
            bb_center            - [x, y, z] center of 3D bounding box (µm); pass as `coords`
                                   to flamingo_tools.extract_block for block visualisation
            min_overlap_fraction - value passed in (or None when not used)
            min_overlap_volume   - value passed in (or None when not used)
            hull_vertices        - list of anchor coordinates (µm) forming the convex hull
                                   boundary; shape (k, 2) for mode='2d' (projection axes),
                                   (k, 3) for mode='3d' (x/y/z). None when the hull could
                                   not be computed. Pass to `hull_to_mask` to generate a
                                   binary napari-compatible mask for the density region.
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
    comp_table = filter_table(table, column_subset=component_list, column="component_labels")
    if comp_table.empty:
        raise ValueError(f"No rows found for component_list={component_list}")

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

    filtered_labels = []
    # Optional: filter by actual voxel overlap with the slice sub-volume.
    if segmentation is not None and (min_overlap_fraction is not None or min_overlap_volume is not None):
        in_slice, filtered_out = _filter_by_segmentation_overlap(
            in_slice, segmentation, slice_min, slice_max,
            axis, voxel_size, min_overlap_fraction=min_overlap_fraction,
            min_overlap_volume=min_overlap_volume,
        )
        filtered_labels = list(filtered_out["label_id"]),

    n_sgns = len(in_slice)
    labels_in = [int(i) for i in list(in_slice["label_id"])]

    # Compute extent (area in 2D, volume in 3D) via convex hull.
    if mode == "2d":
        proj_axes = _AXIS_PROJECTION[axis]
        hull_coords = in_slice[[f"anchor_{a}" for a in proj_axes]].values
        extent_key = "area"
    else:
        hull_coords = in_slice[["anchor_x", "anchor_y", "anchor_z"]].values
        extent_key = "volume"

    extent, hull_vertices = _compute_hull(hull_coords)
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
        "label_ids": labels_in,
        "label_removed": filtered_labels,
        "min_overlap_fraction": min_overlap_fraction,
        "min_overlap_volume": min_overlap_volume,
        "hull_vertices": hull_vertices.tolist() if hull_vertices is not None else None,
    }


def _compute_hull(coords: np.ndarray) -> Tuple[float, Optional[np.ndarray]]:
    """Return (extent, hull_vertices) for the given anchor coordinates.

    extent is the convex-hull area (2D input) or volume (3D input) in µm²/µm³.
    hull_vertices is the subset of `coords` that form the hull, shape (k, ndim), in µm.
    Returns (fallback_extent, None) when the hull cannot be computed.
    """
    if len(coords) == 0:
        warnings.warn("No SGN instances in slice; extent is 0.", stacklevel=3)
        return 0.0, None

    ndim = coords.shape[1]
    if len(coords) < ndim + 1:
        warnings.warn(
            f"Only {len(coords)} SGN instance(s) in slice; falling back to bounding-box extent.",
            stacklevel=3,
        )
        return _bbox_extent(coords), None

    try:
        hull = ConvexHull(coords)
        return float(hull.volume), coords[hull.vertices]  # area for 2D, volume for 3D
    except Exception:
        warnings.warn("ConvexHull failed; falling back to bounding-box extent.", stacklevel=3)
        return _bbox_extent(coords), None


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
    min_overlap_fraction: Optional[float] = None,
    min_overlap_volume: Optional[float] = None,
) -> pd.DataFrame:
    """Filter SGN instances by their voxel overlap with the slice sub-volume.

    Loads a slab from `segmentation` (a zarr array or numpy array in (Z, Y, X) order),
    counts how many voxels of each label_id fall inside, and keeps only rows where
    `voxels_in_slice / n_pixels >= min_overlap_fraction`.

    The function auto-detects whether `segmentation` is a full volume or a pre-extracted
    crop by comparing its physical extent (shape × voxel_size) along the slice axis to the
    maximum anchor coordinate in `table`. When the physical extent is smaller than the
    maximum anchor coordinate the array is treated as a crop and used in full; otherwise
    the appropriate slab is extracted with offset 0.

    Args:
        table: SGN table (already reduced to candidate instances).
        segmentation: Segmentation array shaped (Z, Y, X).
        slice_min: Lower bound of the slice in µm along `axis`.
        slice_max: Upper bound of the slice in µm along `axis`.
        axis: Physical axis name ('x', 'y', or 'z') used for slicing.
        voxel_size: Voxel size in µm per axis (x, y, z order).
        min_overlap_fraction: Minimum overlap fraction for inclusion.

    Returns:
        Filtered DataFrame.
    """
    # Map physical axis (x/y/z) to its index in the (x,y,z) tuple and to zarr dim (Z,Y,X).
    phys_index = {"x": 0, "y": 1, "z": 2}[axis]
    dim = _AXIS_ZYX_DIM[axis]
    vs_a = voxel_size[phys_index]

    if min_overlap_fraction is not None and min_overlap_volume is not None:
        raise ValueError("Choose either a minimal overlap fraction or a minimal volume [µm³], not both.")

    # Auto-detect crop vs. full volume: if the array's physical extent along the slice axis
    # is smaller than the maximum anchor coordinate in the table, the array is a crop and
    # all of its voxels are already within the region of interest.
    seg_physical_extent = segmentation.shape[dim] * vs_a
    max_anchor = float(table[f"anchor_{axis}"].max())
    if max_anchor > seg_physical_extent:
        sub_vol = np.asarray(segmentation)
    else:
        px_start = max(0, math.floor(slice_min / vs_a))
        px_end = min(segmentation.shape[dim], math.ceil(slice_max / vs_a) + 1)
        idx = [slice(None)] * 3
        idx[dim] = slice(px_start, px_end)
        sub_vol = np.asarray(segmentation[tuple(idx)])

    # Count voxels per label in the sub-volume using regionprops (efficient label scan).
    voxels_in_slice = {prop.label: prop.area for prop in regionprops(sub_vol)}

    label_ids = table["label_id"].astype(int).values
    n_pixels = table["n_pixels"].astype(int).values
    vox_in_slice = np.array([voxels_in_slice.get(lid, 0) for lid in label_ids])
    if min_overlap_fraction is not None:
        overlap = np.where(n_pixels > 0, vox_in_slice / n_pixels, 0.0)
        filtered = table[overlap >= min_overlap_fraction]
        filtered_out = table[overlap < min_overlap_fraction]
    elif min_overlap_volume is not None:
        voxel_vol = voxel_size[0] * voxel_size[1] * voxel_size[2]
        volume_in_slice = vox_in_slice * voxel_vol
        filtered = table[volume_in_slice >= min_overlap_volume]
        filtered_out = table[volume_in_slice < min_overlap_volume]
    else:
        raise ValueError("Choose a minimal overlap to exclude instances.")

    if filtered.empty:
        warnings.warn(
            "All SGN instances were removed by the segmentation overlap filter.",
            stacklevel=4,
        )
    return filtered, filtered_out


def sgn_density_profile(
    table: pd.DataFrame,
    positions: Optional[List[Union[float, str]]] = None,
    slice_thickness: float = 10.0,
    run_length_tolerance: float = 0.1,
    component_list: List[int] = [1],
    axis: str = "z",
    length_fraction_column: str = "length_fraction",
    mode: str = "2d",
    segmentation=None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    min_overlap_fraction: Optional[float] = None,
    min_overlap_volume: Optional[float] = None,
) -> dict:
    """Compute SGN density at multiple cochlear positions.

    Args:
        table: SGN segmentation table (see sgn_density_at_position for required columns).
        positions: List of positions to evaluate. Each entry is either a preset string
                   ('apex', 'mid', 'base') or a float in [0, 1]. Default: ['apex', 'mid', 'base'].
        slice_thickness: Total slice thickness in µm (default 10 µm).
        run_length_tolerance: Max abs difference in length_fraction for inclusion (default 0.1).
        component_list: Component label(s) of the main RC component (default 1).
        axis: Volume axis perpendicular to the slice plane (default 'z').
        length_fraction_column: Column name for the run-length fraction (default 'length_fraction').
        mode: '2d' (default) or '3d' — see sgn_density_at_position.
        segmentation: Optional segmentation array (Z, Y, X) — see sgn_density_at_position.
        voxel_size: Voxel size in µm per axis (x, y, z order; default 0.38 µm isotropic).
        min_overlap_fraction: Minimum voxel overlap fraction — see sgn_density_at_position.

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
            component_list=component_list,
            axis=axis,
            length_fraction_column=length_fraction_column,
            mode=mode,
            segmentation=segmentation,
            voxel_size=voxel_size,
            min_overlap_fraction=min_overlap_fraction,
            min_overlap_volume=min_overlap_volume,
        )
    return results


def _auto_roi_halo(
    bb_min: List[float],
    bb_max: List[float],
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    axis: str = "z",
    slice_thickness: Optional[float] = None,
    min_halo: int = 1,
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
            entry["label_ids"] = pos_result.get("label_ids"),
            entry["label_removed"] = pos_result.get("label_removed"),
            entry["hull_vertices"] = pos_result.get("hull_vertices"),

        result_list.append(entry)

    return result_list


def hull_to_mask(
    hull_vertices: Union[List, np.ndarray],
    bb_center: List[float],
    roi_halo: List[int],
    voxel_size: Tuple[float, float, float],
    axis: str = "z",
    mode: str = "2d",
    **_,
) -> np.ndarray:
    """Convert convex hull vertices to a binary ZYX mask aligned with an extracted crop.

    The returned mask has the same spatial extent as a crop produced by
    `flamingo_tools.extract_block` with the given `bb_center` and `roi_halo`, so it can be
    loaded directly as a napari Labels layer alongside the image crop.

    Args:
        hull_vertices: Convex hull vertex coordinates in µm. Shape (N, 2) for mode='2d'
                       (columns ordered as the two projection axes of `_AXIS_PROJECTION[axis]`,
                       e.g. x/y for axis='z'). Shape (N, 3) for mode='3d' (x/y/z order).
                       Typically the ``hull_vertices`` value from a `sgn_density_at_position`
                       result dict.
        bb_center: [x, y, z] physical centre of the crop in µm (``bb_center`` from the
                   density result dict).
        roi_halo: [halo_x, halo_y, halo_z] half-extents of the crop in pixels.
        voxel_size: Voxel size in µm per axis (x, y, z order).
        axis: Slice axis used for the density calculation ('x', 'y', or 'z'). Only
              relevant in mode='2d' to determine the projection plane.
        mode: '2d' rasterises the 2D polygon and extrudes it along the slice axis.
              '3d' performs a half-space test for every voxel centre inside the crop.

    Returns:
        Boolean numpy array of shape (2*halo_z, 2*halo_y, 2*halo_x) — True inside the hull.
    """
    from skimage.draw import polygon as sk_polygon

    verts = np.asarray(hull_vertices, dtype=float)
    phys_idx = {"x": 0, "y": 1, "z": 2}
    mask_shape_zyx = (2 * roi_halo[2], 2 * roi_halo[1], 2 * roi_halo[0])

    if mode == "2d":
        proj_ax0, proj_ax1 = _AXIS_PROJECTION[axis]
        i0, i1 = phys_idx[proj_ax0], phys_idx[proj_ax1]

        origin0 = bb_center[i0] - roi_halo[i0] * voxel_size[i0]
        origin1 = bb_center[i1] - roi_halo[i1] * voxel_size[i1]

        vert_col = (verts[:, 0] - origin0) / voxel_size[i0]
        vert_row = (verts[:, 1] - origin1) / voxel_size[i1]
        n_rows = 2 * roi_halo[i1]
        n_cols = 2 * roi_halo[i0]
        plane_mask = np.zeros((n_rows, n_cols), dtype=bool)
        rr, cc = sk_polygon(vert_row, vert_col, shape=(n_rows, n_cols))
        plane_mask[rr, cc] = True

        # Extrude along the slice axis by inserting a broadcast dimension there.
        slice_dim = _AXIS_ZYX_DIM[axis]
        new_shape = list(plane_mask.shape)
        new_shape.insert(slice_dim, 1)
        return np.broadcast_to(plane_mask.reshape(new_shape), mask_shape_zyx).copy()

    else:  # mode == "3d"
        gx = (bb_center[0] - roi_halo[0] * voxel_size[0]) + np.arange(2 * roi_halo[0]) * voxel_size[0]
        gy = (bb_center[1] - roi_halo[1] * voxel_size[1]) + np.arange(2 * roi_halo[1]) * voxel_size[1]
        gz = (bb_center[2] - roi_halo[2] * voxel_size[2]) + np.arange(2 * roi_halo[2]) * voxel_size[2]

        GZ, GY, GX = np.meshgrid(gz, gy, gx, indexing="ij")  # each (2hz, 2hy, 2hx)
        pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

        hull = ConvexHull(verts)
        # A point is inside the convex hull when all half-space constraints are satisfied.
        inside = np.all(pts @ hull.equations[:, :3].T + hull.equations[:, 3] <= 0, axis=1)
        return inside.reshape(mask_shape_zyx)


def calc_sgn_density(
    output: str,
    seg_table_path: Optional[str] = None,
    json_input: Optional[str] = None,
    json_output: Optional[str] = None,
    force_overwrite: bool = False,
    positions: List[Union[str, float]] = ["apex", "mid", "base"],
    slice_thickness: float = 10.0,
    run_length_tolerance: float = 0.1,
    component_list: List[int] = [1],
    axis: str = "z",
    length_fraction_column: str = "length_fraction",
    density_mode: str = "2d",
    roi_halo: Optional[List[int]] = None,
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    mobie_dir: str = None,
    seg_path: Optional[str] = None,
    seg_key: str = "s0",
    min_overlap_fraction: Optional[float] = None,
    min_overlap_volume: Optional[float] = None,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Calcualte SGN density for 2D and 3D sub-volumes.

    Args:
        output: Output path for JSON file with density results.
        seg_table_path: Input path to SGN segmentation table (TSV). Must contain length_fraction column
                        (produced by flamingo_tools.tonotopic_mapping).
                        Optional when --json_input is supplied.
        json_input: Input JSON file with metadata information(dataset_name, image_channel,
                         segmentation_channel, …). When 'seg_table_path' is absent the table path
                         is derived from dataset_name and segmentation_channel via 'mobie_dir'.
        json_output: Optional output path for block-extraction JSON compatible with
                     flamingo_tools.extract_block --json_info.
                     Contains crop_centers derived from the density bounding boxes.
        force_overwrite: Forcefully overwrite output.
        positions: Cochlear positions to evaluate. Use preset names ('apex', 'mid', 'base') or
                   floats in [0, 1]. Default: apex mid base
        slice_thickness: Total thickness of the horizontal slice in µm. Default: 10.0
        run_length_tolerance: Maximum allowed length_fraction difference to include an SGN instance.
                              Reduces contamination from other cochlear turns. Default: 0.1
        component_list: Component label(s) of the main Rosenthal's Canal component. Default: 1
        axis: Volume axis perpendicular to the slice plane. Default: z
        length_fraction_column: Column name for the run-length fraction in the table. Default: length_fraction.
        density_mode: Density mode: '2d' computes density per cross-sectional area (SGN/µm²);
                      '3d' computes density per convex-hull volume (SGN/µm³). Default: 2d
        roi_halo: ROI halo in pixels [x y z] for the block-extraction JSON output, applied to all
                  positions. Overrides the value from 'json_input'.
                  If omitted, the halo is computed automatically from each position's bounding box.
        voxel_size: Voxel size in µm used to convert bounding box extents to pixels when
                    computing the automatic roi_halo. Provide 1 value (isotropic) or 3 values (x y z).
        mobie_dir: Local MoBIE project directory used to locate the table when 'json_input' is given
                   and 'seg_table_path' is absent.
        seg_path: Path to the SGN segmentation volume (local TIF, N5/Zarr, or S3 OME-ZARR).
                  When omitted and 'json_input' is given, the path is derived automatically as
                  <dataset_name>/images/ome-zarr/<segmentation_channel>.ome.zarr.
                  Only used when 'min_overlap_fraction' is set.
        seg_key: Internal key for N5/Zarr/OME-ZARR segmentation (default: s0).
        min_overlap_fraction: Minimum fraction of an SGN's voxels (n_pixels) that must lie within the
                              slice sub-volume to count the instance. Range (0, 1].
                              Default: None (no segmentation-based filtering). Whether the
                              segmentation is a pre-extracted crop or a full volume is detected
                              automatically from the array shape.
        min_overlap_volume: Analogous to min_overlap_fraction with explicit volume in µm³
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    # Resolve table path and optional JSON metadata.
    json_params = None
    if json_input is not None:
        with open(json_input) as f:
            json_params = json.load(f)
        if isinstance(json_params, list):
            json_params = json_params[0]

    if mobie_dir is None:
        mobie_dir = os.getcwd()

    if seg_table_path is not None:
        table_path = seg_table_path
    elif json_params is not None:
        dataset_name = json_params["dataset_name"]
        seg_channel = json_params.get("segmentation_channel", "SGN_v2")
        if s3:
            table_path = f"{dataset_name}/tables/{seg_channel}/default.tsv"
        else:
            table_path = os.path.join(mobie_dir, dataset_name, "tables", seg_channel, "default.tsv")
            if not os.path.isfile(table_path):
                raise ValueError(f"Table path {table_path} does not exist. Use explicit path or check MoBIE folder.")
    else:
        raise ValueError("Provide 'seg_table_path' or 'json_input'.")

    # Convert numeric position strings to floats.
    positions_list = []
    for p in positions:
        try:
            positions_list.append(float(p))
        except ValueError:
            positions_list.append(p)

    if s3:
        tsv_path, fs = get_s3_path(
            table_path,
            credential_file=s3_credentials,
            bucket_name=s3_bucket_name,
            service_endpoint=s3_service_endpoint,
        )
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table = pd.read_csv(table_path, sep="\t")

    # Resolve voxel size.
    if len(voxel_size) == 1:
        voxel_size = voxel_size * 3
    voxel_size = tuple(voxel_size)

    # Resolve and load segmentation if overlap filtering is requested.
    segmentation = None
    if min_overlap_fraction is not None or min_overlap_volume is not None:
        if seg_path is None and json_params is not None:
            dataset_name = json_params["dataset_name"]
            seg_channel = json_params.get("segmentation_channel", "SGN_v2")
            if s3:
                seg_path = f"{dataset_name}/images/ome-zarr/{seg_channel}.ome.zarr"
            else:
                seg_path = os.path.join(
                    mobie_dir, dataset_name, "images", "ome-zarr",
                    f"{seg_channel}.ome.zarr",
                )
        if seg_path is not None:
            segmentation = read_image_data(
                seg_path, seg_key,
                from_s3=s3,
                credential_file=s3_credentials,
                bucket_name=s3_bucket_name,
                service_endpoint=s3_service_endpoint,
            )

    results = sgn_density_profile(
        table,
        positions=positions_list,
        slice_thickness=slice_thickness,
        run_length_tolerance=run_length_tolerance,
        component_list=component_list,
        axis=axis,
        length_fraction_column=length_fraction_column,
        mode=density_mode,
        segmentation=segmentation,
        voxel_size=voxel_size,
        min_overlap_fraction=min_overlap_fraction,
        min_overlap_volume=min_overlap_volume,
    )

    export_dictionary_as_json(results, output, force_overwrite=force_overwrite)

    if json_output is not None:
        block_list = _build_block_extraction_dict(
            results,
            input_json_params=json_params,
            roi_halo=roi_halo,
            voxel_size=voxel_size,
        )
        export_dictionary_as_json(block_list, json_output, force_overwrite=force_overwrite)
