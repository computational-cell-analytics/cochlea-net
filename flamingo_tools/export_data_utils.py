"""@private
"""

import json
import os
from glob import glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

# Short aliases for source names that differ between cochleae, mapped to
# (source type, name prefix, preferred source names). A cochlea often holds several versions of a
# segmentation, e.g. IHC_v4b and IHC_v11. The preferred names pin the version that is used for the
# analysis; they are tried first, before the alias name itself and before the name prefix, which
# only resolves an unambiguous match.
SOURCE_ALIASES = {
    "SGN": ("segmentation", "SGN", ("SGN_v2",)),
    "IHC": ("segmentation", "IHC", ("IHC_v11",)),
    "synapses": ("spots", "synapse", ("synapse_v3_ihc_v11",)),
}

# Physical size of one cell of the down-sampled cochlea filter volume, in µm.
# 48 * 0.38 reproduces the down-sampling factor that was hard-coded for isotropic 0.38 µm data.
FILTER_CELL_SIZE = 48 * 0.38


def normalize_voxel_size(voxel_size: Union[float, Sequence[float]]) -> Tuple[float, float, float]:
    """Normalize a voxel size to a (x, y, z) tuple of three values.

    Args:
        voxel_size: Voxel size in µm, either a single value for isotropic data, a sequence with a
            single value, or a sequence of three values in (x, y, z) order.

    Returns:
        Voxel size as a tuple of three values in (x, y, z) order.
    """
    if isinstance(voxel_size, (int, float)):
        return (float(voxel_size),) * 3
    values = tuple(float(v) for v in voxel_size)
    if len(values) == 1:
        values = values * 3
    if len(values) != 3:
        raise ValueError(f"voxel_size must have 1 or 3 values, got {len(values)}.")
    return values


def filter_volume_downscale_factors(
    voxel_size: Union[float, Sequence[float]],
    cell_size: float = FILTER_CELL_SIZE,
) -> Tuple[int, int, int]:
    """Compute the per-axis down-sampling factor for the cochlea filter volume.

    The factors are chosen so that one cell of the down-sampled volume covers `cell_size` µm on
    every axis. This keeps the physical extent of a dilation step independent of the voxel size.

    Args:
        voxel_size: Voxel size in µm, scalar or (x, y, z).
        cell_size: Target physical size of one cell of the down-sampled volume, in µm.

    Returns:
        Down-sampling factor in pixels per axis, in (x, y, z) order.
    """
    return tuple(max(1, int(round(cell_size / v))) for v in normalize_voxel_size(voxel_size))


def compute_crop_bb(
    crop_center: List[float],
    roi_halo: Optional[List[int]],
    voxel_size: Union[float, Sequence[float]],
    scale: int,
    shape: Tuple[int, ...],
    axis: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute bounding box start/stop for a crop in ZYX pixel coordinates.

    Args:
        crop_center: Crop center position as [x, y, z] in µm.
        roi_halo: Halo around the center as [halo_x, halo_y, halo_z] in pixels at the target
            scale. Optional when `axis` is given: the two axes not collapsed by `axis` then
            span the full array extent (`shape`), i.e. the whole cross-sectional plane is
            cropped. Required otherwise.
        voxel_size: Voxel size in µm at full resolution (scale 0), scalar for isotropic data or
            three values in (x, y, z) order.
        scale: Scale level (0 = full resolution, each step doubles the effective voxel size).
        shape: Array shape in ZYX order at the target scale.
        axis: Optional axis index into (x, y, z), i.e. 0, 1, or 2. When given, that axis is
            collapsed to a single-pixel slice starting at the crop center, producing a 2D
            crop. Default: None (3D crop; requires roi_halo).

    Returns:
        start: ZYX start pixel coordinates, clamped to zero.
        stop: ZYX stop pixel coordinates, clamped to shape.
    """
    if roi_halo is None and axis is None:
        raise ValueError("roi_halo is required unless axis is also given.")

    voxel_size_zyx = np.array(normalize_voxel_size(voxel_size)[::-1])
    center_zyx = np.round(np.array(crop_center[::-1]) / (voxel_size_zyx * 2 ** scale)).astype(int)

    if roi_halo is None:
        start = np.zeros(3, dtype=int)
        stop = np.array(shape, dtype=int)
    else:
        halo_zyx = np.array(roi_halo[::-1])
        start = np.maximum(0, center_zyx - halo_zyx)
        stop = np.minimum(np.array(shape), center_zyx + halo_zyx)

    if axis is not None:
        if axis not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1, or 2, got {axis}")
        # crop_center/roi_halo are given in (x, y, z) order; the pixel arrays above are ZYX.
        axis_zyx = 2 - axis
        start[axis_zyx] = center_zyx[axis_zyx]
        stop[axis_zyx] = center_zyx[axis_zyx] + 1

    return start, stop


def crop_suffix(crop_center: List[float], axis: Optional[int] = None, suffix: Optional[str] = None) -> str:
    """Build the output filename suffix for a crop.

    Args:
        crop_center: Crop center position as [x, y, z] in µm.
        axis: Optional axis index into (x, y, z) used to collapse the crop to a 2D slice
            (see `compute_crop_bb`). Appended as "_axis-<axis>" when given.
        suffix: Optional extra label appended last, e.g. a tonotopic position name
            ("apex", "mid", "base").

    Returns:
        Suffix of the form "_crop_<x>-<y>-<z>[_axis-<axis>][_<suffix>]".
    """
    coord_string = "-".join(str(int(round(c))).zfill(4) for c in crop_center)
    parts = [f"_crop_{coord_string}"]
    if axis is not None:
        parts.append(f"_axis-{axis}")
    if suffix is not None:
        parts.append(f"_{suffix}")
    return "".join(parts)


def export_output_path(
    out_folder: str,
    name: str,
    ome_zarr: bool = False,
    crop_center: Optional[List[float]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
) -> str:
    """Build the output path for an exported channel.

    Args:
        out_folder: Folder for the exported data.
        name: Name of the exported channel or segmentation.
        ome_zarr: Whether the output is written as OME-Zarr instead of TIF.
        crop_center: Optional crop center as [x, y, z] in µm. When given, the crop suffix is
            appended to the file name, so crops at different positions do not overwrite each other.
        axis: Optional axis index into (x, y, z), see `crop_suffix`.
        suffix: Optional extra label, see `crop_suffix`.

    Returns:
        The output path.
    """
    base = name if crop_center is None else f"{name}{crop_suffix(crop_center, axis, suffix)}"
    extension = "ome.zarr" if ome_zarr else "tif"
    return os.path.join(out_folder, f"{base}.{extension}")


def crop_filter_volume(
    filter_volume: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    us_factor: Union[float, Sequence[float]],
) -> np.ndarray:
    """Extract the sub-region of a down-sampled cochlea filter volume that covers a crop.

    Instead of upscaling the entire filter volume to full resolution and then slicing, this
    function maps each target pixel to its filter volume cell, which is far more memory-efficient
    when exporting high-resolution crops.

    Args:
        filter_volume: Down-sampled boolean cochlea mask in ZYX order.
        start: Crop start in scale-s pixel coordinates, ZYX.
        stop: Crop stop in scale-s pixel coordinates, ZYX.
        us_factor: Size of one filter_volume cell in scale-s pixels, either a single value or one
            value per axis in ZYX order (i.e. ds_factor / 2**scale). May be non-integer.

    Returns:
        Boolean mask aligned to the crop region, shape == stop - start. Regions of the crop
        that fall outside `filter_volume`'s covered extent (it only spans the segmentation
        table's anchor points plus a fixed padding, see `filter_cochlea_volume_single`/
        `filter_cochlea_volume`, and can be smaller than the full image -- e.g. for a
        whole-plane 2D crop) are zero-padded, i.e. treated as "not part of the cochlea".
    """
    factors = np.broadcast_to(np.asarray(us_factor, dtype=float), (3,))
    if np.any(factors <= 0):
        raise ValueError(f"us_factor must be positive, got {us_factor}.")

    indices, valid = [], []
    for ax in range(3):
        index = np.floor(np.arange(start[ax], stop[ax]) / factors[ax]).astype(int)
        valid.append((index >= 0) & (index < filter_volume.shape[ax]))
        indices.append(np.clip(index, 0, filter_volume.shape[ax] - 1))

    cropped = filter_volume[np.ix_(*indices)].astype(bool)
    cropped &= valid[0][:, None, None]
    cropped &= valid[1][None, :, None]
    cropped &= valid[2][None, None, :]
    return cropped


def source_types(cochlea: str) -> Dict[str, str]:
    """Read the source names and types of a cochlea's MoBIE dataset.

    Args:
        cochlea: Cochlea name on the S3 bucket.

    Returns:
        Dict of {source name: source type}, with "image", "segmentation" or "spots" as type.
    """
    s3 = create_s3_target()
    with s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8") as f:
        info = json.loads(f.read())
    return {name: next(iter(source)) for name, source in info["sources"].items()}


def resolve_source_name(sources: Dict[str, str], name: str, kind: Optional[str] = None) -> str:
    """Resolve a source name or short alias to the source name of a specific cochlea.

    Source names are not consistent across cochleae, e.g. an SGN segmentation is called "SGN",
    "sgn" or "SGN_v2". This resolves an exact name, a name with different capitalization, or one
    of the `SOURCE_ALIASES` keys. It never guesses between several candidates.

    For an alias, a pinned version from `SOURCE_ALIASES` wins over a source that is literally
    named like the alias, since a cochlea can hold a legacy source called e.g. "IHC" next to the
    IHC_v11 that is used for the analysis. Such a legacy source usually lacks the columns that
    are added by the post-processing (component_labels, length_fraction, ...), so the alias has
    to mean the analysis version, not the source that happens to carry the alias as its name.

    Args:
        sources: Source names and types of the cochlea, see `source_types`.
        name: Source name, or an alias from `SOURCE_ALIASES` such as "SGN", "IHC" or "synapses".
        kind: Required source type, i.e. "image", "segmentation" or "spots". Takes precedence
            over the type of an alias.

    Returns:
        The source name for this cochlea.
    """
    def of_kind(candidates):
        return [candidate for candidate in candidates if kind is None or sources[candidate] == kind]

    prefix, preferred = name, ()
    if name in SOURCE_ALIASES:
        alias_kind, prefix, preferred = SOURCE_ALIASES[name]
        kind = alias_kind if kind is None else kind

        pinned = of_kind([source for source in preferred if source in sources])
        if pinned:
            return pinned[0]
    elif name in sources:
        if kind is not None and sources[name] != kind:
            raise ValueError(f"Source '{name}' has the type '{sources[name]}', but '{kind}' is required.")
        return name

    matches = of_kind([source for source in sources if source.lower() == name.lower()])
    if len(matches) != 1:
        matches = of_kind([source for source in sources if source.lower().startswith(prefix.lower())])

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Source '{name}' matches more than one source: {sorted(matches)}. Use the exact name."
        )
    raise ValueError(
        f"Source '{name}' does not match any source of the cochlea. "
        f"Available: {sorted(of_kind(sources))}."
    )


def synapse_source_for_ihc(sources: Dict[str, str], ihc_name: str) -> str:
    """Find the synapse detection source that was matched to an IHC segmentation.

    Synapse sources are named synapse_<synapse version>_ihc_<IHC version>, and the "matched_ihc"
    column of a synapse table refers to the label IDs of that IHC segmentation only.

    Args:
        sources: Source names and types of the cochlea, see `source_types`.
        ihc_name: Name of the IHC segmentation, e.g. "IHC_v4c".

    Returns:
        The name of the synapse source for this IHC segmentation.
    """
    if "_" not in ihc_name:
        raise ValueError(f"Cannot derive a synapse source from the IHC segmentation '{ihc_name}'.")

    suffix = f"_ihc_{ihc_name.split('_', 1)[1]}".lower()
    candidates = [name for name, kind in sources.items() if kind == "spots" and name.lower().endswith(suffix)]

    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        pinned = [name for name in SOURCE_ALIASES["synapses"][2] if name in candidates]
        if pinned:
            return pinned[0]
        raise ValueError(
            f"Several synapse sources are matched to '{ihc_name}': {sorted(candidates)}. Use the exact name."
        )
    raise ValueError(
        f"No synapse source is matched to '{ihc_name}'. Available: "
        f"{sorted(name for name, kind in sources.items() if kind == 'spots')}."
    )


def find_crop_files(
    folder: str,
    crop_center: List[float],
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
) -> Dict[str, List[str]]:
    """Find the exported files of one crop, grouped by the folder they are in.

    All files of a crop end with the same crop suffix, see `crop_suffix`. The grouping separates
    the scale levels, which the export functions write to one sub-folder each.

    Args:
        folder: Folder to search, searched recursively.
        crop_center: Crop center position as [x, y, z] in µm.
        axis: Optional axis index into (x, y, z) used for the crop, see `crop_suffix`.
        suffix: Optional extra label used for the crop, see `crop_suffix`.

    Returns:
        Dict of {sub-folder: sorted file paths}, sorted by sub-folder.
    """
    pattern = os.path.join(folder, "**", f"*{crop_suffix(crop_center, axis, suffix)}.tif")

    grouped = {}
    for path in sorted(glob(pattern, recursive=True)):
        grouped.setdefault(os.path.dirname(path), []).append(path)
    return {key: grouped[key] for key in sorted(grouped)}


def layer_kind(sources: Dict[str, str], file_name: str) -> str:
    """Determine whether an exported file holds an intensity image or labels.

    Args:
        sources: Source names and types of the cochlea, see `source_types`.
        file_name: Name of the exported file, e.g. "IHC_v4c_crop_0823-1012-0495_apex.tif".

    Returns:
        Either "labels", for a segmentation or a spots source, or "image".
    """
    matches = [source for source in sources if file_name.startswith(source)]
    if not matches:
        return "image"
    source = max(matches, key=len)
    return "labels" if sources[source] in ("segmentation", "spots") else "image"
