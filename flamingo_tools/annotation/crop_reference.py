"""Find the position of an annotation crop in its cochlea and read the S3 reference segmentation there.

The crop name gives the cochlea and a center triplet. The legacy training crops are listed in `doc/data/*.tsv`
with their standardized names. The triplet is usually the crop center in micrometer in (x, y, z) order, but not
for all crops: `M226L_*` crops give the center in s0 pixels in (z, y, x) order, and `MOTOF1L_0580-1175-1000`
holds the last 330 slices of a 512 box. So each position hypothesis is checked by comparing the center patch of
the crop image with the image source of its stain on S3.
"""
import glob
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import tifffile

STAINS = {"Vglut3": "ihc", "MYO7A": "ihc", "Myo7a": "ihc", "PV": "sgn", "CR": "sgn"}
TABLE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "doc", "data")
# The edge of the checked center patch in pixels.
PATCH = 32
# The box of the crop export in pixels. A crop that is shorter in z can be the start or the end of the box.
EXPORT_BOX = 512
# The reference segmentations, newest model first. A crop uses the first one that its cochlea has.
REFERENCE_ORDER = {"ihc": ("IHC_v11b", "IHC_v11", "IHC_v10", "IHC_v9"), "sgn": ("SGN_v3", "SGN_v2")}
RESIZED_REFERENCE_ORDER = {"sgn": ("SGN_resized_v2",)}
# The IHC model segmentations on S3, for the region proposal.
IHC_MODEL = re.compile(r"^IHC_v(\d+)([a-z]*)$")


@dataclass
class CropInfo:
    """The position of a crop: the start in s0 pixels in (z, y, x) order."""
    name: str
    cochlea: str
    start: Tuple[int, int, int]
    shape: Tuple[int, int, int]
    stain: Optional[str]
    resized: bool = False


def _table_entry(name: str) -> Optional[Dict[str, str]]:
    """The row of a crop in the tables of the training data, by its original name."""
    for path in sorted(glob.glob(os.path.join(TABLE_FOLDER, "*.tsv"))):
        table = pd.read_csv(path, sep="\t", dtype=str)
        if "Original" not in table.columns:
            continue
        match = table[table["Original"] == name]
        if len(match):
            return match.iloc[0].to_dict()
    return None


def stain_of(name: str) -> Optional[str]:
    tokens = re.split(r"[_\-]", name)
    return next((token for token in reversed(tokens) if token in STAINS), None)


def cell_type_of(stain: Optional[str]) -> Optional[str]:
    return STAINS.get(stain) if stain else None


def crop_candidates(name: str) -> Tuple[Optional[str], Optional[np.ndarray], Optional[str], bool]:
    """The cochlea, the center triplet, the stain and the resized flag of a crop name.

    The triplet is returned as it is written, see `position_hypotheses` for its meaning.
    """
    from flamingo_tools.export_data_utils import parse_crop_file

    entry = _table_entry(name)
    standardized = entry["Standardized"] if entry else name
    resized = "resized" in standardized
    stain = stain_of(standardized) or stain_of(name)
    info = parse_crop_file(standardized)
    if info is None:
        match = re.search(r"_(\d{3,4})-(\d{3,4})-(\d{3,4})", standardized)
        if match is None:
            return None, None, stain, resized
        triplet = np.array([float(v) for v in match.groups()])
        source = standardized[:match.start()].split("_")[0]
    else:
        triplet = np.array(info["crop_center"])
        source = info["source"]
    cochlea = re.sub(r"[_\-]resized$", "", source.split("_crop")[0]).replace("-", "_")
    return cochlea, triplet, stain, resized


def position_hypotheses(triplet: np.ndarray, shape: Sequence[int], voxel_size: float) -> List[Tuple[int, ...]]:
    """Candidate crop starts in (z, y, x) order, the usual one first.

    - the triplet is the center in micrometer in (x, y, z) order;
    - the triplet is the center in s0 pixels in (z, y, x) order;
    - as the first, but the crop is the end or the start of an EXPORT_BOX box in z.
    """
    shape = np.asarray(shape)
    center_um = np.round(np.asarray(triplet)[::-1] / voxel_size).astype(int)
    hypotheses = [center_um - shape // 2, np.asarray(triplet).astype(int) - shape // 2]
    if shape[0] < EXPORT_BOX:
        for z_start in (center_um[0] + EXPORT_BOX // 2 - shape[0], center_um[0] - EXPORT_BOX // 2):
            hypotheses.append(np.array([z_start, *(center_um[1:] - shape[1:] // 2)]))
    return [tuple(int(v) for v in h) for h in hypotheses]


def read_padded(array, start: Sequence[int], shape: Sequence[int]) -> np.ndarray:
    """Read a box from an array-like volume. Voxels outside the volume are zero."""
    start, shape = np.asarray(start), np.asarray(shape)
    low = np.maximum(start, 0)
    high = np.minimum(start + shape, array.shape)
    out = np.zeros(tuple(shape), dtype=array.dtype)
    if (high <= low).any():
        return out
    target = tuple(slice(lo - s, hi - s) for lo, hi, s in zip(low, high, start))
    out[target] = array[tuple(slice(lo, hi) for lo, hi in zip(low, high))]
    return out


def _matches(patch: np.ndarray, reference: np.ndarray, min_correlation: float = 0.95) -> bool:
    if patch.shape != reference.shape:
        return False
    if np.array_equal(patch, reference):
        return True
    a, b = patch.astype(float).ravel(), reference.astype(float).ravel()
    if a.std() == 0 or b.std() == 0:
        return False
    return float(np.corrcoef(a, b)[0, 1]) > min_correlation


def open_source(cochlea: str, source: str):
    """The s0 array of an S3 source, as a lazy zarr array."""
    import zarr
    from flamingo_tools.s3_utils import get_s3_path

    store, _ = get_s3_path(f"{cochlea}/images/ome-zarr/{source}.ome.zarr")
    return zarr.open(store, mode="r")["s0"]


def verify_position(
    image: np.ndarray, cochlea: str, stain: str, hypotheses: List[Tuple[int, ...]], open_array=open_source,
) -> Optional[Tuple[int, ...]]:
    """The first crop start whose center patch equals the S3 image of the stain, or None."""
    volume = open_array(cochlea, stain)
    shape = np.asarray(image.shape)
    offset = np.maximum(shape // 2 - PATCH // 2, 0)
    size = np.minimum(shape, PATCH)
    patch = image[tuple(slice(o, o + s) for o, s in zip(offset, size))]
    for start in hypotheses:
        reference = read_padded(volume, np.asarray(start) + offset, size)
        if _matches(patch, reference):
            return start
    return None


def reference_candidates(cell_type: str, resized: bool = False) -> Tuple[str, ...]:
    """The default reference segmentations of a cell type, newest model first."""
    order = RESIZED_REFERENCE_ORDER if resized else REFERENCE_ORDER
    if cell_type not in order:
        raise ValueError(f"No reference segmentations are known for {cell_type} (resized: {resized}).")
    return order[cell_type]


def select_reference(sources: Dict[str, str], candidates: Sequence[str]) -> Optional[str]:
    """The first candidate that is a segmentation source of the cochlea, or None."""
    return next((name for name in candidates if sources.get(name) == "segmentation"), None)


def model_rank(name: str) -> Tuple[int, str]:
    """The age of an IHC model source: a larger rank is a newer model (IHC_v11b > IHC_v11 > IHC_v10 > IHC_v4b)."""
    match = IHC_MODEL.match(name)
    if match is None:
        raise ValueError(f"{name} is not an IHC model source (IHC_v<number><suffix>).")
    return int(match.group(1)), match.group(2)


def ihc_sources(sources: Dict[str, str]) -> List[str]:
    """The IHC model segmentations of a cochlea, newest first."""
    names = [name for name, kind in sources.items() if kind == "segmentation" and IHC_MODEL.match(name)]
    return sorted(names, key=model_rank, reverse=True)


def reference_cache_path(cache_dir: str, name: str, source: str) -> str:
    """The path of the cached reference segmentation of a crop."""
    return os.path.join(cache_dir, f"{name}_{source}.tif")


def locate_crop(
    name: str,
    image: np.ndarray,
    cochlea: Optional[str] = None,
    center: Optional[Sequence[float]] = None,
    voxel_size: float = 0.38,
) -> Tuple[Optional[CropInfo], Dict[str, str]]:
    """The position of a crop in its cochlea, checked against the S3 image of the stain, and the S3 sources.

    Args:
        name: The crop name, the image file name without '.tif'.
        image: The crop image, to check the position.
        cochlea: The cochlea, if the name does not give it.
        center: The crop center in micrometer in (x, y, z) order, if the name does not give it.
        voxel_size: The voxel size of s0 in micrometer.

    Returns:
        The crop position, or None if it is unknown (the reason is printed). The source types of the cochlea.
    """
    from flamingo_tools.export_data_utils import source_types

    found_cochlea, triplet, stain, resized = crop_candidates(name)
    cochlea = cochlea or found_cochlea
    if center is not None:
        triplet = np.asarray(center, dtype=float)
    if cochlea is None or triplet is None:
        print(f"{name}: the crop position is unknown, so there is no S3 reference.")
        return None, {}

    hypotheses = position_hypotheses(triplet, image.shape, voxel_size)
    if center is not None:
        hypotheses = hypotheses[:1]
    sources = source_types(cochlea)
    if stain in sources:
        start = verify_position(image, cochlea, stain, hypotheses)
        if start is None:
            print(f"{name}: the crop image does not match {cochlea}/{stain} at any position, so there is no S3 "
                  "reference. Give the cochlea and the center.")
            return None, sources
    else:
        start = hypotheses[0]
        print(f"{name}: {cochlea} has no image source {stain}, so the position is not checked.")
    return CropInfo(name, cochlea, start, tuple(image.shape), stain, resized), sources


def read_reference(info: CropInfo, source: str, cache_dir: Optional[str] = None) -> np.ndarray:
    """The segmentation of an S3 source at the crop position, as uint32, from the cache if it is there."""
    cache_path = reference_cache_path(cache_dir, info.name, source) if cache_dir else None
    if cache_path and os.path.exists(cache_path):
        return tifffile.imread(cache_path)
    reference = read_padded(open_source(info.cochlea, source), info.start, info.shape)
    if reference.max() >= 2 ** 32:
        raise ValueError(f"The label ids of {info.cochlea}/{source} do not fit into uint32.")
    reference = reference.astype("uint32")
    if cache_path:
        tifffile.imwrite(cache_path, reference, compression="zlib")
    return reference


def load_reference(
    name: str,
    image: np.ndarray,
    cell_type: str,
    cache_dir: Optional[str] = None,
    candidates: Optional[Sequence[str]] = None,
    cochlea: Optional[str] = None,
    center: Optional[Sequence[float]] = None,
    voxel_size: float = 0.38,
) -> Tuple[Optional[np.ndarray], Optional[str], Optional[CropInfo]]:
    """Read the S3 reference segmentation of a crop.

    Args:
        name: The crop name, the image file name without '.tif'.
        image: The crop image, to check the position.
        cell_type: 'ihc' or 'sgn'. It selects the default candidates, see `REFERENCE_ORDER`.
        cache_dir: Folder for the cached reference crop.
        candidates: The segmentation sources to try, in order. The crop uses the first one of its cochlea.
        cochlea: The cochlea, if the name does not give it.
        center: The crop center in micrometer in (x, y, z) order, if the name does not give it.
        voxel_size: The voxel size of s0 in micrometer.

    Returns:
        The reference segmentation of the crop (uint32), the source name and the crop position. All are None if
        the crop has no position or no reference. The reason is printed.
    """
    info, sources = locate_crop(name, image, cochlea, center, voxel_size)
    if info is None:
        return None, None, None
    candidates = candidates or reference_candidates(cell_type, info.resized)
    source = select_reference(sources, candidates)
    if source is None:
        family = "IHC" if cell_type == "ihc" else "SGN"
        available = sorted(n for n, kind in sources.items() if kind == "segmentation" and n.startswith(family))
        print(f"{name}: {info.cochlea} has none of the segmentations {', '.join(candidates)}. It has: "
              f"{', '.join(available) or 'none'}.")
        return None, None, info
    return read_reference(info, source, cache_dir), source, info
