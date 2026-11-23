"""Shared code of the IHC count analysis: the annotated crops, the S3 windows and the windowed row correction."""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import tifffile
import zarr
from scipy.spatial import cKDTree
from skimage.measure import label as label_pieces
from skimage.segmentation import relabel_sequential

from flamingo_tools.export_data_utils import parse_crop_file
from flamingo_tools.postprocessing.ihc_morphology import correct_ihc_row, ihc_row_reference
from flamingo_tools.postprocessing.label_components import refine_ihc_row
from flamingo_tools.s3_utils import get_s3_path

SCRIPTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
for folder in ("training", "validation", os.path.join("validation", "IHCs"), "figures"):
    sys.path.append(os.path.join(SCRIPTS, folder))
from export_training_data import DATASETS  # noqa: E402
from run_evaluation import COMPONENT_DICT  # noqa: E402

REPOSITORY = os.path.join(SCRIPTS, "..")
CROP_ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea/IHC_v11_training_data_refined")
VOXEL_SIZE = 0.38
COLUMNS = ["anchor_x", "anchor_y", "anchor_z"]
# Smaller annotations are partial cells at the crop border or annotation debris. The count rule applies the same
# limit to the output instances.
MIN_IHC_SIZE = 5000
# The max_path_deviation of the refined IHC rows, in um.
MAX_PATH_DEVIATION = 15.0
# The cochleae of the IHC test data in scripts/validation/IHCs/run_evaluation.py.
TEST_COCHLEAE = {cochlea for components in COMPONENT_DICT.values() for cochlea in components}
N_BOOTSTRAP = 2000

# The windowed correction of a region (an evaluation plane or a crop), in um along the row path:
# - a larger gap between the row objects of the region starts a new crossing of the row;
# - a crossing is split after MAX_CROSSING, so that the windows stay small;
# - the window holds the row objects within CONTEXT of the crossing, for the reference IHC and the local axis;
# - the path reaches PATH_MARGIN beyond the crossing, and the window HALO beyond the objects.
CROSSING_GAP = 50.0
# ponytail: a merge of two region objects across a split of a long crossing gives two labels. Upgrade: split at
# the largest gap of a long crossing.
MAX_CROSSING = 150.0
CONTEXT = 40.0
PATH_MARGIN = 60.0
HALO = 5.0


def species_of(cochlea):
    return "gerbil" if cochlea.startswith("G") else "mouse"


def gt_crops():
    """The annotated training crops whose name gives the crop center in um, without the test cochleae.

    'val' marks the val crops of the IHC_v11 checkpoint, 'animal' is the cochlea without its side. Names that
    give no crop center (171R HCAT, M226L, MOTOF1L) and empty crops are skipped.
    """
    val_names = set(DATASETS["IHC_v11"]["val"])
    crops = []
    for path in sorted(glob.glob(os.path.join(CROP_ROOT, "*", "*_annotations.tif"))):
        name = os.path.basename(path)[:-len("_annotations.tif")]
        info = parse_crop_file(name)
        if info is None or name.endswith("_empty"):
            continue
        cochlea = info["source"].replace("-", "_")
        if cochlea in TEST_COCHLEAE:
            print(f"Skipping {name}: {cochlea} holds IHC test data.")
            continue
        crops.append({
            "name": name, "path": path, "cochlea": cochlea, "animal": re.sub(r"_[LR]$", "", cochlea),
            "center": info["crop_center"], "val": name in val_names,
        })
    return crops


def crop_info(crop):
    return {"crop": crop["name"], "val": crop["val"], "animal": crop["animal"], "species": species_of(crop["cochlea"])}


def crop_start(crop, shape):
    """The start of a crop in the volume in pixels, in (z, y, x) order."""
    return np.round(np.array(crop["center"][::-1]) / VOXEL_SIZE).astype(int) - np.array(shape) // 2


def annotation_cells(crop):
    """The annotated IHCs of a crop with MIN_IHC_SIZE pixels or more, one label per connected component.

    Like the training transform, a label id that marks two separate cells gives two cells.
    """
    annotation = tifffile.imread(crop["path"])
    pieces = label_pieces(annotation, background=0)
    ids, counts = np.unique(pieces, return_counts=True)
    keep = ids[(ids > 0) & (counts >= MIN_IHC_SIZE)]
    owner = np.zeros(pieces.max() + 1, dtype=annotation.dtype)
    owner[pieces.ravel()] = annotation.ravel()
    owners, n_pieces = np.unique(owner[keep], return_counts=True)
    if (n_pieces > 1).any():
        print(f"{crop['name']}: the annotation ids {owners[n_pieces > 1].tolist()} hold several cells.")
    return relabel_sequential(np.where(np.isin(pieces, keep), pieces, 0))[0].astype("uint32")


def animal_ci(values, animals, rng, denominator=None):
    """The 95% CI of the mean of per-crop values, from resampling the animals.

    With a per-crop `denominator`, the CI of the pooled ratio sum(values) / sum(denominator).
    """
    values = np.asarray(values, dtype="float64")
    denominator = np.ones_like(values) if denominator is None else np.asarray(denominator, dtype="float64")
    groups = [np.flatnonzero(animals == animal) for animal in np.unique(animals)]
    estimates = []
    for _ in range(N_BOOTSTRAP):
        index = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        estimates.append(values[index].sum() / denominator[index].sum())
    return np.percentile(estimates, [2.5, 97.5])


def read_volume(cochlea, segmentation_name, roi, cache_path):
    """Read a region of the S3 segmentation at s0 and cache it as a TIF."""
    if os.path.exists(cache_path):
        return tifffile.imread(cache_path)
    store, _ = get_s3_path(f"{cochlea}/images/ome-zarr/{segmentation_name}.ome.zarr")
    data = zarr.open(store, mode="r")["s0"][roi]
    assert data.max() < 2 ** 32
    data = data.astype("uint32")
    tifffile.imwrite(cache_path, data, compression="zlib")
    return data


def volume_shape(cochlea, segmentation_name):
    store, _ = get_s3_path(f"{cochlea}/images/ome-zarr/{segmentation_name}.ome.zarr")
    return zarr.open(store, mode="r")["s0"].shape


def row_path(table, components):
    """The central path of an IHC row in um, in (x, y, z) order, sampled at 1 um.

    It is the path of `refine_ihc_row`: one path per component, linked in the order of the list. One path through
    all components can miss a part of a row with gaps (M_AMD_N142_R: by 435 um at one crop).
    """
    return refine_ihc_row(table.copy(), components, merge_components=True)[1]


def row_variants(table, components, min_sizes):
    """The label ids and the path (`row_path`) of the IHC row for each variant.

    - 'components': the instances of the listed components;
    - 'refined_<min_size>': `refine_ihc_row`, seeded with the same components, with MAX_PATH_DEVIATION and the
      row merged into one component.
    """
    label_ids = table["label_id"].values.astype("int64")
    row = table["component_labels"].isin(components).values
    variants = {}
    for min_size in min_sizes:
        refined, path, _ = refine_ihc_row(table.copy(), components, min_size, MAX_PATH_DEVIATION,
                                          merge_components=True)
        variants[f"refined_{min_size}"] = (label_ids[refined["component_labels"].values == 1], path)
    return {"components": (label_ids[row], path), **variants}


def crossings(region_ids, position):
    """Group the region objects into runs along the path. `position` maps the row ids to the path index (um)."""
    order = position.loc[region_ids].sort_values()
    groups, start, last = [], None, None
    for label_id, index in order.items():
        if start is None or index - last > CROSSING_GAP or index - start > MAX_CROSSING:
            groups.append([])
            start = index
        groups[-1].append(label_id)
        last = index
    return groups


def window_box(table, ids, shape):
    """The bounding box of the objects plus HALO, in pixels, (z, y, x) order, clipped to the volume."""
    rows = table[table["label_id"].isin(ids)]
    low = np.floor(rows[["bb_min_z", "bb_min_y", "bb_min_x"]].values.min(axis=0) / VOXEL_SIZE - HALO / VOXEL_SIZE)
    high = np.ceil(rows[["bb_max_z", "bb_max_y", "bb_max_x"]].values.max(axis=0) / VOXEL_SIZE + HALO / VOXEL_SIZE)
    low = np.maximum(low.astype(int), 0)
    high = np.minimum(high.astype(int) + 1, shape)
    return low, high


def correct_region(region, start, table, row_ids, path, cochlea, segmentation_name, cache_folder, shape):
    """`correct_ihc_row` in a 3D window around each row crossing of a region.

    Args:
        region: The S3 segmentation in the region, in (z, y, x) order, for example one plane as (1, y, x).
        start: The start of the region in the volume in pixels, in (z, y, x) order.
        table: The segmentation table of the cochlea.
        row_ids: The label ids of the IHC row.
        path: The row path in um, in (x, y, z) order, sampled at 1 um.
        cochlea, segmentation_name: The S3 segmentation, for the windows.
        cache_folder: The folder of the cached windows.
        shape: The volume shape.

    Returns:
        The corrected region. Only the voxels of the row objects of the region take the corrected labels; one id
        per output cell. The counts of the actions of `correct_ihc_row` for these objects, and 'failed' windows.
    """
    region_ids = np.intersect1d(np.unique(region), row_ids)
    rows = table[table["label_id"].isin(row_ids)]
    position = pd.Series(cKDTree(path).query(rows[COLUMNS].values)[1], index=rows["label_id"].values.astype("int64"))
    start = np.asarray(start)
    corrected = np.zeros(region.shape, dtype="uint32")
    actions, offset = {"failed": 0}, 0
    for group in crossings(region_ids, position):
        low_index, high_index = position.loc[group].min(), position.loc[group].max()
        context = position.index[(position >= low_index - CONTEXT) & (position <= high_index + CONTEXT)]
        low, high = window_box(table, context, shape)
        key = "_".join(f"{a}-{b}" for a, b in zip(low, high))
        window = read_volume(cochlea, segmentation_name, tuple(slice(a, b) for a, b in zip(low, high)),
                             os.path.join(cache_folder, f"{cochlea}_{segmentation_name}_{key}.tif"))
        window, forward, _ = relabel_sequential(np.where(np.isin(window, row_ids), window, 0))
        window = window.astype("uint32")
        local_path = path[max(low_index - int(PATH_MARGIN), 0):high_index + int(PATH_MARGIN) + 1]
        local_path = local_path[:, ::-1] / VOXEL_SIZE - low
        group_ids = forward[np.array(group)]
        overlap_low = np.maximum(low, start)
        overlap_high = np.minimum(high, start + np.array(region.shape))
        in_window = tuple(slice(a - w, b - w) for a, b, w in zip(overlap_low, overlap_high, low))
        in_region = tuple(slice(a - s, b - s) for a, b, s in zip(overlap_low, overlap_high, start))
        mask = np.isin(window[in_window], group_ids)
        try:
            reference = ihc_row_reference([window], local_path, VOXEL_SIZE)
            labels, report, _ = correct_ihc_row(window, local_path, reference, VOXEL_SIZE)
        except Exception as error:
            # One failed window keeps its input, so the other crossings of the region are still scored.
            print(f"  correct_ihc_row failed for {len(group)} region objects, window {key}: {error!r}")
            actions["failed"] += 1
            labels, report = window, None
        output = labels[in_window]
        target = corrected[in_region]
        target[mask] = np.where(output[mask] > 0, output[mask] + offset, 0)
        offset += int(labels.max()) + 1
        if report is not None:
            for action, count in report[report["label_id"].isin(group_ids)]["action"].value_counts().items():
                actions[action] = actions.get(action, 0) + int(count)
        print(f"  crossing of {len(group)} region objects, window {high - low} px")
    return corrected, actions
