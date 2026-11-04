"""Refine the annotations of the IHC_v11 training data.

The script copies IHC_v11_training_data to IHC_v11_training_data_refined, with the same train / val split and
file names. It changes only the annotations:
- IDs above the uint32 maximum get new IDs. The annotations are saved as uint32.
- The edits of `EDITS` fix Martin's comments in ~/Documents/MWfLS/ihc_crops/comments_training_data.md. They use
  the IDs and positions (z, y, x in pixels) as napari shows them in the source annotation.
- `split_reused_ids` makes each ID one connected component. If the crop name gives the crop position, the piece
  at the anchor of the same ID in the S3 table keeps the ID.
- The micro-sam deltas of 2026-11_ihc_micro_sam.py (MICRO_SAM_DELTAS/<crop>.tif) set the voxels of the accepted
  micro-sam corrections. 2026-11_ihc_micro_sam.py computes them on the refined annotations without this step: to
  compute them again, move the deltas away and run the refinement first.

Each change is a row of changes.tsv in the output folder.

The step 'validate_tips' cuts the tips of annotated cells and measures how well `grow_tips` adds them again from
the Vglut3 signal, for the settings TIP_PERCENTILES x TIP_REACHES. The refinement does not use `grow_tips`: with
a threshold low enough to restore a cut tip, it also grows the intact cells, and in the crops with many cut tips
the neighbouring cells give no correct tip signal. The step stays as the baseline for a tip correction with
micro-sam (point 6 of ihc_segmentation_to-do.md).
"""
import argparse
import glob
import json
import os
import re
import shutil

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import binary_erosion, binary_opening, binary_propagation, distance_transform_edt, gaussian_filter
from scipy.ndimage import label as label_structures
from skimage.measure import regionprops
from skimage.morphology import ball
from skimage.segmentation import watershed

from flamingo_tools.analysis.training_data_utils import split_reused_ids
from flamingo_tools.export_data_utils import parse_crop_file
from flamingo_tools.s3_utils import default_table_path, read_table

ROOT = os.path.expanduser("~/Documents/lightsheet-cochlea")
SOURCE = os.path.join(ROOT, "IHC_v11_training_data")
TARGET = os.path.join(ROOT, "IHC_v11_training_data_refined")
COMMENTS = os.path.expanduser("~/Documents/MWfLS/ihc_crops/comments_training_data.md")
# Outside TARGET: the training loaders read every .tif below the data folder.
MICRO_SAM_DELTAS = os.path.join(ROOT, "ihc_micro_sam", "deltas")
TRAINING_CROP_JSONS = os.path.join(os.path.dirname(__file__), "..", "..", "reproducibility", "training_crops")
VOXEL_SIZE = 0.38
UINT32_MAX = np.iinfo("uint32").max
# Smaller pieces of a reused ID are annotation debris, larger pieces are cells (MIN_IHC_SIZE of the quality script).
MIN_CELL_SIZE = 5000
# The S3 anchor of a reused ID selects the piece that keeps the ID if it is at most this far away, in um. The anchors
# of the traced IDs are 0-7 um from one piece; the anchors of the other IDs are more than 200 um away.
MAX_ANCHOR_DISTANCE = 10.0
# IDs above the uint32 maximum are replaced by the next free IDs after the largest ID below this value.
SHORT_ID_LIMIT = 1_000_000
# validate_tips: the threshold of grow_tips is a percentile of the smoothed Vglut3 signal on the surface of the end
# zones of the neighbouring complete cells, and a tip grows at most the reach in um. The tips are cut by TIP_CUTS um.
TIP_PERCENTILES = [25, 50, 75]
TIP_REACHES = [3.0, 5.0]
TIP_CUTS = [2.0, 4.0]
# The fraction of the cell length at each end in which a cell can grow, and in which the neighbours give the
# intensity of a tip.
END_FRACTION = 0.2
SMOOTHING = 1.0

# Crops without a tip comment for the validation of grow_tips.
VALIDATION_CROPS = [
    "M-AMD-000088-R_crop_0723-0383-0381_Vglut3", "M-AMD-000088-R_crop_1106-0696-0771_Vglut3",
    "M-AMD-000112-L_crop_0499-1081-0595_Vglut3", "M-AMD-N153-L_crop_0704-0583-1064_Vglut3",
    "M-AMD-N97-L_crop_0370-1084-0961_Vglut3", "M-AMD-N89-L_crop_0902-0574-0987_Vglut3",
    "M-LR-000249-R_crop_0609-1430-0651_Vglut3", "M-AMD-N75-L_crop_0866-0564-1193_Vglut3",
    "G_EK_000233_L_crop_0593-1018-0276_Vglut3", "G_EK_000233_L_crop_0674-1048-1574_Vglut3",
    "G_EK_000233_L_crop_0788-0937-0255_Vglut3", "G-LR-000234-R_crop_0909-1103-1225_Vglut3",
    "G-LR-000234-L_crop_1246-1024-0782_Vglut3",
]

# The edits per crop: (operation, keyword arguments, Martin's comment). Positions are (z, y, x) in pixels.
EDITS = {
    "G-LR-000234-R_crop_0253-0964-1213_Vglut3": [
        ("remove", dict(label_id=2527), "ID 2527 could be removed. It is very small."),
    ],
    "M-AMD-000088-R_crop_0397-1135-0871_Vglut3": [
        ("merge", dict(label_ids=[474657], to=474), "IDs 474657 and 474 should be merged."),
        ("move", dict(label_id=506, box=(0, 240, None, None, None, None), to=0),
         "ID 506 should not exist below the slice 240."),
    ],
    "M-AMD-N142-R_crop_1213-0848-0665_Vglut3": [
        ("merge", dict(label_ids=[1140], to=1144), "ID 1140 and 1144 should be merged."),
    ],
    "M-AMD-N75-L_crop_0864-0643-0355_Vglut3": [
        ("move_piece", dict(label_id=236, at=(252, 241, 141), to=300),
         "The second piece of 236 has the napari color of 300 and belongs to 300."),
    ],
    "M-AMD-N142-R_crop_0922-0428-0309_Vglut3": [
        ("move_nearest_axis", dict(label_id=803, to=480, contact=2.0, box=(272, None, None, None, None, None)),
         "ID 803 seems to be protruding a bit too much into 480."),
    ],
    "M-AMD-000112-L_crop_1088-0500-1142_Vglut3": [
        ("trim", dict(label_id=765, at=(216, 343, 137), radius=4, to=761),
         "Part of ID 765 has part of its volume inside the 761 instance."),
    ],
    "G_EK_000233_L_crop_1330-1422-0713_Vglut3": [
        ("move", dict(label_id=3, box=(6, 7, None, None, None, 264), to=1),
         "ID 3 has one protruding label into ID 1 in a single slice."),
    ],
    "G_EK_000233_L_crop_1257-1712-0943_Vglut3": [
        ("trim", dict(label_id=4, at=(70, 153, 135), radius=4, to=0), "Weird protrusions of 4, 3, 2."),
        ("trim", dict(label_id=3, at=(46, 140, 136), radius=4, to=0), "Weird protrusions of 4, 3, 2."),
        ("trim", dict(label_id=2, at=(27, 131, 154), radius=5, to=0), "Weird protrusions of 4, 3, 2."),
    ],
    "M-AMD-N142-R_crop_0818-1353-0331_Vglut3": [
        ("trim", dict(label_id=504, at=(219, 266, 79), radius=5, to=0),
         "The small protrusion of ID 504 does not belong to the IHC."),
    ],
    "M-AMD-N153-L_crop_1197-0749-0511_Vglut3": [
        ("trim", dict(label_id=298, at=(243, 205, 107), radius=6, to=0),
         "Two small protrusions of ID 298 and 296 have very little intensity."),
        ("trim", dict(label_id=296, at=(237, 258, 107), radius=5, to=0),
         "Two small protrusions of ID 298 and 296 have very little intensity."),
    ],
    "M-AMD-N153-L_crop_0731-0114-0523_Vglut3": [
        ("trim", dict(label_id=290, at=at, radius=4, to=0), "The small protrusions from ID 290 seem odd.")
        for at in [(238, 253, 119), (255, 238, 111), (276, 260, 132)]
    ],
    "M-AMD-N89-L_crop_1245-1102-1004_Vglut3": [
        ("trim", dict(label_id=866, at=(172, 488, 116), radius=4, to=0), "ID 866 seems to have a weird tip shape."),
    ],
    "MOTOF1L_0580-1175-1000": [
        ("trim", dict(label_id=45842, at=(224, 224, 331), radius=4, to=0),
         "The odd protrusions of 45842 and 4584664854 should be removed."),
        ("trim", dict(label_id=4584664854, at=(253, 211, 347), radius=4, to=0),
         "The odd protrusions of 45842 and 4584664854 should be removed."),
    ],
    "M-AMD-000112-L_crop_0572-0367-0357_Vglut3": [
        ("trim", dict(label_id=232, at=(315, 142, 112), radius=5, to=0),
         "One small protrusion of ID 232 seems like an external small fragment."),
        ("move_piece", dict(label_id=162, at=(109, 510, 88), to=None),
         "No comment: 162 is a cell outside the crop with two slivers at the border, the upper one gets a new ID."),
    ],
    "M226L_Vglut3_2216-0850-1675": [
        ("split_seeds", dict(label_id=979, seed_file="M226L_Vglut3_2216-0850-1675_annotations_edit.tif",
                             seed_slice=42, seed_id=2),
         "ID 979 pretty obviously are two merged instances. I have edited a single slice with label ID 2."),
    ],
}

CHANGE_COLUMNS = ["crop", "split", "step", "label_id", "target", "n_voxels", "z", "y", "x", "comment"]


def crop_paths(root):
    """The annotation paths of a data folder, by crop name."""
    paths = sorted(glob.glob(os.path.join(root, "*", "*_annotations.tif")))
    return {os.path.basename(path)[:-len("_annotations.tif")]: path for path in paths}


def segmentation_channels():
    """The S3 segmentation that the training crops of each cochlea came from."""
    channels = {}
    for path in glob.glob(os.path.join(TRAINING_CROP_JSONS, "*_ihc.json")):
        with open(path) as f:
            params = json.load(f)[0]
        channels[params["dataset_name"]] = params["segmentation_channel"]
    return channels


def s3_anchors(name, labels, channels):
    """The anchor of each label ID in the S3 table of the crop, in crop pixels (z, y, x), or None.

    The crop start is the crop center (um, in the name) in pixels minus half the crop shape, as in the extraction.
    """
    info = parse_crop_file(name)
    if info is None:
        return None
    cochlea = info["source"].replace("-", "_")
    if cochlea not in channels:
        return None
    table = read_table(default_table_path(cochlea, channels[cochlea], s3=True), s3=True)
    start = np.round(np.array(info["crop_center"][::-1]) / VOXEL_SIZE).astype(int) - np.array(labels.shape) // 2
    table = table[table["label_id"].isin(np.unique(labels))]
    positions = table[["anchor_z", "anchor_y", "anchor_x"]].values / VOXEL_SIZE - start
    return dict(zip(table["label_id"].astype(int), positions))


def short_ids(labels):
    """Replace the IDs above the uint32 maximum. Returns the new labels and the mapping of the IDs."""
    ids = np.unique(labels)
    too_large = ids[ids > UINT32_MAX]
    next_id = int(ids[ids < SHORT_ID_LIMIT].max()) + 1
    mapping = {}
    for label_id in too_large:
        while next_id in ids:
            next_id += 1
        mapping[int(label_id)] = next_id
        next_id += 1
    labels = labels.copy()
    for old, new in mapping.items():
        labels[labels == old] = new
    return labels.astype("uint32"), mapping


def _center(voxels):
    return np.round(voxels.mean(axis=0)).astype(int) if len(voxels) else np.zeros(3, dtype=int)


def _box(box, shape):
    return tuple(slice(lo, hi) for lo, hi in zip(box[::2], box[1::2])) if box else tuple(slice(None) for _ in shape)


def _bbox(mask, pad):
    coords = np.argwhere(mask)
    lo = np.maximum(coords.min(axis=0) - pad, 0)
    hi = np.minimum(coords.max(axis=0) + pad + 1, mask.shape)
    return tuple(slice(a, b) for a, b in zip(lo, hi)), lo


def _cell_axis(coords):
    center = coords.mean(axis=0)
    _, vectors = np.linalg.eigh(np.cov((coords - center).T))
    return center, vectors[:, -1]


def _line_distance(coords, center, axis):
    offset = coords - center
    return np.linalg.norm(offset - np.outer(offset @ axis, axis), axis=1)


def _move(labels, voxels, to):
    labels[tuple(voxels.T)] = to
    return [(to, len(voxels), *_center(voxels))]


def op_remove(labels, image, label_id):
    return _move(labels, np.argwhere(labels == label_id), 0)


def op_merge(labels, image, label_ids, to):
    return [row for label_id in label_ids for row in _move(labels, np.argwhere(labels == label_id), to)]


def op_move(labels, image, label_id, box, to):
    """The voxels of `label_id` in the box (z0, z1, y0, y1, x0, x1) go to `to`."""
    box = _box(box, labels.shape)
    start = np.array([sl.start or 0 for sl in box])
    return _move(labels, np.argwhere(labels[box] == label_id) + start, to)


def op_move_piece(labels, image, label_id, at, to):
    """The connected piece of `label_id` nearest to `at` goes to `to`, or to a new ID if `to` is None."""
    to = int(labels.max()) + 1 if to is None else to
    pieces, n_pieces = label_structures(labels == label_id, structure=np.ones((3, 3, 3)))
    voxels = [np.argwhere(pieces == piece) for piece in range(1, n_pieces + 1)]
    nearest = min(voxels, key=lambda v: np.linalg.norm(v - np.array(at), axis=1).min())
    return _move(labels, nearest, to)


def op_trim(labels, image, label_id, at, radius, to):
    """The protrusion of `label_id` nearest to `at` goes to `to`.

    The protrusions are the parts that an opening with a ball of `radius` pixels removes, and that the dilation
    of the opened cell does not restore.
    """
    box, start = _bbox(labels == label_id, radius + 2)
    cell = labels[box] == label_id
    opened, _ = label_structures(binary_opening(cell, ball(radius)))
    opened = opened == np.argmax(np.bincount(opened.ravel())[1:]) + 1
    rest = cell & ~(cell & _dilate(opened, radius))
    pieces, n_pieces = label_structures(rest, structure=np.ones((3, 3, 3)))
    voxels = [np.argwhere(pieces == piece) + start for piece in range(1, n_pieces + 1)]
    nearest = min(voxels, key=lambda v: np.linalg.norm(v - np.array(at), axis=1).min())
    return _move(labels, nearest, to)


def _dilate(mask, radius):
    return distance_transform_edt(~mask) <= radius


def op_move_nearest_axis(labels, image, label_id, to, contact, box):
    """The voxels of `label_id` within `contact` um of `to` and in the box go to `to` if they are nearer to its axis.

    The axis of each cell comes from its voxels more than `contact` um away from the other cell.
    """
    region, start = _bbox(np.isin(labels, [label_id, to]), 4)
    sub = labels[region]
    lines = {}
    for this, other in [(label_id, to), (to, label_id)]:
        far = distance_transform_edt(sub != other) * VOXEL_SIZE > contact
        lines[this] = _cell_axis(np.argwhere((sub == this) & far))
    near = distance_transform_edt(sub != to) * VOXEL_SIZE <= contact
    candidates = np.argwhere((sub == label_id) & near)
    candidates = candidates[_line_distance(candidates, *lines[to]) < _line_distance(candidates, *lines[label_id])]
    inside = np.ones(len(candidates), dtype=bool)
    for dim, sl in enumerate(_box(box, labels.shape)):
        position = candidates[:, dim] + start[dim]
        inside &= (position >= (sl.start or 0)) & (position < (labels.shape[dim] if sl.stop is None else sl.stop))
    return _move(labels, candidates[inside] + start, to)


def op_split_seeds(labels, image, label_id, seed_file, seed_slice, seed_id):
    """Split `label_id` with a seeded watershed on the inverted smoothed signal.

    The seeds are the voxels of `label_id` and `seed_id` in one slice of the seed annotation. The part of
    `seed_id` gets `seed_id`, or the next free ID if `seed_id` is used.
    """
    seeds = tifffile.imread(os.path.join(SOURCE, seed_file))[seed_slice]
    mask = labels == label_id
    markers = np.zeros(labels.shape, dtype="int32")
    markers[seed_slice][(seeds == label_id) & mask[seed_slice]] = 1
    markers[seed_slice][(seeds == seed_id) & mask[seed_slice]] = 2
    new_id = seed_id if not (labels == seed_id).any() else int(labels.max()) + 1
    basins = watershed(-image, markers, mask=mask)
    return _move(labels, np.argwhere(basins == 2), new_id)


OPERATIONS = {
    "remove": op_remove, "merge": op_merge, "move": op_move, "move_piece": op_move_piece, "trim": op_trim,
    "move_nearest_axis": op_move_nearest_axis, "split_seeds": op_split_seeds,
}


def cell_table(labels):
    """The label ID, size, centroid and border flag of each cell of a crop."""
    rows = []
    for prop in regionprops(labels):
        bbox = np.array(prop.bbox)
        border = (bbox[:3] == 0).any() or (bbox[3:] == labels.shape).any()
        rows.append((prop.label, prop.area, *prop.centroid, border))
    return pd.DataFrame(rows, columns=["label_id", "n_voxels", "z", "y", "x", "border"])


def _end_zones(coords, points, fraction=END_FRACTION):
    """Which points are in the outer `fraction` of the cell length at both ends, along the long axis of the cell."""
    center, axis = _cell_axis(coords)
    projection = (coords - center) @ axis
    lo, hi = projection.min(), projection.max()
    margin = fraction * (hi - lo)
    point_projection = (points - center) @ axis
    return (point_projection <= lo + margin) | (point_projection >= hi - margin)


def tip_signal(labels, image, label_id, cells, n_neighbours=6, exclude=()):
    """The signal on the surface of the end zones of the nearest complete cells.

    This is the signal at which the annotators drew the tips. Its percentile is the threshold of grow_tips.
    """
    complete = cells[~cells["border"] & (cells["n_voxels"] >= 0.6 * cells["n_voxels"].median())]
    complete = complete[~complete["label_id"].isin([label_id, *exclude])]
    position = cells.loc[cells["label_id"] == label_id, ["z", "y", "x"]].values[0]
    distance = np.linalg.norm(complete[["z", "y", "x"]].values - position, axis=1)
    values = []
    for neighbour in complete["label_id"].values[np.argsort(distance)[:n_neighbours]]:
        box, start = _bbox(labels == neighbour, 1)
        cell = labels[box] == neighbour
        surface = np.argwhere(cell & ~binary_erosion(cell))
        surface = surface[_end_zones(np.argwhere(cell), surface)]
        values.append(image[box][tuple(surface.T)])
    return np.concatenate(values)


def grow_tips(labels, image, label_id, threshold, reach):
    """The unlabeled voxels to add at the two ends of a cell, as an array of (z, y, x) positions.

    A voxel can be added if its smoothed signal is at least `threshold`, if it is within `reach` um of the cell,
    in the end zones along the long axis of the cell and within the 95th percentile of the radius of the end
    zones. It must connect to the cell through such voxels.
    """
    pad = int(np.ceil(reach / VOXEL_SIZE)) + 2
    box, start = _bbox(labels == label_id, pad)
    cell = labels[box] == label_id
    coords = np.argwhere(cell)
    center, axis = _cell_axis(coords)
    projection = (coords - center) @ axis
    lo, hi = projection.min(), projection.max()
    margin = END_FRACTION * (hi - lo)
    ends = (projection <= lo + margin) | (projection >= hi - margin)
    radius = np.percentile(_line_distance(coords[ends], center, axis), 95)

    grid = np.argwhere(np.ones(cell.shape, dtype=bool))
    grid_projection = (grid - center) @ axis
    in_end = ((grid_projection <= lo + margin) | (grid_projection >= hi - margin)) & (
        _line_distance(grid, center, axis) <= radius)
    candidates = (
        (labels[box] == 0) & (image[box] >= threshold) & in_end.reshape(cell.shape) &
        (distance_transform_edt(~cell) * VOXEL_SIZE <= reach)
    )
    grown = binary_propagation(cell, mask=cell | candidates) & ~cell
    grown = binary_propagation(cell, mask=cell | binary_opening(grown, ball(1))) & ~cell
    return np.argwhere(grown) + start


def smoothed(image):
    return gaussian_filter(image.astype("float32"), SMOOTHING)


def refine_crop(name, path, channels):
    split = os.path.basename(os.path.dirname(path))
    labels = tifffile.imread(path)
    image = tifffile.imread(path.replace("_annotations.tif", ".tif"))
    changes = []

    def record(step, label_id, rows, comment=""):
        changes.extend((name, split, step, label_id, *row, comment) for row in rows)

    if labels.max() > UINT32_MAX:
        labels, mapping = short_ids(labels)
        for old, new in mapping.items():
            record("short_id", old, [(new, int((labels == new).sum()), *_center(np.argwhere(labels == new)))])
    else:
        labels, mapping = labels.astype("uint32"), {}

    signal = smoothed(image) if name in EDITS else None
    for operation, kwargs, comment in EDITS.get(name, []):
        kwargs = dict(kwargs)
        for key in ("label_id", "to"):
            if key in kwargs:
                kwargs[key] = mapping.get(kwargs[key], kwargs[key])
        if "label_ids" in kwargs:
            kwargs["label_ids"] = [mapping.get(i, i) for i in kwargs["label_ids"]]
        rows = OPERATIONS[operation](labels, signal, **kwargs)
        record(operation, kwargs.get("label_id", kwargs.get("label_ids")), rows, comment)

    anchors = s3_anchors(name, labels, channels)
    labels, report = split_reused_ids(
        labels, anchors=anchors, min_size=MIN_CELL_SIZE, max_anchor_distance=MAX_ANCHOR_DISTANCE / VOXEL_SIZE,
    )
    for row in report.itertuples():
        step = f"{row.action}_s3" if row.anchored else row.action
        record(step, row.label_id, [(row.target, row.n_voxels, *np.round([row.z, row.y, row.x]).astype(int))])

    delta_path = os.path.join(MICRO_SAM_DELTAS, f"{name}.tif")
    if os.path.exists(delta_path):
        delta = tifffile.imread(delta_path)
        for label_id in np.unique(delta[delta > 0]):
            voxels = np.argwhere((delta == label_id) & (labels != label_id))
            if len(voxels):
                record("micro_sam", label_id, [(label_id, len(voxels), *_center(voxels))],
                       "micro-sam box prompt, see ihc_micro_sam/targets.tsv")
        labels[delta > 0] = delta[delta > 0]

    labels, report = split_reused_ids(labels, min_size=MIN_CELL_SIZE)
    for row in report.itertuples():
        record(f"final_{row.action}", row.label_id, [(row.target, row.n_voxels, row.z, row.y, row.x)])
    assert split_reused_ids(labels, min_size=MIN_CELL_SIZE)[1].empty, name
    return labels, changes


def update_comments(mapping_rows):
    """Replace the IDs above uint32 in Martin's comments by the new IDs."""
    with open(COMMENTS) as f:
        text = f.read()
    for old, new in mapping_rows:
        text = re.sub(rf"\b{old}\b(?! \(was)", f"{new} (was {old})", text) if f"(was {old})" not in text else text
    with open(COMMENTS, "w") as f:
        f.write(text)


def refine(force):
    channels = segmentation_channels()
    changes = []
    for name, path in crop_paths(SOURCE).items():
        split = os.path.basename(os.path.dirname(path))
        out_dir = os.path.join(TARGET, split)
        os.makedirs(out_dir, exist_ok=True)
        image_out = os.path.join(out_dir, f"{name}.tif")
        if force or not os.path.exists(image_out):
            shutil.copy2(path.replace("_annotations.tif", ".tif"), image_out)
        labels, crop_changes = refine_crop(name, path, channels)
        tifffile.imwrite(os.path.join(out_dir, f"{name}_annotations.tif"), labels, compression="zlib")
        changes.extend(crop_changes)
        print(f"{split}/{name}: {len(crop_changes)} changes")
    changes = pd.DataFrame(changes, columns=CHANGE_COLUMNS)
    changes.to_csv(os.path.join(TARGET, "changes.tsv"), sep="\t", index=False)
    short = changes[changes["step"] == "short_id"]
    update_comments(zip(short["label_id"], short["target"]))
    print(changes.groupby("step").size().to_string())


def validate_tips(n_cells=8, seed=0):
    """Cut the tip of complete cells and measure how much of it grow_tips adds again.

    `recall` is the fraction of the cut voxels that grow_tips adds, `excess` the added voxels outside the cut part
    relative to the cut part. The uncut rows give the voxels that grow_tips adds to an uncut cell, relative to
    its size.
    """
    rng = np.random.default_rng(seed)
    paths = crop_paths(SOURCE)
    rows = []
    for name in VALIDATION_CROPS:
        labels = tifffile.imread(paths[name]).astype("uint32")
        signal = smoothed(tifffile.imread(paths[name].replace("_annotations.tif", ".tif")))
        cells = cell_table(labels)
        median = cells.loc[~cells["border"], "n_voxels"].median()
        picked = cells[~cells["border"] & cells["n_voxels"].between(0.7 * median, 1.3 * median)]
        picked = picked.sample(min(n_cells, len(picked)), random_state=seed)
        for label_id in picked["label_id"]:
            coords = np.argwhere(labels == label_id)
            center, axis = _cell_axis(coords)
            projection = (coords - center) @ axis * (1 if rng.random() < 0.5 else -1)
            for cut in [0.0] + TIP_CUTS:
                cut_voxels = coords[projection > projection.max() - cut / VOXEL_SIZE] if cut else coords[:0]
                cut_labels = labels.copy()
                cut_labels[tuple(cut_voxels.T)] = 0
                cut_set = set(map(tuple, cut_voxels))
                values = tip_signal(cut_labels, signal, label_id, cells)
                for percentile in TIP_PERCENTILES:
                    threshold = np.percentile(values, percentile)
                    for reach in TIP_REACHES:
                        grown = grow_tips(cut_labels, signal, label_id, threshold, reach)
                        recovered = sum(tuple(v) in cut_set for v in grown)
                        rows.append({
                            "crop": name, "species": "gerbil" if name.startswith("G") else "mouse",
                            "label_id": label_id, "cut": cut, "percentile": percentile, "reach": reach,
                            "n_cut": len(cut_voxels), "recall": recovered / len(cut_voxels) if cut else np.nan,
                            "excess": (len(grown) - recovered) / (len(cut_voxels) if cut else len(coords)),
                        })
        print(name, len(picked))
    result = pd.DataFrame(rows)
    os.makedirs(TARGET, exist_ok=True)
    result.to_csv(os.path.join(TARGET, "validate_tips.tsv"), sep="\t", index=False)
    summary = result.groupby(["species", "cut", "percentile", "reach"])[["recall", "excess"]].median()
    print(summary.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-s", "--step", choices=["refine", "validate_tips"], default="refine")
    parser.add_argument("-f", "--force", action="store_true", help="Copy the image files again.")
    args = parser.parse_args()
    if args.step == "refine":
        refine(args.force)
    else:
        validate_tips()


if __name__ == "__main__":
    main()
