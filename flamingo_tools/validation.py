import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import zarr

from scipy.ndimage import distance_transform_edt
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential
from tqdm import tqdm

from .s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT


def _normalize_cochlea_name(name):
    match = re.search(r"\d+", name)
    pos = match.start() if match else None
    assert pos is not None, name
    prefix = name[:pos]
    prefix = f"{prefix[0]}_{prefix[1:]}"
    number = int(name[pos:-1])
    postfix = name[-1]
    return f"{prefix}_{number:06d}_{postfix}"


def _parse_annotation_path(annotation_path):
    fname = os.path.basename(annotation_path)
    name_parts = fname.split("_")
    cochlea = _normalize_cochlea_name(name_parts[0])
    slice_id = int(name_parts[2][1:])
    return cochlea, slice_id


def fetch_data_for_evaluation(
    annotation_path: str,
    cache_path: Optional[str] = None,
    seg_name: str = "SGN_v2",
    z_extent: int = 0,
    components_for_postprocessing: Optional[List[int]] = None,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Fetch segmentation from S3 matching the annotation path for evaluation.

    Args:
        annotation_path: The path to the manual annotations.
        cache_path: An optional path for caching the downloaded segmentation.
        seg_name: The name of the segmentation in the bucket.
        z_extent: Additional z-slices to load from the segmentation.
        components_for_postprocessing: The component ids for restricting the segmentation to.
            Choose [1] for the default componentn containing the helix.

    Returns:
        The segmentation downloaded from the S3 bucket.
        The annotations loaded from pandas and matching the segmentation.
    """
    # Load the annotations and normalize them for the given z-extent.
    annotations = pd.read_csv(annotation_path)
    annotations = annotations.drop(columns="index")
    if z_extent == 0:  # If we don't have a z-extent then we just drop the first axis and rename the other two.
        annotations = annotations.drop(columns="axis-0")
        annotations = annotations.rename(columns={"axis-1": "axis-0", "axis-2": "axis-1"})
    else:  # Otherwise we have to center the first axis.
        # TODO
        raise NotImplementedError

    # Load the segmentaiton from cache path if it is given and if it is already cached.
    if cache_path is not None and os.path.exists(cache_path):
        segmentation = imageio.imread(cache_path)
        return segmentation, annotations

    # Parse which ID and which cochlea from the name.
    cochlea, slice_id = _parse_annotation_path(annotation_path)

    # Open the S3 connection, get the path to the SGN segmentation in S3.
    internal_path = os.path.join(cochlea, "images",  "ome-zarr", f"{seg_name}.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

    # Compute the roi for the given z-extent.
    if z_extent == 0:
        roi = slice_id
    else:
        roi = slice(slice_id - z_extent, slice_id + z_extent)

    # Download the segmentation for this slice and the given z-extent.
    input_key = "s0"
    with zarr.open(s3_store, mode="r") as f:
        segmentation = f[input_key][roi]

    if components_for_postprocessing is not None:
        # Filter the IDs so that only the ones part of 'components_for_postprocessing_remain'.

        # First, we download the MoBIE table for this segmentation.
        internal_path = os.path.join(BUCKET_NAME, cochlea, "tables",  seg_name, "default.tsv")
        with fs.open(internal_path, "r") as f:
            table = pd.read_csv(f, sep="\t")

        # Then we get the ids for the components and us them to filter the segmentation.
        component_mask = np.isin(table.component_labels.values, components_for_postprocessing)
        keep_label_ids = table.label_id.values[component_mask].astype("int64")
        filter_mask = ~np.isin(segmentation, keep_label_ids)
        segmentation[filter_mask] = 0

    segmentation, _, _ = relabel_sequential(segmentation)

    # Cache it if required.
    if cache_path is not None:
        imageio.imwrite(cache_path, segmentation, compression="zlib")

    return segmentation, annotations


# We should use the hungarian based matching, but I can't find the bug in it right now.
def _naive_matching(annotations, segmentation, segmentation_ids, matching_tolerance, coordinates):
    distances, indices = distance_transform_edt(segmentation == 0, return_indices=True)

    matched_ids = {}
    matched_distances = {}
    annotation_id = 0
    for _, row in annotations.iterrows():
        coordinate = tuple(int(np.round(row[coord])) for coord in coordinates)
        object_distance = distances[coordinate]
        if object_distance <= matching_tolerance:
            closest_object_coord = tuple(idx[coordinate] for idx in indices)
            object_id = segmentation[closest_object_coord]
            if object_id not in matched_ids or matched_distances[object_id] > object_distance:
                matched_ids[object_id] = annotation_id
                matched_distances[object_id] = object_distance
        annotation_id += 1

    tp_ids_objects = np.array(list(matched_ids.keys()))
    tp_ids_annotations = np.array(list(matched_ids.values()))
    return tp_ids_objects, tp_ids_annotations


# There is a bug in here that neither I nor o3 can figure out ...
def _assignment_based_matching(annotations, segmentation, segmentation_ids, matching_tolerance, coordinates):
    n_objects, n_annotations = len(segmentation_ids), len(annotations)

    # In order to get the full distance matrix, we compute the distance to all objects for each annotation.
    # This is not very efficient, but it's the most straight-forward and most rigorous approach.
    scores = np.zeros((n_objects, n_annotations), dtype="float")
    i = 0
    for _, row in tqdm(annotations.iterrows(), total=n_annotations, desc="Compute pairwise distances"):
        coordinate = tuple(int(np.round(row[coord])) for coord in coordinates)
        distance_input = np.ones(segmentation.shape, dtype="bool")
        distance_input[coordinate] = False
        distances = distance_transform_edt(distance_input)

        props = regionprops_table(segmentation, intensity_image=distances, properties=("label", "min_intensity"))
        distances = props["min_intensity"]
        assert len(distances) == scores.shape[0]
        scores[:, i] = distances
        i += 1

    # Find the assignment of points to objects.
    # These correspond to the TP ids in the point / object annotations.
    tp_ids_objects, tp_ids_annotations = linear_sum_assignment(scores)
    match_ok = scores[tp_ids_objects, tp_ids_annotations] <= matching_tolerance
    tp_ids_objects, tp_ids_annotations = tp_ids_objects[match_ok], tp_ids_annotations[match_ok]
    tp_ids_objects = segmentation_ids[tp_ids_objects]

    return tp_ids_objects, tp_ids_annotations


def compute_matches_for_annotated_slice(
    segmentation: np.typing.ArrayLike,
    annotations: pd.DataFrame,
    matching_tolerance: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Computes the ids of matches and non-matches for a annotated validation slice.

    Computes true positive ids (for objects and annotations), false positive ids and false negative ids
    by solving a linear cost assignment of distances between objects and annotations.

    Args:
        segmentation: The segmentation for this slide. We assume that it is relabeled consecutively.
        annotations: The annotations, marking cell centers.
        matching_tolerance: The maximum distance for matching an annotation to a segmented object.

    Returns:
        A dictionary with keys 'tp_objects', 'tp_annotations' 'fp' and 'fn', mapping to the respective ids.
    """
    assert segmentation.ndim in (2, 3)
    coordinates = ["axis-1", "axis-2"] if segmentation.ndim == 2 else ["axis-0", "axis-1", "axis-2"]
    segmentation_ids = np.unique(segmentation)[1:]

    # Crop to the minimal enclosing bounding box of points and segmented objects.
    seg_mask = segmentation != 0
    if seg_mask.sum() > 0:
        bb_seg = np.where(seg_mask)
        bb_seg = tuple(slice(int(bb.min()), int(bb.max())) for bb in bb_seg)
        bb_points = tuple(
            slice(int(np.floor(annotations[coords].min())), int(np.ceil(annotations[coords].max())) + 1)
            for coords in coordinates
        )
        bbox = tuple(slice(min(bbs.start, bbp.start), max(bbs.stop, bbp.stop)) for bbs, bbp in zip(bb_seg, bb_points))
    else:
        print("The segmentation is empty!!!")
        bbox = tuple(
            slice(int(np.floor(annotations[coords].min())), int(np.ceil(annotations[coords].max())) + 1)
            for coords in coordinates
        )
    segmentation = segmentation[bbox]

    annotations = annotations.copy()
    for coord, bb in zip(coordinates, bbox):
        annotations[coord] -= bb.start
        assert (annotations[coord] <= bb.stop).all()

    # tp_ids_objects, tp_ids_annotations =\
    #     _assignment_based_matching(annotations, segmentation, segmentation_ids, matching_tolerance, coordinates)
    tp_ids_objects, tp_ids_annotations =\
        _naive_matching(annotations, segmentation, segmentation_ids, matching_tolerance, coordinates)
    assert len(tp_ids_objects) == len(tp_ids_annotations)

    # Find the false positives: objects that are not part of the matches.
    fp_ids = np.setdiff1d(segmentation_ids, tp_ids_objects)

    # Find the false negatives: annotations that are not part of the matches.
    fn_ids = np.setdiff1d(np.arange(len(annotations)), tp_ids_annotations)

    return {"tp_objects": tp_ids_objects, "tp_annotations": tp_ids_annotations, "fp": fp_ids, "fn": fn_ids}


def compute_scores_for_annotated_slice(
    segmentation: np.typing.ArrayLike,
    annotations: pd.DataFrame,
    matching_tolerance: float = 0.0,
) -> Dict[str, int]:
    """Computes the scores for a annotated validation slice.

    Computes true positives, false positives and false negatives for scoring.

    Args:
        segmentation: The segmentation for this slide. We assume that it is relabeled consecutively.
        annotations: The annotations, marking cell centers.
        matching_tolerance: The maximum distance for matching an annotation to a segmented object.

    Returns:
        A dictionary with keys 'tp', 'fp' and 'fn', mapping to the respective counts.
    """
    result = compute_matches_for_annotated_slice(segmentation, annotations, matching_tolerance)

    # To determine the TPs, FPs and FNs.
    tp = len(result["tp_objects"])
    fp = len(result["fp"])
    fn = len(result["fn"])
    return {"tp": tp, "fp": fp, "fn": fn}


def create_consensus_annotations(
    annotation_paths: Dict[str, str],
    matching_distance: float = 5.0,
    min_matches_for_consensus: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create a consensus annotation from multiple manual annotations.

    Args:
        annotation_paths: A dictionary that maps annotator names to the path to the manual annotations.
        matching_distance: The maximum distance for matching annotations to a consensus annotation.
        min_matches_for_consensus: The minimum number of matching annotations to consider an annotation as consensus.

    Returns:
        A dataframe with the consensus annotations.
        A dataframe with the unmatched annotations.
    """
    dfs, coords, ann_id = [], [], []
    for name, path in annotation_paths.items():
        df = pd.read_csv(path, usecols=["axis-0", "axis-1", "axis-2"])
        df["annotator"] = name
        dfs.append(df)
    big = pd.concat(dfs, ignore_index=True)
    coords = big[["axis-0", "axis-1", "axis-2"]].values
    ann_id = big["annotator"].values

    trees, idx_by_ann = {}, {}
    for ann in np.unique(ann_id):
        idx = np.where(ann_id == ann)[0]
        idx_by_ann[ann] = idx
        trees[ann] = cKDTree(coords[idx])

    edges = []
    for i, annA in enumerate(trees):
        idxA, treeA = idx_by_ann[annA], trees[annA]
        for annB in list(trees)[i+1:]:
            idxB, treeB = idx_by_ann[annB], trees[annB]

            # A -> B
            dAB, jB = treeB.query(coords[idxA], distance_upper_bound=matching_distance)
            # B -> A
            dBA, jA = treeA.query(coords[idxB], distance_upper_bound=matching_distance)

            for k, (d, j) in enumerate(zip(dAB, jB)):
                if np.isfinite(d):
                    a_idx = idxA[k]
                    b_idx = idxB[j]
                    # check reciprocity
                    if jA[j] == k and np.isfinite(dBA[j]):
                        edges.append((a_idx, b_idx))

    # --- union–find to group ---------------------------------
    parent = np.arange(len(coords))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)

    # --- collect results -------------------------------------
    cluster = defaultdict(list)
    for i in range(len(coords)):
        cluster[find(i)].append(i)

    consensus_rows, unmatched = [], []
    for members in cluster.values():
        if len(members) >= min_matches_for_consensus:
            anns = {ann_id[m] for m in members}
            # by construction anns are unique
            subset = coords[members]
            rep_pt = subset.mean(0)
            consensus_rows.append({
                "axis-0": rep_pt[0],
                "axis-1": rep_pt[1],
                "axis-2": rep_pt[2],
                "annotators": anns,
                "member_indices": members
            })
        else:
            unmatched.extend(members)

    consensus_df = pd.DataFrame(consensus_rows)
    unmatched_df = big.iloc[unmatched].reset_index(drop=True)
    return consensus_df, unmatched_df


def for_visualization(segmentation, annotations, matches):
    green_red = ["#00FF00", "#FF0000"]

    seg_vis = np.zeros_like(segmentation)
    tps, fps = matches["tp_objects"], matches["fp"]
    seg_vis[np.isin(segmentation, tps)] = 1
    seg_vis[np.isin(segmentation, fps)] = 2

    seg_props = dict(colormap={1: green_red[0], 2: green_red[1]})

    point_vis = annotations.copy()
    tps = matches["tp_annotations"]
    match_properties = ["tp" if aid in tps else "fn" for aid in range(len(annotations))]
    # The color cycle assigns the first color to the first property etc.
    # So we need to set the first color to red if the first id is a false negative and vice versa.
    color_cycle = green_red[::-1] if match_properties[0] == "fn" else green_red
    point_props = dict(
        properties={
            "id": list(range(len(annotations))),
            "match": match_properties,
        },
        face_color="match",
        face_color_cycle=color_cycle,
        border_width=0.25,
        size=10,
    )

    return seg_vis, point_vis, seg_props, point_props
