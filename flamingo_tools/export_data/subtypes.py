"""Export the SGN subtypes of a segmentation, defined by the marker labels of the subtype stains.
"""
import os
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from flamingo_tools.export_data_utils import (
    cochlea_table, export_bounding_box, export_name, export_output_path, normalize_scales, normalize_voxel_size,
    write_volume,
)
from flamingo_tools.file_utils import open_scale_level
from flamingo_tools.postprocessing.sgn_subtype_utils import COCHLEAE, STAIN_TO_TYPE
from flamingo_tools.s3_utils import MOBIE_FOLDER, default_image_path

# Merged masks use consecutive IDs after they select the types that apply to one cochlea.
# This order gives N98L the IDs 1=Ia, 2=Ib, 3=Ic and 4=II.
SUBTYPE_ID_ORDER = ("Type Ia", "Type Ib", "Type IbIc", "Type Ic", "Type I", "Type II", "inconclusive")


def types_for_stain(stains):
    stains = sorted(stains)
    if len(stains) not in (1, 2):
        raise ValueError(f"Give one or two subtype stains, got {stains}.")
    if len(stains) == 1:
        combinations = [f"{stains[0]}+", f"{stains[0]}-"]
    else:
        combinations = [
            f"{stains[0]}+/{stains[1]}+",
            f"{stains[0]}+/{stains[1]}-",
            f"{stains[0]}-/{stains[1]}+",
            f"{stains[0]}-/{stains[1]}-"
        ]
    unknown = [combination for combination in combinations if combination not in STAIN_TO_TYPE]
    if unknown:
        raise ValueError(
            f"No subtype is defined for the stain combinations {unknown}, see "
            "flamingo_tools.postprocessing.sgn_subtype_utils.STAIN_TO_TYPE."
        )
    types = list(set([STAIN_TO_TYPE[stain] for stain in combinations]))
    return types


def subtype_ids_for_stains(stains):
    """Return consecutive label IDs for the subtype classes defined by the stains."""
    subtypes = set(types_for_stain(stains))
    unknown = sorted(subtypes.difference(SUBTYPE_ID_ORDER))
    if unknown:
        raise ValueError(f"No subtype ID order is defined for: {unknown}.")
    ordered = [subtype for subtype in SUBTYPE_ID_ORDER if subtype in subtypes]
    return {subtype: type_id for type_id, subtype in enumerate(ordered, start=1)}


def stain_expression_from_subtype(subtype, stains):
    assert len(stains) in (1, 2)
    dic_list = []
    if len(stains) == 1:
        possible_key = [
            key for key in STAIN_TO_TYPE.keys()
            if STAIN_TO_TYPE[key] == subtype and len(key.split("/")) != 2 and stains[0] in key
        ][0]
        dic = {stains[0]: possible_key[-1:]}
        dic_list.append(dic)

    else:
        possible_keys = [
            key for key in STAIN_TO_TYPE.keys()
            if STAIN_TO_TYPE[key] == subtype and len(key.split("/")) > 1 and all([stain in key for stain in stains])
        ]
        for key in possible_keys:
            stain1 = key.split("/")[0][:-1]
            stain2 = key.split("/")[1][:-1]
            expression1 = key.split("/")[0][-1:]
            expression2 = key.split("/")[1][-1:]
            dic = {stain1: expression1, stain2: expression2}
            dic_list.append(dic)

    return dic_list


def label_ids_for_subtype(table_seg, subtype, stains):
    """Return the segmentation instance IDs assigned to one subtype.

    A subtype can have several stain expressions, e.g. "Type II" for Prph+/Tuj1+ and Prph+/Tuj1-.
    The result is the union of the instances of all expressions.
    """
    label_ids = []
    for expression in stain_expression_from_subtype(subtype, stains):
        subset = table_seg
        for stain, sign in expression.items():
            expression_value = 1 if sign == "+" else 2
            subset = subset.loc[subset[f"marker_{stain}"] == expression_value]
        label_ids.extend(subset["label_id"].tolist())
    return label_ids


def merge_subtype_masks(segmentation, table_seg, stains, subtype_ids):
    """Replace segmentation instance IDs with subtype IDs in one label volume."""
    merged = np.zeros(segmentation.shape, dtype="uint8")
    for subtype, type_id in subtype_ids.items():
        label_ids = label_ids_for_subtype(table_seg, subtype, stains)
        subtype_mask = np.isin(segmentation, label_ids)
        if np.any(merged[subtype_mask] != 0):
            raise ValueError(f"Subtype '{subtype}' overlaps a subtype that was already assigned.")
        merged[subtype_mask] = type_id
        print(f"subtype {subtype}: id={type_id}, {len(label_ids)} instances")
    return merged


def subtype_groups(stains: Sequence[str]) -> Dict[str, List[str]]:
    """Get the subtypes that are exported as one binary mask each, for the stains of a cochlea.

    Args:
        stains: The subtype stains, one or two.

    Returns:
        Dict of {name in the file name: subtypes}. Type Ib and Type Ic are also exported together.
    """
    subtypes = sorted(types_for_stain(stains))
    groups = {subtype.replace(" ", ""): [subtype] for subtype in subtypes}
    if "Type Ib" in subtypes and "Type Ic" in subtypes:
        groups["TypeIbTypeIc"] = ["Type Ib", "Type Ic"]
    return groups


def export_subtypes(
    input_path: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    table: pd.DataFrame,
    stains: Sequence[str],
    input_key: Optional[str] = "s0",
    name: Optional[str] = None,
    merge_masks: bool = False,
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    voxel_size: Union[float, Sequence[float]] = (0.38, 0.38, 0.38),
    force: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> None:
    """Export the SGN subtypes of a segmentation at lower resolutions.

    Without `merge_masks`, the output for each scale s is one binary mask per subtype,
    <output_folder>/scale<s>/<name>_<subtype><crop suffix>.tif, see `subtype_groups`. With `merge_masks`, it is
    one label volume <name>_subtypes<crop suffix>.tif, with the IDs of `subtype_ids_for_stains`.

    Args:
        input_path: Path to the segmentation: a tif file, or a zarr/n5 container.
        output_folder: Output folder.
        scale: Scale level(s), see `flamingo_tools.file_utils.open_scale_level`.
        table: Segmentation table with a column "marker_<stain>" (1 = positive, 2 = negative) for each stain.
        stains: The subtype stains, one or two, e.g. ["CR", "Ntng1"].
        input_key: Key of the segmentation in the container. Ignored for a tif file.
        name: Name of the export. Default: the file name of the input without its extension.
        merge_masks: Write one label volume with all subtypes instead of one binary mask per subtype.
        crop_center: Optional crop center as [x, y, z] in µm, see
            `flamingo_tools.export_data.lower_resolution.export_lower_resolution`.
        roi_halo: Halo around the crop center in pixels at the target scale.
        axis: Optional axis index into (x, y, z) to crop as a single-pixel 2D slice at the crop center.
        suffix: Optional label appended to the name of a crop.
        voxel_size: Voxel size of the input in µm, scalar or (x, y, z).
        force: Replace an existing output. Otherwise, an existing output is kept.
        s3: Read the input from the S3 bucket.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    if merge_masks:
        subtype_ids = subtype_ids_for_stains(stains)
        outputs = {"subtypes": None}
    else:
        outputs = subtype_groups(stains)
    missing_columns = sorted({"label_id", *(f"marker_{stain}" for stain in stains)}.difference(table.columns))
    if missing_columns:
        raise ValueError(f"The subtype table is missing required columns: {missing_columns}.")

    voxel_size = normalize_voxel_size(voxel_size)
    name = export_name(input_path) if name is None else name
    print(f"Subtype stains: {list(stains)}.")
    if merge_masks:
        print(f"Merged subtype IDs: {subtype_ids}")

    for s in normalize_scales(scale):
        out_folder = os.path.join(output_folder, f"scale{s}")
        os.makedirs(out_folder, exist_ok=True)

        out_paths = {}
        for output in outputs:
            out_path = export_output_path(out_folder, f"{name}_{output}", False, crop_center, axis, suffix)
            if os.path.exists(out_path) and not force:
                print(f"Skipping {out_path}. File already exists.")
            else:
                out_paths[output] = out_path
        if not out_paths:
            continue

        level = open_scale_level(
            input_path, input_key, s, from_s3=s3, bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint,
            credential_file=s3_credentials,
        )
        start, stop = export_bounding_box(level.shape, crop_center, roi_halo, voxel_size, s, axis)
        segmentation = level.read(start, stop, labels=True)
        print("Data shape", segmentation.shape)

        for output, out_path in out_paths.items():
            if merge_masks:
                data = merge_subtype_masks(segmentation, table, stains, subtype_ids)
            else:
                label_ids = []
                for subtype in outputs[output]:
                    label_ids.extend(label_ids_for_subtype(table, subtype, stains))
                print(f"subtypes {'/'.join(outputs[output])} with {len(label_ids)} instances")
                data = np.isin(segmentation, label_ids).astype("float32")
            write_volume(out_path, data, voxel_size, s)


def export_subtypes_cochlea(
    cochlea: str,
    output_folder: str,
    scale: Union[int, Sequence[int]],
    segmentation: Optional[str] = None,
    stains: Optional[Sequence[str]] = None,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **kwargs,
) -> None:
    """Export the SGN subtypes of a cochlea in a MoBIE project.

    The output is written to <output_folder>/<cochlea>, see `export_subtypes`.

    Args:
        cochlea: Name of the cochlea.
        output_folder: Output folder.
        scale: Scale level(s).
        segmentation: Name of the SGN segmentation. Default: the "output_seg" or the "seg_data" entry of the
            cochlea in `flamingo_tools.postprocessing.sgn_subtype_utils.COCHLEAE`.
        stains: The subtype stains. Default: the "subtype_stains" entry of the cochlea in `COCHLEAE`.
        mobie_dir: Local MoBIE project directory. Ignored when s3 is set.
        s3: Read the data from the S3 bucket instead of the local MoBIE project.
        s3_credentials: File path to credentials for the S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        kwargs: Further arguments of `export_subtypes`, e.g. merge_masks, crop_center, roi_halo, axis, suffix,
            voxel_size or force.
    """
    info = COCHLEAE.get(cochlea, {})
    if segmentation is None:
        segmentation = info.get("output_seg") or info.get("seg_data")
    if stains is None:
        stains = info.get("subtype_stains")
    if segmentation is None or stains is None:
        raise ValueError(
            f"Cochlea {cochlea} has no entry in COCHLEAE. Give the segmentation and the stains."
        )

    s3_kwargs = dict(
        s3=s3, s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name, s3_service_endpoint=s3_service_endpoint,
    )
    export_subtypes(
        default_image_path(cochlea, segmentation, s3=s3, mobie_dir=mobie_dir),
        os.path.join(output_folder, cochlea),
        scale,
        cochlea_table(cochlea, segmentation, mobie_dir=mobie_dir, **s3_kwargs),
        stains,
        input_key="s0",
        name=segmentation,
        **s3_kwargs,
        **kwargs,
    )
