import argparse
import os
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import tifffile
import zarr

from flamingo_tools.export_data_utils import compute_crop_bb, crop_suffix
from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
from flamingo_tools.postprocessing.sgn_subtype_utils import STAIN_TO_TYPE, COCHLEAE
# from skimage.segmentation import relabel_sequential


# Merged masks use consecutive IDs after they select the types that apply to one cochlea.
# This order gives N98L the IDs 1=Ia, 2=Ib, 3=Ic and 4=II.
SUBTYPE_ID_ORDER = ("Type Ia", "Type Ib", "Type IbIc", "Type Ic", "Type I", "Type II", "inconclusive")


def types_for_stain(stains):
    stains = sorted(stains)
    assert len(stains) in (1, 2)
    if len(stains) == 1:
        combinations = [f"{stains[0]}+", f"{stains[0]}-"]
    else:
        combinations = [
            f"{stains[0]}+/{stains[1]}+",
            f"{stains[0]}+/{stains[1]}-",
            f"{stains[0]}-/{stains[1]}+",
            f"{stains[0]}-/{stains[1]}-"
        ]
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
    """Return the segmentation instance IDs assigned to one subtype."""
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


def read_subtype_table(cochlea, seg_name):
    """Read the default table for a subtype segmentation."""
    internal_path = os.path.join(cochlea, "tables", seg_name, "default.tsv")
    tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    with fs.open(tsv_path, "r") as f:
        return pd.read_csv(f, sep="\t")


def stain_to_type(stain):
    # Normalize the staining string.
    stains = stain.replace(" ", "").split("/")
    assert len(stains) in (1, 2)

    if len(stains) == 1:
        stain_norm = stain
    else:
        s1, s2 = sorted(stains)
        stain_norm = f"{s1}/{s2}"

    if stain_norm not in STAIN_TO_TYPE:
        breakpoint()
        raise ValueError(f"Invalid stain combination: {stain_norm}")

    return STAIN_TO_TYPE[stain_norm], stain_norm


def filter_subtypes(cochlea, segmentation, seg_name, subtype):
    """Filter segmentation with marker labels.
    Positive segmentation instances are set to 1, negative to 2.
    """
    table_seg = read_subtype_table(cochlea, seg_name)

    # get stains
    stains = [column.split("_")[1] for column in list(table_seg.columns) if "marker_" in column]
    stains.sort()

    if isinstance(subtype, str):
        stain_dict = stain_expression_from_subtype(subtype, stains)
        if len(stain_dict) == 0:
            raise ValueError("The dictionary containing stain information must have at least one entry. "
                             "Check parameters.")

        subset = table_seg.copy()

        for dic in stain_dict:
            for stain in dic.keys():
                expression_value = 1 if dic[stain] == "+" else 2
                subset = subset.loc[subset[f"marker_{stain}"] == expression_value]

        label_ids_subtype = list(subset["label_id"])
        print(f"subtype {subtype} with {len(label_ids_subtype)} instances")

    else:
        label_ids_subtype = []
        for sub in subtype:
            stain_dict = stain_expression_from_subtype(sub, stains)
            if len(stain_dict) == 0:
                raise ValueError("The dictionary containing stain information must have at least one entry. "
                                 "Check parameters.")

            subset = table_seg.copy()

            for dic in stain_dict:
                for stain in dic.keys():
                    expression_value = 1 if dic[stain] == "+" else 2
                    subset = subset.loc[subset[f"marker_{stain}"] == expression_value]

            label_ids_subtype.extend(list(subset["label_id"]))

        subtypes_str = "/".join(subtype)
        print(f"subtypes {subtypes_str} with {len(label_ids_subtype)} instances")

    subtype_mask = np.isin(segmentation, label_ids_subtype)
    segmentation[~subtype_mask] = 0
    segmentation[subtype_mask] = 1

    return segmentation.astype("float32")


def export_lower_resolution(
    cochlea: str,
    scale: List[int],
    output_folder: str,
    stains: Optional[List[str]] = None,
    force: bool = False,
    merge_masks: bool = False,
    crop_center: Optional[List[float]] = None,
    roi_halo: Optional[List[int]] = None,
    axis: Optional[int] = None,
    suffix: Optional[str] = None,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
):
    subtype_stains = stains
    force_overwrite = force
    crop = crop_center is not None
    # iterate through exporting lower resolutions
    for s in scale:
        out_folder = os.path.join(output_folder, cochlea, f"scale{s}")
        os.makedirs(out_folder, exist_ok=True)
        if cochlea in COCHLEAE.keys():
            if subtype_stains is None:
                subtype_stains = COCHLEAE[cochlea]["subtype_stains"]
            if "output_seg" in list(COCHLEAE[cochlea].keys()):
                seg_name = COCHLEAE[cochlea]["output_seg"]
            else:
                seg_name = COCHLEAE[cochlea]["seg_data"]
        else:
            raise ValueError(f"Cochlea {cochlea} is not in the dictionary. Check values.")

        print(f"Subtype stains: {subtype_stains}.")
        subtypes = types_for_stain(subtype_stains)
        subtypes.sort()

        if merge_masks:
            subtype_ids = subtype_ids_for_stains(subtype_stains)
            print(f"Merged subtype IDs: {subtype_ids}")
            if crop:
                out_path = os.path.join(
                    out_folder, f"{seg_name}_subtypes{crop_suffix(crop_center, axis, suffix)}.tif"
                )
            else:
                out_path = os.path.join(out_folder, f"{seg_name}_subtypes.tif")
            if os.path.exists(out_path) and not force_overwrite:
                print(f"Skipping {out_path}. File already exists.")
                continue

            input_key = f"s{s}"
            internal_path = os.path.join(cochlea, "images", "ome-zarr", f"{seg_name}.ome.zarr")
            s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
            f = zarr.open(s3_store, mode="r")
            if crop:
                start, stop = compute_crop_bb(
                    crop_center, roi_halo, voxel_size=voxel_size, scale=s, shape=f[input_key].shape, axis=axis,
                )
                data = f[input_key][start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            else:
                data = f[input_key][:]

            print("Data shape", data.shape)
            table_seg = read_subtype_table(cochlea, seg_name)
            data = merge_subtype_masks(data, table_seg, subtype_stains, subtype_ids)
            tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")
            continue

        if "Type Ib" in subtypes and "Type Ic" in subtypes:
            subtypes.append(["Type Ib", "Type Ic"])

        for subtype in subtypes:
            if isinstance(subtype, str):
                subtype_str = subtype.replace(" ", "")
            else:
                subtype_str = "".join([s.replace(" ", "") for s in subtype])

            if crop:
                out_path = os.path.join(
                    out_folder, f"{seg_name}_{subtype_str}{crop_suffix(crop_center, axis, suffix)}.tif"
                )
            else:
                out_path = os.path.join(out_folder, f"{seg_name}_{subtype_str}.tif")
            if os.path.exists(out_path) and not force_overwrite:
                continue

            input_key = f"s{s}"
            internal_path = os.path.join(cochlea, "images", "ome-zarr", f"{seg_name}.ome.zarr")
            s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
            f = zarr.open(s3_store, mode="r")
            if crop:
                start, stop = compute_crop_bb(
                    crop_center, roi_halo, voxel_size=voxel_size, scale=s, shape=f[input_key].shape, axis=axis,
                )
                data = f[input_key][start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
            else:
                data = f[input_key][:]

            print("Data shape", data.shape)

            print(f"Filtering subtype: {subtype}.")
            data = filter_subtypes(cochlea, data, seg_name=seg_name, subtype=subtype)
            tifffile.imwrite(out_path, data, bigtiff=True, compression="zlib")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cochlea", required=True)
    parser.add_argument("-s", "--scale", nargs="+", type=int, required=True)
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--stains", nargs="+", type=str, default=None)
    parser.add_argument("-f", "--force", action="store_true")
    parser.add_argument("--merge_masks", action="store_true",
                        help="Write one categorical subtype label volume instead of one binary mask per subtype.")
    parser.add_argument("--crop_center", nargs=3, type=float, default=None,
                        help="Crop center as x y z in µm. Requires --roi_halo.")
    parser.add_argument("--roi_halo", nargs=3, type=int, default=None,
                        help="Halo around the crop center as halo_x halo_y halo_z in pixels at the target scale. "
                        "Optional when --axis is given: the whole plane is cropped in that case.")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None,
                        help="Axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at the crop center. "
                        "Requires --crop_center.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Extra label appended to the output filename after the crop/axis suffix, "
                        "e.g. a position name such as 'apex'.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    args = parser.parse_args()
    if args.crop_center is not None:
        if args.roi_halo is None and args.axis is None:
            parser.error("--crop_center requires --roi_halo, unless --axis is also given "
                         "(the whole plane is cropped in that case).")
    elif args.axis is not None:
        parser.error("--axis requires --crop_center.")

    export_lower_resolution(**vars(args))


if __name__ == "__main__":
    main()
