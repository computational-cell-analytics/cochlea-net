"""Create IHC training crops for the MWfLS cochleae, at the IHCs that IHC_v11 misses.

The reference is the curated IHC row of IHC_v9 (component 1). The pipeline settings of both models are
equal, so the misses come from the model. Each crop holds PV, Vglut3, IHC_v9 and IHC_v11 for the correction in napari.
The table crop_context.tsv in the output folder gives the position, tonotopy and object counts of each crop.
"""
import argparse
import json
import os
import sys

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from flamingo_tools.analysis.training_data_utils import correct_ihc_row, find_crop_centers_missed, ihc_row_reference
from flamingo_tools.extract_block_util import extract_block_json_wrapper
from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT
from flamingo_tools.s3_utils import create_s3_target, default_table_path, read_table

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "figures"))
from data import cochlea_components, get_mapped_table, read_source_table  # noqa: E402
from util import COCHLEA_DICT  # noqa: E402

COCHLEAE = ["M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R"]
REFERENCE = "IHC_v9"
SEGMENTATION = "IHC_v11"
HALO = [128, 256, 256]  # (x, y, z) in pixels, the default of export_crop_centers
VOXEL_SIZE = 0.38
# A row IHC with a smaller fraction of its voxels in IHC_v11 objects counts as missed.
MIN_COVERAGE = 0.2
# The cases of ~/Documents/MWfLS/ihc_crops/comments.md, with the ids of the input segmentation:
# - tips: a tip within 3 µm of each napari coordinate (z, y, x);
# - cells: the instances hold this number of cells, and 80% of their voxels stay;
# - kept: each instance keeps 95% of its voxels;
# - removed: the instances are removed.
COMMENT_CASES = [
    ("M-AMD-000126-R_crop_0448-1443-0611", "IHC_v11", "tips",
     [[220, 313, 120], [238, 321, 115], [258, 312, 121], [280, 306, 123], [306, 298, 130]], None),
    ("M-AMD-000126-R_crop_1086-0853-0293", "IHC_v11", "cells", [1132, 1134, 1135, 1136, 1137, 1138], 3),
    ("M-AMD-000126-R_crop_1086-0853-0293", "IHC_v11", "removed", [516, 520, 752, 944, 946, 1275, 1279], None),
    ("M-AMD-000127-L_crop_0826-1150-1139", "IHC_v11", "cells", [1385, 1541], 1),
    ("M-AMD-000127-L_crop_1136-1138-1082", "IHC_v11", "kept", [1361, 1362, 1371, 1374, 1375], None),
    # Good cells with a protrusion: the trim may remove the protrusion, but the cell stays one cell.
    ("M-AMD-000127-L_crop_1136-1138-1082", "IHC_v11", "cells", [1369], 1),
    ("M-AMD-000127-L_crop_1136-1138-1082", "IHC_v11", "cells", [1370], 1),
    ("M-AMD-000127-L_crop_1136-1138-1082", "IHC_v11", "cells", [1376], 1),
    ("M-AMD-000127-L_crop_1136-1138-1082", "IHC_v11", "removed", [1136, 1357, 1394], None),
    ("M-AMD-000126-L_crop_1216-0631-0806", "IHC_v11", "removed", [1276, 1280], None),
    ("M-AMD-000126-L_crop_1216-0631-0806", "IHC_v9", "removed", [3456, 3457, 3464, 3474, 3556, 3560], None),
    ("M-AMD-000126-L_crop_1084-0437-0642", "IHC_v9", "removed", [2999, 3075, 3076, 3138, 3145, 3147, 3148], None),
    ("M-AMD-000126-L_crop_0541-0868-0259", "IHC_v11", "removed", [377], None),
]


def export_missed_crops(cochlea: str, out_dir: str, n_crops: int, force_overwrite: bool = False) -> str:
    output_path = os.path.join(out_dir, f"{cochlea}_missed_{SEGMENTATION.lower()}.json")
    if os.path.isfile(output_path) and not force_overwrite:
        print(f"JSON dictionary {output_path} already exists. Skipping creation.")
        return output_path

    reference = read_table(default_table_path(cochlea, REFERENCE, s3=True), s3=True)
    reference = reference[reference["component_labels"] == 1]
    table = read_table(default_table_path(cochlea, SEGMENTATION, s3=True), s3=True)
    crop_size = [2 * h * VOXEL_SIZE for h in HALO]
    centers, n_missed = find_crop_centers_missed(reference, table, n_crops, crop_size)
    print(f"{cochlea}: crops at {centers} with {n_missed} missed IHCs.")

    crop_dict = {
        "dataset_name": cochlea,
        "image_channel": ["PV", "Vglut3", REFERENCE, SEGMENTATION],
        "segmentation_channel": SEGMENTATION,
        "cell_type": "ihc",
        "n_blocks": len(centers),
        "roi_halo": HALO,
        "component_list": [1],
        "crop_centers": centers,
        "missed_ihcs": n_missed,
    }
    with open(output_path, "w") as f:
        json.dump([crop_dict], f, indent='\t', separators=(',', ': '))
    return output_path


def _crop_name(cochlea: str, center: list) -> str:
    return f"{cochlea.replace('_', '-')}_crop_{'-'.join(str(c).zfill(4) for c in center)}"


def _n_objects(segmentation: np.ndarray) -> int:
    return int(np.count_nonzero(np.unique(segmentation)))


def compare_crop(reference: np.ndarray, segmentation: np.ndarray, row_ids: np.ndarray) -> dict:
    """Count how the segmentation fails on the reference row IHCs of one crop."""
    in_row = np.isin(reference, row_ids)
    pairs = pd.DataFrame({"row": reference[in_row], "seg": segmentation[in_row]})
    coverage = (pairs["seg"] > 0).groupby(pairs["row"]).mean()
    missed = coverage < MIN_COVERAGE
    covered = pairs[pairs["row"].isin(coverage.index[~missed]) & (pairs["seg"] > 0)]
    # The main segmentation object of a row IHC is the one with the largest overlap.
    main_object = covered.groupby(["row", "seg"]).size().groupby(level="row").idxmax().map(lambda key: key[1])
    seg_ids = np.unique(segmentation[segmentation > 0])
    return {
        "IHC_v9_row": len(coverage),
        "Row_missed": int(missed.sum()),
        "Row_merged": int(main_object.duplicated(keep=False).sum()),
        "IHC_v11_extra": len(np.setdiff1d(seg_ids, pairs["seg"].unique())),
    }


def write_crop_context(json_files: list, crop_root: str, out_path: str) -> pd.DataFrame:
    """Write one table row per extracted crop with its position, tonotopy, object counts and Vglut3 levels."""
    s3 = create_s3_target()
    rows = []
    for json_file in json_files:
        with open(json_file) as f:
            params = json.load(f)[0]
        cochlea = params["dataset_name"]
        mapped = get_mapped_table(s3, cochlea, "IHC", REFERENCE)
        tree = cKDTree(mapped[["anchor_x", "anchor_y", "anchor_z"]].values)
        full_shape = tuple(2 * h for h in reversed(params["roi_halo"]))
        for center, run_missed in zip(params["crop_centers"], params["missed_ihcs"]):
            name = _crop_name(cochlea, center)
            prefix = os.path.join(crop_root, cochlea, name)
            reference = imageio.imread(f"{prefix}_{REFERENCE.replace('_', '-')}.tif")
            segmentation = imageio.imread(f"{prefix}_{SEGMENTATION.replace('_', '-')}.tif")
            vglut3 = imageio.imread(f"{prefix}_Vglut3.tif")
            center_ihc = mapped.iloc[tree.query(center)[1]]
            row_mask = np.isin(reference, mapped["label_id"].values)
            rows.append({
                "Name": name, "Cochlea": cochlea, "Alias": COCHLEA_DICT[cochlea]["alias"],
                "Protocol": SYNAPSE_DICT[cochlea]["protocol"],
                "Center_x": center[0], "Center_y": center[1], "Center_z": center[2],
                "Shape_zyx": "x".join(str(s) for s in reference.shape), "Clipped": reference.shape != full_shape,
                "Length_fraction": round(center_ihc["length_fraction"], 3),
                "Frequency_kHz": round(center_ihc["frequency[kHz]"], 1),
                "Run_missed": run_missed,
                "IHC_v9_objects": _n_objects(reference),
                "IHC_v11_objects": _n_objects(segmentation),
                **compare_crop(reference, segmentation, mapped["label_id"].values),
                "Vglut3_row_median": float(np.median(vglut3[row_mask])),
                "Vglut3_background_median": float(np.median(vglut3[(reference == 0) & (segmentation == 0)])),
            })
    table = pd.DataFrame(rows)
    table.to_csv(out_path, sep="\t", index=False)
    return table


def write_proposals(json_files: list, crop_root: str, tip_method: str = "path") -> pd.DataFrame:
    """Write a correction proposal of IHC_v9 and IHC_v11 for each crop, its IHC tips, and the table ihc_correction.tsv.

    The reference IHC of a crop is pooled from both segmentations. The central path is the one of the
    in-memory tonotopic mapping of the IHC_v9 row. The tips are napari points in <name>_<version>-tips.csv.
    """
    s3 = create_s3_target()
    reports, checks = [], []
    for json_file in json_files:
        with open(json_file) as f:
            params = json.load(f)[0]
        cochlea = params["dataset_name"]
        table = read_source_table(s3, cochlea, REFERENCE)
        components = cochlea_components(cochlea, "IHC", REFERENCE)
        _, path = tonotopic_mapping(table, component_label=components, cell_type="ihc")
        for center in params["crop_centers"]:
            name = _crop_name(cochlea, center)
            prefix = os.path.join(crop_root, cochlea, name)
            # The same crop start as extract_block_single, in (z, y, x) order.
            start = np.round(np.array(center[::-1]) / VOXEL_SIZE).astype(int) - np.array(params["roi_halo"][::-1])
            crop_path = path[["z", "y", "x"]].values / VOXEL_SIZE - np.maximum(start, 0)
            segmentations = {
                version: imageio.imread(f"{prefix}_{version.replace('_', '-')}.tif")
                for version in (REFERENCE, SEGMENTATION)
            }
            reference = ihc_row_reference(list(segmentations.values()), crop_path, VOXEL_SIZE)
            for version, segmentation in segmentations.items():
                labels, report, tips = correct_ihc_row(
                    segmentation, crop_path, reference, VOXEL_SIZE, tip_method=tip_method
                )
                imageio.imwrite(f"{prefix}_{version.replace('_', '-')}-proposal.tif", labels, compression="zlib")
                tips.rename(columns={"z": "axis-0", "y": "axis-1", "x": "axis-2"}).to_csv(
                    f"{prefix}_{version.replace('_', '-')}-tips.csv", index=False
                )
                checks.extend(check_comments(name, version, segmentation, labels, tips))
                actions = report["action"].value_counts().to_dict()
                print(f"{name} {version}: reference {reference['n_pixels']:.0f} px, actions {actions}, "
                      f"review {int(report['review'].sum())}")
                report.insert(0, "Version", version)
                report.insert(0, "Name", name)
                reports.append(report)
    report = pd.concat(reports, ignore_index=True)
    report.to_csv(os.path.join(crop_root, "ihc_correction.tsv"), sep="\t", index=False)
    if checks:
        checks = pd.DataFrame(checks)
        print(f"Cases of comments.md ({tip_method} tips): {int(checks['pass'].sum())} of {len(checks)} pass")
        print(checks.to_string(index=False))
    return report


def check_comments(name: str, version: str, segmentation: np.ndarray, labels: np.ndarray, tips: pd.DataFrame) -> list:
    """Check the proposal of one crop against the cases of comments.md."""
    rows = []
    for crop, case_version, kind, items, expected in COMMENT_CASES:
        if crop != name or case_version != version:
            continue
        if kind == "tips":
            distances = cKDTree(tips[["z", "y", "x"]].values).query(items)[0] * VOXEL_SIZE
            result, passed = np.round(distances, 1).tolist(), bool((distances <= 3.0).all())
        elif kind == "cells":
            outputs = labels[np.isin(segmentation, items)]
            cells, counts = np.unique(outputs[outputs > 0], return_counts=True)
            result = int((counts >= 0.1 * len(outputs)).sum())
            passed = result == expected and np.mean(outputs > 0) >= 0.8
        else:
            fractions = [np.mean(labels[segmentation == i] == (i if kind == "kept" else 0)) for i in items]
            result, passed = np.round(fractions, 2).tolist(), all(f >= 0.95 for f in fractions)
        rows.append({"Name": name, "Version": version, "case": kind, "ids": items if kind != "tips" else "",
                     "expected": expected, "result": result, "pass": passed})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--cochlea", nargs="+", type=str, default=COCHLEAE, help="Cochlea(e) to analyze.")
    parser.add_argument("-j", "--json_dir", type=str, required=True,
                        help="Output directory for JSON dictionaries which feature parameters for crop extraction.")
    parser.add_argument("-n", "--n_crops", type=int, default=3, help="Maximal number of crops per cochlea.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite JSON dictionaries.")
    parser.add_argument("-o", "--output_folder", type=str, default=None,
                        help="Output directory for extracted crops in TIF format.")
    parser.add_argument("-p", "--proposals", action="store_true",
                        help="Write the correction proposals of the crops and ihc_correction.tsv to the output folder.")
    parser.add_argument("-t", "--tip_method", choices=["path", "height"], default="path",
                        help="The IHC tip detection of the correction proposals.")
    args = parser.parse_args()
    if args.proposals and args.output_folder is None:
        parser.error("--proposals needs --output_folder.")

    os.makedirs(args.json_dir, exist_ok=True)
    json_files = []
    for cochlea in args.cochlea:
        json_file = export_missed_crops(cochlea, args.json_dir, args.n_crops, force_overwrite=args.force)
        json_files.append(json_file)
        if args.output_folder is not None:
            output_dir = os.path.join(args.output_folder, cochlea)
            os.makedirs(output_dir, exist_ok=True)
            extract_block_json_wrapper(output_path=output_dir, json_file=json_file, s3=True, input_key="s0")

    if args.output_folder is not None:
        table = write_crop_context(json_files, args.output_folder, os.path.join(args.output_folder, "crop_context.tsv"))
        print(table.to_string(index=False))
        if args.proposals:
            write_proposals(json_files, args.output_folder, args.tip_method)


if __name__ == "__main__":
    main()
