"""Regenerate the Otof marker labels of the four M_AMD_OTOF27/28 cochleae.

The script is specific to this cohort. It holds the two sets of per-crop thresholds that were
established for it, and assigns the marker labels from either of them. The assignment follows the
annotation path: each crop center is mapped onto the "length fraction" of the cochlea, and the crop
governs the band up to the middle of the distance to its neighbour.
"""
import argparse
import copy
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flamingo_tools.analysis.seg_table_utils import filter_table
from flamingo_tools.intensity_annotation.eval_annotations import (
    apply_nearest_threshold, get_length_fraction_from_center, length_fraction_limits, percentage,
)
from flamingo_tools.json_util import export_dictionary_as_json
from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER

# Component labels copied from reproducibility/object_measures/MAMDOTOF*_IHC.json.
COCHLEAE = {
    "M_AMD_OTOF27_L": [1],
    "M_AMD_OTOF27_R": [2, 4, 10],
    "M_AMD_OTOF28_L": [5, 9, 1, 3, 4, 14, 8, 15],
    "M_AMD_OTOF28_R": [2, 1, 3, 4],
}

SEG_NAME = "IHC_v11"
MARKER_NAME = "Otof"
# Both threshold sets apply to the "mean" of the object measures. They differ in the table: the
# optimal set uses the plain measures, the annotator set the ones with a background mask.
COLUMN = "mean"

# Thresholds fitted by a leave-one-instance-out sweep on the plain "mean", with a reference-positive
# instance weighted three times. They reproduce the annotated calls better than any single threshold
# per cochlea, but they cannot be derived by hand from an annotation.
OPTIMAL_THRESHOLDS = {
    "M_AMD_OTOF27_L": {"0568-0112-0692": 187.1, "0653-1306-0537": 417.4, "0709-0690-1017": 162.3,
                       "0795-0604-0099": 195.6, "1085-1389-0594": 197.5, "1259-0662-0447": 268.9},
    "M_AMD_OTOF27_R": {"0625-0676-0122": float("inf"), "0659-1149-0749": float("inf"),
                       "0795-0468-1234": float("inf"), "1005-0278-0513": float("inf"),
                       "1195-1045-0730": float("inf"), "1204-0549-1311": float("inf")},
    "M_AMD_OTOF28_L": {"0181-1327-0785": 363.9, "0405-0603-0943": 215.2, "0472-1367-0217": 319.4,
                       "0662-1598-0752": 198.8, "0733-0749-0275": 342.0, "0733-0989-0838": 340.8},
    "M_AMD_OTOF28_R": {"0311-1418-0473": 570.2, "0397-0162-0613": float("inf"),
                       "0405-0891-0963": 185.1, "0527-0802-0061": 242.9,
                       "0725-1540-0481": 256.7, "0944-0849-0402": 228.3},
}

# Thresholds derived from the annotations of both annotators on the background-subtracted "mean",
# each one the value between the clearly positive and the clearly negative population of its crop.
# This is the rule of `get_crop_parameters`, so every value can be reproduced from the annotation.
# A crop in which the annotators found no positive instance carries 1.5 times the highest measured
# value of its cochlea, which is above every instance, so that its whole band stays negative.
ANNOTATOR_THRESHOLDS = {
    "M_AMD_OTOF27_L": {
        "0568-0112-0692": 53.26315841619322,
        "0653-1306-0537": 194.39733318768327,
        "0709-0690-1017": 58.89357464675676,
        "0795-0604-0099": 76.21486313501688,
        "1085-1389-0594": 89.54114028632449,
        "1259-0662-0447": 104.9538179255346,
    },
    "M_AMD_OTOF27_R": {
        "0625-0676-0122": 130.09116770522894,
        "0659-1149-0749": 130.09116770522894,
        "0795-0468-1234": 130.09116770522894,
        "1005-0278-0513": 130.09116770522894,
        "1195-1045-0730": 130.09116770522894,
        "1204-0549-1311": 130.09116770522894,
    },
    "M_AMD_OTOF28_L": {
        "0181-1327-0785": 212.34494611847026,
        "0405-0603-0943": 79.32732810483077,
        "0472-1367-0217": 185.29890858956566,
        "0662-1598-0752": 84.90613256046865,
        "0733-0749-0275": 218.91884909889228,
        "0733-0989-0838": 188.79711612481222,
    },
    "M_AMD_OTOF28_R": {
        "0311-1418-0473": 365.26131092992637,
        "0397-0162-0613": 1527.4094900976788,
        "0405-0891-0963": 69.1054971795348,
        "0527-0802-0061": 111.3035059775494,
        "0725-1540-0481": 123.45252114722476,
        "0944-0849-0402": 124.0269535018329,
    },
}

THRESHOLD_SETS = {
    "local_optimal": {
        "thresholds": OPTIMAL_THRESHOLDS,
        "use_bg_mask": False,
        "description": "fitted on the plain 'mean' by a leave-one-out sweep",
    },
    "local_annotator": {
        "thresholds": ANNOTATOR_THRESHOLDS,
        "use_bg_mask": True,
        "description": "between the annotated populations of the background-subtracted 'mean'",
    },
}


def _read_table(
    table_path: str,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> pd.DataFrame:
    """Read a TSV table from a local path or from the S3 bucket."""
    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            return pd.read_csv(f, sep="\t")
    return pd.read_csv(table_path, sep="\t")


def _table_paths(
    cochlea: str,
    use_bg_mask: bool,
    s3: bool = False,
    mobie_dir: str = MOBIE_FOLDER,
    meas_dir: Optional[str] = None,
) -> Tuple[str, str, bool]:
    """Build the paths of the segmentation table and of the object measures.

    Returns:
        The segmentation table path, the measurement table path, and whether the measurement table
        is on the S3 bucket.
    """
    seg_string = SEG_NAME.replace("_", "-")
    suffix = "-bg-mask" if use_bg_mask else ""
    meas_name = f"{MARKER_NAME}_{seg_string}_object-measures{suffix}.tsv"

    if s3:
        seg_path = f"{cochlea}/tables/{SEG_NAME}/default.tsv"
    else:
        seg_path = os.path.join(mobie_dir, cochlea, "tables", SEG_NAME, "default.tsv")

    if meas_dir is not None:
        cochlea_str = cochlea.replace("_", "-")
        meas_path = os.path.join(meas_dir, f"{cochlea_str}_{meas_name}")
        if not os.path.exists(meas_path):
            meas_path = os.path.join(meas_dir, meas_name)
        return seg_path, meas_path, False

    if s3:
        return seg_path, f"{cochlea}/tables/{SEG_NAME}/{meas_name}", True
    return seg_path, os.path.join(mobie_dir, cochlea, "tables", SEG_NAME, meas_name), False


def apply_local_thresholds(
    table_seg: pd.DataFrame,
    table_meas: pd.DataFrame,
    thresholds: Dict[str, float],
    component_list: List[int],
    column: str = COLUMN,
    halo_size: int = 20,
) -> Tuple[pd.DataFrame, dict]:
    """Assign marker labels from one threshold per annotation crop.

    The crop centers are mapped onto the "length fraction" of the cochlea, and each crop governs the
    band up to the middle of the distance to its neighbour, exactly as in the annotation path of
    `scripts/measurements/eval_marker_annotations.py`.

    The mapping runs on the instances of the connected components only. An instance outside them
    carries a placeholder length fraction of 0, which would pull the crop positions toward the start
    of the cochlea. Such an instance keeps the label 0.

    Args:
        table_seg: Segmentation table of the whole cochlea.
        table_meas: Table with the object measures.
        thresholds: Threshold per crop center string.
        component_list: List of connected components.
        column: Column of the measurement table that the thresholds apply to.
        halo_size: Halo in micrometer to find the instances around a crop center.

    Returns:
        The segmentation table with the "marker_labels" column, and the per-crop breakdown.
    """
    intensity_dic = {center: {"median_intensity": value} for center, value in thresholds.items()}
    table_component = filter_table(table_seg, component_list).copy()

    labeled = apply_nearest_threshold(
        copy.deepcopy(intensity_dic), table_component, table_meas, column=column, halo_size=halo_size,
    )
    for center in intensity_dic:
        intensity_dic[center]["length_fraction"] = get_length_fraction_from_center(
            table_component, center, halo_size=halo_size
        )

    assignment = dict(zip(labeled["label_id"], labeled["marker_labels"]))
    table_seg["marker_labels"] = table_seg["label_id"].map(assignment).fillna(0).astype(int)

    ordered = sorted(intensity_dic.items(), key=lambda item: item[1]["length_fraction"])
    limits = length_fraction_limits([entry["length_fraction"] for _, entry in ordered])
    crop_counts = {}
    for num, (center, entry) in enumerate(ordered):
        in_band = labeled[
            (labeled["length_fraction"] > limits[num]) & (labeled["length_fraction"] < limits[num + 1])
        ]
        n_positive = int((in_band["marker_labels"] == 1).sum())
        n_negative = int((in_band["marker_labels"] == 2).sum())
        threshold = entry["median_intensity"]
        crop_counts[center] = {
            "threshold": None if not np.isfinite(threshold) else float(threshold),
            "length_fraction": round(float(entry["length_fraction"]), 4),
            "band": [round(float(limits[num]), 4), round(float(limits[num + 1]), 4)],
            "n_positive": n_positive,
            "n_negative": n_negative,
            "percent_positive": percentage(n_positive, n_positive + n_negative),
        }
    return table_seg, crop_counts


def _plot_local_thresholds(
    table_seg: pd.DataFrame,
    table_meas: pd.DataFrame,
    crop_counts: dict,
    column: str,
    name: str,
    save_path: str,
) -> None:
    """Plot the marker intensity over the cochlea, with the threshold of every crop over its band."""
    merged = table_seg[["label_id", "length_fraction", "marker_labels"]].merge(
        table_meas[["label_id", column]], on="label_id", how="inner"
    )
    merged = merged[merged["marker_labels"] > 0]

    fig, ax = plt.subplots(1, figsize=(7, 4))
    for label, color, text in ((2, "tab:orange", "negative"), (1, "tab:blue", "positive")):
        subset = merged[merged["marker_labels"] == label]
        ax.scatter(subset["length_fraction"], subset[column], s=6, c=color, label=text)
    for entry in crop_counts.values():
        if entry["threshold"] is not None:
            ax.plot(entry["band"], [entry["threshold"]] * 2, color="red", linestyle="--")

    n_positive = int((merged["marker_labels"] == 1).sum())
    ax.set_xlabel("length_fraction")
    ax.set_ylabel(column)
    ax.legend()
    ax.set_title(f"{name}\npositive: {percentage(n_positive, len(merged))} %", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def regenerate_thresholds(
    threshold_set: str,
    output_dir: str,
    cochleae: Optional[List[str]] = None,
    mobie_dir: str = MOBIE_FOLDER,
    meas_dir: Optional[str] = None,
    force_overwrite: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> None:
    """Assign the Otof marker labels of the OTOF cochleae from one of the stored threshold sets.

    Args:
        threshold_set: Name of the threshold set, a key of THRESHOLD_SETS.
        output_dir: Output directory for the segmentation table, the thresholds and the plot.
        cochleae: Cochleae to process. Defaults to all cochleae of COCHLEAE.
        mobie_dir: Local MoBIE directory used for creating data paths.
        meas_dir: Directory with the object-measures tables, instead of the MoBIE or S3 location.
        force_overwrite: Whether to overwrite already existing results.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    if threshold_set not in THRESHOLD_SETS:
        raise ValueError(f"Choose one of the threshold sets {list(THRESHOLD_SETS.keys())}, not '{threshold_set}'.")

    info = THRESHOLD_SETS[threshold_set]
    cochleae = list(COCHLEAE.keys()) if cochleae is None else cochleae
    seg_string = SEG_NAME.replace("_", "-")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Threshold set '{threshold_set}': {info['description']}.")

    for cochlea in cochleae:
        if cochlea not in COCHLEAE:
            raise ValueError(f"Cochlea {cochlea} is not one of {list(COCHLEAE.keys())}.")
        if cochlea not in info["thresholds"]:
            raise ValueError(f"The threshold set '{threshold_set}' has no entry for {cochlea}.")

        name = f"{cochlea.replace('_', '-')}_{MARKER_NAME}_{seg_string}"
        out_path = os.path.join(output_dir, f"{name}.tsv")
        if os.path.exists(out_path) and not force_overwrite:
            print(f"Skipping {out_path}. Table already exists.")
            continue

        seg_path, meas_path, s3_meas = _table_paths(
            cochlea, info["use_bg_mask"], s3=s3, mobie_dir=mobie_dir, meas_dir=meas_dir,
        )
        table_seg = _read_table(seg_path, s3, s3_credentials, s3_bucket_name, s3_service_endpoint)
        table_meas = _read_table(meas_path, s3_meas, s3_credentials, s3_bucket_name, s3_service_endpoint)
        table_meas["label_id"] = table_meas["label_id"].astype("int64")

        for required in ("component_labels", "length_fraction"):
            if required not in table_seg.columns:
                raise ValueError(f"The column '{required}' is not in {seg_path}. "
                                 "Run the component labeling and the tonotopic mapping first.")
        if COLUMN not in table_meas.columns:
            raise ValueError(f"The column '{COLUMN}' is not in {meas_path}.")

        thresholds = dict(info["thresholds"][cochlea])
        table_seg, crop_counts = apply_local_thresholds(
            table_seg, table_meas, thresholds, COCHLEAE[cochlea],
        )

        n_positive = int((table_seg["marker_labels"] == 1).sum())
        n_negative = int((table_seg["marker_labels"] == 2).sum())
        print(f"{cochlea}: {len(thresholds)} crop thresholds on '{COLUMN}', "
              f"{percentage(n_positive, n_positive + n_negative)} % positive "
              f"of {n_positive + n_negative} instances.")
        table_seg.to_csv(out_path, sep="\t", index=False)

        param_dict = {
            "cochlea": cochlea,
            "marker": MARKER_NAME,
            "segmentation": SEG_NAME,
            "threshold_set": threshold_set,
            "description": info["description"],
            "column": COLUMN,
            "use_bg_mask": info["use_bg_mask"],
            "measurement_table": meas_path,
            "component_list": COCHLEAE[cochlea],
            "n_total": len(table_seg),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_unassigned": int(len(table_seg) - n_positive - n_negative),
            "percent_positive": percentage(n_positive, n_positive + n_negative),
            "percent_negative": percentage(n_negative, n_positive + n_negative),
            "crops": crop_counts,
        }
        export_dictionary_as_json(param_dict, os.path.join(output_dir, f"{name}.json"), force_overwrite=True)
        _plot_local_thresholds(table_seg, table_meas, crop_counts, COLUMN, name,
                               os.path.join(output_dir, f"{name}.png"))


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate the Otof marker labels of the M_AMD_OTOF27/28 cochleae "
                    "from one of the stored per-crop threshold sets."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--local_optimal", action="store_true",
                       help="Thresholds fitted on the plain 'mean' by a leave-one-out sweep.")
    group.add_argument("--local_annotator", action="store_true",
                       help="Thresholds between the annotated populations of the "
                            "background-subtracted 'mean'.")

    parser.add_argument("-o", "--output_dir", type=str, required=True,
                        help="Output directory for the segmentation table, the thresholds and the plot.")
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=None,
                        help="Cochlea(e) to process. Default: all four.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--meas_dir", type=str, default=None,
                        help="Directory with the object-measures tables.")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project.")

    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()
    threshold_set = "local_optimal" if args.local_optimal else "local_annotator"

    regenerate_thresholds(
        threshold_set=threshold_set,
        output_dir=args.output_dir,
        cochleae=args.cochlea,
        mobie_dir=args.mobie_dir,
        meas_dir=args.meas_dir,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
