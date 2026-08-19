import argparse
import copy
import os
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu

from flamingo_tools.analysis.seg_table_utils import filter_table
from flamingo_tools.intensity_annotation.eval_annotations import (
    apply_nearest_threshold, get_length_fraction_from_center, length_fraction_limits, percentage,
)
from flamingo_tools.json_util import export_dictionary_as_json
from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER

# Cochleae with an Otof marker on IHC segmentation.
# The component labels are copied from reproducibility/object_measures/MAMDOTOF*_IHC.json.
COCHLEAE_OTOF = {
    "M_AMD_OTOF27_L": {"channels": ["Vglut3", "Alphatag", "Otof"], "component_list": [1]},
    "M_AMD_OTOF27_R": {"channels": ["Vglut3", "Alphatag", "Otof"], "component_list": [2, 4, 10]},
    "M_AMD_OTOF28_L": {"channels": ["Vglut3", "Alphatag", "Otof"],
                       "component_list": [5, 9, 1, 3, 4, 14, 8, 15]},
    "M_AMD_OTOF28_R": {"channels": ["Vglut3", "Alphatag", "Otof"], "component_list": [2, 1, 3, 4]},
}

# Cochleae with a ChReef marker on SGN segmentation.
# The component labels are copied from reproducibility/object_measures/ChReef_*.json.
COCHLEAE_CHREEF = {
    "M_LR_000143_L": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000143_R": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000144_L": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000144_R": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000145_L": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000145_R": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000153_L": {"channels": ["GFP"], "component_list": [1, 2, 3]},
    "M_LR_000153_R": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000155_L": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000155_R": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000189_L": {"channels": ["GFP"], "component_list": [1]},
    "M_LR_000189_R": {"channels": ["GFP"], "component_list": [1]},
    "G_EK_000049_L": {"channels": ["GFP"], "component_list": [1, 3, 4, 5]},
    "G_EK_000049_R": {"channels": ["GFP"], "component_list": [1, 2]},
}

# The f-Chrimson marker uses the GFP stain on SGN segmentation, like ChReef.
# Add the cochleae and their component labels once the data is available.
COCHLEAE_CHRIMSON = {}

# Marker family to cochleae, segmentation name, marker stain and the default intensity column
# used when the cochlea has no entry in THRESHOLD_DICT.
MARKER_GROUPS = {
    "otof": {"cochleae": COCHLEAE_OTOF, "seg_name": "IHC_v11", "marker_name": "Otof",
             "intensity_column": "median"},
    "chreef": {"cochleae": COCHLEAE_CHREEF, "seg_name": "SGN_v2", "marker_name": "GFP",
               "intensity_column": "median"},
    "chrimson": {"cochleae": COCHLEAE_CHRIMSON, "seg_name": "SGN_v2", "marker_name": "GFP",
                 "intensity_column": "median"},
}

# Features derived from the object measures, in addition to the plain table columns.
# "median_bg" is the median of the background-subtracted table, the other features are
# differences of percentiles of the plain table, which cancel the local background offset.
DERIVED_FEATURES = {
    "p95_sub_p5": ("percentile-95", "percentile-5"),
    "p90_sub_p10": ("percentile-90", "percentile-10"),
    "p90_sub_median": ("percentile-90", "median"),
    "iqr": ("percentile-75", "percentile-25"),
}

BG_FEATURE = "median_bg"

# Fixed marker thresholds per cochlea. An instance is positive when it reaches every threshold
# of its cochlea, so several columns can be combined.
#
# For the OTOF cochleae the rule is a level test on the background-subtracted median plus a
# shared contrast gate on "p95_sub_p5". The contrast gate is what separates the populations:
# the level alone calls IHCs positive that have a raised background but no bright substructure.
# The values come from the sweep in scripts/measurements/eval_marker_thresholds.py against the
# annotated crops: 18 of 499 annotated instances misclassified, and 20.2 under a repeated
# 5-fold cross validation, against 18 and 25.0 for a single threshold on "percentile-90".
# M_AMD_OTOF27_R is an all-negative control. Its highest "p95_sub_p5" is 116, so the gate alone
# keeps it at zero positives and its level threshold does not influence the result.
THRESHOLD_DICT = {
    "M_AMD_OTOF27_L": {"median_bg": 23, "p95_sub_p5": 240},
    "M_AMD_OTOF27_R": {"median_bg": 34, "p95_sub_p5": 240},
    "M_AMD_OTOF28_L": {"median_bg": 128, "p95_sub_p5": 240},
    "M_AMD_OTOF28_R": {"median_bg": 51, "p95_sub_p5": 240},
}

# Feature that carries the local thresholds. It comes from the plain object-measures table, so the
# local rule does not depend on the table computed with a background mask.
LOCAL_THRESHOLD_FEATURE = "mean"

# Threshold per cochlea and crop center, from the sweep in scripts/measurements/eval_marker_thresholds.py.
# The crops sit at six positions along the cochlea, so these thresholds follow the local imaging
# conditions, which one threshold per cochlea cannot. A crop that the annotators marked entirely
# negative has an infinite threshold, so that its whole length-fraction band stays negative. This
# replaces the earlier convention of 1.5 times the highest median, which depended on a maximum
# measured elsewhere in the cochlea.
# Scored leave-one-instance-out with positives weighted 3 times, the local thresholds miss 10 of the
# 89 annotated positives, against 15 for a single threshold on the same feature.
LOCAL_THRESHOLD_DICT = {
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

PLOT_OUT = "./marker_threshold_plots"


def marker_threshold(intensities: np.ndarray) -> float:
    """Split marker intensities into a negative and a positive population.

    The threshold is the Otsu threshold of the intensity distribution.
    Instances at or above the threshold are positive.

    Args:
        intensities: Marker intensity per segmentation instance.

    Returns:
        The threshold between the negative and the positive population.
    """
    values = np.asarray(intensities, dtype="float64")
    values = values[np.isfinite(values)]
    if len(np.unique(values)) < 2:
        raise ValueError("The intensities do not contain two distinct values, so no threshold can be computed.")
    return float(threshold_otsu(values))


def build_features(table_plain: Optional[pd.DataFrame], table_bg: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Combine the plain and the background-subtracted object measures into one feature table.

    The plain table contributes its own columns and the derived features of DERIVED_FEATURES.
    The background-subtracted table contributes its median as "median_bg".

    Args:
        table_plain: Object measures without a background mask.
        table_bg: Object measures with a background mask.

    Returns:
        Feature table with a "label_id" column.
    """
    if table_plain is None and table_bg is None:
        raise ValueError("Provide at least one table of object measures.")

    if table_plain is None:
        features = table_bg[["label_id"]].copy()
    else:
        features = table_plain.copy()
        for name, (left, right) in DERIVED_FEATURES.items():
            if left in features.columns and right in features.columns:
                features[name] = features[left] - features[right]

    if table_bg is not None:
        bg = table_bg[["label_id", "median"]].rename(columns={"median": BG_FEATURE})
        features = features.merge(bg, on="label_id", how="inner" if table_plain is not None else "left")

    features["label_id"] = features["label_id"].astype("int64")
    return features


def apply_thresholds(features: pd.DataFrame, thresholds: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply one threshold per feature and combine them with a logical and.

    Args:
        features: Feature table with a "label_id" column.
        thresholds: Threshold per feature name.

    Returns:
        The label ids that carry a value for every feature, and their positive mask.
    """
    missing = [name for name in thresholds if name not in features.columns]
    if missing:
        raise KeyError(f"The features {missing} are not in the measurement tables. "
                       f"Available: {sorted(features.columns)}")

    values = features[list(thresholds)].to_numpy(dtype="float64")
    is_finite = np.isfinite(values).all(axis=1)
    label_ids = features["label_id"].to_numpy()[is_finite]
    is_positive = np.ones(int(is_finite.sum()), dtype=bool)
    for num, name in enumerate(thresholds):
        is_positive &= values[is_finite, num] >= thresholds[name]
    return label_ids, is_positive


def apply_local_thresholds(
    table_seg: pd.DataFrame,
    features: pd.DataFrame,
    thresholds: Dict[str, float],
    component_list: List[int],
    feature: str = LOCAL_THRESHOLD_FEATURE,
    halo_size: int = 20,
) -> Tuple[pd.DataFrame, dict]:
    """Assign marker labels from one threshold per annotation crop.

    The crop centers are mapped onto the "length fraction" of the cochlea, and each crop governs
    the band up to the middle of the distance to its neighbour, exactly as in the annotation based
    path of `scripts/measurements/eval_marker_annotations.py`.

    The mapping runs on the instances of the connected components only. Instances outside them
    carry a placeholder length fraction of 0, which would pull the crop positions toward the start
    of the cochlea. They keep the label 0 in the result.

    Args:
        table_seg: Segmentation table of the whole cochlea.
        features: Feature table with a "label_id" column.
        thresholds: Threshold per crop center string. An infinite threshold marks a crop in which
            the annotators found no positive instance.
        component_list: List of connected components.
        feature: Feature that the thresholds apply to.
        halo_size: Halo in micrometer to find the instances around a crop center.

    Returns:
        The segmentation table with the "marker_labels" column, and the per-crop breakdown.
    """
    intensity_dic = {center: {"median_intensity": value} for center, value in thresholds.items()}
    table_component = filter_table(table_seg, component_list).copy()

    labeled = apply_nearest_threshold(
        copy.deepcopy(intensity_dic), table_component, features, column=feature, halo_size=halo_size,
    )
    # apply_nearest_threshold adds the length fraction to the dictionary that it receives.
    mapped = {center: entry for center, entry in intensity_dic.items()}
    for center in mapped:
        mapped[center]["length_fraction"] = get_length_fraction_from_center(
            table_component, center, halo_size=halo_size
        )

    table_seg["marker_labels"] = 0
    assignment = dict(zip(labeled["label_id"], labeled["marker_labels"]))
    table_seg["marker_labels"] = table_seg["label_id"].map(assignment).fillna(0).astype(int)

    ordered = sorted(mapped.items(), key=lambda item: item[1]["length_fraction"])
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
    features: pd.DataFrame,
    crop_counts: dict,
    feature: str,
    name: str,
    save_path: str,
) -> None:
    """Plot the feature over the cochlea, with the threshold of every crop over its own band."""
    merged = table_seg[["label_id", "length_fraction", "marker_labels"]].merge(
        features[["label_id", feature]], on="label_id", how="inner"
    )
    merged = merged[merged["marker_labels"] > 0]

    fig, ax = plt.subplots(1, figsize=(7, 4))
    for label, color, text in ((2, "tab:orange", "negative"), (1, "tab:blue", "positive")):
        subset = merged[merged["marker_labels"] == label]
        ax.scatter(subset["length_fraction"], subset[feature], s=6, c=color, label=text)
    for entry in crop_counts.values():
        if entry["threshold"] is None:
            continue
        ax.plot(entry["band"], [entry["threshold"]] * 2, color="red", linestyle="--")
    n_positive = int((merged["marker_labels"] == 1).sum())
    ax.set_xlabel("length_fraction")
    ax.set_ylabel(feature)
    ax.legend()
    ax.set_title(f"{name}\nlocal thresholds on {feature}\n"
                 f"positive: {percentage(n_positive, len(merged))} %", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


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


def _find_meas_tables(
    cochlea: str,
    seg_name: str,
    marker_name: str,
    meas_dir: Optional[str] = None,
    mobie_dir: str = MOBIE_FOLDER,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> Dict[str, str]:
    """Find the plain and the background-subtracted object-measures table of a marker stain.

    A table of object measures is named "<marker>_<seg>_object-measures[-bg-mask].tsv" in the MoBIE
    project and on the S3 bucket, and "<cochlea>_<marker>_<seg>_object-measures[-bg-mask].tsv" in a
    directory passed as meas_dir. The "object-measures" part is required, so that tables of marker
    labels, e.g. "GFP_SGN-v2.tsv", are not mistaken for a table of object measures.

    Returns:
        Dictionary with the keys "plain" and "bg" for the tables that exist.
    """
    seg_string = seg_name.replace("_", "-")
    cochlea_str = cochlea.replace("_", "-")

    if meas_dir is not None:
        table_dir = meas_dir
        file_names = [entry.name for entry in os.scandir(table_dir)]
        prefixes = (f"{cochlea_str}_{marker_name}_", f"{marker_name}_")
    elif s3:
        table_dir = f"{cochlea}/tables/{seg_name}"
        dir_store, fs = get_s3_path(table_dir, bucket_name=s3_bucket_name,
                                    service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        file_names = [os.path.basename(path) for path in fs.ls(dir_store.path, detail=False)]
        prefixes = (f"{marker_name}_",)
    else:
        table_dir = os.path.join(mobie_dir, cochlea, "tables", seg_name)
        file_names = [entry.name for entry in os.scandir(table_dir)]
        prefixes = (f"{marker_name}_",)

    matches = sorted(
        name for name in file_names
        if name.startswith(prefixes) and name.endswith(".tsv") and
        "object-measures" in name and seg_string in name
    )
    tables = {}
    for name in matches:
        key = "bg" if "bg-mask" in name else "plain"
        tables.setdefault(key, os.path.join(table_dir, name))
    if not tables:
        raise FileNotFoundError(f"No object-measures table for the channel '{marker_name}' in {table_dir}.")
    return tables


def _resolve_thresholds(
    cochlea: str,
    features: pd.DataFrame,
    intensity_column: str,
    threshold: Optional[float] = None,
) -> Tuple[Dict[str, float], str]:
    """Choose the thresholds of a cochlea and report how they were obtained.

    The precedence is an explicit threshold, then the fixed thresholds of THRESHOLD_DICT,
    then an Otsu threshold on the intensity column.

    Returns:
        Threshold per feature name, and the method, one of "given", "fixed" or "otsu".
    """
    if threshold is not None:
        return {intensity_column: float(threshold)}, "given"
    if cochlea in THRESHOLD_DICT:
        return {name: float(value) for name, value in THRESHOLD_DICT[cochlea].items()}, "fixed"

    warnings.warn(f"Cochlea {cochlea} has no entry in THRESHOLD_DICT. Using an Otsu threshold on "
                  f"'{intensity_column}', which is not validated against annotations.")
    if intensity_column not in features.columns:
        raise KeyError(f"The column '{intensity_column}' is not in the measurement tables.")
    values = features[intensity_column].to_numpy(dtype="float64")
    return {intensity_column: marker_threshold(values)}, "otsu"


def _plot_thresholds(
    features: pd.DataFrame,
    thresholds: Dict[str, float],
    is_positive: np.ndarray,
    name: str,
    method: str,
    save_path: str,
) -> None:
    """Plot the marker intensity distribution with the thresholds.

    A single feature gives a histogram with the threshold as a vertical line. Two or more
    features give a scatter plot of the first two features with both thresholds.
    """
    pos_percent = percentage(int(is_positive.sum()), len(is_positive))
    title = f"{name}\n{method}: " + ", ".join(f"{k} >= {round(v, 4)}" for k, v in thresholds.items())
    title += f"\npositive: {pos_percent} %"
    names = list(thresholds)

    fig, ax = plt.subplots(1)
    if len(names) == 1:
        values = features[names[0]].to_numpy(dtype="float64")
        ax.hist(values[np.isfinite(values)], bins=48)
        ax.axvline(x=thresholds[names[0]], color="red", linestyle="--")
        ax.set_xlabel(names[0])
        ax.set_ylabel("Count")
    else:
        x = features[names[0]].to_numpy(dtype="float64")
        y = features[names[1]].to_numpy(dtype="float64")
        keep = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[keep][~is_positive], y[keep][~is_positive], s=6, c="tab:orange", label="negative")
        ax.scatter(x[keep][is_positive], y[keep][is_positive], s=6, c="tab:blue", label="positive")
        ax.axvline(x=thresholds[names[0]], color="red", linestyle="--")
        ax.axhline(y=thresholds[names[1]], color="red", linestyle="--")
        ax.set_xlabel(names[0])
        ax.set_ylabel(names[1])
        ax.legend()
        if len(names) > 2:
            title += f"\n(further thresholds not shown: {', '.join(names[2:])})"
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def threshold_marker(
    cochleae: List[str],
    output_dir: Optional[str] = None,
    threshold_save_dir: Optional[str] = None,
    plot_dir: str = PLOT_OUT,
    group: str = "otof",
    seg_name: Optional[str] = None,
    marker_name: Optional[str] = None,
    component_list: Optional[List[int]] = None,
    table_seg_path: Optional[str] = None,
    table_meas_path: Optional[str] = None,
    meas_dir: Optional[str] = None,
    intensity_column: Optional[str] = None,
    threshold: Optional[float] = None,
    local: bool = False,
    mobie_dir: str = MOBIE_FOLDER,
    force_overwrite: bool = False,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
) -> None:
    """Assign a marker label to segmentation instances with fixed or automatic thresholds.

    An instance is positive when it reaches every threshold of its cochlea in THRESHOLD_DICT.
    A cochlea without an entry falls back to an Otsu threshold on the intensity column and a
    warning. Only instances within the connected components of the cochlea are considered.
    The assignment is written to the "marker_labels" column of the output segmentation table,
    with 1 for positive, 2 for negative, and 0 for an instance outside the components or
    without an object measure.

    Args:
        cochleae: List of cochleae.
        output_dir: Output directory for the segmentation table with "marker_labels", in the format
            <cochlea>_<marker>_<seg>.tsv. Without an output directory, the table is saved in the
            appropriate location in the MoBIE project. Required for data on the S3 bucket.
        threshold_save_dir: Optional directory for saving the thresholds.
        plot_dir: Directory for the intensity plots.
        group: Marker family, which selects the cochleae, the segmentation and the marker stain.
        seg_name: Identifier for the segmentation. Overrides the value of the marker family.
        marker_name: Identifier for the marker stain. Overrides the value of the marker family.
        component_list: List of connected components. Overrides the value of the cochlea dictionary.
        table_seg_path: Path to the segmentation table.
        table_meas_path: Path to a single table with object measures.
        meas_dir: Directory with the object-measures tables of all cochleae. Use this to read the
            measures from disk while the segmentation table comes from the S3 bucket.
        intensity_column: Column for the Otsu fallback. Overrides the value of the marker family.
        threshold: Fixed threshold on the intensity column, which overrides THRESHOLD_DICT.
        local: Whether to use one threshold per annotation crop from LOCAL_THRESHOLD_DICT, instead
            of one threshold per cochlea. This follows the local imaging conditions along the
            cochlea. A cochlea without local thresholds falls back to the thresholds per cochlea.
        mobie_dir: Local MoBIE directory used for creating data paths.
        force_overwrite: Whether to overwrite already existing results.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
    """
    if group not in MARKER_GROUPS:
        raise ValueError(f"Choose one of the marker groups {list(MARKER_GROUPS.keys())}, not '{group}'.")
    if output_dir is None and s3:
        raise ValueError("Specify an output directory, when data is accessed from the S3 bucket.")

    group_info = MARKER_GROUPS[group]
    cochlea_dict = group_info["cochleae"]
    seg_name = group_info["seg_name"] if seg_name is None else seg_name
    marker_name = group_info["marker_name"] if marker_name is None else marker_name
    intensity_column = group_info["intensity_column"] if intensity_column is None else intensity_column
    seg_string = seg_name.replace("_", "-")

    for cochlea in cochleae:
        cochlea_str = cochlea.replace("_", "-")
        name = f"{cochlea_str}_{marker_name}_{seg_string}"

        if cochlea in cochlea_dict:
            components = cochlea_dict[cochlea]["component_list"] if component_list is None else component_list
        elif component_list is None:
            warnings.warn(f"Cochlea {cochlea} is not in the '{group}' dictionary. "
                          "Pass the connected components with --components.")
            continue
        else:
            components = component_list

        if output_dir is None:
            out_dir = os.path.join(mobie_dir, cochlea, "tables", seg_name)
            out_path = os.path.join(out_dir, f"{marker_name}_{seg_string}.tsv")
        else:
            out_dir = output_dir
            out_path = os.path.join(out_dir, f"{name}.tsv")
        os.makedirs(out_dir, exist_ok=True)

        if os.path.exists(out_path) and not force_overwrite:
            print(f"Skipping {out_path}. Table already exists.")
            continue

        if table_seg_path is None:
            if s3:
                seg_table = f"{cochlea}/tables/{seg_name}/default.tsv"
            else:
                seg_table = os.path.join(mobie_dir, cochlea, "tables", seg_name, "default.tsv")
        else:
            seg_table = table_seg_path

        try:
            table_seg = _read_table(seg_table, s3, s3_credentials, s3_bucket_name, s3_service_endpoint)
            s3_meas = s3 and meas_dir is None and table_meas_path is None
            if table_meas_path is not None:
                key = "bg" if "bg-mask" in os.path.basename(table_meas_path) else "plain"
                meas_tables = {key: table_meas_path}
            else:
                meas_tables = _find_meas_tables(
                    cochlea, seg_name, marker_name, meas_dir=meas_dir, mobie_dir=mobie_dir, s3=s3,
                    s3_credentials=s3_credentials, s3_bucket_name=s3_bucket_name,
                    s3_service_endpoint=s3_service_endpoint,
                )
            loaded = {
                key: _read_table(path, s3_meas, s3_credentials, s3_bucket_name, s3_service_endpoint)
                for key, path in meas_tables.items()
            }
        except FileNotFoundError as e:
            warnings.warn(f"Skipping cochlea {cochlea}. {e}")
            continue

        if "component_labels" not in table_seg.columns:
            warnings.warn(f"Skipping cochlea {cochlea}. The column 'component_labels' is not in {seg_table}. "
                          "Run the component labeling first.")
            continue

        features = build_features(loaded.get("plain"), loaded.get("bg"))
        valid_ids = filter_table(table_seg, components).label_id
        if len(valid_ids) == 0:
            warnings.warn(f"Skipping cochlea {cochlea}. No instance is in the components {components}.")
            continue
        features = features[features["label_id"].isin(valid_ids)]

        use_local = local and cochlea in LOCAL_THRESHOLD_DICT
        if local and not use_local:
            warnings.warn(f"Cochlea {cochlea} has no entry in LOCAL_THRESHOLD_DICT. "
                          "Using the thresholds of the whole cochlea instead.")

        if "marker_labels" in table_seg.columns:
            print(f"{cochlea}: Replacing the existing 'marker_labels' column in the output table.")

        crop_counts = None
        if use_local:
            if "length_fraction" not in table_seg.columns:
                warnings.warn(f"Skipping cochlea {cochlea}. The column 'length_fraction' is not in {seg_table}. "
                              "Run the tonotopic mapping first.")
                continue
            if LOCAL_THRESHOLD_FEATURE not in features.columns:
                warnings.warn(f"Skipping cochlea {cochlea}. The feature '{LOCAL_THRESHOLD_FEATURE}' is not in "
                              "the measurement tables.")
                continue
            thresholds, method = dict(LOCAL_THRESHOLD_DICT[cochlea]), "local"
            table_seg, crop_counts = apply_local_thresholds(
                table_seg, features, thresholds, components, feature=LOCAL_THRESHOLD_FEATURE,
            )
            rule = f"{LOCAL_THRESHOLD_FEATURE}, {len(thresholds)} crop thresholds"
        else:
            try:
                thresholds, method = _resolve_thresholds(cochlea, features, intensity_column, threshold)
                label_ids, is_positive = apply_thresholds(features, thresholds)
            except (KeyError, ValueError) as e:
                warnings.warn(f"Skipping cochlea {cochlea}. {e}")
                continue

            n_missing = len(valid_ids) - len(label_ids)
            if n_missing > 0:
                print(f"{cochlea}: {n_missing} of {len(valid_ids)} instances have no object measure "
                      "and stay unassigned.")
            table_seg["marker_labels"] = 0
            table_seg.loc[table_seg["label_id"].isin(label_ids[is_positive]), "marker_labels"] = 1
            table_seg.loc[table_seg["label_id"].isin(label_ids[~is_positive]), "marker_labels"] = 2
            rule = ", ".join(f"{k} >= {round(v, 4)}" for k, v in thresholds.items())

        n_positive = int((table_seg["marker_labels"] == 1).sum())
        n_negative = int((table_seg["marker_labels"] == 2).sum())
        n_total = len(table_seg)
        print(f"{cochlea}: {method} [{rule}], "
              f"{percentage(n_positive, n_positive + n_negative)} % positive of {n_positive + n_negative} instances.")

        table_seg.to_csv(out_path, sep="\t", index=False)

        if threshold_save_dir is not None:
            os.makedirs(threshold_save_dir, exist_ok=True)
            per_feature = {} if use_local else {
                feature: percentage(int((features[feature].to_numpy(dtype="float64") >= value).sum()), len(features))
                for feature, value in thresholds.items()
            }
            param_dict = {
                "cochlea": cochlea,
                "marker": marker_name,
                "segmentation": seg_name,
                "method": method,
                "scope": "local" if use_local else "cochlea",
                "feature": LOCAL_THRESHOLD_FEATURE if use_local else None,
                "thresholds": {feature: (None if not np.isfinite(value) else float(value))
                               for feature, value in thresholds.items()},
                "measurement_tables": meas_tables,
                "component_list": [int(comp) for comp in components],
                "n_total": int(n_total),
                "n_positive": n_positive,
                "n_negative": n_negative,
                "n_unassigned": int(n_total - n_positive - n_negative),
                "percent_positive": percentage(n_positive, n_positive + n_negative),
                "percent_negative": percentage(n_negative, n_positive + n_negative),
                "percent_passing_each_threshold": None if use_local else per_feature,
                "crops": crop_counts,
                "all_negative_crops": sorted(
                    center for center, value in thresholds.items() if not np.isfinite(value)
                ) if use_local else None,
            }
            export_dictionary_as_json(param_dict, os.path.join(threshold_save_dir, f"{name}.json"),
                                      force_overwrite=True)

        os.makedirs(plot_dir, exist_ok=True)
        plot_path = os.path.join(plot_dir, f"{name}.png")
        if use_local:
            _plot_local_thresholds(table_seg, features, crop_counts, LOCAL_THRESHOLD_FEATURE, name, plot_path)
        else:
            _plot_thresholds(features, thresholds, is_positive, name, method, plot_path)


def main():
    parser = argparse.ArgumentParser(
        description="Assign each segmentation instance a marker label based on intensity thresholds."
    )

    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=None,
                        help="Cochlea(e) to process. Default: all cochleae of the marker group.")
    parser.add_argument("-g", "--group", type=str, default="otof", choices=list(MARKER_GROUPS.keys()),
                        help="Marker group, which selects the cochleae, the segmentation and the marker stain.")
    parser.add_argument("-o", "--output", type=str, help="Output directory.")
    parser.add_argument("-t", "--threshold_save_dir", type=str, default=None,
                        help="Output directory for the thresholds.")
    parser.add_argument("-p", "--plot_dir", type=str, default=PLOT_OUT,
                        help="Output directory for the intensity plots.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    parser.add_argument("--components", type=int, nargs="+", default=None,
                        help="Connected components, which override the entry of the cochlea dictionary.")
    parser.add_argument("--intensity_column", type=str, default=None,
                        help="Column for the Otsu fallback, used when the cochlea is not in THRESHOLD_DICT.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Fixed threshold on the intensity column, which overrides THRESHOLD_DICT.")
    parser.add_argument("--local", action="store_true",
                        help="Use one threshold per annotation crop from LOCAL_THRESHOLD_DICT.")

    # options for specific data paths
    parser.add_argument("--seg_table", type=str, default=None, help="Path to segmentation table.")
    parser.add_argument("--meas_table", type=str, default=None, help="Path to a single object-measures table.")
    parser.add_argument("--meas_dir", type=str, default=None,
                        help="Directory with the object-measures tables of all cochleae.")

    # options for creating data paths automatically
    parser.add_argument("--seg_name", type=str, default=None,
                        help="Identifier for the segmentation, which overrides the marker group.")
    parser.add_argument("--marker_name", type=str, default=None,
                        help="Identifier for the marker stain, which overrides the marker group.")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project.")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    cochleae = list(MARKER_GROUPS[args.group]["cochleae"].keys()) if args.cochlea is None else args.cochlea

    threshold_marker(
        cochleae=cochleae,
        output_dir=args.output,
        threshold_save_dir=args.threshold_save_dir,
        plot_dir=args.plot_dir,
        group=args.group,
        seg_name=args.seg_name,
        marker_name=args.marker_name,
        component_list=args.components,
        table_seg_path=args.seg_table,
        table_meas_path=args.meas_table,
        meas_dir=args.meas_dir,
        intensity_column=args.intensity_column,
        threshold=args.threshold,
        local=args.local,
        mobie_dir=args.mobie_dir,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":
    main()
