import argparse
import json
import os

import numpy as np
import pandas as pd

import flamingo_tools.intensity_annotation.eval_annotations as eval_utils
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.postprocessing.sgn_subtype_utils import STAIN_TO_TYPE, COCHLEAE, subtype_measurement_table
# from skimage.segmentation import relabel_sequential


def types_for_stain(stains):
    stains.sort()
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


def read_table(table_path):
    """Read a table from the S3 bucket."""
    tsv_path, fs = get_s3_path(table_path)
    with fs.open(tsv_path, "r") as f:
        return pd.read_csv(f, sep="\t")


def filter_subtypes(table_seg, subtype, stains=None):
    """Filter segmentation with marker labels.
    Positive segmentation instances are set to 1, negative to 2.
    """
    # get stains
    if stains is None:
        stains = [column.split("_")[1] for column in list(table_seg.columns) if "marker_" in column]
        stains.sort()

    stain_dict = stain_expression_from_subtype(subtype, stains)
    if len(stain_dict) == 0:
        raise ValueError("The dictionary containing stain information must have at least one entry. Check parameters.")

    label_ids_subtype = []
    for dic in stain_dict:
        subset = table_seg.copy()
        for stain in dic.keys():
            expression_value = 1 if dic[stain] == "+" else 2
            subset = subset[subset[f"marker_{stain}"] == expression_value]

        label_ids_subtype.extend(list(subset["label_id"]))
    return label_ids_subtype


def seg_name_for_cochlea(cochlea):
    """Get the segmentation that the subtype labels are written for."""
    if "output_seg" in list(COCHLEAE[cochlea].keys()):
        return COCHLEAE[cochlea]["output_seg"]
    return COCHLEAE[cochlea]["seg_data"]


def subtype_assignment_dict(cochlea, subtype_column="subtype_label"):
    """Get the stains of every subtype label column of a cochlea."""
    if "label_stains" in COCHLEAE[cochlea].keys():
        return COCHLEAE[cochlea]["label_stains"]
    return {subtype_column: COCHLEAE[cochlea]["subtype_stains"]}


def assign_subtypes(cochlea, output_folder, subtype_column="subtype_label"):
    seg_name = seg_name_for_cochlea(cochlea)

    for subtype_column, subtype_stains in subtype_assignment_dict(cochlea, subtype_column).items():

        subtype_stains = sorted(subtype_stains)
        out_path = os.path.join(output_folder, f"{cochlea}_subtypes.tsv")

        table = read_table(f"{cochlea}/tables/{seg_name}/default.tsv")

        print(f"Subtype stains: {subtype_stains}.")
        subtypes = types_for_stain(subtype_stains)
        subtypes.sort()

        # Subtype labels
        subtype_labels = ["None" for _ in range(len(table))]
        table[subtype_column] = subtype_labels
        for subtype in subtypes:

            label_ids_subtype = filter_subtypes(table, subtype=subtype, stains=subtype_stains)
            print(f"Subtype '{subtype}' with {len(label_ids_subtype)} instances.")
            table.loc[table["label_id"].isin(label_ids_subtype), subtype_column] = subtype

        table.to_csv(out_path, sep="\t", index=False)


def variance_file_path(variance_dir, cochlea, stain, seg_name):
    """Get the default path of the threshold variance file written by eval_subtype_annotations.py."""
    cochlea_str = cochlea.replace("_", "-")
    seg_string = seg_name.replace("_", "-")
    return os.path.join(variance_dir, f"{cochlea_str}_{stain}_{seg_string}_variance.json")


def scenario_thresholds(variance_dic, scenario):
    """Get the threshold per crop that one scenario of a variance file uses."""
    thresholds = {}
    for center, crop_dic in variance_dic["crops"].items():
        if scenario in crop_dic:
            thresholds[center] = crop_dic[scenario]["threshold"]
    return thresholds


def count_subtypes(table_seg, stains, subtypes, component_labels=None):
    """Count the segmentation instances per subtype.

    Only instances with a positive or negative marker label in every stain are counted, which is the
    same set of instances that the subtype fraction plots use.

    Args:
        table_seg: Segmentation table with a marker_<stain> column per stain.
        stains: The stains of the subtype pairing.
        subtypes: The subtypes that the stain pairing can produce.
        component_labels: Optional connected components to restrict the count to.

    Returns:
        Dictionary with the instance counts and percentages per subtype.
    """
    if component_labels is not None:
        table_seg = table_seg[table_seg["component_labels"].isin(component_labels)]

    assigned = table_seg
    for stain in stains:
        assigned = assigned[assigned[f"marker_{stain}"].isin([1, 2])]

    counts = {subtype: len(filter_subtypes(assigned, subtype=subtype, stains=list(stains)))
              for subtype in subtypes}
    n_assigned = len(assigned)
    return {
        "n_assigned": n_assigned,
        "n_unassigned": int(len(table_seg) - n_assigned),
        "counts": counts,
        "percent": {subtype: eval_utils.percentage(count, n_assigned) for subtype, count in counts.items()},
    }


def subtype_variance_for_pairing(cochlea, seg_name, stains, variance_dics, table_seg, subtypes):
    """Compare the subtype percentages that the thresholds of the individual annotators produce.

    Args:
        cochlea: The name of the cochlea.
        seg_name: Identifier for the segmentation.
        stains: The stains of the subtype pairing.
        variance_dics: The content of the variance file per stain.
        table_seg: Segmentation table of the cochlea.
        subtypes: The subtypes that the stain pairing can produce.

    Returns:
        Dictionary with the subtype percentages per scenario and their variance over the annotators.
    """
    # an annotator must have annotated every stain of the pairing to give a consistent subtype
    annotators = set.intersection(*[set(variance_dics[stain]["annotators"]) for stain in stains])
    for stain in stains:
        missing = sorted(set(variance_dics[stain]["annotators"]) - annotators)
        for annotator in missing:
            print(f"Skipping annotator {annotator} for the pairing {'/'.join(stains)} of cochlea {cochlea}. "
                  f"The annotator is missing for stain {stain}.")
    annotators = sorted(annotators)

    measurements = {}
    for stain in stains:
        meas_table, column = subtype_measurement_table(cochlea, stain, seg_name, s3=True)
        measurements[stain] = (read_table(meas_table), column)

    component_labels = COCHLEAE[cochlea].get("component_list", [1])
    scenarios = {}
    for scenario in ["median"] + annotators:
        table_scenario = table_seg.copy()
        for stain in stains:
            thresholds = scenario_thresholds(variance_dics[stain], scenario)
            if len(thresholds) == 0:
                print(f"Skipping scenario {scenario} for stain {stain} of cochlea {cochlea}. No threshold available.")
                table_scenario = None
                break
            table_meas, column = measurements[stain]
            intensity_dic = {center: {"median_intensity": threshold} for center, threshold in thresholds.items()}
            table_scenario = eval_utils.apply_nearest_threshold(
                intensity_dic, table_scenario, table_meas, column=column, suffix=stain,
            )
        if table_scenario is None:
            continue
        scenarios[scenario] = count_subtypes(
            table_scenario, stains, subtypes, component_labels=component_labels,
        )

    evaluated = [annotator for annotator in annotators if annotator in scenarios]

    def percent_difference(first, second):
        return {subtype: float(round(first["percent"][subtype] - second["percent"][subtype], 4))
                for subtype in subtypes}

    deviation = {}
    if "median" in scenarios:
        deviation = {annotator: percent_difference(scenarios[annotator], scenarios["median"])
                     for annotator in evaluated}

    pairwise_difference = {}
    for num, first in enumerate(evaluated):
        for second in evaluated[num + 1:]:
            pairwise_difference[f"{first}-{second}"] = percent_difference(scenarios[first], scenarios[second])

    percentages = {
        subtype: [scenarios[annotator]["percent"][subtype] for annotator in evaluated] for subtype in subtypes
    }
    variance = {subtype: float(round(np.var(values), 4)) if len(values) > 0 else None
                for subtype, values in percentages.items()}
    deviation_std = {subtype: float(round(np.std(values), 4)) if len(values) > 0 else None
                     for subtype, values in percentages.items()}

    return {
        "stains": list(stains),
        "annotators": evaluated,
        "custom_thresholds": any(variance_dics[stain].get("custom_thresholds", False) for stain in stains),
        "scenarios": scenarios,
        "deviation_from_median": deviation,
        "pairwise_difference": pairwise_difference,
        "variance": variance,
        "std": deviation_std,
    }


def subtype_variance(cochlea, output_folder, variance_dir):
    """Evaluate how much the subtype percentages of a cochlea depend on the annotator.

    The thresholds of the individual annotators are read from the variance files that
    eval_subtype_annotations.py writes per subtype stain. They are applied again to the whole
    cochlea, so that the subtype pairing can be evaluated for every annotator separately.

    Args:
        cochlea: The name of the cochlea.
        output_folder: Output directory for <cochlea>_subtypes_variance.json.
        variance_dir: Directory containing the variance files of the subtype stains.
    """
    seg_name = seg_name_for_cochlea(cochlea)
    table_seg = None
    labels = {}

    for subtype_column, subtype_stains in subtype_assignment_dict(cochlea).items():
        stains = sorted(subtype_stains)
        variance_paths = {stain: variance_file_path(variance_dir, cochlea, stain, seg_name) for stain in stains}
        missing = [path for path in variance_paths.values() if not os.path.exists(path)]
        if len(missing) > 0:
            print(f"Skipping the pairing {'/'.join(stains)} of cochlea {cochlea}. "
                  f"Missing variance file(s): {', '.join(missing)}.")
            continue

        variance_dics = {}
        for stain, path in variance_paths.items():
            with open(path, "r") as f:
                variance_dics[stain] = json.load(f)

        if table_seg is None:
            table_seg = read_table(f"{cochlea}/tables/{seg_name}/default.tsv")

        subtypes = types_for_stain(list(stains))
        subtypes.sort()
        print(f"Evaluating the variance of the subtypes {subtypes} for the stains {stains}.")

        pairing = subtype_variance_for_pairing(cochlea, seg_name, stains, variance_dics, table_seg, subtypes)
        pairing["variance_files"] = {stain: os.path.basename(path) for stain, path in variance_paths.items()}
        labels[subtype_column] = pairing

    if len(labels) == 0:
        print(f"No variance file found for cochlea {cochlea} in {variance_dir}.")
        return

    out_path = os.path.join(output_folder, f"{cochlea}_subtypes_variance.json")
    result = {
        "cochlea": cochlea,
        "segmentation": seg_name,
        "component_labels": COCHLEAE[cochlea].get("component_list", [1]),
        "labels": labels,
    }
    os.makedirs(output_folder, exist_ok=True)
    # The keys are not sorted, so that the per-scenario results stay in the order of the evaluation.
    with open(out_path, "w") as f:
        json.dump(result, f, indent=4)
    print(f"Saved the subtype variance to {out_path}.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", required=True, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output_folder", required=True)
    parser.add_argument("--variance", type=str, default=None,
                        help="Directory containing the threshold variance files of the subtype stains. "
                        "If given, the subtype percentages are evaluated per annotator and saved as "
                        "<cochlea>_subtypes_variance.json.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    for cochlea in args.cochlea:
        assign_subtypes(cochlea, args.output_folder)
        if args.variance is not None:
            subtype_variance(cochlea, args.output_folder, args.variance)


if __name__ == "__main__":
    main()
