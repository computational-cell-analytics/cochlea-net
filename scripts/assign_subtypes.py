import argparse
import os

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
# from skimage.segmentation import relabel_sequential

COCHLEA_DICT = {
    "M_LR_000098_L": {"seg_data": "SGN_v2", "subtype": ["CR", "Ntng1"]},
    "M_LR_000099_L": {"seg_data": "PV_SGN_v2", "subtype": ["Calb1", "Lypd1"]},
    "M_LR_000184_L": {"seg_data": "SGN_v2b", "subtype": ["Prph"]},
    "M_LR_000184_R": {"seg_data": "SGN_v2b", "subtype": ["Prph"]},
    "M_LR_000260_L": {"seg_data": "SGN_v2", "subtype": ["Prph", "Tuj1"]},
    "M_LR_N110_L": {"seg_data": "SGN_v2", "subtype": ["Calb1", "Ntng1"]},
    "M_LR_N110_R": {"seg_data": "SGN_v2", "subtype": ["Calb1", "Ntng1"]},
    "M_LR_N152_L": {"seg_data": "SGN_v2", "subtype": ["CR", "Ntng1"]},
    "M_AMD_N180_L": {"seg_data": "SGN_merged", "subtype": ["CR", "Ntng1"]},
    "M_AMD_N180_R": {"seg_data": "SGN_merged", "subtype": ["CR", "Ntng1"]},
}


STAIN_TO_TYPE = {
    # Combinations of Calb1 and CR:
    "CR+/Calb1+": "Type Ib",
    "CR-/Calb1+": "Type IbIc",  # Calb1 is expressed at Ic less than Lypd1 but more then CR
    "CR+/Calb1-": "Type Ia",
    "CR-/Calb1-": "Type II",

    # Combinations of Calb1 and Lypd1:
    "Calb1+/Lypd1+": "Type IbIc",
    "Calb1+/Lypd1-": "Type Ib",
    "Calb1-/Lypd1+": "Type Ic",
    "Calb1-/Lypd1-": "inconclusive",  # Can be Type Ia or Type II

    # Combinations of Prph and Tuj1:
    "Prph+/Tuj1+": "Type II",
    "Prph+/Tuj1-": "Type II",
    "Prph-/Tuj1+": "Type I",
    "Prph-/Tuj1-": "inconclusive",

    # Prph is isolated.
    "Prph+": "Type II",
    "Prph-": "Type I",

    # Combinations of CR and Ntng1
    "CR+/Ntng1+": "Type Ib",
    "CR+/Ntng1-": "Type Ia",
    "CR-/Ntng1+": "Type Ic",
    "CR-/Ntng1-": "inconclusive",

    # Combinations of Calb1 and Ntng1
    "Calb1+/Ntng1+": "Type Ib",
    "Calb1+/Ntng1-": "inconclusive",
    "Calb1-/Ntng1+": "Type Ic",
    "Calb1-/Ntng1-": "inconclusive",
}


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


def filter_subtypes(cochlea, seg_name, subtype, stains=None):
    """Filter segmentation with marker labels.
    Positive segmentation instances are set to 1, negative to 2.
    """
    internal_path = os.path.join(cochlea, "tables",  seg_name, "default.tsv")
    tsv_path, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
    with fs.open(tsv_path, "r") as f:
        table_seg = pd.read_csv(f, sep="\t")

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


def export_lower_resolution(args):

    cochlea = args.cochlea
    subtype_stains = COCHLEA_DICT[cochlea]["subtype"]
    subtype_stains.sort()
    seg_name = COCHLEA_DICT[cochlea]["seg_data"]

    out_path = os.path.join(args.output_folder, f"{cochlea}_subtypes.tsv")

    table_seg_path = f"{cochlea}/tables/{seg_name}/default.tsv"
    table_path_s3, fs = get_s3_path(table_seg_path)
    with fs.open(table_path_s3, "r") as f:
        table = pd.read_csv(f, sep="\t")

    print(f"Subtype stains: {subtype_stains}.")
    subtypes = types_for_stain(subtype_stains)
    subtypes.sort()

    # Subtype labels
    subtype_labels = ["None" for _ in range(len(table))]
    table["subtype_label"] = subtype_labels
    for subtype in subtypes:

        label_ids_subtype = filter_subtypes(cochlea, seg_name=seg_name, subtype=subtype, stains=subtype_stains)
        print(f"Subtype '{subtype}' with {len(label_ids_subtype)} instances.")
        table.loc[table["label_id"].isin(label_ids_subtype), "subtype_label"] = subtype

    table.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", "-c", required=True)
    parser.add_argument("--output_folder", "-o", required=True)
    args = parser.parse_args()

    export_lower_resolution(args)


if __name__ == "__main__":
    main()
