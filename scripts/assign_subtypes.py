import argparse
import os

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
from flamingo_tools.segmentation.sgn_subtype_utils import STAIN_TO_TYPE, COCHLEAE
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


def assign_subtypes(cochlea, output_folder, subtype_column="subtype_label"):
    if "label_stains" in COCHLEAE[cochlea].keys():
        for subtype_column, subtype_stains in COCHLEAE[cochlea]["label_stains"].items():

            subtype_stains.sort()
            if "output_seg" in list(COCHLEAE[cochlea].keys()):
                seg_name = COCHLEAE[cochlea]["output_seg"]
            else:
                seg_name = COCHLEAE[cochlea]["seg_data"]

            out_path = os.path.join(output_folder, f"{cochlea}_subtypes.tsv")

            table_seg_path = f"{cochlea}/tables/{seg_name}/default.tsv"
            table_path_s3, fs = get_s3_path(table_seg_path)
            with fs.open(table_path_s3, "r") as f:
                table = pd.read_csv(f, sep="\t")

            print(f"Subtype stains: {subtype_stains}.")
            subtypes = types_for_stain(subtype_stains)
            subtypes.sort()

            # Subtype labels
            subtype_labels = ["None" for _ in range(len(table))]
            table[subtype_column] = subtype_labels
            for subtype in subtypes:

                label_ids_subtype = filter_subtypes(cochlea, seg_name=seg_name, subtype=subtype, stains=subtype_stains)
                print(f"Subtype '{subtype}' with {len(label_ids_subtype)} instances.")
                table.loc[table["label_id"].isin(label_ids_subtype), subtype_column] = subtype

            table.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", required=True, help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output_folder", required=True)
    args = parser.parse_args()

    for cochlea in args.cochlea:
        assign_subtypes(cochlea, args.output_folder)


if __name__ == "__main__":
    main()
