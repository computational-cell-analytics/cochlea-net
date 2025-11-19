import argparse
import json
import os
from typing import Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.postprocessing import label_components_sgn, label_components_ihc


def label_custom_components(tsv_table, custom_dict):
    """Label IHC components using multiple post-processing configurations and combine the
    results into final components.
    The function applies successive post-processing steps defined in a `custom_dic`
    configuration. Each entry under `label_dicts` specifies:
    - `label_params`: a list of parameter sets. The segmentation is processed once for
    each parameter set (e.g., {"min_size": 500, "max_edge_distance": 65, "min_component_length": 5}).
    - `components`: lists of label IDs to extract from each corresponding post-processing run.
    Label IDs collected from all runs are merged to form the final component (e.g., key "1").
    Global filtering is applied using `min_size_global`, and any `missing_ids`
    (e.g., 4800 or 4832) are added explicitly to the final component.
    Example `custom_dic` structure:
    {
        "min_size_global": 500,
        "missing_ids": [4800, 4832],
        "label_dicts": {
            "1": {
                "label_params": [
                    {"min_size": 500, "max_edge_distance": 65, "min_component_length": 5},
                    {"min_size": 400, "max_edge_distance": 45, "min_component_length": 5}
                ],
                "components": [[18, 22], [1, 45, 83]]
            }
        }
    }

    Args:
        tsv_table: Pandas dataframe of the MoBIE segmentation table.
        custom_dict: Custom dictionary featuring post-processing parameters.

    Returns:
        Pandas dataframe featuring labeled components.
    """
    min_size = custom_dict["min_size_global"]
    component_labels = [0 for _ in range(len(tsv_table))]
    tsv_table.loc[:, "component_labels"] = component_labels
    for custom_comp, label_dict in custom_dict["label_dicts"].items():
        label_params = label_dict["label_params"]
        label_components = label_dict["components"]

        combined_label_ids = []
        for comp, other_kwargs in zip(label_components, label_params):
            tsv_table_tmp = label_components_ihc(tsv_table.copy(), **other_kwargs)
            label_ids = list(tsv_table_tmp.loc[tsv_table_tmp["component_labels"].isin(comp), "label_id"])
            combined_label_ids.extend(label_ids)
            print(f"{comp}", len(combined_label_ids))

        combined_label_ids = list(set(combined_label_ids))

        tsv_table.loc[tsv_table["label_id"].isin(combined_label_ids), "component_labels"] = int(custom_comp)

    tsv_table.loc[tsv_table["n_pixels"] < min_size, "component_labels"] = 0
    if "missing_ids" in list(custom_dict.keys()):
        for m in custom_dict["missing_ids"]:
            tsv_table.loc[tsv_table["label_id"] == m, "component_labels"] = 1

    return tsv_table


def repro_label_components(
    ddict: dict,
    output_dir: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    default_cell_type = "sgn"
    default_component_list = [1]
    default_iterations_erode = None
    default_max_edge_distance = 30
    default_min_length = 50
    default_min_size = 1000
    default_seg_channel = "SGN_v2"
    default_threshold_erode = None

    with open(ddict, "r") as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    for dic in param_dicts:
        cochlea = dic["cochlea"]
        print(f"\n{cochlea}")

        cell_type = dic.get("cell_type", default_cell_type)
        component_list = dic.get("component_list", default_component_list)
        iterations_erode = dic.get("iterations_erode", default_iterations_erode)
        max_edge_distance = dic.get("max_edge_distance", default_max_edge_distance)
        min_component_length = dic.get("min_component_length", default_min_length)
        min_size = dic.get("min_size", default_min_size)
        table_name = dic.get("segmentation_channel", default_seg_channel)
        threshold_erode = dic.get("threshold_erode", default_threshold_erode)

        s3_path = os.path.join(f"{cochlea}", "tables", table_name, "default.tsv")
        tsv_path, fs = get_s3_path(s3_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")

        if "custom_dic" in list(dic.keys()):
            print(len(table[table["component_labels"] == 1]))
            tsv_table = label_custom_components(table, dic["custom_dic"])
        else:
            if cell_type == "sgn":
                tsv_table = label_components_sgn(table, min_size=min_size,
                                                 threshold_erode=threshold_erode,
                                                 min_component_length=min_component_length,
                                                 max_edge_distance=max_edge_distance,
                                                 iterations_erode=iterations_erode)
            elif cell_type == "ihc":
                tsv_table = label_components_ihc(table, min_size=min_size,
                                                 min_component_length=min_component_length,
                                                 max_edge_distance=max_edge_distance)
            else:
                raise ValueError("Choose a supported cell type. Either 'sgn' or 'ihc'.")

        custom_comp = len(tsv_table[tsv_table["component_labels"].isin(component_list)])
        print(f"Total {cell_type.upper()}s: {len(tsv_table)}")
        if component_list == [1]:
            print(f"Largest component has {custom_comp} {cell_type.upper()}s.")
        else:
            for comp in component_list:
                print(f"Component {comp} has {len(tsv_table[tsv_table["component_labels"] == comp])} instances.")
            print(f"Custom component(s) have {custom_comp} {cell_type.upper()}s.")

        cochlea_str = "-".join(cochlea.split("_"))
        table_str = "-".join(table_name.split("_"))
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "_".join([cochlea_str, f"{table_str}.tsv"]))

        tsv_table.to_csv(out_path, sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Script to label segmentation using a segmentation table and graph connected components.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Input JSON dictionary.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory.")

    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    repro_label_components(
        args.input, args.output,
        args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
