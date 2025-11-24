import argparse
import json
import os
from typing import List, Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.postprocessing.label_components import label_components_sgn, label_components_ihc


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


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def label_components_single(
    table_path: str,
    out_path: str,
    cell_type: str = "sgn",
    component_list: List[int] = [1],
    max_edge_distance: float = 30,
    min_component_length: int = 50,
    min_size: int = 1000,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    custom_dic: Optional[dict] = None,
    **_
):
    """Process a single cochlea using one set of parameters or a custom_dic.
    """
    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
    with fs.open(tsv_path, "r") as f:
        table = pd.read_csv(f, sep="\t")

    if custom_dic is not None:
        tsv_table = label_custom_components(table, custom_dic)
    else:
        if cell_type == "sgn":
            tsv_table = label_components_sgn(table, min_size=min_size,
                                             min_component_length=min_component_length,
                                             max_edge_distance=max_edge_distance)
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
            num_instances = len(tsv_table[tsv_table["component_labels"] == comp])
            print(f"Component {comp} has {num_instances} instances.")
        print(f"Custom component(s) have {custom_comp} {cell_type.upper()}s.")

    tsv_table.to_csv(out_path, sep="\t", index=False)


def repro_label_components(
    output_path: str,
    table_path: Optional[str] = None,
    ddict: Optional[str] = None,
    **kwargs
):
    """Wrapper function for labeling connected components using a segmentation table.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.
    """
    if ddict is None:
        label_components_single(table_path, output_path, **kwargs)
    else:
        param_dicts = _load_json_as_list(ddict)
        for params in param_dicts:

            cochlea = params["cochlea"]
            print(f"\n{cochlea}")
            seg_channel = params["segmentation_channel"]
            table_path = os.path.join(f"{cochlea}", "tables", seg_channel, "default.tsv")

            if os.path.isdir(output_path):
                cochlea_str = "-".join(cochlea.split("_"))
                table_str = "-".join(seg_channel.split("_"))
                save_path = os.path.join(output_path, "_".join([cochlea_str, f"{table_str}.tsv"]))
            else:
                save_path = output_path
            label_components_single(table_path=table_path, out_path=save_path, **params)


def main():
    parser = argparse.ArgumentParser(
        description="Script to label segmentation using a segmentation table and graph connected components.")

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path. Either directory or specific file.")

    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")

    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'.")

    # options for post-processing
    parser.add_argument("--min_size", type=int, default=1000,
                        help="Minimal number of pixels for filtering small instances.")
    parser.add_argument("--min_component_length", type=int, default=50,
                        help="Minimal length for filtering out connected components.")
    parser.add_argument("--max_edge_distance", type=float, default=30,
                        help="Maximal distance in micrometer between points to create edges for connected components.")
    parser.add_argument("-c", "--components", type=str, nargs="+", default=[1], help="List of connected components.")

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

    repro_label_components(
        output_path=args.output,
        table_path=args.input,
        ddict=args.json,
        cell_type=args.cell_type,
        component_list=args.components,
        max_edge_distance=args.max_edge_distance,
        min_component_length=args.min_component_length,
        min_size=args.min_size,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
