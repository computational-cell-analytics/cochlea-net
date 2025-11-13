import argparse
import json
import os
from typing import Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.postprocessing import label_components_sgn, label_components_ihc
from flamingo_tools.segmentation.cochlea_mapping import tonotopic_mapping


def repro_label_components(
    ddict: dict,
    output_dir: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    apply_tonotopic_mapping: bool = False,
):
    min_size = 50
    default_threshold_erode = None
    default_min_length = 50
    default_max_edge_distance = 30
    default_iterations_erode = None
    default_cell_type = "sgn"
    default_component_list = [1]

    with open(ddict, "r") as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    for dic in param_dicts:
        cochlea = dic["cochlea"]
        print(f"Labeling components for {cochlea}.")

        threshold_erode = dic["threshold_erode"] if "threshold_erode" in dic else default_threshold_erode
        min_component_length = dic["min_component_length"] if "min_component_length" in dic else default_min_length
        max_edge_distance = dic["max_edge_distance"] if "max_edge_distance" in dic else default_max_edge_distance
        iterations_erode = dic["iterations_erode"] if "iterations_erode" in dic else default_iterations_erode
        cell_type = dic["cell_type"] if "cell_type" in dic else default_cell_type
        component_list = dic["component_list"] if "component_list" in dic else default_component_list

        # The table name sometimes has to be over-written.
        # table_name = "PV_SGN_V2_DA"
        # table_name = "CR_SGN_v2"
        # table_name = "Ntng1_SGN_v2"
        table_name = "SGN_detect_v10"

        # table_name = f"{cell_type.upper()}_{unet_version}"

        s3_path = os.path.join(f"{cochlea}", "tables", table_name, "default.tsv")
        tsv_path, fs = get_s3_path(s3_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")

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

        largest_comp = len(tsv_table[tsv_table["component_labels"].isin(component_list)])
        print(f"The segmentation features {len(tsv_table)} {cell_type.upper()}s.")
        if component_list == [1]:
            print(f"Largest component has {largest_comp} {cell_type.upper()}s.")
        else:
            print(f"Custom component(s) have {largest_comp} {cell_type.upper()}s.")

        if apply_tonotopic_mapping:
            tsv_table = tonotopic_mapping(tsv_table, cell_type=cell_type)

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
    parser.add_argument("-t", "--tonotopic_mapping", action="store_true", help="Also compute the tonotopic mapping.")

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
        apply_tonotopic_mapping=args.tonotopic_mapping,
    )


if __name__ == "__main__":

    main()
