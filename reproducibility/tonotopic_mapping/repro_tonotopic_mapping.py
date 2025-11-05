import argparse
import json
import os
from typing import Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.segmentation.cochlea_mapping import tonotopic_mapping


def repro_tonotopic_mapping(
    ddict: dict,
    output_dir: str,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    force_overwrite: Optional[bool] = None,
):
    default_cell_type = "ihc"
    default_apex_position = "apex_higher"
    default_max_edge_distance = 30
    default_component_list = [1]

    remove_columns = ["tonotopic_label",
                      "tonotopic_value[kHz]",
                      "distance_to_path[µm]",
                      "length_fraction",
                      "run_length[µm]",
                      "centrality"]

    with open(ddict, 'r') as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    for dic in param_dicts:
        cochlea = dic["cochlea"]
        seg_channel = dic["segmentation_channel"]
        if "OTOF" in cochlea:
            otof = True
        else:
            otof = False

        if cochlea[0] in ["M", "m"]:
            animal = "mouse"
        elif cochlea[0] in ["G", "g"]:
            animal = "gerbil"
        else:
            animal = "mouse"
            # raise ValueError("Cochlea does not have expected name format 'M_[...]' or 'G_[...]'.")

        cochlea_str = "-".join(cochlea.split("_"))
        seg_str = "-".join(seg_channel.split("_"))
        os.makedirs(output_dir, exist_ok=True)
        output_table_path = os.path.join(output_dir, f"{cochlea_str}_{seg_str}.tsv")

        s3_path = os.path.join(f"{cochlea}", "tables", f"{seg_channel}", "default.tsv")
        print(f"Tonotopic mapping for {cochlea}.")

        tsv_path, fs = get_s3_path(s3_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, 'r') as f:
            table = pd.read_csv(f, sep="\t")

        cell_type = dic["type"] if "type" in dic else default_cell_type
        component_list = dic["component_list"] if "component_list" in dic else default_component_list
        component_mapping = dic["component_mapping"] if "component_mapping" in dic else component_list
        apex_position = dic["apex_position"] if "apex_position" in dic else default_apex_position
        max_edge_distance = dic["max_edge_distance"] if "max_edge_distance" in dic else default_max_edge_distance

        apex_higher = (apex_position == "apex_higher")

        for column in remove_columns:
            if column in list(table.columns):
                table = table.drop(column, axis=1)

        if not os.path.isfile(output_table_path) or force_overwrite:
            table = tonotopic_mapping(table, component_label=component_list, animal=animal,
                                      cell_type=cell_type, component_mapping=component_mapping,
                                      apex_higher=apex_higher, max_edge_distance=max_edge_distance,
                                      otof=otof)

            table.to_csv(output_table_path, sep="\t", index=False)

        else:
            print(f"Skipping {output_table_path}. Table already exists.")


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('-i', '--input', type=str, required=True, help="Input JSON dictionary.")
    parser.add_argument('-o', "--output", type=str, required=True, help="Output directory.")

    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    repro_tonotopic_mapping(
        args.input, args.output,
        args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
        args.force,
    )


if __name__ == "__main__":

    main()
