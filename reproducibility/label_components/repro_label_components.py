import argparse
import json
import os
from typing import List, Optional

from flamingo_tools.postprocessing.label_components import label_components_single


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def wrapper_label_components(
    output_path: str,
    table_path: Optional[str] = None,
    ddict: Optional[str] = None,
    s3: bool = False,
    **kwargs
):
    """Wrapper function for labeling connected components using a segmentation table.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.
    """
    if ddict is None:
        label_components_single(table_path, output_path, s3=s3, **kwargs)
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
            label_components_single(table_path=table_path, out_path=save_path, s3=s3,
                                    **params)


def main():
    parser = argparse.ArgumentParser(
        description="Script to label segmentation using a segmentation table and graph connected components.")

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path. Either directory (for --json) or specific file otherwise.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")

    # options for post-processing
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'.")
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

    wrapper_label_components(
        output_path=args.output,
        table_path=args.input,
        ddict=args.json,
        cell_type=args.cell_type,
        component_list=args.components,
        max_edge_distance=args.max_edge_distance,
        min_component_length=args.min_component_length,
        min_size=args.min_size,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
