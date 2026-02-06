import argparse
import json
import os
from typing import List, Optional

from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping_single


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def wrapper_tonotopic_mapping(
    output_path: str,
    table_path: Optional[str] = None,
    ddict: Optional[str] = None,
    force_overwrite: bool = False,
    animal: str = "mouse",
    otof: bool = False,
    s3: bool = False,
    **kwargs
):
    """Wrapper function for tonotopic mapping using a segmentation table.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.

    Args:
        output_path: Output path to segmentation table with new column "component_labels".
        table_path: File path to segmentation table.
        ddict: Data dictionary containing parameters for tonotopic mapping.
        force_overwrite: Forcefully overwrite existing output path.
        animal: Animal specifier for species specific frequency mapping. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
    """
    if ddict is None:
        tonotopic_mapping_single(table_path, out_path=output_path, animal=animal, ototf=otof,
                                 force_overwrite=force_overwrite, s3=s3, **kwargs)
    else:
        if output_path is None:
            raise ValueError("Specify an output path when supplying a JSON dictionary.")
        param_dicts = _load_json_as_list(ddict)
        for params in param_dicts:

            cochlea = params["cochlea"]
            print(f"\n{cochlea}")
            seg_channel = params["segmentation_channel"]
            table_path = os.path.join(f"{cochlea}", "tables", seg_channel, "default.tsv")

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

            if os.path.isdir(output_path):
                cochlea_str = "-".join(cochlea.split("_"))
                table_str = "-".join(seg_channel.split("_"))
                save_path = os.path.join(output_path, "_".join([cochlea_str, f"{table_str}.tsv"]))
            else:
                save_path = output_path

            tonotopic_mapping_single(table_path=table_path, out_path=save_path, animal=animal, otof=otof,
                                     force_overwrite=force_overwrite, s3=s3, **params)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path. Directory (for --json) or specific file. Default: Overwrite input table.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for tonotopic mapping
    parser.add_argument("--animal", type=str, default="mouse",
                        help="Animal type to be used for frequency mapping. Either 'mouse' or 'gerbil'.")
    parser.add_argument("--otof", action="store_true", help="Use frequency mapping for OTOF cochleae.")
    parser.add_argument(
        "--apex_position", type=str, default="apex_higher",
        help="Apex is set to node with higher y-value. Use 'apex_lower' to reverse default mapping.",
    )

    # options for post-processing
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'.")
    parser.add_argument("--max_edge_distance", type=float, default=30,
                        help="Maximal distance in micrometer between points to create edges for connected components.")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1], help="List of connected components.")

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

    wrapper_tonotopic_mapping(
        output_path=args.output,
        table_path=args.input,
        ddict=args.json,
        force_overwrite=args.force,
        cell_type=args.cell_type,
        animal=args.animal,
        otof=args.otof,
        max_edge_distance=args.max_edge_distance,
        component_list=args.components,
        apex_position=args.apex_position,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
