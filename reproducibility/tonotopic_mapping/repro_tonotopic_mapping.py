import argparse
import json
import os
from typing import List, Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def tonotopic_mapping_single(
    table_path: str,
    out_path: str,
    force_overwrite: bool = False,
    cell_type: str = "sgn",
    animal: str = "mouse",
    otof: bool = False,
    apex_position: str = "apex_higher",
    component_list: List[int] = [1],
    component_mapping: Optional[List[int]] = None,
    max_edge_distance: float = 30,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **_
):
    """Tonotopic mapping of a single cochlea.
    Each segmentation instance within a given component list is assigned a frequency[kHz], a run length and an offset.
    The components used for the mapping itself can be a subset of the component list to adapt to broken components
    along the Rosenthal's canal.
    If the cochlea is broken in the direction of the Rosenthal's canal, the components have to be provided in a
    continuous order which reflects the positioning within 3D.
    The frequency is calculated using the Greenwood function using animal specific parameters.
    The orientation of the mapping can be reversed using the apex position in reference to the y-coordinate.

    Args:
        table_path: File path to segmentation table.
        out_path: Output path to segmentation table with new column "component_labels".
        force_overwrite: Forcefully overwrite existing output path.
        cell_type: Cell type of the segmentation. Currently supports "sgn" and "ihc".
        animal: Animal for species specific frequency mapping. Either "mouse" or "gerbil".
        otof: Use mapping by *Mueller, Hearing Research 202 (2005) 63-73* for OTOF cochleae.
        apex_position: Identify position of apex and base. Apex is set to node with higher y-value per default.
        component_list: List of components. Can be passed to obtain the number of instances within the component list.
        components_mapping: Components to use for tonotopic mapping. Ignore components torn parallel to main canal.
        max_edge_distance: Maximal edge distance between graph nodes to create an edge between nodes.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """
    if os.path.isdir(out_path):
        raise ValueError(f"Output path {out_path} is a directory. Provide a path to a single output file.")

    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table = pd.read_csv(table_path, sep="\t")

    apex_higher = (apex_position == "apex_higher")

    # overwrite input file
    if os.path.realpath(out_path) == os.path.realpath(table_path) and not s3:
        force_overwrite = True

    if os.path.isfile(out_path) and not force_overwrite:
        print(f"Skipping {out_path}. Table already exists.")

    else:
        table = tonotopic_mapping(table, component_label=component_list, animal=animal,
                                  cell_type=cell_type, component_mapping=component_mapping,
                                  apex_higher=apex_higher, max_edge_distance=max_edge_distance,
                                  otof=otof)

        table.to_csv(out_path, sep="\t", index=False)


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
        tonotopic_mapping_single(table_path, output_path, animal=animal, ototf=otof, force_overwrite=force_overwrite,
                                 s3=s3, **kwargs)
    else:
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

    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path. Either directory or specific file.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")

    # options for tonotopic mapping
    parser.add_argument("--animal", type=str, default="mouse",
                        help="Animyl type to be used for frequency mapping. Either 'mouse' or 'gerbil'.")
    parser.add_argument("--otof", action="store_true", help="Use frequency mapping for OTOF cochleae.")
    parser.add_argument("--apex_position", type=str, default="apex_higher",
                        help="Use frequency mapping for OTOF cochleae.")

    # options for post-processing
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'.")
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
