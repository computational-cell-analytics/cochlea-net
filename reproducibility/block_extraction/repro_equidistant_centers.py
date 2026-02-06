import argparse
import json
import os
from typing import List, Optional

import pandas as pd
from flamingo_tools.s3_utils import get_s3_path
from flamingo_tools.postprocessing.cochlea_mapping import equidistant_centers, equidistant_centers_single


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def wrapper_equidistant_centers(
    input_path: str,
    output_path: Optional[str] = None,
    ddict: Optional[str] = None,
    n_blocks: int = 10,
    cell_type: str = "sgn",
    component_list: List[int] = [1],
    force_overwrite: bool = False,
    offset_blocks: bool = True,
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **kwargs,
):
    """Wrapper function for extracting blocks from volumetric data.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.
    """
    if ddict is None:
        equidistant_centers_single(input_path, output_path, s3=s3, n_blocks=n_blocks,
                                   cell_type=cell_type, component_list=component_list,
                                   force_overwrite=force_overwrite,
                                   offset_blocks=offset_blocks, **kwargs)
    else:
        param_dicts = _load_json_as_list(ddict)

        out_dict = []
        if output_path is None:
            output_path = input_path
            force_overwrite = True

        if output_path is None:
            output_path = ddict
            force_overwrite = True

        if os.path.isfile(output_path) and not force_overwrite:
            print(f"Skipping {output_path}. File already exists.")

        for params in param_dicts:
            cochlea = params["cochlea"]
            seg_channel = params["segmentation_channel"]

            s3_path = os.path.join(f"{cochlea}", "tables", f"{seg_channel}", "default.tsv")
            print(f"Finding equidistant centers for {cochlea}.")

            tsv_path, fs = get_s3_path(s3_path, bucket_name=s3_bucket_name,
                                       service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            with fs.open(tsv_path, 'r') as f:
                table = pd.read_csv(f, sep="\t")

            centers = equidistant_centers(
                table, component_label=component_list, cell_type=cell_type,
                n_blocks=n_blocks,
                offset_blocks=offset_blocks,
            )
            centers = [[round(c) for c in center] for center in centers]

            params["crop_centers"] = centers
            out_dict.append(params)

        with open(output_path, "w") as f:
            json.dump(out_dict, f, indent='\t', separators=(',', ': '))


def main():
    parser = argparse.ArgumentParser(
        description="Script to find a number of equidistant centers within an SGN or IHC segmentation to then "
                    "extract region of interest (ROI) blocks around these center coordinates.")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for JSON dictionary. Optional for --json: Table is overwritten.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")

    # options for equidistant centers
    parser.add_argument('-n', "--n_blocks", type=int, default=6,
                        help="Number of blocks to find equidistant centers for. Default: 6")
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'. Default: sgn")
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

    wrapper_equidistant_centers(
        input_path=args.input,
        output_path=args.output,
        ddict=args.json,
        n_blocks=args.n_blocks,
        cell_type=args.cell_type,
        component_list=args.components,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
