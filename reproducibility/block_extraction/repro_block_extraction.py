import argparse
import json
import os
from typing import List, Optional

from flamingo_tools.extract_block_util import extract_block_single


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def wrapper_extract_block(
    input_path: str,
    output_path: str,
    center_coords: Optional[str] = None,
    ddict: Optional[str] = None,
    s3: bool = False,
    **kwargs
):
    """Wrapper function for extracting blocks from volumetric data.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.
    """
    if ddict is None:
        if len(center_coords) != 3:
            print(f"Coordinates {center_coords} are not suited for 3D block extraction.")
        extract_block_single(input_path, center_coords, output_path, s3=s3, **kwargs)
    else:
        param_dicts = _load_json_as_list(ddict)
        input_key = "s0" if s3 else None

        for params in param_dicts:
            cochlea = params["cochlea"]

            print(f"\n{cochlea}")
            if isinstance(params["image_channel"], list):
                image_channels = params["image_channel"]
            else:
                image_channels = [params["image_channel"]]

            for image_channel in image_channels:
                input_path = os.path.join(cochlea, "images", "ome-zarr", image_channel + ".ome.zarr")
                for coords in params["crop_centers"]:
                    extract_block_single(input_path, coords, output_path, input_key=input_key, s3=s3, **params)


def main():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument('-o', "--output", type=str, required=True, help="Output directory or file.")
    parser.add_argument('-i', '--input', type=str, default=None,
                        help="Input path to data in n5/ome-zarr/TIF format.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")

    # options for block etraction
    parser.add_argument("-c", "--coords", type=int, nargs="+", default=[],
                        help="3D coordinate as center of extracted block [µm].")
    parser.add_argument('-k', "--input_key", type=str, default=None,
                        help="Input key for data in input file with n5/OME-ZARR format.")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Output key for data in output file with n5 format. Default: TIF file.")
    parser.add_argument('-r', "--resolution", type=float, default=0.38,
                        help="Resolution of input in micrometer [µm]. Default: 0.38")
    parser.add_argument("--roi_halo", type=int, nargs="+", default=[128, 128, 64],
                        help="ROI halo around center coordinate [pixel]. Default: 128 128 64")

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

    wrapper_extract_block(
        output_path=args.output,
        table_path=args.input,
        center_coords=args.coords,
        ddict=args.json,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
