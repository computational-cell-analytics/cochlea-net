import argparse
import json
import os
from multiprocessing import cpu_count
from typing import List, Optional

import numpy as np
import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.s3_utils import MOBIE_FOLDER
from flamingo_tools.measurements import compute_object_measures, compute_sgn_background_mask


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def object_measures_single(
    table_path: str,
    seg_path: str,
    image_paths: List[str],
    out_paths: List[str],
    force_overwrite: bool = False,
    component_list: List[int] = [1],
    background_mask: Optional[np.typing.ArrayLike] = None,
    resolution: List[float] = [0.38, 0.38, 0.38],
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **_
):
    input_key = "s0"

    if not isinstance(resolution, float):
        if len(resolution) == 1:
            resolution = resolution * 3
        assert len(resolution) == 3
        resolution = np.array(resolution)[::-1]
    else:
        resolution = (resolution,) * 3

    for (img_path, out_path) in zip(image_paths, out_paths):
        n_threads = int(os.environ.get("SLURM_CPUS_ON_NODE", cpu_count()))

        # overwrite input file
        if os.path.realpath(out_path) == os.path.realpath(table_path) and not s3:
            force_overwrite = True

        if os.path.isfile(out_path) and not force_overwrite:
            print(f"Skipping {out_path}. Table already exists.")

        else:
            if background_mask is None:
                feature_set = "default"
                dilation = None
                median_only = False
            else:
                print("Using background mask for calculating object measures.")
                feature_set = "default_background_subtract"
                dilation = 4
                median_only = True

                if s3:
                    img_path, fs = s3_utils.get_s3_path(img_path, bucket_name=s3_bucket_name,
                                                        service_endpoint=s3_service_endpoint,
                                                        credential_file=s3_credentials)
                    seg_path, fs = s3_utils.get_s3_path(seg_path, bucket_name=s3_bucket_name,
                                                        service_endpoint=s3_service_endpoint,
                                                        credential_file=s3_credentials)

                mask_cache_path = os.path.join(os.path.dirname(out_path), "bg-mask.zarr")
                background_mask = compute_sgn_background_mask(
                    image_path=img_path,
                    segmentation_path=seg_path,
                    image_key=input_key,
                    segmentation_key=input_key,
                    n_threads=n_threads,
                    cache_path=mask_cache_path,
                )

            compute_object_measures(
                image_path=img_path,
                segmentation_path=seg_path,
                segmentation_table_path=table_path,
                output_table_path=out_path,
                image_key=input_key,
                segmentation_key=input_key,
                feature_set=feature_set,
                s3_flag=s3,
                component_list=component_list,
                dilation=dilation,
                median_only=median_only,
                background_mask=background_mask,
                n_threads=n_threads,
                resolution=resolution,
            )


def wrapper_object_measures(
    out_paths: List[str],
    image_paths: Optional[List[str]] = None,
    table_path: Optional[str] = None,
    seg_path: Optional[str] = None,
    ddict: Optional[str] = None,
    force_overwrite: bool = False,
    s3: bool = False,
    **kwargs
):
    """Wrapper function for calculationg object measures for different image channels using a segmentation table.
    The function is used to distinguish between a passed parameter dictionary in JSON format
    and the explicit setting of parameters.

    Args:
        output_paths: Output path(s) to table containing object measures.
        image_paths: Path(s) to one or multiple image channels in ome.zarr format.
        table_path: File path to segmentation table.
        seg_path: Input path to segmentation channel in ome.zarr format.
        ddict: Data dictionary containing parameters for tonotopic mapping.
        force_overwrite: Forcefully overwrite existing output path.
        s3: Use S3 bucket.
    """
    out_paths = [os.path.realpath(o) for o in out_paths]
    if ddict is None:
        object_measures_single(table_path, seg_path, image_paths, out_paths, force_overwrite=force_overwrite,
                               s3=s3, **kwargs)

    else:
        param_dicts = _load_json_as_list(ddict)
        for num, params in enumerate(param_dicts):
            cochlea = params["cochlea"]
            print(f"\n{cochlea}")
            seg_channel = params["segmentation_channel"]
            image_channels = params["image_channel"]
            table_path = os.path.join(f"{cochlea}", "tables", seg_channel, "default.tsv")
            if len(out_paths) == 1 and os.path.isdir(out_paths[0]):

                c_str = "-".join(cochlea.split("_"))
                s_str = "-".join(seg_channel.split("_"))
                out_paths_tmp = []
                for img_channel in image_channels:
                    i_str = "-".join(img_channel.split("_"))
                    out_paths_tmp.append(os.path.join(out_paths[0], f"{c_str}_{i_str}_{s_str}_object-measures.tsv"))

            else:
                assert len(image_channels) == len(out_paths)
                out_paths_tmp = out_paths.copy()

            if s3:
                image_paths = [f"{cochlea}/images/ome-zarr/{ch}.ome.zarr" for ch in image_channels]
                seg_path = f"{cochlea}/images/ome-zarr/{seg_channel}.ome.zarr"
                seg_table = f"{cochlea}/tables/{seg_channel}/default.tsv"
            else:
                image_paths = [f"{MOBIE_FOLDER}/{cochlea}/images/ome-zarr/{ch}.ome.zarr" for ch in image_channels]
                seg_path = f"{MOBIE_FOLDER}/{cochlea}/images/ome-zarr/{seg_channel}.ome.zarr"
                seg_table = f"{MOBIE_FOLDER}/{cochlea}/tables/{seg_channel}/default.tsv"

            object_measures_single(
                table_path=seg_table,
                seg_path=seg_path,
                image_paths=image_paths,
                out_paths=out_paths_tmp,
                force_overwrite=force_overwrite,
                s3=s3,
                **params,
            )


def main():
    parser = argparse.ArgumentParser(
        description="Script to compute object measures for different stainings.")

    parser.add_argument("-o", "--output", type=str, nargs="+", required=True,
                        help="Output path(s). Either directory or specific file(s).")
    parser.add_argument("-i", "--image_paths", type=str, nargs="+", default=None,
                        help="Input path to one or multiple image channels in ome.zarr format.")
    parser.add_argument("-t", "--seg_table", type=str, default=None,
                        help="Input path to segmentation table.")
    parser.add_argument("-s", "--seg_path", type=str, default=None,
                        help="Input path to segmentation channel in ome.zarr format.")
    parser.add_argument("-j", "--json", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("--force", action="store_true", help="Forcefully overwrite output.")

    # options for object measures
    parser.add_argument("-c", "--components", type=str, nargs="+", default=[1], help="List of components.")
    parser.add_argument("-r", "--resolution", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Resolution of input in micrometer.")
    parser.add_argument("--bg_mask", action="store_true", help="Use background mask for calculating object measures.")

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

    wrapper_object_measures(
        out_paths=args.output,
        image_paths=args.image_paths,
        table_path=args.seg_table,
        seg_path=args.seg_path,
        ddict=args.json,
        force_overwrite=args.force,
        s3=args.s3,
    )


if __name__ == "__main__":

    main()
