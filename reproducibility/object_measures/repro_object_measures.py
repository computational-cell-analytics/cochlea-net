import argparse
import json
import os
from multiprocessing import cpu_count
from typing import Optional

import numpy as np
import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.measurements import compute_object_measures, compute_sgn_background_mask


def repro_object_measures(
    json_file: str,
    output_dir: str,
    force_overwrite: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    s3_flag = True
    input_key = "s0"
    default_resolution = 0.38
    default_component_list = [1]
    default_bg_mask = None

    with open(json_file, "r") as myfile:
        data = myfile.read()
    param_dicts = json.loads(data)

    for dic in param_dicts:
        cochlea = dic["cochlea"]
        image_channels = dic["image_channel"] if isinstance(dic["image_channel"], list) else [dic["image_channel"]]
        seg_channel = dic["segmentation_channel"]
        resolution = tuple(dic["resolution"]) if "resolution" in dic else default_resolution
        component_list = dic["component_list"] if "component_list" in dic else default_component_list
        bg_mask = dic["background_mask"] if "background_mask" in dic else default_bg_mask
        print(f"Processing cochlea {cochlea}")

        if not isinstance(resolution, float):
            assert len(resolution) == 3
            resolution = np.array(resolution)[::-1]

        for img_channel in image_channels:

            print(f"Processing image channel {img_channel}")
            cochlea_str = "-".join(cochlea.split("_"))
            img_str = "-".join(img_channel.split("_"))
            seg_str = "-".join(seg_channel.split("_"))
            output_table_path = os.path.join(output_dir, f"{cochlea_str}_{img_str}_{seg_str}_object-measures.tsv")

            img_s3 = f"{cochlea}/images/ome-zarr/{img_channel}.ome.zarr"
            seg_s3 = f"{cochlea}/images/ome-zarr/{seg_channel}.ome.zarr"
            seg_table_s3 = f"{cochlea}/tables/{seg_channel}/default.tsv"

            img_path, fs = s3_utils.get_s3_path(img_s3, bucket_name=s3_bucket_name,
                                                service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
            seg_path, fs = s3_utils.get_s3_path(seg_s3, bucket_name=s3_bucket_name,
                                                service_endpoint=s3_service_endpoint, credential_file=s3_credentials)

            n_threads = int(os.environ.get("SLURM_CPUS_ON_NODE", cpu_count()))
            if os.path.isfile(output_table_path) and not force_overwrite:
                print(f"Skipping creation of {output_table_path}. File already exists.")

            else:
                if bg_mask is None:
                    feature_set = "default"
                    dilation = None
                    median_only = False
                else:
                    print("Using background mask for calculating object measures.")
                    feature_set = "default_background_subtract"
                    dilation = 4
                    median_only = True
                    mask_cache_path = os.path.join(output_dir, f"{cochlea_str}_{img_str}_{seg_str}_bg-mask.zarr")
                    bg_mask = compute_sgn_background_mask(
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
                    segmentation_table_path=seg_table_s3,
                    output_table_path=output_table_path,
                    image_key=input_key,
                    segmentation_key=input_key,
                    feature_set=feature_set,
                    s3_flag=s3_flag,
                    component_list=component_list,
                    dilation=dilation,
                    median_only=median_only,
                    background_mask=bg_mask,
                    n_threads=n_threads,
                    resolution=resolution,
                )


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

    repro_object_measures(
        args.input, args.output, args.force,
        args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
