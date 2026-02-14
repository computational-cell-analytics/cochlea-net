import argparse
import json
import os
from typing import List, Optional

from flamingo_tools.s3_utils import MOBIE_FOLDER
from flamingo_tools.measurements import object_measures_single


def _load_json_as_list(ddict_path: str) -> List[dict]:
    with open(ddict_path, "r") as f:
        data = json.loads(f.read())
    # ensure the result is always a list
    return data if isinstance(data, list) else [data]


def wrapper_object_measures(
    out_paths: List[str],
    image_paths: Optional[List[str]] = None,
    table_path: Optional[str] = None,
    seg_path: Optional[str] = None,
    ddict: Optional[str] = None,
    force_overwrite: bool = False,
    s3: bool = False,
    use_bg_mask: bool = False,
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
    if ddict is None:
        object_measures_single(table_path, seg_path, image_paths, out_paths, force_overwrite=force_overwrite,
                               s3=s3, **kwargs)

    else:
        out_paths = [os.path.realpath(o) for o in out_paths]
        param_dicts = _load_json_as_list(ddict)
        for params in param_dicts:
            cochlea = params["cochlea"]
            print(f"\n{cochlea}")
            seg_channel = params["segmentation_channel"]
            image_channels = params["image_channel"]
            table_path = os.path.join(cochlea, "tables", seg_channel, "default.tsv")
            if len(out_paths) == 1 and os.path.isdir(out_paths[0]):

                c_str = cochlea.replace('_', '-')
                s_str = seg_channel.replace('_', '-')
                out_paths_tmp = []
                for img_channel in image_channels:
                    i_str = img_channel.replace('_', '-')
                    out_paths_tmp.append(os.path.join(out_paths[0], f"{c_str}_{i_str}_{s_str}_object-measures.tsv"))

            else:
                assert len(image_channels) == len(out_paths)
                out_paths_tmp = out_paths.copy()

            if "use_bg_mask" in list(params.keys()):
                if params["use_bg_mask"] in ["yes", "Yes"]:
                    use_bg_mask = True

            if s3:
                image_paths = [os.path.join(cochlea, "images", "ome-zarr", f"{ch}.ome.zarr")
                               for ch in image_channels]
                seg_path = os.path.join(cochlea, "images", "ome-zarr", f"{seg_channel}.ome.zarr")
                seg_table = os.path.join(cochlea, "tables", f"{seg_channel}", "default.tsv")
            else:
                image_paths = [os.path.join(MOBIE_FOLDER, cochlea, "images", "ome-zarr", f"{ch}.ome.zarr")
                               for ch in image_channels]
                seg_path = os.path.join(MOBIE_FOLDER, cochlea, "images", "ome-zarr",
                                        f"{seg_channel}.ome.zarr")
                seg_table = os.path.join(MOBIE_FOLDER, cochlea, "tables", seg_channel, "default.tsv")

            object_measures_single(
                table_path=seg_table,
                seg_path=seg_path,
                image_paths=image_paths,
                out_paths=out_paths_tmp,
                force_overwrite=force_overwrite,
                use_bg_mask=use_bg_mask,
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
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for object measures
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1], help="List of components.")
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
        component_list=args.components,
        resolution=args.resolution,
        use_bg_mask=args.bg_mask,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


if __name__ == "__main__":

    main()
