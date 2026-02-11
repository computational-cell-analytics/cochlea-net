"""@private
"""

import json
import os
from typing import Optional, List, Union, Tuple

import imageio.v3 as imageio
import numpy as np
import zarr
from skimage.transform import rescale

from flamingo_tools.file_utils import read_image_data
from flamingo_tools.s3_utils import get_s3_path, MOBIE_FOLDER
from flamingo_tools.postprocessing.cochlea_mapping import equidistant_centers_single


def extract_block_single(
    input_path: str,
    coords: List[int],
    output_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    channel_name: Optional[str] = None,
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    resolution: Union[float, Tuple[float, float, float]] = 0.38,
    roi_halo: List[int] = [128, 128, 64],
    s3: Optional[bool] = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    scale_factor: Optional[Tuple[float, float, float]] = None,
    force_overwrite: bool = False,
    **_,
) -> None:
    """Extract block around coordinate from input data according to a given halo.
    Either from a local file or from an S3 bucket.

    Args:
        input_path: Input folder in n5 / ome-zarr format.
        coords: Center coordinates of extracted 3D volume.
        output_dir: Output directory for saving output as <basename>_crop.n5. Default: input directory.
        output_path: Output directory or file for saving output as <basename>_crop.n5. Default: input directory.
        input_key: Input key for data in input file.
        output_key: Output key for data in n5 format. If None is supplied, output is TIF file.
        roi_halo: ROI halo of extracted 3D volume.
        s3: Flag for accessing data stored on S3 bucket.
        s3_credentials: File path to credentials for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        scale_factor: Optional factor for rescaling the extracted data.
        force_overwrite: Flag for forcefully overwriting output files.
    """
    coord_string = "-".join([str(int(round(c))).zfill(4) for c in coords])

    # Dimensions are inversed to view in MoBIE (x y z) -> (z y x)
    # Make sure the coords / roi_halo are not modified in-place.
    coords = coords.copy()
    roi_halo = roi_halo.copy()
    coords.reverse()
    roi_halo.reverse()

    # get components of output file
    prefix = ""
    suffix = ""
    if dataset_name is not None:
        dataset_str = "-".join(dataset_name.split("_"))
        prefix = f"{dataset_str}_"
    if channel_name is not None:
        channel_str = "-".join(channel_name.split("_"))
        suffix = f"_{channel_str}"

    if os.path.isdir(output_path):
        if output_key is None:
            output_name = f"{prefix}crop_{coord_string}{suffix}.tif"
        else:
            output_name = f"{prefix}crop_{coord_string}{suffix}.n5"

        output_path = os.path.join(output_path, output_name)

    if os.path.isfile(output_path) and not force_overwrite:
        print(f"Skipping block extration because {output_path} already exists.")

    coords = np.array(coords).astype("float")
    if not isinstance(resolution, float):
        assert len(resolution) == 3
        resolution = np.array(resolution)[::-1]
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))

    if s3:
        input_path, fs = get_s3_path(
            input_path, bucket_name=s3_bucket_name,
            service_endpoint=s3_service_endpoint, credential_file=s3_credentials
        )

    data_ = read_image_data(input_path, input_key)
    data_roi = data_[roi]
    if scale_factor is not None:
        kwargs = {"preserve_range": True}
        # Check if this is a segmentation.
        if data_roi.dtype in (np.dtype("int32"), np.dtype("uint32"), np.dtype("int64"), np.dtype("uint64")):
            kwargs.update({"order": 0, "anti_aliasing": False})
        data_roi = rescale(data_roi, scale_factor, **kwargs).astype(data_roi.dtype)

    if output_key is None:
        imageio.imwrite(output_path, data_roi, compression="zlib")
    else:
        f_out = zarr.open(output_path, mode="w")
        f_out.create_dataset(output_key, data=data_roi, compression="gzip")


def extract_block_json_wrapper(
    output_path: str,
    input_path: Optional[str] = None,
    json_file: Optional[str] = None,
    coords: List[int] = [],
    mobie_dir: str = MOBIE_FOLDER,
    force: bool = False,
    s3: Optional[bool] = False,
    **kwargs,
):
    """Wrapper function for extracting blocks based on a dictionary in a JSON file.

    Args:
        output_path: Output path for storing extracted blocks.
        input_path: Input path for image channel.
        json_file: JSON file containing parameter dictionary.
        coords: List of center coordinates for extracting blocks.
        mobie_dir: Local MoBIE directory used for creating data paths when a JSON dict is provided.
        force: Flag for forcefully overwriting output files.
        s3: Flag for accessing data stored on S3 bucket.
    """
    if json_file is not None:
        input_key = "s0"
        with open(json_file, "r") as f:
            params = json.loads(f.read())
        cochlea = params["dataset_name"]
        if isinstance(params["image_channel"], list):
            image_channels = params["image_channel"]
        else:
            image_channels = [params["image_channel"]]

        for image_channel in image_channels:
            if s3:
                input_path = os.path.join(cochlea, "images", "ome-zarr", f"{image_channel}.ome.zarr")
            else:
                input_path = os.path.join(mobie_dir, cochlea, "images", "ome-zarr", f"{image_channel}.ome.zarr")

            for coords in params["crop_centers"]:
                kwargs.update(params)
                extract_block_single(
                    input_path=input_path,
                    input_key=input_key,
                    coords=coords,
                    output_path=output_path,
                    channel_name=image_channel,
                    force_overwrite=force,
                    s3=s3,
                    **kwargs,
                )

    else:
        if input_path is None:
            raise ValueError("An input path to image data is required, if no JSON file is supplied.")
        extract_block_single(
            input_path=input_path,
            output_path=output_path,
            force_overwrite=force,
            **kwargs,
        )


def extract_central_block_from_json(
    json_file: str,
    output_path: str,
    force_overwrite: bool = False,
    s3: bool = False,
    mobie_dir: str = MOBIE_FOLDER,
    **kwargs,
):
    """Extract central blocks based on parameters in a JSON file.
    This function combines the search of equidistant center coordinates and the subsequent block extraction.

    Args:
        json_file: JSON with parameter dictionary. Can be created using 'flamingo_tools.json_block_extraction'.
        output_path: Output directory for storing extracted blocks.
        force_overwrite: Force overwrite of extracted blocks.
        s3: Flag for accessing data on the S3 bucket.
        mobie_dir: Local MoBIE directory used for creating data paths.
    """
    with open(json_file, "r") as f:
        dic = json.loads(f.read())

    if s3:
        table_path = os.path.join(dic["dataset_name"], "tables", dic["segmentation_channel"], "default.tsv")
    else:
        table_path = os.path.join(mobie_dir, dic["dataset_name"], "tables", dic["segmentation_channel"], "default.tsv")

    equidistant_centers_single(
        table_path=table_path,
        output_path=json_file,
        n_blocks=dic["n_blocks"],
        cell_type=dic["cell_type"],
        component_list=dic["component_list"],
        s3=s3,
        **kwargs,
    )

    os.makedirs(output_path, exist_ok=True)

    extract_block_json_wrapper(
        output_path=output_path,
        json_file=json_file,
        s3=s3,
        mobie_dir=mobie_dir,
        force=force_overwrite,
        **kwargs,
    )
