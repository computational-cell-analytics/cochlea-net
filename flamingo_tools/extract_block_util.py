"""@private
"""

import os
from typing import Optional, List, Union, Tuple

import imageio.v3 as imageio
import numpy as np
import zarr
from skimage.transform import rescale

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.file_utils import read_image_data


def extract_block(
    input_path: str,
    coords: List[int],
    output_dir: Optional[str] = None,
    input_key: Optional[str] = None,
    output_key: Optional[str] = None,
    resolution: Union[float, Tuple[float, float, float]] = 0.38,
    roi_halo: List[int] = [128, 128, 64],
    tif: bool = False,
    s3: Optional[bool] = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    scale_factor: Optional[Tuple[float, float, float]] = None,
) -> None:
    """Extract block around coordinate from input data according to a given halo.
    Either from a local file or from an S3 bucket.

    Args:
        input_path: Input folder in n5 / ome-zarr format.
        coords: Center coordinates of extracted 3D volume.
        output_dir: Output directory for saving output as <basename>_crop.n5. Default: input directory.
        input_key: Input key for data in input file.
        output_key: Output key for data in n5 output or used as suffix for tif output.
        roi_halo: ROI halo of extracted 3D volume.
        tif: Flag for tif output
        s3: Flag for considering input_path for S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
        scale_factor: Optional factor for rescaling the extracted data.
    """
    coord_string = "-".join([str(int(round(c))).zfill(4) for c in coords])

    # Dimensions are inversed to view in MoBIE (x y z) -> (z y x)
    # Make sure the coords / roi_halo are not modified in-place.
    coords = coords.copy()
    roi_halo = roi_halo.copy()
    coords.reverse()
    roi_halo.reverse()

    input_content = list(filter(None, input_path.split("/")))

    if s3:
        image_name = input_content[-1].split(".")[0]
        image_prefix = image_name
        basename = input_content[0]
    else:
        basename = "".join(input_content[-1].split(".")[:-1])
        image_prefix = basename.split("_")[-1]

    input_dir = input_path.split(basename)[0]
    input_dir = os.path.abspath(input_dir)

    if output_dir == "":
        output_dir = input_dir
    os.makedirs(output_dir, exist_ok=True)

    if tif:
        if output_key is None:
            output_name = basename + "_crop_" + coord_string + "_" + image_prefix + ".tif"
        else:
            output_name = basename + "_" + image_prefix + "_crop_" + coord_string + "_" + output_key + ".tif"

        output_file = os.path.join(output_dir, output_name)
    else:
        output_key = "raw" if output_key is None else output_key
        output_file = os.path.join(output_dir, basename + "_crop_" + coord_string + ".n5")

    coords = np.array(coords).astype("float")
    if not isinstance(resolution, float):
        assert len(resolution) == 3
        resolution = np.array(resolution)[::-1]
    coords = coords / resolution
    coords = np.round(coords).astype(np.int32)

    roi = tuple(slice(co - rh, co + rh) for co, rh in zip(coords, roi_halo))

    if s3:
        input_path, fs = s3_utils.get_s3_path(
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

    if tif:
        imageio.imwrite(output_file, data_roi, compression="zlib")
    else:
        f_out = zarr.open(output_file, mode="w")
        f_out.create_dataset(output_key, data=data_roi, compression="gzip")
