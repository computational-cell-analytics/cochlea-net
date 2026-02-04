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

    if s3:
        # MoBIE format <cochlea>/images/ome-zarr/<stain>.ome.zarr
        input_content = list(filter(None, os.path.normpath(input_path).split(os.path.sep)))
        channel_name = input_content[-1].split(".")[0] if channel_name is None else channel_name
        dataset_name = input_content[0] if dataset_name is None else dataset_name

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

    if output_key is None:
        imageio.imwrite(output_path, data_roi, compression="zlib")
    else:
        f_out = zarr.open(output_path, mode="w")
        f_out.create_dataset(output_key, data=data_roi, compression="gzip")
