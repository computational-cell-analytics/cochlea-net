from concurrent import futures
from multiprocessing import cpu_count
from typing import Optional

import numpy as np
import pandas as pd
from elf.io import open_file
from scipy.ndimage import binary_opening
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label
from tqdm import tqdm

from ..file_utils import read_image_data
from ..measurements import _get_bounding_box_and_center
from .postprocessing import compute_table_on_the_fly


def _naive_nucleus_segmentation_impl(image, segmentation, table, output, n_threads, resolution):
    opening_iterations = 3

    # Compute the table on the fly if it wasn't given.
    if table is None:
        table = compute_table_on_the_fly(segmentation, resolution=resolution)

    def segment_nucleus(seg_id):
        bb = _get_bounding_box_and_center(table, seg_id, resolution, image.shape, dilation=None)
        image_local, seg_local = image[bb], segmentation[bb]
        mask = seg_local == seg_id

        # Smooth before computing the threshold.
        image_local = gaussian(image_local)
        # Compute threshold only in the mask.
        threshold = threshold_otsu(image_local[mask])

        nucleus_mask = np.logical_and(image_local < threshold, mask)
        nucleus_mask = label(nucleus_mask)
        ids, sizes = np.unique(nucleus_mask, return_counts=True)
        ids, sizes = ids[1:], sizes[1:]
        nucleus_mask = (nucleus_mask == ids[np.argmax(sizes)])
        nucleus_mask = binary_opening(nucleus_mask, iterations=opening_iterations)
        output[bb][nucleus_mask] = seg_id

    n_threads = cpu_count() if n_threads is None else n_threads
    seg_ids = table.label_id.values
    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tqdm(tp.map(segment_nucleus, seg_ids), total=len(seg_ids), desc="Segment nuclei"))

    return output


def naive_nucleus_segmentation(
    image_path: str,
    segmentation_path: str,
    segmentation_table_path: Optional[str],
    output_path: str,
    output_key: str,
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    n_threads: Optional[int] = None,
    resolution: float = 0.38,
):
    """Segment nuclei per object with an otsu threshold.

    This assumes that the nucleus is stained significantly less.

    Args:
        image_path: The filepath to the image data. Either a tif or hdf5/zarr/n5 file.
        segmentation_path: The filepath to the segmentation data. Either a tif or hdf5/zarr/n5 file.
        segmentation_table_path: The path to the segmentation table in MoBIE format.
        output_path: The path for saving the nucleus segmentation.
        output_key: The key for saving the nucleus segmentation.
        image_key: The key (= internal path) for the image data. Not needed fir tif.
        segmentation_key: The key (= internal path) for the segmentation data. Not needed for tif.
        n_threads: The number of threads to use for computation.
        resolution: The resolution / voxel size of the data.
    """
    # First, we load the pre-computed segmentation table from MoBIE.
    if segmentation_table_path is None:
        table = None
    else:
        table = pd.read_csv(segmentation_table_path, sep="\t")

    # Then, open the volumes.
    image = read_image_data(image_path, image_key)
    segmentation = read_image_data(segmentation_path, segmentation_key)

    # Create the output volume.
    with open_file(output_path, mode="a") as f:
        output = f.create_dataset(
            output_key, shape=segmentation.shape, dtype=segmentation.dtype, compression="gzip",
            chunks=segmentation.chunks
        )

        # And run the nucleus segmentation.
        _naive_nucleus_segmentation_impl(image, segmentation, table, output, n_threads, resolution)
