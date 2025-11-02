import os
from typing import Tuple

import imageio.v3 as imageio
import pooch
import requests
from skimage.data import binary_blobs, cells3d
from skimage.measure import label

from .file_utils import get_cache_dir
from .segmentation.postprocessing import compute_table_on_the_fly

SEGMENTATION_URL = "https://owncloud.gwdg.de/index.php/s/kwoGRYiJRRrswgw/download"


def get_test_volume_and_segmentation(folder: str) -> Tuple[str, str, str]:
    """Download a small volume with nuclei and corresponding segmentation.

    Args:
        folder: The test data folder. The data will be downloaded to this folder.

    Returns:
        The path to the image, stored as tif.
        The path to the segmentation, stored as tif.
        The path to the segmentation table, stored as tsv.
    """
    os.makedirs(folder, exist_ok=True)

    segmentation_path = os.path.join(folder, "segmentation.tif")
    resp = requests.get(SEGMENTATION_URL)
    resp.raise_for_status()

    with open(segmentation_path, "wb") as f:
        f.write(resp.content)

    nuclei = cells3d()[20:40, 1]
    segmentation = imageio.imread(segmentation_path)
    assert nuclei.shape == segmentation.shape

    image_path = os.path.join(folder, "image.tif")
    imageio.imwrite(image_path, nuclei)

    table_path = os.path.join(folder, "default.tsv")
    table = compute_table_on_the_fly(segmentation, resolution=0.38)
    table.to_csv(table_path, sep="\t", index=False)

    return image_path, segmentation_path, table_path


def create_image_data_and_segmentation(folder: str, size: int = 256) -> Tuple[str, str, str]:
    """Create test data containing an image, a corresponding segmentation and segmentation table.

    Args:
        folder: The test data folder. The data will be written to this folder.

    Returns:
        The path to the image, stored as tif.
        The path to the segmentation, stored as tif.
        The path to the segmentation table, stored as tsv.
    """
    os.makedirs(folder, exist_ok=True)
    data = binary_blobs(size, n_dim=3).astype("uint8") * 255
    seg = label(data)

    image_path = os.path.join(folder, "image.tif")
    segmentation_path = os.path.join(folder, "segmentation.tif")
    imageio.imwrite(image_path, data)
    imageio.imwrite(segmentation_path, seg)

    table_path = os.path.join(folder, "default.tsv")
    table = compute_table_on_the_fly(seg, resolution=0.38)
    table.to_csv(table_path, sep="\t", index=False)

    return image_path, segmentation_path, table_path


# TODO add metadata
def create_test_data(root: str, size: int = 256, n_channels: int = 2, n_tiles: int = 4) -> None:
    """Create test data in the flamingo data format.

    Args:
        root: Directory for saving the data.
        size: The axis length for the data.
        n_channels The number of channels to create:
        n_tiles: The number of tiles to create.
    """
    channel_folders = [f"channel{chan_id}" for chan_id in range(n_channels)]
    file_name_pattern = "volume_R%i_C%i_I0.tif"
    for chan_id, channel_folder in enumerate(channel_folders):
        out_folder = os.path.join(root, channel_folder)
        os.makedirs(out_folder, exist_ok=True)
        for tile_id in range(n_tiles):
            out_path = os.path.join(out_folder, file_name_pattern % (tile_id, chan_id))
            data = binary_blobs(size, n_dim=3).astype("uint8") * 255
            imageio.imwrite(out_path, data)


def _sample_registry():
    urls = {
        "PV": "https://owncloud.gwdg.de/index.php/s/JVZCOpkILT70sdv/download",
        "VGlut3": "https://owncloud.gwdg.de/index.php/s/LvGXh0xQR9IKvNk/download",
        "CTBP2": "https://owncloud.gwdg.de/index.php/s/qaffCaF1sGpqlT3/download",
    }
    registry = {
        "PV": "fbf50cc9119f2dd2bd4dac7d76b746b7d42cab33b94b21f8df304478dd51e632",
        "VGlut3": "6a3af6ffce3d06588ffdc73df356ac64b83b53aaf6aabeabd49ef6d11d927e20",
        "CTBP2": "8dcd5f1ebb35194f328788594e275f2452de0e28c85073578dac7100d83c45fc",
    }
    cache_dir = get_cache_dir()
    data_registry = pooch.create(
        path=os.path.join(cache_dir, "data"),
        base_url="",
        registry=registry,
        urls=urls,
    )
    return data_registry


def sample_data_pv():
    data_path = _sample_registry().fetch("PV")
    data = imageio.imread(data_path, extension=".tif")
    add_image_kwargs = {"name": "PV", "colormap": "gray"}
    return [(data, add_image_kwargs)]


def sample_data_vglut3():
    data_path = _sample_registry().fetch("VGlut3")
    data = imageio.imread(data_path, extension=".tif")
    add_image_kwargs = {"name": "VGlut3", "colormap": "gray"}
    return [(data, add_image_kwargs)]


def sample_data_ctbp2():
    data_path = _sample_registry().fetch("CTBP2")
    data = imageio.imread(data_path, extension=".tif")
    add_image_kwargs = {"name": "CTBP2", "colormap": "gray"}
    return [(data, add_image_kwargs)]
