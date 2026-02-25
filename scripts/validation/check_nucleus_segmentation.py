import os

import imageio.v3 as imageio
import numpy as np

ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/croppings/226R_SGN_crop"
IMAGE_PATH = os.path.join(ROOT, "M_LR_000226_R_crop_0802-1067-0776_PV.tif")
SEG_PATH = os.path.join(ROOT, "M_LR_000226_R_crop_0802-1067-0776_SGN_v2.tif")
NUC_PATH = os.path.join(ROOT, "M_LR_000226_R_crop_0802-1067-0776_NUCLEI.tif")


def segment_nuclei():
    from flamingo_tools.segmentation.nucleus_segmentation import _naive_nucleus_segmentation_impl

    image = imageio.imread(IMAGE_PATH)
    segmentation = imageio.imread(SEG_PATH)

    nuclei = np.zeros_like(segmentation, dtype=segmentation.dtype)
    _naive_nucleus_segmentation_impl(image, segmentation, table=None, output=nuclei, n_threads=8, voxel_size=0.38)

    imageio.imwrite(NUC_PATH, nuclei, compression="zlib")


def check_segmentation():
    import napari

    image = imageio.imread(IMAGE_PATH)
    segmentation = imageio.imread(SEG_PATH)
    nuclei = imageio.imread(NUC_PATH)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(segmentation)
    v.add_labels(nuclei)
    napari.run()


def main():
    segment_nuclei()
    check_segmentation()


if __name__ == "__main__":
    main()
