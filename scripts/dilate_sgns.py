import os

import numpy as np
import tifffile

from elf.parallel import seeded_watershed, distance_transform

LOWER_RESOLUTION_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/lower_resolution"


def dilate_sgns(in_path, out_path, dilation_distance, block_shape=[64, 128, 128], halo=[8, 16, 16]):
    print(f"Dilating {in_path} with dilation distance {dilation_distance}.")
    seg = tifffile.imread(in_path).astype("uint32")

    distances = distance_transform(seg == 0, halo=halo, sampling=(1, 1, 1), block_shape=block_shape, verbose=True)
    extension_mask = distances < dilation_distance

    extended_seg = np.zeros_like(seg)
    extended_seg = seeded_watershed(
        distances, seg, out=extended_seg, mask=extension_mask, block_shape=block_shape, halo=halo, verbose=True
    )

    tifffile.imwrite(out_path, extended_seg.astype("float32"), bigtiff=True, compression="zlib")


def main():
    cochlea = "G_EK_000049_L"
    scale = 3
    suffixes = ["_marker_positive", "_marker_negative", ""]
    dilation_distances = [2, 4]

    for suffix in suffixes:
        for dilation_distance in dilation_distances:
            in_path = os.path.join(f"{LOWER_RESOLUTION_DIR}", cochlea, f"scale{scale}", f"SGN_v2{suffix}.tif")
            out_path = os.path.join(f"{LOWER_RESOLUTION_DIR}", cochlea, f"scale{scale}",
                                    f"SGN_v2{suffix}_{scale}_dilated_{dilation_distance}.tif")
            dilate_sgns(in_path, out_path, dilation_distance)


if __name__ == "__main__":
    main()
