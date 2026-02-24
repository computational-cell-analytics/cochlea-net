import os

import imageio.v3 as imageio
from mobie import add_image

INPUT_ROOT = "/mnt/ceph-hdd/cold/nim00007/cochlea-lightsheet/keppeler-et-al/marmoset"
MOBIE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
DS_NAME = "LaVision-Mar05"
VOXEL_SIZE = (3.0, 1.887779, 1.887779)


# Marmoset_cochlea_05_LSFM_ch1_raw.tif
def add_marmoset_05():
    channel_names = ("PV", "7-AAD", "MYO")

    scale_factors = 4 * [[2, 2, 2]]
    chunks = (96, 96, 96)

    for channel_id, channel_name in enumerate(channel_names, 1):
        input_path = os.path.join(INPUT_ROOT, f"Marmoset_cochlea_05_LSFM_ch{channel_id}_raw.tif")
        print("Load image data ...")
        input_data = imageio.imread(input_path)
        print(input_data.shape)
        add_image(
            input_path=input_data, input_key=None, root=MOBIE_ROOT,
            dataset_name=DS_NAME, image_name=channel_name, resolution=VOXEL_SIZE,
            scale_factors=scale_factors, chunks=chunks, unit="micrometer", use_memmap=False,
        )


def main():
    add_marmoset_05()


if __name__ == "__main__":
    main()
