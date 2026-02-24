import os

import imageio.v3 as imageio
from mobie import add_image
from mobie.metadata import read_dataset_metadata

INPUT_ROOT = "/mnt/ceph-hdd/cold/nim00007/cochlea-lightsheet/keppeler-et-al/mouse"
MOBIE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
DS_NAME = "LaVision-M04"
VOXEL_SIZE = (3.0, 1.887779, 1.887779)


# Mouse_cochlea_04_LSFM_ch1_raw.tif  Mouse_cochlea_04_LSFM_ch2_raw.tif  Mouse_cochlea_04_LSFM_ch3_raw.tif
def add_mouse_lsm():
    channel_names = ("PV", "7-AAD", "MYO")

    scale_factors = 4 * [[2, 2, 2]]
    chunks = (96, 96, 96)

    for channel_id, channel_name in enumerate(channel_names, 1):
        mobie_ds_folder = os.path.join(MOBIE_ROOT, DS_NAME)
        ds_metadata = read_dataset_metadata(mobie_ds_folder)
        if channel_name in ds_metadata["sources"]:
            print(channel_name, "is already in MoBIE")
            continue

        input_path = os.path.join(INPUT_ROOT, f"Mouse_cochlea_04_LSFM_ch{channel_id}_raw.tif")
        print("Load image data ...")
        input_data = imageio.imread(input_path)
        print(input_data.shape)
        add_image(
            input_path=input_data, input_key=None, root=MOBIE_ROOT,
            dataset_name=DS_NAME, image_name=channel_name, resolution=VOXEL_SIZE,
            scale_factors=scale_factors, chunks=chunks, unit="micrometer", use_memmap=False,
        )


def main():
    add_mouse_lsm()


if __name__ == "__main__":
    main()
