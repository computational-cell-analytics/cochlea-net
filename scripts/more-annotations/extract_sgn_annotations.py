import os
from flamingo_tools.extract_block_util import extract_block_single

RESOLUTION_LA_VISION = (1.887779, 1.887779, 3.000000)
RESOLUTION_FLAMINGO = (0.38, 0.38, 0.38)

POSITIONS = [
    [2451.991261845771, 2497.0312568466725, 504.00000000000017],
    [2364.0285060661868, 2104.541310616445, 684.2966391951725],
    [2579.872281689804, 2266.294057961108, 532.8474622712272],
    [2251.404321115024, 1972.6189003459485, 313.69577047550024],
]

EMPTY_POSITIONS = [
    [3091.354274253545, 1396.2702622881343, 443.21051449917223],
    [1052.8399693103918, 2180.579279395121, 437.81154147679354],
    [3621.8731222257875, 1602.0695390382377, 620.0517925327181],
]


def download_lavision_crops():
    input_path = "LaVision-M04/images/ome-zarr/PV.ome.zarr"
    input_key = "s0"
    output_key = None

    output_folder = "./LA_VISION_M04"
    os.makedirs(output_folder, exist_ok=True)
    for pos in POSITIONS:
        halo = [128, 128, 32]
        extract_block_single(
            input_path, pos, output_folder, input_key, output_key, RESOLUTION_LA_VISION, halo,
            tif=True, s3=True,
        )

    output_folder = "./LA_VISION_M04_empty"
    os.makedirs(output_folder, exist_ok=True)
    for pos in EMPTY_POSITIONS:
        halo = [128, 128, 32]
        extract_block_single(
            input_path, pos, output_folder, input_key, output_key, RESOLUTION_LA_VISION, halo,
            tif=True, s3=True,
        )


def downscale_segmentation():
    # Scale levels:
    # 0: 0.38
    # 1: 0.76
    # 2: 1.52
    cochleae_and_positions = {
        "M_LR_000226_R": [
            [709.1792864323405, 277.94313087502087, 790.2787473759703],
            [684.1211492422168, 551.0610808519966, 972.7784147188805],
            [855.2911547522649, 893.5164525605765, 781.6745184537485],
            [805.856486020322, 1087.1388983637446, 872.4092720709023],
        ],
        "M_LR_000226_L": [
            [728.811391169819, 787.126384246222, 765.5121274338735],
            [310.8110214721421, 503.69151338122936, 433.37560298279783],
            [409.56553632355974, 773.9536143837831, 926.4997632186463],
        ],
        "M_LR_000227_R": [
            [802.695142936733, 928.040906650113, 787.9300000000001],
            [539.6960827733835, 837.7146969656125, 787.9300000000001],
            [460.70492230292973, 366.6096043369565, 909.2776827283466],
        ],
        "M_LR_000227_L": [
            [583.3657358293676, 835.4967569151809, 754.4900000000004],
            [846.4954841793519, 963.2748384826734, 706.7658788868116],
            [927.8483264319711, 746.0723412831164, 578.0355803590329],
        ],
    }

    input_key = "s2"
    halo = [128, 128, 128]
    resolution = [1.52,] * 3
    output_key = None

    image_out_folder = "./downscaled_sgn_labels/images"
    label_out_folder = "./downscaled_sgn_labels/labels"

    for cochlea, positions in cochleae_and_positions.items():
        print("Extracting blocks for", cochlea)
        input_path = f"{cochlea}/images/ome-zarr/PV.ome.zarr"
        seg_path = f"{cochlea}/images/ome-zarr/SGN_v2.ome.zarr"
        for position in positions:
            extract_block_single(
                input_path, position, image_out_folder, input_key, output_key, resolution, halo,
                tif=True, s3=True, scale_factor=(0.5, 1, 1),
            )
            extract_block_single(
                seg_path, position, label_out_folder, input_key, output_key, resolution, halo,
                tif=True, s3=True, scale_factor=(0.5, 1, 1),
            )


# Note: consider different normalization strategy for these cochleae and normalize by local intensity
# rather than by global values.
# Also double check empty positions again and make sure they don't contain SGNs
def download_lavision_crops2():
    input_key = "s0"
    output_key = None

    # Additional positions for LaVision annotations:
    input_path = "LaVision-M04/images/ome-zarr/PV.ome.zarr"
    new_positions_m04 = [
        [2031.0655170248258, 1925.206039671767, 249.14546086048554],
        [2378.3720460599393, 2105.471228531872, 303.9285928812524],
        [1619.3251178227529, 3444.7351705689553, 271.2360278843609],
        [2358.2784398426843, 1503.2211953830192, 762.7325586759833],
    ]

    output_folder = "./LA_VISION_M04_2"
    os.makedirs(output_folder, exist_ok=True)
    for pos in new_positions_m04:
        halo = [128, 128, 32]
        extract_block_single(
            input_path, pos, output_folder, input_key, output_key, RESOLUTION_LA_VISION, halo,
            tif=True, s3=True,
        )

    # Position in Marmoset:
    new_positions_mar05 = [
        [2462.7875134103206, 2818.067344942212, 1177.1380214828991]
    ]
    input_path = "LaVision-Mar05/images/ome-zarr/PV.ome.zarr"
    output_folder = "./LA_VISION_Mar05"
    os.makedirs(output_folder, exist_ok=True)
    for pos in new_positions_mar05:
        halo = [128, 128, 32]
        extract_block_single(
            input_path, pos, output_folder, input_key, output_key, RESOLUTION_LA_VISION, halo,
            tif=True, s3=True,
        )


def main():
    # download_lavision_crops()
    # downscale_segmentation()
    download_lavision_crops2()


if __name__ == "__main__":
    main()
