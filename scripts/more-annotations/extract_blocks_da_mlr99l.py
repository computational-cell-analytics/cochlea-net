from flamingo_tools.extract_block_util import extract_block_single


positions_with_signal = [
    [1215.074063453085, 912.697256780485, 1036.814204517708],
    [1030.351117830933, 1262.3358840155736, 1123.2581736686361],
    [1192.167776682008, 354.058713359485, 767.1544606203263],
    [916.9294364078347, 754.7061965177552, 923.607923806173],
]
positions_without_signal = [
    [1383.4288658807268, 783.0008672288084, 467.5426478786816],
]

halo = [256, 256, 64]


for pos in positions_with_signal + positions_without_signal:
    extract_block_single(
        input_path="M_LR_000099_L/images/ome-zarr/PV.ome.zarr",
        coords=pos,
        output_dir="./MLR99L_for_DA",
        input_key="s0",
        roi_halo=halo,
        tif=True,
        s3=True
    )
