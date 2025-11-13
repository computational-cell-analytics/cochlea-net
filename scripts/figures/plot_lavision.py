import os
from typing import List
from concurrent import futures

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd
import zarr

from elf.parallel import isin
from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
from magicgui import magicgui
from nifty.tools import blocking
from scipy.ndimage import binary_dilation
from tqdm import tqdm

IHC_SETTINGS = {
    "LaVision-M02": {
        "components": [1, 3, 6, 8, 22, 24],
        "min_size": 500
    },
    "LaVision-M03": {
        "components": [1, 2],
        "min_size": 500
    },
    "LaVision-M04": {
        "components": [1],
        "min_size": 500
    },
    "LaVision-Mar05": {
        "components": [1, 2, 3, 5, 7, 9, 11, 13],  # 100
        "min_size": 250
    },
    "LaVision-OTOF23R": {
        "components": [4, 7, 18],
        "min_size": 250
    },
    "LaVision-OTOF25R": {
        "components": [1, 23, 40, 51, 86, 112],
        "min_size": 250
    },
    # "LaVision-OTOF36R": {
    #     "components": [1],
    #     "min_size": 250
    # },
}


def _mask_in_parallel(seg, mask):
    blocks = blocking([0, 0, 0], mask.shape, [32, 128, 128])

    def apply_mask(block_id):
        block = blocks.getBlock(block_id)
        bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))
        mask_bb = mask[bb]
        if mask_bb.sum() == 0:
            return
        seg_bb = seg[bb]
        if seg_bb.sum() == 0:
            return
        seg_bb[mask_bb] = 0
        seg[bb] = seg_bb

    n_blocks = blocks.numberOfBlocks
    with futures.ThreadPoolExecutor(8) as tp:
        list(tqdm(tp.map(apply_mask, range(n_blocks)), total=n_blocks))

    return seg


def download_seg(ds, seg_channel, output_folder, apply_filter=True, scale=1):
    if ds == "LaVision-Mar05" and "SGN" in seg_channel:
        input_key = f"s{scale + 1}"
    else:
        input_key = f"s{scale}"

    if ds == "LaVision-Mar05" and scale == 1:
        output_path = os.path.join(output_folder, f"{seg_channel}_s1.tif")
    else:
        output_path = os.path.join(output_folder, f"{seg_channel}.tif")

    if os.path.exists(output_path):
        return

    internal_path = os.path.join(ds, "images",  "ome-zarr", f"{seg_channel}.ome.zarr")
    s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)

    table_path = os.path.join(BUCKET_NAME, ds, "tables",  seg_channel, "default.tsv")
    with fs.open(table_path, "r") as f:
        table = pd.read_csv(f, sep="\t")

    with zarr.open(s3_store, mode="r") as f:
        data = f[input_key][:]

    if apply_filter:
        valid_ids = table[table.component_labels == 1].label_id
        print(seg_channel, ":", len(valid_ids))
        mask = np.zeros_like(data, dtype="bool")
        mask = ~isin(data, valid_ids, out=mask, verbose=True, block_shape=(32, 256, 256))
        data = _mask_in_parallel(data, mask)
        v = napari.Viewer()
        v.add_labels(data)
        napari.run()
    else:
        table_out = os.path.join(output_folder, f"{seg_channel}.tsv")
        table.to_csv(table_out, sep="\t", index=False)

    imageio.imwrite(output_path, data, compression="zlib")


def download_dataset(
    ds,
    channels=["PV", "MYO"],
    seg_channels=["SGN_LOWRES-v5c", "IHC_LOWRES-v3"],
    apply_filters=[True, False],
    scale=1,
):
    print(ds)

    output_folder = os.path.join("data", ds)
    os.makedirs(output_folder, exist_ok=True)
    input_key = f"s{scale}"

    for channel in channels:
        if ds == "LaVision-Mar05" and scale == 1:
            output_path = os.path.join(output_folder, f"{channel}_s1.tif")
        else:
            output_path = os.path.join(output_folder, f"{channel}.tif")
        if os.path.exists(output_path):
            continue

        internal_path = os.path.join(ds, "images",  "ome-zarr", f"{channel}.ome.zarr")
        s3_store, fs = get_s3_path(internal_path, bucket_name=BUCKET_NAME, service_endpoint=SERVICE_ENDPOINT)
        with zarr.open(s3_store, mode="r") as f:
            data = f[input_key][:]
        imageio.imwrite(output_path, data, compression="zlib")

    for seg_channel, apply_filter in zip(seg_channels, apply_filters):
        print(seg_channel, scale)
        download_seg(ds, seg_channel, output_folder, apply_filter=apply_filter, scale=scale)


def download_data():
    # datasets = ["LaVision-M02", "LaVision-M03"]
    # for ds in datasets:
    #     download_dataset(ds)

    # datasets = ["LaVision-Mar05"]
    # for ds in datasets:
    #     for scale in (1, 2):
    #         download_dataset(
    #             ds, scale=scale,
    #             seg_channels=["SGN_LOWRES-v5", "IHC_LOWRES-v3", "IHC_LOWRES-v4"],
    #             apply_filters=[True, False, False],
    #         )

    channels = ["CR", "rbOtof"]
    datasets = ["LaVision-OTOF23R", "LaVision-OTOF25R"]
    for ds in datasets:
        download_dataset(ds, channels=channels, seg_channels=["IHC_LOWRES-v3"], apply_filters=[True], scale=1)


def _filter_ihcs(ihcs, table, components, min_size):
    keep_ids = table[
        table.component_labels.isin(components) & (table.n_pixels > min_size)
    ].label_id
    print("Ihc components:", sorted(components))
    print("Min-size:", min_size)
    print("Number of IHCs:", len(keep_ids))
    ihc_filtered = ihcs.copy()
    ihc_filtered[~np.isin(ihcs, keep_ids)] = 0
    return ihc_filtered


def check_ihcs_ds(ds, channel="MYO", seg_name="IHC_LOWRES-v3"):
    from nifty.tools import takeDict

    myo = imageio.imread(f"data/{ds}/{channel}.tif")
    ihcs = imageio.imread(f"data/{ds}/{seg_name}.tif")

    table = pd.read_csv(f"data/{ds}/{seg_name}.tsv", sep="\t")
    relabel_dict = {0: 0}
    relabel_dict.update({int(k): int(v) for k, v in zip(table.label_id.values, table.component_labels.values)})
    ihc_components = takeDict(relabel_dict, ihcs)

    @magicgui()
    def filter_component(v: napari.Viewer, component_list: List[int] = [1], min_size=500):
        ihc_filtered = _filter_ihcs(ihcs, table, component_list, min_size)
        v.layers["ihc_filtered"].data = ihc_filtered

    if ds in IHC_SETTINGS:
        settings = IHC_SETTINGS[ds]
        components, min_size = settings["components"], settings["min_size"]
        ihc_filtered = _filter_ihcs(ihcs, table, components, min_size)
    else:
        ihc_filtered = np.zeros_like(ihcs)

    v = napari.Viewer()
    v.add_image(myo)
    v.add_labels(ihcs, name="ihcs")
    v.add_labels(ihc_components)
    v.add_labels(ihc_filtered)
    v.title = ds

    v.window.add_dock_widget(filter_component)

    napari.run()


def check_ihcs():
    # datasets = ["LaVision-M02", "LaVision-M03", "LaVision-M04"]
    # datasets = ["LaVision-Mar05"]
    # for ds in datasets:
    #     check_ihcs_ds(ds, channel="MYO", seg_name="IHC_LOWRES-v3")

    # datasets = ["LaVision-OTOF23R", "LaVision-OTOF25R", "LaVision-OTOF36R"]
    datasets = ["LaVision-OTOF23R"]
    for ds in datasets:
        check_ihcs_ds(ds, channel="CR")


def create_rendering_m03(export_for_rendering):
    ds = "LaVision-M03"
    out_folder = f"data/{ds}/for_rendering"

    pv = imageio.imread(f"data/{ds}/PV.tif").astype("float32")
    myo = imageio.imread(f"data/{ds}/MYO.tif").astype("float32")
    sgn = imageio.imread(f"data/{ds}/SGN_LOWRES-v5c.tif")

    ihc_path = f"data/{ds}/IHC_LOWRES-v3-filtered.tif"
    if os.path.exists(ihc_path):
        ihc = imageio.imread(ihc_path)
    else:
        ihc = imageio.imread(f"data/{ds}/IHC_LOWRES-v3.tif")
        table = pd.read_csv(f"data/{ds}/IHC_LOWRES-v3.tsv", sep="\t")
        settings = IHC_SETTINGS[ds]
        components, min_size = settings["components"], settings["min_size"]
        ihc = _filter_ihcs(ihc, table, components, min_size)
        imageio.imwrite(ihc_path, ihc, compression="zlib")

    pv_mask = binary_dilation(sgn > 0, iterations=3)
    myo_mask = binary_dilation(ihc > 0, iterations=2)
    if export_for_rendering:
        os.makedirs(out_folder, exist_ok=True)
        imageio.imwrite(os.path.join(out_folder, "PV.tif"), pv, compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "MYO.tif"), myo, compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "SGN.tif"), sgn.astype("float32"), compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "IHC.tif"), ihc.astype("float32"), compression="zlib")

    factor = 10

    myo = myo / myo.max()
    myo[myo_mask] *= factor

    pv = pv / pv.max()
    pv[pv_mask] *= factor

    if export_for_rendering:
        imageio.imwrite(os.path.join(out_folder, "PV_filtered.tif"), pv, compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "MYO_filtered.tif"), myo, compression="zlib")
        return

    scale = (6, 3.9, 3.9)

    v = napari.Viewer()
    v.add_image(pv, colormap="red", scale=scale, blending="additive")
    v.add_image(myo, colormap="cyan", scale=scale, blending="additive")
    v.add_labels(sgn, scale=scale)
    v.add_labels(ihc, scale=scale)

    v.scale_bar.visible = True
    v.scale_bar.unit = "µm"     # unit to display

    napari.run()


def create_rendering_mar05(apply_mask=True):
    pv = imageio.imread("data/LaVision-Mar05/PV.tif")
    sgn = imageio.imread("data/LaVision-Mar05/SGN_LOWRES-v5.tif")

    myo = imageio.imread("data/LaVision-Mar05/MYO.tif")
    ihc = imageio.imread("data/LaVision-Mar05/IHC_LOWRES-combined.tif")
    # ihc = imageio.imread("data/LaVision-Mar05/IHC_LOWRES-combined2.tif")
    n_ihcs = len(np.unique(ihc)) - 1
    print(n_ihcs)

    factor = 5

    myo = myo / myo.max()
    if apply_mask:
        myo_mask = imageio.imread("data/ihc_mask.tif").astype("bool")
        myo[myo_mask] *= factor

    pv = pv / pv.max()
    if apply_mask:
        pv_mask = binary_dilation(sgn > 0, iterations=4)
        pv[pv_mask] *= factor

    scale = (12, 7.8, 7.8)

    v = napari.Viewer()
    v.add_image(pv, colormap="red", scale=scale, blending="additive")
    v.add_image(myo, colormap="cyan", scale=scale, blending="additive")
    v.add_labels(sgn, scale=scale)
    v.add_labels(ihc, scale=scale)

    v.scale_bar.visible = True
    v.scale_bar.unit = "µm"     # unit to display

    napari.run()


def create_rendering_otof(ds="LaVision-OTOF25R", apply_mask=True, for_rendering=False):
    cr = imageio.imread(f"data/{ds}/CR.tif")
    rb_otof = imageio.imread(f"data/{ds}/rbOtof.tif")

    ihc_path = f"data/{ds}/IHC_LOWRES-v3_filtered.tif"
    if os.path.exists(ihc_path):
        ihc = imageio.imread(ihc_path)
    else:
        ihc = imageio.imread(f"data/{ds}/IHC_LOWRES-v3.tif")

    out_folder = f"data/{ds}/for_rendering"
    os.makedirs(out_folder, exist_ok=True)
    if for_rendering:
        imageio.imwrite(os.path.join(out_folder, "CR.tif"), cr.astype("float32"), compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "rbOtof.tif"), rb_otof.astype("float32"), compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "IHC-v3.tif"), ihc.astype("float32"), compression="zlib")

    factor = 5

    ihc_mask = binary_dilation(ihc > 0, iterations=3)
    cr = cr / cr.max()
    if apply_mask:
        cr[ihc_mask] *= factor

    rb_otof = rb_otof / rb_otof.max()
    if apply_mask:
        rb_otof[ihc_mask] *= factor

    if for_rendering and apply_mask:
        imageio.imwrite(os.path.join(out_folder, "CR_filtered.tif").astype("float32"), cr, compression="zlib")
        imageio.imwrite(os.path.join(out_folder, "rbOtof_filtered.tif").astype("float32"), rb_otof, compression="zlib")
        return

    scale = (6, 3.8, 3.8)
    v = napari.Viewer()
    v.add_image(cr, colormap="red", scale=scale, blending="additive")
    v.add_image(rb_otof, colormap="cyan", scale=scale, blending="additive")
    v.add_labels(ihc, scale=scale)

    v.scale_bar.visible = True
    v.scale_bar.unit = "µm"     # unit to display

    napari.run()


# LaVision-M02
# SGN_detect-v5b : 9009
# IHC_LOWRES-v3 : 583

# LaVision-M03
# SGN_detect-v5b : 9687

# LaVision-OTOF23R
# IHC_LOWRES-v3: 623

# LaVision-OTOF25R
# IHC_LOWRES-v3: 551
def main():
    # download_data()
    # check_ihcs()

    # create_rendering_m03(export_for_rendering=True)
    # create_rendering_mar05()

    create_rendering_otof(ds="LaVision-OTOF23R", for_rendering=True)


# Merges and missing IDS:

# LaVision-M02: Looks Good
# LaVision-M03: Looks Good

# LaVision-M04:
# Merged / Missing: 428, 650, 1128, 2121, 4653, 121, 122, 123, 735, 745, 725, 925
# 4285, 4286, 4282, 4412, 4413, 4405, 4402, 4464, 4513, 4511, 4512, 4510

# LaVision-OTOF 23R: Looks good

# LaVision-OTOF 25R:
# Merged: 4822, 4836, 4835, 4834, 4833
# Missing: 4838, 4837, 4832
# Merged: 4813, 4803
# Missing: 4801, 4802, 4804, 4805, 4799, 4800, 4805
if __name__ == "__main__":
    main()
