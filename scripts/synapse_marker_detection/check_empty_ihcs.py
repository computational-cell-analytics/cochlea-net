import os
from typing import Tuple

# import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd
import s3fs
import zarr

#
# NOTE: This was copied and adapted from 'flamingo_tools.s3_util'
# because of some import errors I have on the jupyter workstation
# that prevent me from importig from the flamingo_tools package.
#

SERVICE_ENDPOINT = "https://s3.fs.gwdg.de/"
BUCKET_NAME = "cochlea-lightsheet"
CREDENTIAL_FILE = os.path.expanduser("~/.aws/credentials")


def read_s3_credentials(credential_file: str) -> Tuple[str, str]:
    access_key, secret_key = None, None
    with open(credential_file) as f:
        for line in f:
            if line.startswith("aws_access_key_id"):
                access_key = line.rstrip("\n").strip().split(" ")[-1]
            if line.startswith("aws_secret_access_key"):
                secret_key = line.rstrip("\n").strip().split(" ")[-1]
    if access_key is None or secret_key is None:
        raise ValueError(f"Invalid credential file {credential_file}")
    return access_key, secret_key


def _get_fs():
    client_kwargs = {"endpoint_url": SERVICE_ENDPOINT}
    key, secret = read_s3_credentials(CREDENTIAL_FILE)
    s3_filesystem = s3fs.S3FileSystem(key=key, secret=secret, client_kwargs=client_kwargs)
    return s3_filesystem


#
# s3 utils go until here.
#


def get_ihcs_wo_syn(cochlea, ihc_name, syn_name):
    """This function returns the table of IHCs without synapse.
    """
    fs = _get_fs()

    table_path = os.path.join(BUCKET_NAME, cochlea, "tables", ihc_name, "default.tsv")
    with fs.open(table_path, "r") as f:
        ihc_table = pd.read_csv(f, sep="\t")
    ihc_table = ihc_table[ihc_table.component_labels == 1]
    ihc_ids = ihc_table.label_id.values

    table_path = os.path.join(BUCKET_NAME, cochlea, "tables", syn_name, "default.tsv")
    with fs.open(table_path, "r") as f:
        syn_table = pd.read_csv(f, sep="\t")

    matched_ihc = syn_table.matched_ihc.values
    matched_ihc = matched_ihc[np.isin(matched_ihc, ihc_ids)]

    ihc_ids_with_syn, syn_count = np.unique(matched_ihc, return_counts=True)
    ihc_ids_wo_syn_id = np.setdiff1d(ihc_ids, ihc_ids_with_syn)
    ihc_ids_wo_syn = ihc_table[ihc_table.label_id.isin(ihc_ids_wo_syn_id)]
    return ihc_ids_wo_syn


def _get_mobie_ds(cochlea, data_name):
    internal_path = os.path.join(BUCKET_NAME, cochlea, "images", "ome-zarr", f"{data_name}.ome.zarr")
    s3_filesystem = _get_fs()
    s3_path = zarr.storage.FSStore(internal_path, fs=s3_filesystem)
    return zarr.open(s3_path, mode="r")["s0"]


# TODO also get the synapse detections and visualize them.
def check_empty_ihcs(cochlea, ihc_name, syn_name, pred_path=None):
    ihc_ids_wo_syn = get_ihcs_wo_syn(cochlea, ihc_name, syn_name)
    print(len(ihc_ids_wo_syn), "IHCs don't have synapses.")

    halo = (96, 192, 192)
    resolution = [0.38] * 3

    ds_vglut = _get_mobie_ds(cochlea, "Vglut3")
    ds_ctbp2 = _get_mobie_ds(cochlea, "CTBP2")
    ds_seg = _get_mobie_ds(cochlea, ihc_name)
    if pred_path is not None:
        ds_pred = zarr.open(pred_path, "r")["prediction"]

    for i, row in ihc_ids_wo_syn.iterrows():
        position = [row.anchor_z, row.anchor_y, row.anchor_x]
        center = [int(pos / res) for pos, res in zip(position, resolution)]
        bb = tuple(slice(ce - ha, ce + ha) for ce, ha in zip(center, halo))
        print("Load ID:", row.label_id, "from", center)

        ctbp2 = ds_ctbp2[bb]
        vglut3 = ds_vglut[bb]

        seg = ds_seg[bb]
        if pred_path is not None:
            pred = ds_pred[bb]

        pos_string = "-".join(str(int(pos)) for pos in position[::-1])

        v = napari.Viewer()
        v.add_image(vglut3)
        v.add_image(ctbp2)
        v.add_labels(seg)
        if pred_path is not None:
            v.add_image(pred)
        v.title = f"ID: {row.label_id} @ {pos_string}"
        napari.run()

        # TODO enable saving of the crops via a magicgui plugin
        # if save_raw:
        #     imageio.imwrite(f"{cochlea}_VGlut3_{'-'.join(str(int(pos)) for pos in position)}.tif", vglut3)
        #     imageio.imwrite(f"{cochlea}_CTBP2_{'-'.join(str(int(pos)) for pos in position)}.tif", ctbp2)
        #     continue


def check_gerbil():
    cochlea = "G_LR_000233_R"
    ihc_name = "IHC_v6"
    syn_name = "synapse_v3_ihc_v6"

    pred_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions/G_LR_000233_R/synapses_v3"
    pred_path = os.path.join(pred_root, "predictions.zarr")
    if not os.path.exists(pred_path):
        pred_path = None

    check_empty_ihcs(cochlea, ihc_name, syn_name, pred_path=pred_path)


# For the Gerbil cochlea:
# Synapses are not detected in both of these areas due to the very low contrast
# of the CTBP2 signal
# positions = [
#     [620.0,  1050.0, 1760.0],
#     [1116.2177082702497, 1255.3013776373136, 298.84620835303593]
# ]
def main():
    # TODO create an argparser for this instead, so that we can use it to inspect other cochleae.
    check_gerbil()


if __name__ == "__main__":
    main()
