import os
from glob import glob

import h5py
import imageio.v3 as imageio
import napari
import numpy as np

IHC_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/croppings/IHC_crop"
IHC_SEG = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/croppings/IHC_seg"


def inspect_all_data():

    images = sorted(glob(os.path.join(IHC_ROOT, "**/*.tif"), recursive=True))
    segmentations = sorted(glob(os.path.join(IHC_SEG, "**/*.tif"), recursive=True))

    skip_names = ["Calretinin"]

    for im_path, seg_path in zip(images, segmentations):
        print("Loading", im_path)
        root, fname = os.path.split(im_path)
        folder = os.path.basename(root)
        if folder in skip_names:
            continue

        try:
            im = imageio.imread(im_path)
            seg = imageio.imread(seg_path).astype("uint32")

            v = napari.Viewer()
            v.add_image(im)
            v.add_labels(seg)
            v.title = f"{folder}/{fname}"
            napari.run()
        except ValueError:
            continue


def _require_prediction(im, image_path, with_mask):
    model_path = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/IHC/v2_cochlea_distance_unet_IHC_supervised_2025-05-21"  # noqa

    root, fname = os.path.split(image_path)
    folder = os.path.basename(root)

    cache_path = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/croppings/predictions/{folder}"
    os.makedirs(cache_path, exist_ok=True)
    cache_path = os.path.join(cache_path, fname.replace(".tif", ".h5"))

    output_key = "pred_masked" if with_mask else "pred"

    if os.path.exists(cache_path):
        with h5py.File(cache_path, "r") as f:
            if output_key in f:
                pred = f[output_key][:]
                return pred

    from torch_em.util import load_model
    from torch_em.util.prediction import predict_with_halo
    from torch_em.transform.raw import standardize

    block_shape = (128, 128, 128)
    halo = (16, 32, 32)
    if with_mask:
        from bioimage_cpp.utils import Blocking

        mask = np.zeros(im.shape, dtype=bool)
        blocking = Blocking([0, 0, 0], im.shape, block_shape)

        for block_id in range(blocking.number_of_blocks):
            block = blocking.get_block(block_id)
            bb = tuple(slice(beg, end) for beg, end in zip(block.begin, block.end))
            data = im[bb]
            max_ = np.percentile(data, 95)
            if max_ > 200:
                mask[bb] = 1
    else:
        mask = None

    im = standardize(im)

    model = load_model(model_path)

    pred = predict_with_halo(
        im, model, gpu_ids=[0], block_shape=block_shape, halo=halo, preprocess=None, mask=mask
    )

    with h5py.File(cache_path, "a") as f:
        f.create_dataset(output_key, data=pred, compression="lzf")


def check_block_artifacts():
    image_path = os.path.join(IHC_ROOT, "Calretinin/M61L_CR_IHC_forannotations_C1.tif")
    im = imageio.imread(image_path)
    predictions = _require_prediction(im, image_path, with_mask=False)

    seg_path = os.path.join(IHC_SEG, "Calretinin/M61L_CR_IHC_forannotations_C1.tif")
    seg_old = imageio.imread(seg_path)

    v = napari.Viewer()
    v.add_image(im)
    v.add_image(predictions)
    v.add_labels(seg_old)
    napari.run()


def _get_ihc_v_sgn_mask(seg, props, threshold, criterion="ratio"):
    sgn_ids = props.label[props[criterion] < threshold].values
    ihc_ids = props.label[props[criterion] >= threshold].values

    ihc_v_sgn = np.zeros_like(seg, dtype="uint32")
    ihc_v_sgn[np.isin(seg, ihc_ids)] = 1
    ihc_v_sgn[np.isin(seg, sgn_ids)] = 2

    return ihc_v_sgn


# From inspection:
# - CR looks quite good, but also shows the blocking artifacts, and some merges:
#   Calretinin/M61L_CR_IHC_forannotations_C1.tif (blocking artifacts)
#   Calretinin/M63R_CR640_apexIHC_C2.tif (merges, but also weird looking stain)
#   Calretinin/M78L_CR488_apexIHC2_C6.tif (background structures are segmented)
#   Background is the case for some others too; it segments the hairs.
# - Myo7a, looks good, but as we discussed the stain is not specific
#   Myo7a/3.1L_Myo7a_apex_HCAT_reslice_C2.tif (good candidate for filtering)
#   Myo7a/3.1L_Myo7a_mid_HCAT_reslice_C4.tif (good candidate for filtering)
# - PV: Stain looks quite different, segmentations don't look so good.
def main():
    inspect_all_data()
    # check_block_artifacts()


if __name__ == "__main__":
    main()
