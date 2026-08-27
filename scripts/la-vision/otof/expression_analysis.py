import json
import os
import pickle

import imageio.v3 as imageio
import napari
import numpy as np
import pandas as pd
import nifty.tools as nt

from elf.parallel import seeded_watershed, distance_transform, isin
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
from skimage.measure import regionprops
from sklearn.ensemble import RandomForestClassifier

DATA_ROOT = "/home/pape/Work/my_projects/flamingo-tools/scripts/figures/data"
SEG_NAME = "IHC_LOWRES-v3"

COCHLEA = {
    "LaVision-OTOF23R": {
        "positive": [2040, 1951, 1952, 1949, 1935, 1936, 1885, 1894, 1915, 5110, 3103, 3104, 3316,
                     3196, 3204, 2859, 2611, 2604, 469, 102, 106, 82, 161],
        "negative": [2038, 2036, 2037, 1946, 1882, 1934, 1900, 1913, 2046, 3755, 4728, 3250, 3256, 3269, 3290,
                     3308, 3074, 2340, 264, 92, 3217, 2610, 762, 2606],
    },
    "LaVision-OTOF25R": {
        "positive": [4804, 4811, 4784, 4838, 5005, 5001, 5223, 3883, 3874, 4204, 4489, 4358, 4224, 3911, 3264,
                     2621, 2533, 2544, 2545, 4431, 4433, 4427, 4439, 4493, 4369, 4405, 3265, 3265, 3883, 5005],
        "negative": [4815, 4820, 4837, 4997, 5003, 5004, 3588, 3869, 4187, 4333, 4463, 4491, 4490, 4359, 3438,
                     3442, 2540, 1305, 2798, 2799, 2599, 2506],
    },
}


def _get_table(cochlea):
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]
    seg_source = sources[SEG_NAME]

    table_folder = os.path.join(
        BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
    )
    table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
    table = pd.read_csv(table_content, sep="\t")

    return table


def get_expression_mask(ihcs, table, marker_column):
    remapping = {0: 0}
    remapping.update({int(label_id): int(expression_value)
                      for label_id, expression_value
                      in zip(table.label_id.values, table[marker_column].values)})
    return nt.takeDict(remapping, ihcs)


def dilate_ihcs(seg, dilation=2):
    block_shape = [64, 128, 128]
    halo = [8, 16, 16]

    dilation_factor = 2
    distances = distance_transform(seg == 0, halo=halo, sampling=(1, 1, 1), block_shape=block_shape, verbose=True)
    extension_mask = distances < dilation_factor

    extended_seg = np.zeros_like(seg)
    extended_seg = seeded_watershed(
        distances, seg, out=extended_seg, mask=extension_mask, block_shape=block_shape, halo=halo, verbose=True
    )
    return extended_seg


def _update_table(segmentation, signal, table):
    def percentiles(mask, sign):
        extracted = sign[mask]
        return [
            np.percentile(extracted, 1),
            np.percentile(extracted, 10),
            np.percentile(extracted, 25),
            np.median(extracted),
            np.percentile(extracted, 75),
            np.percentile(extracted, 90),
            np.percentile(extracted, 99),
        ]

    print("Computing regionprops ...")
    props = regionprops(segmentation, signal, extra_properties=[percentiles])
    label_ids = table.label_id.values.astype(int)

    intensity_p1 = {prop.label: prop.percentiles[0] for prop in props}
    table["intensity-p1"] = [intensity_p1.get(label, 0) for label in label_ids]

    intensity_p10 = {prop.label: prop.percentiles[1] for prop in props}
    table["intensity-p10"] = [intensity_p10.get(label, 0) for label in label_ids]

    intensity_p25 = {prop.label: prop.percentiles[2] for prop in props}
    table["intensity-p25"] = [intensity_p25.get(label, 0) for label in label_ids]

    median_intensity = {prop.label: prop.percentiles[3] for prop in props}
    table["median-intensity"] = [median_intensity.get(label, 0) for label in label_ids]

    intensity_p75 = {prop.label: prop.percentiles[4] for prop in props}
    table["intensity-p75"] = [intensity_p75.get(label, 0) for label in label_ids]

    intensity_p90 = {prop.label: prop.percentiles[5] for prop in props}
    table["intensity-p90"] = [intensity_p90.get(label, 0) for label in label_ids]

    intensity_p99 = {prop.label: prop.percentiles[5] for prop in props}
    table["intensity-p99"] = [intensity_p99.get(label, 0) for label in label_ids]

    mean = {prop.label: prop.mean_intensity for prop in props}
    table["mean-intensity"] = [mean.get(label, 0) for label in label_ids]

    std = {prop.label: prop.std_intensity for prop in props}
    table["std-intensity"] = [std.get(label, 0) for label in label_ids]

    valid_ids = np.array([prop.label for prop in props])
    return table, valid_ids


def _classify_expression(table, ds, valid_ids, train_rf):
    feature_names = ["intensity-p1", "intensity-p10", "intensity-p25", "median-intensity",
                     "intensity-p75", "intensity-p90", "intensity-p99", "mean-intensity", "std-intensity"]
    label_ids = table.label_id.values.astype(int)
    features = table[feature_names].values

    output_path = f"./data/rf_{ds}.pkl"
    if train_rf:
        positive_ids = np.unique(COCHLEA[ds]["positive"])
        negative_ids = np.unique(COCHLEA[ds]["negative"])

        positive_mask = np.isin(label_ids, positive_ids)
        negative_mask = np.isin(label_ids, negative_ids)

        train_features = np.concatenate([
            features[positive_mask], features[negative_mask]
        ], axis=0)
        train_labels = np.array([0] * len(positive_ids) + [1] * len(negative_ids))

        print("Train classifier ...")
        rf = RandomForestClassifier(n_estimators=50, max_depth=8)
        rf.fit(train_features, train_labels)
        with open(output_path, "wb") as f:
            pickle.dump(rf, f)
    else:
        with open(output_path, "rb") as f:
            rf = pickle.load(f)

    valid_mask = np.isin(label_ids, valid_ids)
    valid_features = features[valid_mask]
    print("Predict classifier ...")
    classification = rf.predict(valid_features) + 1
    classification = {label: cls for label, cls in zip(valid_ids, classification)}

    # Over-ride for labels:
    if train_rf:
        for pos_id in positive_ids:
            classification[pos_id] = 1
        for neg_id in negative_ids:
            classification[neg_id] = 2

    marker_column = "expression_classification"
    table[marker_column] = [classification.get(label, 0) for label in label_ids]

    return table, marker_column


def get_masks(seg, table, marker_column):
    pos_ids = table[table[marker_column] == 1].label_id.values
    neg_ids = table[table[marker_column] == 2].label_id.values

    pos_mask = np.zeros(seg.shape, dtype="uint8")
    pos_mask = isin(seg, pos_ids, out=pos_mask, block_shape=(32, 128, 128))
    neg_mask = np.zeros(seg.shape, dtype="uint8")
    neg_mask = isin(seg, neg_ids, out=neg_mask, block_shape=(32, 128, 128))

    return pos_mask.astype("float32"), neg_mask.astype("float32")


def expression_analysis(ds, export, train_rf):
    folder = os.path.join(DATA_ROOT, ds)
    rb_otof = imageio.imread(os.path.join(folder, "rbOtof.tif"))
    if ds == "LaVision-OTOF23R":
        ihcs = imageio.imread(os.path.join(folder, "IHC_LOWRES-v3_filtered.tif"))
    else:
        ihcs = imageio.imread(os.path.join(folder, "IHC_LOWRES-v3.tif"))
    table = _get_table(ds)

    ihcs_dilated = dilate_ihcs(ihcs, dilation=2)
    table, valid_ids = _update_table(ihcs_dilated, rb_otof, table)
    table, marker_column = _classify_expression(table, ds, valid_ids, train_rf)
    expression_mask = get_expression_mask(ihcs_dilated, table, marker_column=marker_column)

    if export:
        output_folder = f"./data/{ds}"
        os.makedirs(output_folder, exist_ok=True)
        # Save the table.
        table.to_csv(os.path.join(output_folder, "expression_classification.tsv"), sep="\t", index=False)

        # Save the dilated IHCs and the individual expression masks for Lennart.
        imageio.imwrite(os.path.join(output_folder, "ihcs_dilated.tif"), ihcs_dilated, compression="zlib")

        pos_mask, neg_mask = get_masks(ihcs, table, marker_column)
        imageio.imwrite(os.path.join(output_folder, "ihcs_positive.tif"), pos_mask, compression="zlib")
        imageio.imwrite(os.path.join(output_folder, "ihcs_negative.tif"), neg_mask, compression="zlib")

        pos_mask, neg_mask = get_masks(ihcs_dilated, table, marker_column)
        imageio.imwrite(os.path.join(output_folder, "ihcs_positive_dilated.tif"), pos_mask, compression="zlib")
        imageio.imwrite(os.path.join(output_folder, "ihcs_negative_dilated.tif"), neg_mask, compression="zlib")

    v = napari.Viewer()
    v.add_image(rb_otof)
    v.add_labels(ihcs, visible=False)
    v.add_labels(ihcs_dilated)
    v.add_labels(expression_mask)
    napari.run()


def main():
    # ds = "LaVision-OTOF25R"
    # expression_analysis(ds, export=True, train_rf=True)

    ds = "LaVision-OTOF23R"
    expression_analysis(ds, export=True, train_rf=True)


if __name__ == "__main__":
    main()
