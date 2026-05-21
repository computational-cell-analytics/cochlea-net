import os
import sys
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import numpy as np
import pandas as pd
import pickle

from skimage.measure import regionprops
from sklearn.linear_model import LogisticRegression
from flamingo_tools.s3_utils import get_s3_path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_AMD = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/Result_AMD"
ROOT_EK = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/Result_EK"

COCHLEA_DICT = {
    "MLR99L": {"cochlea_long": "M_LR_000099_L", "seg_name": "PV_SGN_v2"}
}


def load_annotations(pattern):
    paths = sorted(glob(pattern))
    annotations = [path[len(pattern):] for path in paths]
    channels = [annotation.split("_")[0] for annotation in annotations]
    return paths, channels


# Get the features and per channel labels from this crop.
def extract_crop_data(cochlea, crop_table, crop_root):
    table = pd.read_csv(crop_table, sep="\t")
    prefix = Path(crop_table).stem

    # Get the paths to all annotations.
    paths_amd, channels_amd = load_annotations(os.path.join(ROOT_AMD, f"positive-negative_{prefix}*"))
    paths_ek, channels_ek = load_annotations(os.path.join(ROOT_EK, f"positive-negative_{prefix}*"))
    channel_names = list(set(channels_amd))
    channel_names.sort()

    cochlea_long = COCHLEA_DICT[cochlea]["cochlea_long"]
    seg_name = COCHLEA_DICT[cochlea]["seg_name"]

    for channel in channel_names:
        s3_path = f"{cochlea_long}/tables/{seg_name}/{channel}_{"-".join(seg_name.split("_"))}_object-measures.tsv"
        tsv_path, fs = get_s3_path(s3_path)
        with fs.open(tsv_path, 'r') as f:
            table_measure = pd.read_csv(f, sep="\t")

        table = table.merge(
            table_measure[["label_id", "median"]],
            on="label_id",
            how="left"
        )
        # Rename the merged column
        table.rename(columns={"median": f"intensity_{channel}"}, inplace=True)

    # Load the segmentation.
    seg_path = os.path.join(crop_root, f"{prefix}_{seg_name}.tif")
    seg = imageio.imread(seg_path)

    # Load the features (= intensity and PV intensity ratios) for both channels.
    features = table[
        [f"marker_{channel_names[0]}", f"{channel_names[0]}_ratio_PV"] +
        [f"marker_{channel_names[1]}", f"{channel_names[1]}_ratio_PV"]
    ].values

    # Load the labels, derived from the annotations.
    labels = {channel: None for channel in channel_names}
    # total_channels = channels_amd + channels_ek
    # total_paths = paths_amd + paths_ek

    for channel, path in zip(channels_amd, paths_amd):
        data = imageio.imread(path)
        props = regionprops(seg, data)
        labeling = np.array([prop.max_intensity for prop in props], dtype="int32")
        if labels[channel] is None:
            labels[channel] = labeling
        else:
            # Combine labels so that we only keep the labels that agree, set others to zero
            # (in order to filter them out later).
            prev_labeling = labels[channel]
            disagreement = prev_labeling != labeling
            labeling[disagreement] = 0
            labels[channel] = labeling

    return features, labels


def process_cochlea(cochlea):
    # The root folders for tables and crop data for this cochlea.
    table_root = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/tables_{cochlea}"
    crop_root = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/{cochlea}"
    model_root = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/classifier/{cochlea}"

    # Getthe tables for all crops in this cochlea.
    tables = sorted(glob(os.path.join(table_root, "*.tsv")))

    # Iterate over the crops, load the features and the labels per channel.
    features = []
    labels = {}
    for table in tables:
        crop_features, crop_labels = extract_crop_data(cochlea, table, crop_root)
        features.append(crop_features)
        # Concatenate the labels per channel.
        for channel, labeling in crop_labels.items():
            if channel in labels:
                labels[channel] = np.concatenate([labels[channel], labeling], axis=0)
            else:
                labels[channel] = labeling
    features = np.concatenate(features, axis=0)

    # Train and evaluate logistic regression per channel.
    start, stop = 0, 2
    for channel, labeling in labels.items():
        # Exclude labels with value zero.
        label_mask = labeling != 0
        # Get the features for this channel.
        this_features = features[:, start:stop][label_mask]
        this_labels = labeling[label_mask]

        labels = list(set(this_labels))
        for l in labels:
            print(f"label {l} occurences: {list(this_labels).count(l)}")

        # Create a train and test split.
        train_features, test_features, train_labels, test_labels = train_test_split(
            this_features, this_labels, test_size=0.3
        )

        # Train and evaluate the classifier.
        classifier = LogisticRegression(penalty="l2")
        classifier.fit(train_features, train_labels)

        prediction = classifier.predict(test_features)
        model_path = os.path.join(model_root, f"logistic_{channel}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(classifier, f)

        accuracy = accuracy_score(test_labels, prediction)
        print("Channel:", channel)
        print("Accuracy:", accuracy)
        for index in [1, 2]:
            label_list = []
            pred_list = []
            for l, p in zip(test_labels, prediction):
                if l == index:
                    label_list.append(l)
                    pred_list.append(p)
            print(f"Accuracy label {index}: {accuracy_score(np.array(label_list), np.array(pred_list))}")

        start += 2
        stop += 2

    # Note: we could do some other things here:
    # - Train a single classifier for subtype prediction (= 4 classes) using all channels.
    # - Use different classifier (e.g. RandomForest); however, accuracy from logistic regression looks fine.
    # - To better understand results we could also look at the confusion matrix.
    # - A better evaluation would be to train and test on separate blocks.

    # The classifier can be saved and loaded with pickle, to apply it to all SGNs in the cochlea later.


def apply_model(cochlea):
    model_root = f"/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/SGN_subtypes/classifier/{cochlea}"
    models = [entry.path for entry in os.scandir(model_root) if ".pkl" in entry.name]

    cochlea_long = COCHLEA_DICT[cochlea]["cochlea_long"]
    seg_name = COCHLEA_DICT[cochlea]["seg_name"]

    s3_path = os.path.join(f"{cochlea_long}", "tables", f"{seg_name}", "default.tsv")
    tsv_path, fs = get_s3_path(s3_path)
    with fs.open(tsv_path, 'r') as f:
        table = pd.read_csv(f, sep="\t")

    for model_path in models:
        channel = os.path.basename(model_path).split(".pkl")[0].split("_")[1]

        s3_path = f"{cochlea_long}/tables/{seg_name}/{channel}_{"-".join(seg_name.split("_"))}_object-measures.tsv"
        tsv_path, fs = get_s3_path(s3_path)
        with fs.open(tsv_path, 'r') as f:
            table_measure = pd.read_csv(f, sep="\t")

        table = table.merge(
            table_measure[["label_id", "median"]],
            on="label_id",
            how="left"
        )
        table.rename(columns={"median": f"intensity_{channel}"}, inplace=True)

        subset = table.loc[table[f"marker_{channel}"].isin([1, 2])]
        features = subset[
            [f"marker_{channel}", f"{channel}_ratio_PV"]
        ].values

        with open(model_path, "rb") as f:
            classifier = pickle.load(f)

        prediction = classifier.predict(features)
        # switch prediction to be consistent with markers: 1 - positive, 2 - negative
        prediction = [2 if x == 1 else 1 for x in prediction]

        table.loc[:, f"classifier_{channel}"] = 0
        table.loc[subset.index, f"classifier_{channel}"] = prediction

    out_path = os.path.join(model_root, cochlea + ".tsv")
    table.to_csv(out_path, sep="\t", index=False)


def main():
    # Process a cochlea by:
    # - Extracting the features (intensities and intensity ratios) and labels for each crop.
    # - Training a classifier based on the labels and evaluating it.
    cochlea = "MLR99L"
    process_cochlea(cochlea)
    apply_model(cochlea)


if __name__ == "__main__":
    main()
