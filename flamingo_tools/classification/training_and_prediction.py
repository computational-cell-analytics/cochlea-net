from typing import Optional, Sequence

import h5py
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

from ..cpu_utils import available_cpus
from ..measurements import compute_object_measures


def train_classifier(feature_paths: Sequence[str], save_path: str, **rf_kwargs) -> None:
    """Train a random forest classifier on features and labels that were exported via the classification GUI.

    Args:
        feature_paths: The path to the h5 files with features and labels.
        save_path: Where to save the trained random forest.
        rf_kwargs: Keyword arguments for creating the random forest.
    """
    features, labels = [], []
    for path in feature_paths:
        with h5py.File(path, "r") as f:
            for name, group in f.items():
                features.append(group["features"][:])
                labels.append(group["labels"][:])

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(features, labels)

    dump(rf, save_path)


def predict_classifier(
    rf_path: str,
    image_path: str,
    segmentation_path: str,
    feature_table_path: str,
    segmentation_table_path: Optional[str],
    image_key: Optional[str] = None,
    segmentation_key: Optional[str] = None,
    n_threads: Optional[int] = None,
    feature_set: str = "default",
) -> pd.DataFrame:
    """Run prediction with a trained classifier on an input volume with associated segmentation.

    Args:
        rf_path: The path to the trained random forest.
        image_path: The path to the image data.
        segmentation_path: The path to the segmentation.
        feature_table_path: The path for the features used for prediction.
            The features will be computed and saved if this table does not exist.
        segmentation_table_path: The path to the segmentation table (in MoBIE format).
            It will be computed on the fly if it is not given.
        image_key: The key / internal path for the image data. Not needed for tif data.
        segmentation_key: The key / internal path for the segmentation data. Not needed for tif data.
        n_threads: The number of threads for parallelization.
        feature_set: The feature set to use. Refer to `flamingo_tools.measurements.FEATURE_FUNCTIONS` for details.

    Returns:
        A dataframe with the prediction. It contains the columns 'label_id', 'predictions' and
            'probs-0', 'probs-1', ... . The latter columns contain the probabilities for the respective class.
    """
    compute_object_measures(
        image_path=image_path,
        segmentation_path=segmentation_path,
        segmentation_table_path=segmentation_table_path,
        output_table_path=feature_table_path,
        image_key=image_key,
        segmentation_key=segmentation_key,
        n_threads=n_threads,
        feature_set=feature_set,
    )

    features = pd.read_csv(feature_table_path, sep="\t")
    label_ids = features.label_id.values
    features = features.drop(columns=["label_id"]).values

    rf = load(rf_path)
    n_threads = available_cpus() if n_threads is None else n_threads
    rf.n_jobs_ = n_threads

    probs = rf.predict_proba(features)
    result = {"label_id": label_ids, "prediction": np.argmax(probs, axis=1)}
    result.update({"probs-{i}": probs[:, i] for i in range(probs.shape[1])})
    return pd.DataFrame(result)
