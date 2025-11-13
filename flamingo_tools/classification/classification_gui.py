import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Optional

import h5py
import imageio.v3 as imageio
import napari
import numpy as np

from joblib import dump
from magicgui import magic_factory

try:
    import micro_sam.sam_annotator.object_classifier as classifier_util
    from micro_sam.object_classification import project_prediction_to_segmentation
    from micro_sam.sam_annotator._widgets import _generate_message
except ImportError:
    micro_sam = None

from ..measurements import compute_object_measures_impl

IMAGE_LAYER_NAME = None
SEGMENTATION_LAYER_NAME = None
FEATURES = None
SEG_IDS = None
CLASSIFIER = None
LABELS = None
FEATURE_SET = None


def _compute_features(segmentation, image):
    features = compute_object_measures_impl(image, segmentation, feature_set=FEATURE_SET)
    seg_ids = features.label_id.values.astype(int)
    features = features.drop(columns="label_id").values
    return features, seg_ids


@magic_factory(call_button="Train and predict")
def _train_and_predict_rf_widget(viewer: "napari.viewer.Viewer") -> None:
    global FEATURES, SEG_IDS, CLASSIFIER, LABELS

    annotations = viewer.layers["annotations"].data
    segmentation = viewer.layers[SEGMENTATION_LAYER_NAME].data
    labels = classifier_util._accumulate_labels(segmentation, annotations)
    LABELS = labels

    if FEATURES is None:
        print("Computing features ...")
        image = viewer.layers[IMAGE_LAYER_NAME].data
        FEATURES, SEG_IDS = _compute_features(segmentation, image)

    print("Training random forest ...")
    rf = classifier_util._train_rf(FEATURES, labels, n_estimators=200, max_depth=10, n_jobs=cpu_count())
    CLASSIFIER = rf

    # Run and set the prediction.
    print("Run prediction ...")
    pred = rf.predict(FEATURES)
    prediction_data = project_prediction_to_segmentation(segmentation, pred, SEG_IDS)
    viewer.layers["prediction"].data = prediction_data


@magic_factory(call_button="Export Classifier")
def _create_export_rf_widget(export_path: Optional[Path] = None) -> None:
    rf = CLASSIFIER
    if rf is None:
        return _generate_message("error", "You have not run training yet.")
    if export_path is None or export_path == "":
        return _generate_message("error", "You have to provide an export path.")
    # Do we add an extension? .joblib?
    dump(rf, export_path)


@magic_factory(call_button="Export Features")
def _create_export_feature_widget(export_path: Optional[Path] = None) -> None:

    if FEATURES is None or LABELS is None:
        return _generate_message("error", "You have not run training yet.")
    if export_path is None or export_path == "":
        return _generate_message("error", "You have to provide an export path.")

    valid = LABELS != 0
    features, labels = FEATURES[valid], LABELS[valid]

    export_path = Path(export_path).with_suffix(".h5")
    with h5py.File(export_path, "a") as f:
        g = f.create_group(IMAGE_LAYER_NAME)
        g.attrs["feature_set"] = FEATURE_SET
        g.create_dataset("features", data=features, compression="lzf")
        g.create_dataset("labels", data=labels, compression="lzf")


def run_classification_gui(
    image_path: str,
    segmentation_path: str,
    image_name: Optional[str] = None,
    segmentation_name: Optional[str] = None,
    feature_set: str = "default",
) -> None:
    """Start the classification GUI.

    Args:
        image_path: The path to the image data.
        segmentation_path: The path to the segmentation.
        image_name: The name for the image layer. Will use the filename if not given.
        segmentation_name: The name of the label layer with the segmentation.
            Will use the filename if not given.
        feature_set: The feature set to use. Refer to `flamingo_tools.measurements.FEATURE_FUNCTIONS` for details.
    """
    global IMAGE_LAYER_NAME, SEGMENTATION_LAYER_NAME, FEATURE_SET

    image = imageio.imread(image_path)
    segmentation = imageio.imread(segmentation_path)

    image_name = os.path.basename(image_path) if image_name is None else image_name
    segmentation_name = os.path.basename(segmentation_path) if segmentation_name is None else segmentation_name

    IMAGE_LAYER_NAME = image_name
    SEGMENTATION_LAYER_NAME = segmentation_name
    FEATURE_SET = feature_set

    viewer = napari.Viewer()
    viewer.add_image(image, name=image_name)
    viewer.add_labels(segmentation, name=segmentation_name)

    shape = image.shape
    viewer.add_labels(name="prediction", data=np.zeros(shape, dtype="uint8"))
    viewer.add_labels(name="annotations", data=np.zeros(shape, dtype="uint8"))

    # Add the gui elements.
    train_widget = _train_and_predict_rf_widget()
    rf_export_widget = _create_export_rf_widget()
    feature_export_widget = _create_export_feature_widget()

    viewer.window.add_dock_widget(train_widget)
    viewer.window.add_dock_widget(feature_export_widget)
    viewer.window.add_dock_widget(rf_export_widget)

    napari.run()
