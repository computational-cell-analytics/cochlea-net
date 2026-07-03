import os
from check_ihc_seg import IHC_ROOT, IHC_SEG


def run_object_classifier():
    from flamingo_tools.classification import run_classification_gui

    image_path = os.path.join(IHC_ROOT, "Myo7a/3.1L_Myo7a_apex_HCAT_reslice_C2.tif")
    seg_path = os.path.join(IHC_SEG, "Myo7a/3.1L_Myo7a_apex_HCAT_reslice_C2.tif")

    run_classification_gui(image_path, seg_path, segmentation_name="IHCs")


def train_random_forest():
    from flamingo_tools.classification import train_classifier

    feature_path = "data/features.h5"
    save_path = "data/rf.joblib"
    train_classifier([feature_path], save_path=save_path)


def apply_random_forest():
    from flamingo_tools.classification import predict_classifier

    image_path = os.path.join(IHC_ROOT, "Myo7a/3.1L_Myo7a_mid_HCAT_reslice_C4.tif")
    seg_path = os.path.join(IHC_SEG, "Myo7a/3.1L_Myo7a_mid_HCAT_reslice_C4.tif")

    rf_path = "data/rf.joblib"
    results = predict_classifier(
        rf_path, image_path, seg_path, feature_table_path="data/features.csv",
        segmentation_table_path=None, n_threads=4,
    )

    import imageio.v3 as imageio
    import napari
    from bioimage_cpp.utils import take_dict

    image = imageio.imread(image_path)
    seg = imageio.imread(seg_path)

    relabel_dict = {label_id: pred + 1 for label_id, pred in zip(results.label_id, results.prediction)}
    relabel_dict[0] = 0
    pred = take_dict(relabel_dict, seg)

    v = napari.Viewer()
    v.add_image(image)
    v.add_labels(seg)
    v.add_labels(pred)
    napari.run()


def main():
    # 1.) Start the classifier GUI to extract features for a random forest.
    # run_object_classifier()

    # 2.) Train a random forest on the features.
    # train_random_forest()

    # 3.) Apply the random forest to another dataset.
    apply_random_forest()


if __name__ == "__main__":
    main()
