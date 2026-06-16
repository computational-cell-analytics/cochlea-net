import argparse
import os
from shutil import copyfile

from sklearn.model_selection import train_test_split


TRAINING_COCHLEAE = [
    "M_AMD_N142_R",
    "M_AMD_N153_L",
    "M_AMD_N75_L",
    "M_AMD_N89_L",
    "M_AMD_N97_L",
    "M_LR_000249_R",
    "M_AMD_000088_R",
    "M_AMD_000112_L"
]

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
REVISED_ANNOTATIONS = os.path.join(COCHLEA_DIR, "AnnotatedImageCrops/IHC_training_crops_2026-04/revised_annotations")
IMAGE_DIR = os.path.join(COCHLEA_DIR, "AnnotatedImageCrops/IHC_training_crops_2026-04/data")
IHC_TRAINING = os.path.join(COCHLEA_DIR, "training_data/IHC")
TRAINING_DICT = {
    "IHC_v8": {
        "training_data": os.path.join(IHC_TRAINING, "IHC_v8_2026-05"),
        "image_channel": "Vglut3",
    },
    "IHC_PV-v1": {
        "training_data": os.path.join(IHC_TRAINING, "IHC_PV-v1_2026-05"),
        "image_channel": "PV",
    },
}


def match_image_data_to_annotation(
    annotation_path: str,
    data_dir: str,
    image_channel: str = "Vglut3",
) -> str:
    """Match image channel and its file path to a given annotation path.

    Args:
        annotation_path: Path to annotation TIF file.
        data_dir: Directory containing image data in individual folders for each cochlea.
        image_channel: Image channel for IHC segmentation. Either Vglut3 or PV.

    Returns:
        File path to image data.
    """
    content = os.path.basename(annotation_path).split("_")
    cochlea_str = content[0]
    cochlea_crop_prefix = "_".join(content[:3])
    cochlea_dir_prefix = "_".join(cochlea_str.split("-"))
    if cochlea_dir_prefix not in TRAINING_COCHLEAE:
        print(f"{cochlea_dir_prefix} not in expected training cochleae.")

    cochlea_dirs = [entry.path for entry in os.scandir(data_dir) if
                    os.path.isdir(entry.path) and
                    cochlea_dir_prefix in entry.name]
    if len(cochlea_dirs) == 0:
        raise ValueError(f"No matching directory for {annotation_path}.")
    elif len(cochlea_dirs) != 1:
        raise ValueError(f"Found{len(cochlea_dirs)} directories for {annotation_path}.")
    else:
        cochlea_dir = cochlea_dirs[0]

    image_paths = [entry.path for entry in os.scandir(cochlea_dir) if
                   cochlea_crop_prefix in entry.name and
                   image_channel in entry.name]

    if len(image_paths) == 0:
        raise ValueError(f"No matching image for {annotation_path}.")
    elif len(image_paths) != 1:
        raise ValueError(f"Found{len(image_paths)} images for {annotation_path}.")
    else:
        return image_paths[0]


def prepare_training_data(
    annotation_dir: str,
    data_dir: str,
    image_channel: str,
    training_data: str,
) -> None:
    """Prepare training data by transferring image data and annotations to the data directory.

    Args:
        annotation_path: Path to annotation TIF file.
        data_dir: Directory containing image data in individual folders for each cochlea.
        image_channel: Image channel for IHC segmentation. Either Vglut3 or PV.
        training_data: Directory containing training and validation data for network training.
    """
    annotation_paths = [entry.path for entry in os.scandir(annotation_dir)]
    annotation_paths.sort()

    training_dir = os.path.join(training_data, "train")
    os.makedirs(training_dir, exist_ok=True)
    validation_dir = os.path.join(training_data, "val")
    os.makedirs(validation_dir, exist_ok=True)

    # separate files into train and validation
    train_fraction = 0.85
    n_train = int(train_fraction * len(annotation_paths))
    print(f"Using {n_train} images for training and {len(annotation_paths) - n_train} for validation.")
    annotations_train, _ = train_test_split(
        annotation_paths, train_size=n_train, random_state=42,
    )

    for annotation_path in annotation_paths:
        image_path = match_image_data_to_annotation(annotation_path, data_dir, image_channel)
        annotation_dst_name = f"{os.path.splitext(os.path.basename(image_path))[0]}_annotations.tif"

        # separate output into train and validation dataset for training
        if annotation_path in annotations_train:
            image_dst = os.path.join(training_dir, os.path.basename(image_path))
            annotation_dst = os.path.join(training_dir, annotation_dst_name)
        else:
            image_dst = os.path.join(validation_dir, os.path.basename(image_path))
            annotation_dst = os.path.join(validation_dir, annotation_dst_name)

        copyfile(image_path, image_dst)
        copyfile(annotation_path, annotation_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Script to create training and validation data for IHC network.")

    parser.add_argument("-i", "--input", type=str, default=REVISED_ANNOTATIONS,
                        help="Input directory containing annotations.")
    parser.add_argument("-d", "--data_dir", type=str, default=IMAGE_DIR,
                        help="Directory containing subfolders for each cochlea with image data.")
    parser.add_argument("-c", "--channel", type=str, default="Vglut3",
                        help="Channel name of image data. Default: Vglut3")
    parser.add_argument("-t", "--train", type=str, default=None,
                        help="Directory for train and validation data.")
    args = parser.parse_args()

    if args.train is None:
        for key, items in TRAINING_DICT.items():
            print(f"Creating training data for {key}.")
            prepare_training_data(
                annotation_dir=args.input,
                data_dir=args.data_dir,
                training_data=items["training_data"],
                image_channel=items["image_channel"],
            )
    else:

        prepare_training_data(
            annotation_dir=args.input,
            data_dir=args.data_dir,
            image_channel=args.channel,
            training_data=args.train,
        )


if __name__ == "__main__":
    main()
