import os
from glob import glob

from flamingo_tools.synapse_detection.read_imaris_data import extract_training_data


# Files that look good for training:
# - 4.1L_apex_IHCribboncount_Z.ims
# - 4.1L_base_IHCribbons_Z.ims
# - 4.1L_mid_IHCribboncount_Z.ims
# - 4.2R_apex_IHCribboncount_Z.ims
# - 4.2R_apex_IHCribboncount_Z.ims
# - 6.2R_apex_IHCribboncount_Z.ims  (very small crop)
# - 6.2R_base_IHCribbons_Z.ims
def process_training_data_v1():
    files = sorted(glob("./data/synapse_stains/*.ims"))
    for ff in files:
        extract_training_data(ff, output_folder="./training_data")


def _match_tif(imaris_file):
    folder = os.path.split(imaris_file)[0]

    fname = os.path.basename(imaris_file)
    parts = fname.split("_")
    cochlea = parts[0].upper()
    region = parts[1]

    tif_name = f"{cochlea}_{region}_CTBP2.tif"
    tif_path = os.path.join(folder, tif_name)
    assert os.path.exists(tif_path), tif_path

    return tif_path


def process_training_data_v2(visualize=True):
    input_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ImageCropsIHC_synapses"

    train_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v2"  # noqa
    test_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test/v2"  # noqa

    train_folders = ["M78L_IHC-synapse_crops"]
    test_folders = ["M226L_IHC-synapse_crops", "M226R_IHC-synapsecrops"]

    valid_files = [
        "m78l_apexp2718_cr-ctbp2.ims",
        "m226r_apex_p1268_pv-ctbp2.ims",
        "m226r_base_p800_vglut3-ctbp2.ims",
    ]

    for folder in train_folders + test_folders:

        if visualize:
            output_folder = None
        elif folder in train_folders:
            output_folder = train_output
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = test_output
            os.makedirs(output_folder, exist_ok=True)

        imaris_files = sorted(glob(os.path.join(input_root, folder, "*.ims")))
        for imaris_file in imaris_files:
            if os.path.basename(imaris_file) not in valid_files:
                continue
            extract_training_data(imaris_file, output_folder, tif_file=None, crop=True, scale=True)


# We have fixed the imaris data extraction problem and can use all the crops!
def process_training_data_v3(visualize=True):
    input_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ImageCropsIHC_synapses"

    train_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v3"  # noqa
    test_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v3"  # noqa

    train_folders = ["synapse_stains", "M78L_IHC-synapse_crops", "M226R_IHC-synapsecrops"]
    test_folders = ["M226L_IHC-synapse_crops"]

    exclude_names = ["220824_Ex3IL_rbCAST1635_mCtBP2580_chCR488_cell1_CtBP2spots.ims"]

    for folder in train_folders + test_folders:

        if visualize:
            output_folder = None
        elif folder in train_folders:
            output_folder = train_output
            os.makedirs(output_folder, exist_ok=True)
        else:
            output_folder = test_output
            os.makedirs(output_folder, exist_ok=True)

        imaris_files = sorted(glob(os.path.join(input_root, folder, "*.ims")))
        for imaris_file in imaris_files:
            if os.path.basename(imaris_file) in exclude_names:
                print("Skipping", imaris_file)
                continue
            extract_training_data(imaris_file, output_folder, tif_file=None, crop=True)


def process_training_data_v4(visualize=False):
    input_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/Synapses_2026-04"

    train_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/training_data/v4"  # noqa
    test_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v4"  # noqa

    train_folders_v4 = [input_root]

    input_v3 = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/ImageCropsIHC_synapses"
    train_folders = ["synapse_stains", "M78L_IHC-synapse_crops", "M226R_IHC-synapsecrops"]
    test_folders = ["M226L_IHC-synapse_crops"]

    train_folders_v3 = [os.path.join(input_v3, tf) for tf in train_folders]
    test_folders_v3 = [os.path.join(input_v3, tf) for tf in test_folders]

    train_folders_total = train_folders_v3 + train_folders_v4
    for folder in train_folders_total + test_folders_v3:
        if folder in train_folders_total:
            output_folder = train_output
        else:
            output_folder = test_output
        exclude_names = []
        if visualize:
            output_folder = None
        else:
            output_folder = train_output
            os.makedirs(output_folder, exist_ok=True)

        imaris_files = sorted(glob(os.path.join(folder, "*.ims")))
        for imaris_file in imaris_files:
            if os.path.basename(imaris_file) in exclude_names:
                print("Skipping", imaris_file)
                continue
            extract_training_data(imaris_file, output_folder, tif_file=None, crop=True)


def process_eval_data(visualize=False):
    input_root = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/AnnotatedImageCrops/SynapseValidation"  # noqa
    test_folders = ["M227R_IHC-synapsecrops", "MLR227L_IHC-synapsecrops"]
    test_output = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data/synapses/test_data/v3"  # noqa
    exclude_names = []

    for folder in test_folders:

        if visualize:
            output_folder = None
        else:
            output_folder = test_output

        imaris_files = sorted(glob(os.path.join(input_root, folder, "*.ims")))
        for imaris_file in imaris_files:
            if os.path.basename(imaris_file) in exclude_names:
                print("Skipping", imaris_file)
                continue
            extract_training_data(imaris_file, output_folder, tif_file=None, crop=True)


def main():
    # process_training_data_v1()
    # process_training_data_v2(visualize=True)
    # process_training_data_v3(visualize=False)
    # process_eval_data(visualize=False)
    process_training_data_v4(visualize=False)


if __name__ == "__main__":
    main()
