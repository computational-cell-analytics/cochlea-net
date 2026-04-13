import argparse
import os
from typing import List, Tuple

from flamingo_tools.analysis.training_data_utils import export_crop_centers
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT
from flamingo_tools.extract_block_util import extract_block_json_wrapper

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


def create_crop_dictionaries(
    cochleae: List[str],
    out_dir: str,
    force_overwrite: bool = False,
) -> Tuple[List[str], List[str]]:
    """Create a JSON dictionary featuring information for creating crops from an IHC segmentation.
    Per default, the dictionary is used to extract the PV, Vglut3, and IHC_v4b channels.
    The component list and the staining protocol of the cochlea are taken from the SYNAPSE_DICT.
    Existing dictionaries are not overwritten.

    Args:
        cochleae: List of cochleae.
        out_dir: Output directory for JSON dictionaries. Dictionary names are automatically generated.
    """
    cochlea_names = []
    json_files = []
    for cochlea in cochleae:
        if cochlea not in SYNAPSE_DICT.keys():
            raise ValueError(f"Cochlea {cochlea} is not in SYNAPSE_DICT."
                             "Please add an entry in flamingo_tools/postprocessing/synapse_per_ihc_utils.py")
        if "component_list" in list(SYNAPSE_DICT[cochlea].keys()):
            component_labels = SYNAPSE_DICT[cochlea]["component_list"]
        else:
            component_labels = [1]
        suffix = SYNAPSE_DICT[cochlea]["protocol"]
        json_file = export_crop_centers(
            cochlea, component_labels, out_dir, suffix=suffix, force_overwrite=force_overwrite,
        )
        cochlea_names.append(f"{cochlea}_{suffix}")
        json_files.append(json_file)
    return cochlea_names, json_files


def main():
    parser = argparse.ArgumentParser(
        description="Script to create JSON dictionaries and crops for IHC annotations.")

    parser.add_argument("-c", "--cochlea", nargs="+", type=str, default=TRAINING_COCHLEAE,
                        help="Cochlea(e) to analyze.")
    parser.add_argument("-j", "--json_dir", type=str, required=True,
                        help="Output directory for JSON dictionaries which feature parameters for crop extraction.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite JSON dictionaries.")
    parser.add_argument("-o", "--output_folder", type=str, default=None,
                        help="Output directory for extracted crops in TIF format.")
    args = parser.parse_args()

    # create JSON dictionaries with parameters for crop extraction
    cochlea_names, json_files = create_crop_dictionaries(
        cochleae=args.cochlea,
        out_dir=args.json_dir,
        force_overwrite=args.force,
    )

    # from flamingo_tools.analysis.training_data_utils import create_xlsx_for_crops
    # save_path = os.path.join(args.json_dir, "ihc_center_location.xlsx")
    # export_position_for_crop_centers(json_files, save_path=save_path)

    if args.output_folder is not None:
        # extract crops for cochlea(e)
        os.makedirs(args.output_folder, exist_ok=True)
        for (cochlea_name, json_file) in zip(cochlea_names, json_files):
            print(f"Extracting crops for cochlea {cochlea_name}.")
            output_dir = os.path.join(args.output_folder, cochlea_name)
            os.makedirs(output_dir, exist_ok=True)
            extract_block_json_wrapper(output_path=output_dir, json_file=json_file, s3=True, input_key="s0")


if __name__ == "__main__":
    main()
