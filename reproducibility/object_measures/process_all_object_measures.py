import json
import os
import subprocess
import zarr

import flamingo_tools.s3_utils as s3_utils

OUTPUT_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements3"  # noqa
# JSON_ROOT = "/user/pape41/u12086/Work/my_projects/flamingo-tools/reproducibility/object_measures"
JSON_ROOT = "/user/schilling40/u15000/flamingo-tools/reproducibility/object_measures"
COCHLEAE = [
    "M_LR_000143_L",
    "M_LR_000144_L",
    "M_LR_000145_L",
    "M_LR_000153_L",
    "M_LR_000155_L",
    "M_LR_000189_L",
    "M_LR_000143_R",
    "M_LR_000144_R",
    "M_LR_000145_R",
    "M_LR_000153_R",
    "M_LR_000155_R",
    "M_LR_000189_R",
]


def process_cochlea(cochlea, start_slurm):
    short_name = cochlea.replace("_", "").replace("0", "")

    # Check if this cochlea has been processed already.
    output_name = cochlea.replace("_", "-")
    output_path = os.path.join(OUTPUT_ROOT, f"{output_name}_GFP_SGN-v2_object-measures.tsv")
    if os.path.exists(output_path):
        print(cochlea, "has been processed already.")
        return

    # Check if the raw data for this cochlea is accessible.
    img_name = f"{cochlea}/images/ome-zarr/GFP.ome.zarr"
    img_path, _ = s3_utils.get_s3_path(img_name)
    try:
        zarr.open(img_path, mode="r")
    except Exception:
        print("The data for", cochlea, "at", img_name, "does not exist.")
        return

    # Then generate the json file if it does not yet exist.
    template_path = os.path.join(JSON_ROOT, "ChReef_MLR143L.json")
    with open(template_path, "r") as f:
        json_template = json.load(f)

    json_path = os.path.join(JSON_ROOT, f"ChReef_{short_name}.json")
    if not os.path.exists(json_path):
        print("Write json to", json_path)
        # TODO: We may need to replace the component list for some.
        json_template[0]["dataset_name"] = cochlea
        with open(json_path, "w") as f:
            json.dump(json_template, f, indent=4)

    print(cochlea, "is not yet processed")
    # Then start the slurm job.
    if not start_slurm:
        return

    print("Submit slurm job for", cochlea)
    subprocess.run(["sbatch", "slurm_template.sbatch", json_path, OUTPUT_ROOT])


def main():
    start_slurm = True
    for cochlea in COCHLEAE:
        process_cochlea(cochlea, start_slurm)


if __name__ == "__main__":
    main()
