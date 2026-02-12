#!/usr/bin/python
# -- coding: utf-8 --
import os
import subprocess


mobie_dir = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"

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
seg = "SGN_v2"

COCHLEAE = [
    "M_LR_000226_L",
    "M_LR_000227_L",
    "M_LR_000226_R",
    "M_LR_000227_R",
]
seg = "SGN_v2"
# seg = "IHC_v4c"


COCHLEAE = [
    "M_AMD_N62_L",
    "M_AMD_Runx1_L",
    "M_LR_000099_L",
    "M_LR_000214_L",
]
seg = "PV_SGN_v2"

if "SGN" in seg:
    tonotopic_dir = os.path.join(mobie_dir, "tables/tonotopic_sgn")
elif "IHC" in seg:
    tonotopic_dir = os.path.join(mobie_dir, "tables/tonotopic_ihc")
else:
    raise ValueError("Choose either a segmentation channel with 'SGN' or 'IHC'.")

dry_run = False

for cochlea in COCHLEAE:
    cochlea_table = cochlea.replace('_', '-')
    seg_table = seg.replace('_', '-')
    in_path = os.path.join(tonotopic_dir, f"{cochlea_table}_{seg_table}.tsv")
    out_path = f"cochlea-lightsheet:cochlea-lightsheet/{cochlea}/tables/{seg}/default.tsv"
    print(out_path)

    # "--dry-run"
    subprocess.run(["rclone", "copyto", in_path, out_path], stdout=subprocess.PIPE).stdout.decode('utf-8')
