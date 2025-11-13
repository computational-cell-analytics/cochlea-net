import json
import os

import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
from sgn_subtypes import COCHLEAE


def get_overview(cochlea, seg_name, component_ids):
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]

    # Load the segmentation table.
    seg_source = sources[seg_name]
    table_folder = os.path.join(
        BUCKET_NAME, cochlea, seg_source["segmentation"]["tableData"]["tsv"]["relativePath"]
    )
    table_content = s3.open(os.path.join(table_folder, "default.tsv"), mode="rb")
    table = pd.read_csv(table_content, sep="\t")

    table = table[table.component_labels.isin(component_ids)]
    subtype_labels = table.subtype_label.dropna().values

    n_sgns = float(len(subtype_labels))
    label_names, label_counts = np.unique(subtype_labels, return_counts=True)

    print("N-SGNs:", int(n_sgns))
    for name, count in zip(label_names, label_counts):
        print(name, ":", np.round(100 * count / n_sgns, 2), "%")


def run_overview():
    for cochlea, info in COCHLEAE.items():
        if cochlea == "M_AMD_Runx1_L":
            continue
        seg_name = info["output_seg"] if "output_seg" in info else info["seg_data"]
        component_ids = info.get("component_list", [1])
        print(cochlea)
        # print(component_ids)
        try:
            get_overview(cochlea, seg_name, component_ids)
        except Exception as e:
            print("Failed due to", e)
        print()


def main():
    # cochlea = "M_LR_000099_L"
    # get_overview(cochlea, seg_name="PV_SGN_v2")
    run_overview()


if __name__ == "__main__":
    main()
