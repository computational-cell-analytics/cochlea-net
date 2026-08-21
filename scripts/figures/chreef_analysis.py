import json
import os
import pickle

# import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
# import tifffile
# import zarr
# from matplotlib import cm, colors

from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target

from util import cohort_cochleae

INTENSITY_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet/tables/measurements"  # noqa
# The cochleae for the ChReef analysis.
COCHLEAE = cohort_cochleae("chreef_mouse")


def download_data():
    s3 = create_s3_target()
    source_name = "SGN_v2"

    cache_path = "./chreef_data.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    chreef_data = {}
    for cochlea in COCHLEAE:
        print("Processsing cochlea:", cochlea)
        content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
        info = json.loads(content.read())
        sources = info["sources"]

        # Load the seg table and filter the compartments.
        source = sources[source_name]["segmentation"]
        rel_path = source["tableData"]["tsv"]["relativePath"]
        table_content = s3.open(os.path.join(BUCKET_NAME, cochlea, rel_path, "default.tsv"), mode="rb")
        table = pd.read_csv(table_content, sep="\t")

        # May need to be adjusted for some cochleae.
        table = table[table.component_labels == 1]
        # The relevant values for analysis.
        try:
            values = table[["label_id", "length[µm]", "frequency[kHz]", "marker_labels"]]
        except KeyError:
            print("Could not find the values for", cochlea, "it will be skippped.")
            continue

        fname = f"{cochlea.replace('_', '-')}_GFP_SGN-v2_object-measures.tsv"
        intensity_file = os.path.join(INTENSITY_ROOT, fname)
        assert os.path.exists(intensity_file), intensity_file
        intensity_table = pd.read_csv(intensity_file, sep="\t")
        values = values.merge(intensity_table, on="label_id")

        chreef_data[cochlea] = values

    with open(cache_path, "wb") as f:
        chreef_data = pickle.dump(chreef_data, f)
    return chreef_data


def analyze_transduction(chreef_data):
    breakpoint()
    pass


def main():
    chreef_data = download_data()
    analyze_transduction(chreef_data)


if __name__ == "__main__":
    main()
