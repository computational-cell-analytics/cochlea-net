import json
import os

import numpy as np
import pandas as pd
from flamingo_tools.s3_utils import create_s3_target, BUCKET_NAME

COCHLEAE = ["G_EK_000233_L"]
SGN_COMPONENTS = {}
IHC_COMPONENTS = {"G_EK_000233_L": [1, 2, 3, 4, 5, 8]}


def open_json(fs, path):
    s3_path = os.path.join(BUCKET_NAME, path)
    with fs.open(s3_path, "r") as f:
        content = json.load(f)
    return content


def open_tsv(fs, path):
    s3_path = os.path.join(BUCKET_NAME, path)
    with fs.open(s3_path, "r") as f:
        table = pd.read_csv(f, sep="\t")
    return table


def measure_sgns(fs):
    print("SGNs:")
    seg_name = "SGN_v2"
    for dataset in COCHLEAE:
        print("Cochlea:", dataset)
        dataset_info = open_json(fs, os.path.join(dataset, "dataset.json"))
        sources = dataset_info["sources"]
        assert seg_name in sources

        source_info = sources[seg_name]["segmentation"]
        table_path = source_info["tableData"]["tsv"]["relativePath"]
        table = open_tsv(fs, os.path.join(dataset, table_path, "default.tsv"))

        component_labels = table.component_labels.values
        component_ids = SGN_COMPONENTS.get(dataset, [1])
        n_sgns = np.isin(component_labels, component_ids).sum()
        print("N-SGNs:", n_sgns)


def measure_ihcs(fs):
    print("IHCs:")
    seg_name = "IHC_v5"
    for dataset in COCHLEAE:
        print("Cochlea:", dataset)
        dataset_info = open_json(fs, os.path.join(dataset, "dataset.json"))
        sources = dataset_info["sources"]
        assert seg_name in sources

        source_info = sources[seg_name]["segmentation"]
        table_path = source_info["tableData"]["tsv"]["relativePath"]
        table = open_tsv(fs, os.path.join(dataset, table_path, "default.tsv"))

        component_labels = table.component_labels.values
        component_ids = IHC_COMPONENTS.get(dataset, [1])
        n_ihcs = np.isin(component_labels, component_ids).sum()
        print("N-IHCs:", n_ihcs)


def measure_synapses(fs):
    print("Synapses:")
    spot_name = "synapse_v3_ihc_v5"
    seg_name = "IHC_v5"
    for dataset in COCHLEAE:
        print("Cochlea:", dataset)
        dataset_info = open_json(fs, os.path.join(dataset, "dataset.json"))
        sources = dataset_info["sources"]
        assert spot_name in sources

        source_info = sources[spot_name]["spots"]
        table_path = source_info["tableData"]["tsv"]["relativePath"]
        table = open_tsv(fs, os.path.join(dataset, table_path, "default.tsv"))

        source_info = sources[seg_name]["segmentation"]
        table_path = source_info["tableData"]["tsv"]["relativePath"]
        ihc_table = open_tsv(fs, os.path.join(dataset, table_path, "default.tsv"))

        ihc_components = IHC_COMPONENTS.get(dataset, [1])
        valid_ihcs = ihc_table.label_id[ihc_table.component_labels.isin(ihc_components)]
        table = table[table.matched_ihc.isin(valid_ihcs)]

        _, syn_count = np.unique(table.matched_ihc.values, return_counts=True)
        print("Avg Syn. per IHC:")
        print(np.mean(syn_count), "+-", np.std(syn_count))


def main():
    fs = create_s3_target()
    measure_sgns(fs)
    measure_ihcs(fs)
    measure_synapses(fs)


if __name__ == "__main__":
    main()
