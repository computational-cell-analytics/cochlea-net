import argparse
import os

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import get_s3_path, BUCKET_NAME, SERVICE_ENDPOINT
from flamingo_tools.postprocessing.synapse_per_ihc_utils import SYNAPSE_DICT

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
OUT_DIR = f"{COCHLEA_DIR}/mobie_project/cochlea-lightsheet/tables/syn_per_ihc"


def add_syn_per_ihc(args):
    syn_limit = 25

    if args.output_folder is None:
        out_dir = OUT_DIR
    else:
        out_dir = args.output_folder

    for cochlea in args.cochlea:
        if args.seg_version is None:
            seg_version = SYNAPSE_DICT[cochlea]["ihc_table_name"]
        else:
            seg_version = "IHC_v4b"

        print(f"Evaluating cochlea {cochlea}.")

        ihc_version = seg_version.split("_")[1]
        syn_per_ihc_dir = f"{COCHLEA_DIR}/predictions/synapses/ihc_counts_{ihc_version}"

        if args.component_list is None:
            component_list = SYNAPSE_DICT[cochlea].get("component_list", [1])
        else:
            component_list = [1]

        s3_path = os.path.join(f"{cochlea}", "tables", seg_version, "default.tsv")
        tsv_path, fs = get_s3_path(s3_path, bucket_name=BUCKET_NAME,
                                   service_endpoint=SERVICE_ENDPOINT)
        with fs.open(tsv_path, 'r') as f:
            ihc_table = pd.read_csv(f, sep="\t")

        # synapse_table
        syn_path = os.path.join(syn_per_ihc_dir, f"ihc_count_{cochlea}.tsv")
        if not os.path.isfile(syn_path):
            print(f"Skipping cochlea {cochlea}. Synapse table {syn_path} does not exist.")
            continue

        with open(syn_path, 'r') as f:
            syn_table = pd.read_csv(f, sep="\t")

        syn_per_IHC = [-1 for _ in range(len(ihc_table))]
        ihc_table.loc[:, "syn_per_IHC"] = syn_per_IHC

        ihc_table.loc[ihc_table['component_labels'].isin(component_list), 'syn_per_IHC'] = 0
        zero_syn = ihc_table[ihc_table["syn_per_IHC"] == 0]
        print(f"Total IHC in component: {len(zero_syn)}")

        for label_id, syn_count in zip(list(syn_table["label_id"]), list(syn_table["synapse_count"])):
            ihc_table.loc[ihc_table["label_id"] == label_id, "syn_per_IHC"] = syn_count
        zero_syn = ihc_table[ihc_table["syn_per_IHC"] > syn_limit]
        print(f"IHC in component with more than 25 synapses: {len(zero_syn)}")
        zero_syn = ihc_table[ihc_table["syn_per_IHC"] == 0]
        print(f"IHC in component without synapses: {len(zero_syn)}")

        syn_per_IHC = list(ihc_table.loc[ihc_table['component_labels'].isin(component_list), 'syn_per_IHC'])

        if args.ihc_syn:
            syn_per_IHC = [s for s in syn_per_IHC if s != 0]

        print(f"Mean syn_per_IHC: {round(sum(syn_per_IHC) / len(syn_per_IHC), 2)}")
        print(f"Stdv syn_per_IHC: {round(np.std(syn_per_IHC), 2)}")
        out_path = os.path.join(out_dir, cochlea + "_syn-per-ihc.tsv")
        ihc_table.to_csv(out_path, sep="\t", index=False)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cochlea", type=str, nargs="+", default=list(SYNAPSE_DICT.keys()),
                        help="Cochlea(e) to process.")
    parser.add_argument("-o", "--output_folder", type=str, default=None, help="Path to output folder.")
    parser.add_argument("-s", "--seg_version", type=str, default=None, help="Path to output folder.")
    parser.add_argument("--ihc_syn", action="store_true", help="Consider only IHC with synapses.")
    parser.add_argument("--component_list", type=int, nargs="+", default=None,
                        help="List of IHC components.")

    args = parser.parse_args()

    add_syn_per_ihc(args)


if __name__ == "__main__":
    main()
