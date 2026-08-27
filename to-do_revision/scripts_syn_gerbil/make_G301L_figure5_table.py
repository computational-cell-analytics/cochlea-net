"""Create the Figure 5 per-IHC table from the corrected local G301L v3 result.

The production detection retains assignments out to 8 um. The manuscript count uses a 3 um
cutoff and IHC components 1-13 from the finalized dilated-mask IHC_v11 segmentation. Zero-count
IHCs are deliberately retained in this table; Figure 5 applies its explicit zero exclusion when
loading the table, keeping the source measurement complete and reusable.
"""

import argparse
import os

import pandas as pd


WS = "/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools"
COCHLEA = "G_LR_000301_L"
RESULT_FOLDER = os.path.join(
    WS, "synapses-v3", f"{COCHLEA}_IHC_v11_dilated_mask1",
)
DETECTIONS = os.path.join(RESULT_FOLDER, "synapse_detection_filtered.tsv")
IHC_TABLE = os.path.join(
    WS, "prediction/G301L/IHC_v11_dilated_mask1/default_components.tsv",
)
DEFAULT_OUTPUT = os.path.join(RESULT_FOLDER, f"ihc_count_{COCHLEA}.tsv")
COMPONENTS = list(range(1, 14))
MAX_DISTANCE = 3.0


def make_table(detections_path: str, ihc_path: str, output_path: str) -> pd.DataFrame:
    detections = pd.read_csv(detections_path, sep="\t")
    ihcs = pd.read_csv(ihc_path, sep="\t")

    required_detection_columns = {"matched_ihc", "distance_to_ihc"}
    required_ihc_columns = {"label_id", "component_labels"}
    if missing := required_detection_columns.difference(detections.columns):
        raise ValueError(f"Detection table is missing columns: {sorted(missing)}")
    if missing := required_ihc_columns.difference(ihcs.columns):
        raise ValueError(f"IHC table is missing columns: {sorted(missing)}")

    selected = ihcs.loc[ihcs["component_labels"].isin(COMPONENTS)].copy()
    if len(selected) != 946:
        raise ValueError(f"Expected 946 IHCs in components 1-13, found {len(selected)}")
    if selected["label_id"].duplicated().any():
        raise ValueError("The selected IHC table contains duplicate label IDs")

    selected_ids = selected["label_id"].astype(int)
    within_cutoff = detections.loc[
        (detections["distance_to_ihc"] <= MAX_DISTANCE)
        & detections["matched_ihc"].isin(selected_ids)
    ]
    counts = within_cutoff["matched_ihc"].astype(int).value_counts()

    result = pd.DataFrame({
        "label_id": selected_ids,
        "synapse_count": selected_ids.map(counts).fillna(0).astype(int),
        # Keep the historical typo for schema compatibility with measure_synapses.py outputs.
        "snyapse_table": "synapse_v3_ihc_v11",
        "ihc_table": "IHC_v11",
        "max_dist": int(MAX_DISTANCE),
    })

    if int(result["synapse_count"].sum()) != len(within_cutoff):
        raise RuntimeError("Per-IHC counts do not sum to the selected detection count")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    temporary_path = f"{output_path}.tmp"
    result.to_csv(temporary_path, sep="\t", index=False)
    os.replace(temporary_path, output_path)

    positive = result.loc[result["synapse_count"] > 0, "synapse_count"]
    print(f"Output: {output_path}")
    print(f"Selected IHCs: {len(result)}")
    print(f"Mapped synapses at <= {MAX_DISTANCE:g} um: {int(result.synapse_count.sum())}")
    print(f"Zero-count IHCs: {int((result.synapse_count == 0).sum())}")
    print(f"Mean over all selected IHCs: {result.synapse_count.mean():.6f}")
    print(f"Mean after excluding zeros: {positive.mean():.6f}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", default=DETECTIONS)
    parser.add_argument("--ihc-table", default=IHC_TABLE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    make_table(args.detections, args.ihc_table, args.output)


if __name__ == "__main__":
    main()
