"""Combine per-cochlea outputs from the v3 normalization experiment."""

import argparse
import json
import os

import bioimage_cpp as bic
import numpy as np
import pandas as pd
import zarr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    count_tables, stat_tables, differences = [], [], []
    for cochlea, info in manifest["selection"].items():
        folder = os.path.join(args.output_root, cochlea)
        complete = os.path.join(folder, "complete.json")
        if not os.path.exists(complete):
            raise RuntimeError(f"Experiment is not complete for {cochlea}: {complete} is missing")

        count_tables.append(pd.read_csv(os.path.join(folder, "threshold_counts.tsv"), sep="\t"))
        stats = pd.read_csv(os.path.join(folder, "normalization_stats.tsv"), sep="\t")
        stats.insert(0, "cochlea", cochlea)
        stat_tables.append(stats)

        global_prediction = zarr.open(
            os.path.join(folder, "predictions_global.zarr"), mode="r"
        )["prediction"]
        local_prediction = zarr.open(
            os.path.join(folder, "predictions_local.zarr"), mode="r"
        )["prediction"]
        blocking = bic.utils.Blocking(
            [0, 0, 0], list(global_prediction.shape), list(manifest["block_shape"])
        )
        for record in info["blocks"]:
            block = blocking.get_block(int(record["block_id"]))
            bb = tuple(slice(int(beg), int(end)) for beg, end in zip(block.begin, block.end))
            global_block = np.asarray(global_prediction[bb])
            local_block = np.asarray(local_prediction[bb])
            delta = np.abs(global_block - local_block)
            differences.append(
                {
                    "cochlea": cochlea,
                    "block_id": int(record["block_id"]),
                    "regime": record["regime"],
                    "max_abs_difference": float(delta.max()),
                    "mean_abs_difference": float(delta.mean()),
                    "n_different_voxels": int(np.count_nonzero(delta)),
                }
            )

    counts = pd.concat(count_tables, ignore_index=True)
    stats = pd.concat(stat_tables, ignore_index=True)
    differences = pd.DataFrame(differences)
    counts.to_csv(os.path.join(args.output_root, "combined_threshold_counts.tsv"), sep="\t", index=False)
    stats.to_csv(os.path.join(args.output_root, "combined_normalization_stats.tsv"), sep="\t", index=False)
    differences.to_csv(
        os.path.join(args.output_root, "combined_prediction_differences.tsv"), sep="\t", index=False
    )

    aggregate = counts.groupby(
        ["cochlea", "regime", "normalization", "threshold"], as_index=False
    )[["n_all", "n_within_3um", "n_target_ihc"]].sum()
    aggregate.to_csv(os.path.join(args.output_root, "aggregate_counts.tsv"), sep="\t", index=False)

    print("Aggregate detections within 3 um of an IHC:")
    print(
        aggregate.pivot_table(
            index=["cochlea", "regime", "normalization"],
            columns="threshold",
            values="n_within_3um",
        ).to_string()
    )
    print("\nPrediction difference maxima by cochlea:")
    print(differences.groupby("cochlea")["max_abs_difference"].max().to_string())


if __name__ == "__main__":
    main()
