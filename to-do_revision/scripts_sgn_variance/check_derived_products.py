"""Check that a segmentation and its component table exist and look sound.

Used by remove_converted_source.sh when the converted prediction has already been deleted and can
no longer be compared against the original. The watershed reads every voxel of every shard to
produce these, so their existence is evidence that the converted array was complete and readable --
a different argument from the sample comparison, and the reason that path is opt-in.
"""

import argparse
import os

import pandas as pd

MIN_COMPONENT_SIZE = 1000


def check_derived_products(folder: str) -> None:
    segmentation = os.path.join(folder, "segmentation.zarr", "segmentation", "zarr.json")
    if not os.path.exists(segmentation):
        raise SystemExit(f"no {segmentation}")

    table_path = os.path.join(folder, "default_components.tsv")
    if not os.path.exists(table_path):
        raise SystemExit(f"no {table_path}")
    table = pd.read_csv(table_path, sep="\t")
    if "component_labels" not in table.columns:
        raise SystemExit(f"{table_path} has no component_labels column")

    in_component_1 = int((table.component_labels == 1).sum())
    if in_component_1 < MIN_COMPONENT_SIZE:
        raise SystemExit(f"only {in_component_1} objects in component 1")

    print(f"derived products present: {len(table)} objects, {in_component_1} in component 1; "
          "the watershed read every voxel of every shard to produce them")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("folder", help="The version folder, holding segmentation.zarr.")
    args = parser.parse_args()
    check_derived_products(args.folder)


if __name__ == "__main__":
    main()
