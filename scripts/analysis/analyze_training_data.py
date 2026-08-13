import argparse
import json
import os
import re
import sys
from typing import List, Optional

import pandas as pd

from flamingo_tools.analysis.training_data_utils import (
    add_metadata_to_crop_table,
    add_metadata_to_crop_table_synapses,
)

DOC_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "doc", "data"))

# Tables analyzed with the --all flag, together with the directories containing their crops.
TRAINING_DATA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/training_data"
DEFAULT_TABLES = {
    "synapses_v5.tsv": {
        "data_dir": os.path.join(TRAINING_DATA_DIR, "synapses/training_data/v5"),
        "test_dir": os.path.join(TRAINING_DATA_DIR, "synapses/test_data/v5"),
    },
    "IHC_v11.tsv": {
        "data_dir": os.path.join(TRAINING_DATA_DIR, "IHC/IHC_v11_2026-07"),
        "label_dir": None,
    },
    "SGN_v2.tsv": {
        "data_dir": os.path.join(TRAINING_DATA_DIR, "SGN/2025-05_supervised"),  # noqa
        "label_dir": None,
    },
}

TABLE_TYPES = ["IHC", "SGN", "synapse"]

# Species encoded in the first component of the standardized crop name.
SPECIES_PREFIXES = {"M": "mouse", "G": "gerbil"}
UNKNOWN_SPECIES = "unknown"


def table_type(table_path: str) -> str:
    """Derive the type of training data from the file name of the table.

    Args:
        table_path: File path to TSV table.

    Returns:
        The type of training data. Either 'IHC', 'SGN', or 'synapse'.
    """
    file_name = os.path.basename(table_path)
    matches = [t for t in TABLE_TYPES if t.lower() in file_name.lower()]
    if len(matches) != 1:
        raise ValueError(
            f"The name of the table '{file_name}' must contain exactly one of {TABLE_TYPES}, but it contains {matches}."
        )
    return matches[0]


def species_from_name(standardized_name: str) -> str:
    """Derive the species from the first component of a standardized crop name.

    The components are separated by '-' or '_', because some crops keep the older underscore form.

    Args:
        standardized_name: The standardized name of a crop.

    Returns:
        The species that matches the first component, or 'unknown' for a prefix outside
        SPECIES_PREFIXES.
    """
    prefix = re.split(r"[-_]", standardized_name)[0]
    return SPECIES_PREFIXES.get(prefix, UNKNOWN_SPECIES)


def resolve_layout(data_dir: str) -> Optional[str]:
    """Check the layout of the data directory and locate the annotations.

    Args:
        data_dir: Directory containing the crops.

    Returns:
        The directory containing the annotations for an 'images'-'labels' layout.
        None for a 'train'-'val' layout, where the 'Dataset' column gives the sub-directory.
    """
    if all(os.path.isdir(os.path.join(data_dir, sub_dir)) for sub_dir in ("train", "val")):
        return None
    if all(os.path.isdir(os.path.join(data_dir, sub_dir)) for sub_dir in ("images", "labels")):
        return os.path.join(data_dir, "labels")
    raise ValueError(
        f"The data directory '{data_dir}' contains neither the sub-directories 'train' and 'val' "
        "nor the sub-directories 'images' and 'labels'."
    )


def _count_splits(df: pd.DataFrame, count_column: str) -> dict:
    splits = {name: df[df["Dataset"] == name] for name in ("train", "val", "test")}
    instances = {name: int(split[count_column].sum()) for name, split in splits.items()}

    counts = {
        "n_crops": len(splits["train"]) + len(splits["val"]),
        "n_crops_train": len(splits["train"]),
        "n_crops_val": len(splits["val"]),
        "n_instances": instances["train"] + instances["val"],
        "n_instances_train": instances["train"],
        "n_instances_val": instances["val"],
    }
    if len(splits["test"]) != 0:
        counts["n_crops_test"] = len(splits["test"])
        counts["n_instances_test"] = instances["test"]
    return counts


def summarize_table(table_path: str, data_type: str) -> Optional[dict]:
    """Count the crops and instances per dataset of an analyzed table.

    Args:
        table_path: File path to TSV table.
        data_type: The type of training data. Either 'IHC', 'SGN', or 'synapse'.

    Returns:
        The number of crops and instances for training and validation, and for testing if present.
        The entry 'species' repeats these counts for each species present in the table.
        None if the table contains crops without an instance count, which would understate the totals.
    """
    df = pd.read_csv(table_path, sep="\t")
    # For segmentations the instances below the size threshold are artifacts and are not counted.
    count_column = "n_samples" if data_type == "synapse" else "n_samples[>=1000px]"

    unmeasured = df.loc[df[count_column].isna(), "Original"].tolist()
    if unmeasured:
        print(f"{os.path.basename(table_path)}: no instance count for {len(unmeasured)} crops: {unmeasured}")
        return None

    species = df["Standardized"].astype(str).map(species_from_name)
    unknown = df.loc[species == UNKNOWN_SPECIES, "Original"].tolist()
    if unknown:
        print(f"{os.path.basename(table_path)}: unknown species prefix for {len(unknown)} crops: {unknown}")

    summary = _count_splits(df, count_column)
    summary["species"] = {name: _count_splits(group, count_column) for name, group in sorted(df.groupby(species))}
    return summary


def update_overview(output_path: str, key: str, summary: dict) -> None:
    """Add the summary of a table to the overview file and sort the entries alphabetically.

    Args:
        output_path: File path to the JSON overview.
        key: Name of the table with the training data information.
        summary: The crop and instance counts of the table.
    """
    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            overview = json.loads(f.read())
    else:
        overview = {}

    overview[key] = summary
    with open(output_path, "w") as f:
        json.dump(dict(sorted(overview.items())), f, indent=2)
        f.write("\n")


def analyze_table(
    table_path: str,
    data_dir: str,
    output_path: str,
    label_dir: Optional[str] = None,
    test_dir: Optional[str] = None,
    recompute: bool = False,
    n_workers: int = 4,
) -> List[str]:
    """Add the metadata of the crops to a training data table and summarize it in the overview.

    The overview is only updated if all crops of the table were measured.

    Args:
        table_path: File path to TSV table.
        data_dir: Directory containing the crops.
        output_path: File path to the JSON overview.
        label_dir: Directory containing the annotations. Determined from the layout of data_dir if not given.
        test_dir: Directory containing the test crops. Only used for synapses.
        recompute: Measure all crops again, including the rows that are already complete.
        n_workers: Number of threads for reading the annotation crops.

    Returns:
        The names of the crops that could not be measured.
    """
    data_type = table_type(table_path)
    if data_type == "synapse":
        resolve_layout(data_dir)
        failed = add_metadata_to_crop_table_synapses(table_path, data_dir, test_dir=test_dir, recompute=recompute)
    else:
        if label_dir is None:
            label_dir = resolve_layout(data_dir)
        failed = add_metadata_to_crop_table(
            table_path, data_dir, label_dir=label_dir, recompute=recompute, n_workers=n_workers,
        )

    table_name = os.path.basename(table_path)
    summary = summarize_table(table_path, data_type)
    if summary is None:
        print(f"{table_name}: incomplete, the overview is not updated.")
    else:
        update_overview(output_path, os.path.splitext(table_name)[0], summary)
        print(f"{table_name}: {summary}")
    return failed


def main():
    parser = argparse.ArgumentParser(
        description="Analyze the tables documenting the training data and summarize them in a JSON overview."
    )
    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Input path to a TSV table with training data information.")
    parser.add_argument("--all", action="store_true",
                        help=f"Analyze the predefined tables in {DOC_DATA_DIR}. Default behavior without --input.")
    parser.add_argument("-d", "--data_dir", type=str, default=None,
                        help="Directory containing the crops. Features the sub-directories "
                             "'train' and 'val', or 'images' and 'labels'. Required with --input.")
    parser.add_argument("--test_dir", type=str, default=None,
                        help="Directory containing the test crops. Only used for synapses.")
    parser.add_argument("-o", "--output", type=str, default=os.path.join(DOC_DATA_DIR, "overview.json"),
                        help=f"Output path for the JSON overview. Default: {os.path.join(DOC_DATA_DIR, 'overview.json')}")  # noqa
    parser.add_argument("--recompute", action="store_true",
                        help="Measure all crops again, including the rows that are already complete.")
    parser.add_argument("-n", "--n_workers", type=int, default=4,
                        help="Number of threads for reading the annotation crops. Default: 4")

    args = parser.parse_args()

    if args.all or args.input is None:
        tables = {os.path.join(DOC_DATA_DIR, name): kwargs for name, kwargs in DEFAULT_TABLES.items()}
    else:
        if args.data_dir is None:
            parser.error("--data_dir is required when a single table is analyzed with --input.")
        tables = {args.input: {"data_dir": args.data_dir, "test_dir": args.test_dir}}

    failed = {}
    for table_path, kwargs in tables.items():
        failed_crops = analyze_table(
            table_path,
            output_path=args.output,
            recompute=args.recompute,
            n_workers=args.n_workers,
            **kwargs,
        )
        if failed_crops:
            failed[os.path.basename(table_path)] = failed_crops

    # Report the failures after all tables were processed, so that one bad crop does not cost a whole run.
    if failed:
        for table_name, failed_crops in failed.items():
            print(f"{table_name}: {len(failed_crops)} crops could not be measured: {failed_crops}")
        sys.exit(1)


if __name__ == "__main__":
    main()
