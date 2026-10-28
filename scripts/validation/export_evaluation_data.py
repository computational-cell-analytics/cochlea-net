"""Export the evaluation data of Figure 2 (SGN, IHC and synapses) for publication.

The cochlea ID at the start of each crop name is replaced with the sample alias of the CochleaNet paper.
The annotator initials are replaced with the indexes 01 to 03, in alphabetical order of the initials.
Only images and annotations are exported. The network outputs are not part of the export.
"""
import argparse
import os
import shutil
from glob import glob

import pandas as pd
import tifffile
import zarr

# Aliases of the sheet 'Samples' in the sample table of the CochleaNet paper.
ALIASES = {
    "MLR226L": "M_01L",
    "MLR226R": "M_01R",
    "MLR227L": "M_02L",
    "MLR227R": "M_02R",
    "MAMD58L": "M_21L",
    "MLR169R": "M_28R",
}
ANNOTATORS = {"AMD": "01", "EK": "02", "LR": "03"}

_SGN = "SGN_consensus_annotation"
_IHC = "IHC_consensus_annotation/F1ValidationIHCs"
_SYNAPSES = "synapses_consensus_annotation"

# Folders relative to the input root. '{}' in an annotator folder stands for the initials.
# These are the folders that scripts/validation/*/evaluate_consensus.py and run_evaluation.py read.
DATASETS = {
    "SGN_evaluation": {
        "images": f"{_SGN}/for_consensus_annotation",
        "image_extension": ".tif",
        "consensus": f"{_SGN}/final_annotations/final_consensus_annotations",
        "annotators": f"{_SGN}/Annotations{{}}",
        # The two rescaled slabs were annotated again after the rescaling ('ROOT2' in evaluate_consensus.py).
        "annotators_rescaled": f"{_SGN}/for_consensus_annotation/Annotations{{}}",
    },
    "IHC_evaluation": {
        "images": _IHC,
        "image_extension": ".tif",
        "consensus": f"{_IHC}/consensus_annotation",
        "annotators": f"{_IHC}/Annotations_{{}}",
    },
    "synapses_evaluation": {
        # The input of scripts/validation/synapses/prediction.py, with the arrays 'raw' and 'raw_ihc'.
        "images": f"{_SYNAPSES}/synapses_test_data/v5/images",
        "image_extension": ".zarr",
        "consensus": f"{_SYNAPSES}/Synapses_2026-04/consensus_annotation",
        "annotators": f"{_SYNAPSES}/Synapses_2026-04/for_consensus_annotations_synapses_{{}}/labels",
    },
}


def published_name(stem: str) -> str:
    """Replace the cochlea ID with the alias. '_rescaled' is dropped, because it only documents the processing."""
    cochlea_id, crop = stem.split("_", 1)
    return f"{ALIASES[cochlea_id]}_{crop}".removesuffix("_rescaled")


def find_annotation(folder: str, stem: str) -> str:
    matches = [path for path in glob(os.path.join(folder, f"{stem}*.csv")) if "negative" not in os.path.basename(path)]
    if len(matches) != 1:
        raise ValueError(f"Expected one annotation of {stem} in {folder}, found {matches}")
    return matches[0]


def anonymize_annotator(value: str) -> str:
    """Map 'consensus' to itself and 'AMD' or 'Annotations_AMD' to the index of the annotator."""
    return value if value == "consensus" else ANNOTATORS[value.split("_")[-1]]


def export_consensus(source: str, target: str) -> int:
    """Copy a consensus annotation with anonymized annotators and return the number of consensus points."""
    table = pd.read_csv(source, dtype=str, keep_default_na=False)
    if "annotator" not in table.columns:
        shutil.copy2(source, target)
        return len(table)
    table["annotator"] = table["annotator"].map(anonymize_annotator)
    table.to_csv(target, index=False)
    return int((table["annotator"] == "consensus").sum())


def image_shape(path: str) -> tuple:
    if path.endswith(".zarr"):
        return zarr.open(path, mode="r")["raw"].shape
    with tifffile.TiffFile(path) as f:
        return f.series[0].shape


def export_dataset(dataset: str, input_root: str, output_root: str) -> None:
    folders = DATASETS[dataset]
    output_dir = os.path.join(output_root, dataset)
    annotator_dirs = {
        initials: os.path.join(output_dir, f"annotator_{index}") for initials, index in ANNOTATORS.items()
    }
    # Fail on an existing folder, so that a second run cannot leave stale files behind.
    for folder in [os.path.join(output_dir, "images"), os.path.join(output_dir, "consensus"), *annotator_dirs.values()]:
        os.makedirs(folder)

    rows = []
    for consensus_path in sorted(glob(os.path.join(input_root, folders["consensus"], "*.csv"))):
        stem = os.path.splitext(os.path.basename(consensus_path))[0]
        name = published_name(stem)

        image = os.path.join(input_root, folders["images"], stem + folders["image_extension"])
        image_target = os.path.join(output_dir, "images", name + folders["image_extension"])
        if os.path.isdir(image):
            shutil.copytree(image, image_target)
        else:
            shutil.copy2(image, image_target)

        n_consensus = export_consensus(consensus_path, os.path.join(output_dir, "consensus", f"{name}.csv"))

        annotators = folders["annotators_rescaled"] if "rescaled" in stem else folders["annotators"]
        for initials, target_dir in annotator_dirs.items():
            source = find_annotation(os.path.join(input_root, annotators.format(initials)), stem)
            shutil.copy2(source, os.path.join(target_dir, f"{name}.csv"))

        alias = ALIASES[stem.split("_", 1)[0]]
        rows.append({
            "Name": name, "Alias": alias, "Species": "mouse",
            "Shape_zyx": str(tuple(image_shape(image_target))), "n_consensus": n_consensus,
        })

    table = pd.DataFrame(rows).sort_values("Name")
    if not table["Name"].is_unique:
        raise ValueError(f"Duplicate publication names in {dataset}")
    table.to_csv(os.path.join(output_dir, "crops.tsv"), sep="\t", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, help="Root of the original evaluation data.")
    parser.add_argument("-o", "--output", required=True, help="Root of the exported evaluation data.")
    args = parser.parse_args()

    for dataset in DATASETS:
        export_dataset(dataset, args.input, args.output)


if __name__ == "__main__":
    main()
