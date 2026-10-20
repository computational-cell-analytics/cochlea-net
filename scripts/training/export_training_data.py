"""Export the training data of SGN_v2, IHC_v11, synapses_v3, SGN-lowres_v5 and IHC-lowres_v3 for publication.

The crop names come from the 'Standardized' column of the tables in doc/data.
The internal cochlea ID at the start of each name is replaced with the sample alias of the CochleaNet paper.
The output folders follow the train / val split of the published checkpoints.
"""
import argparse
import os
import shutil

import pandas as pd

DOC_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "doc", "data"))

# Aliases of the sheet 'Samples' in the sample table of the CochleaNet paper.
PAPER_ALIASES = {
    "M-LR-000226-L": "M_01L",
    "M-LR-000226-R": "M_01R",
    "M-LR-000227-L": "M_02L",
    "M-LR-000227-R": "M_02R",
    "M-LR-000144-L": "M_05L",
    "M-LR-000145-L": "M_06L",
    "M-LR-000184-L": "M_10L",
    "M-LR-000151-R": "M_17R",
    "M-AMD-000078-L": "M_18L",
    "M-LR-000167-R": "M_19R",
    "M-LR-000172-R": "M_20R",
    "M-AMD-000058-L": "M_21L",
    "M-AMD-OTOF1-L": "M_22L",
    "M-AMD-OTOF2-L": "M_23L",
    "M-LR-000171-R": "M_24R",
    "M-AMD-000010-L": "M_25L",
    "M-AMD-000004-L": "M_26L",
    "M-AMD-000004-R": "M_26R",
    "M-AMD-000006-R": "M_27R",
    "G-EK-000233-L": "G_1L",
    "LaVision-OTOF23R": "LV_M_03",
    "LaVision-OTOF25R": "LV_M_04",
    "LaVision-Mar05": "LV_Mar_01",
}

# Aliases for cochleae that are not in the paper. The numbers continue after the highest alias in use
# (M_31 and G_7 in scripts/figures/util.py, LV_M_04 in the sample table), with one number per animal.
NEW_ALIASES = {
    "M-AMD-000088-R": "M_32R",
    "M-AMD-000112-L": "M_33L",
    "M-AMD-N75-L": "M_34L",
    "M-AMD-N89-L": "M_35L",
    "M-AMD-N97-L": "M_36L",
    "M-AMD-N142-R": "M_37R",
    "M-AMD-N153-L": "M_38L",
    "M-LR-000249-R": "M_39R",
    "G-LR-000234-L": "G_8L",
    "G-LR-000234-R": "G_8R",
    "LaVision-M04": "LV_M_05",
    "LaVision-OTOF36R": "LV_M_06",
}

# The species code is the first part of the alias, after "LV_" for the LaVision cochleae.
SPECIES = {"M": "mouse", "G": "gerbil", "Mar": "marmoset"}

# The validation crops of the published checkpoints, as original names. They differ from the val folders:
# SGN_v2 from 'val_dataset' in best.pt, IHC_v11 from split.json next to the checkpoint, and synapses_v3 from
# train_test_split(test_size=2, random_state=42) in train_synapse_detection.py (commit 39f28d8).
# The sources of the low-res datasets are the train / val pairs of split.json next to the checkpoints,
# linked into one folder per split (see doc/data/training_data.md).
# The published names match the version 1 models on BioImage.IO, which do not show the internal versions.
DATASETS = {
    "SGN_v2": {
        "published": "SGN_v1",
        "source": "SGN_v2/2025-05_supervised",
        "val": [
            "M_LR_000167_R_crop_0951-0742-0602",
            "M_LR_000167_R_crop_0951-0742-0652",
            "M_LR_000167_R_crop_1210-0148-0606",
            "M_LR_000167_R_crop_1305-0341-0526",
            "M_LR_000184_L_resized_crop_0885-0913-1023",
            "MLR167R_PV_C1",
            "M_LR_000145_L_PV_resized_crop_0352-0759-1100_empty",
            "M_LR_000167_R_crop_1137-0669-1044",
        ],
    },
    "IHC_v11": {
        "published": "IHC_v1",
        "source": "IHC_v11/IHC_v11_2026-07",
        "val": [
            "M-AMD-N142-R_crop_0477-0765-0124_Vglut3",
            "M-AMD-N153-L_crop_0912-1329-0709_Vglut3",
            "G_EK_000233_L_crop_1080-1426-1006_Vglut3_empty",
            "G-LR-000234-L_crop_1287-1688-0999_Vglut3",
            "M-AMD-N142-R_crop_1213-0848-0665_Vglut3",
            "171R_Vglut3_apexIHC_HCAT_C1",
            "M-AMD-N89-L_crop_1245-1102-1004_Vglut3",
            "G-LR-000234-R_crop_0874-0165-0374_Vglut3",
            "G-LR-000234-L_crop_1161-1186-1241_Vglut3",
            "G-LR-000234-R_crop_1344-0554-0668_Vglut3",
            "G-LR-000234-L_crop_0636-1728-0590_Vglut3",
            "G_EK_000233_L_crop_1257-1712-0943_Vglut3",
            "M-AMD-000112-L_crop_0572-0367-0357_Vglut3",
            "G-LR-000234-L_crop_0758-1065-0321_Vglut3",
        ],
    },
    "synapses_v3": {
        "published": "synapses_v1",
        "source": "synapses_v3",
        "val": ["m226r_apex_p1268_pv-ctbp2", "m226r_mid_p2824_vglut3-ctbp2"],
    },
    "SGN-lowres_v5": {
        "published": "SGN-lowres_v1",
        "source": "SGN-lowres_v5",
        "val": [
            "LaVision-Mar05_crop_1182-2920-0726_PV_empty",
            "LaVision-Mar05_crop_2463-2818-1177_PV_2",
            "LaVision-M04_crop_3520-1525-0310_PV_empty",
        ],
    },
    "IHC-lowres_v3": {
        "published": "IHC-lowres_v1",
        "source": "IHC-lowres_v3",
        "val": [
            "LaVision-Mar05_crop_3516-2532-1801_MYO",
            "M_LR_000226_R_crop_0557-1215-0501_Vglut3",
            "LaVision-M04_crop_1540-2075-0924_MYO",
            "M_LR_000227_R_crop_0672-1082-0435_VGlut3",
        ],
    },
}


def split_name(standardized_name: str):
    """Split a standardized crop name into the cochlea ID and the crop part.

    The marker '-resized' of the cochlea ID is dropped, because it only documents an early processing error.
    """
    cochlea_id, crop = standardized_name.split("_", 1)
    return cochlea_id.removesuffix("-resized"), crop


def copy_crop(dataset: str, source_dir: str, original: str, target_dir: str, name: str):
    if dataset.startswith("synapses"):
        shutil.copytree(os.path.join(source_dir, "images", f"{original}.zarr"),
                        os.path.join(target_dir, "images", f"{name}.zarr"))
        shutil.copy2(os.path.join(source_dir, "labels", f"{original}.csv"),
                     os.path.join(target_dir, "labels", f"{name}.csv"))
    else:
        for suffix in ("", "_annotations"):
            shutil.copy2(os.path.join(source_dir, f"{original}{suffix}.tif"),
                         os.path.join(target_dir, f"{name}{suffix}.tif"))


def export_dataset(dataset: str, input_root: str, output_root: str) -> pd.DataFrame:
    """Copy the crops of one dataset with publication names and write its crop table.

    Returns:
        The overview of the dataset, which maps the original names to the publication names.
    """
    val = DATASETS[dataset]["val"]
    published = DATASETS[dataset]["published"]
    output_dir = os.path.join(output_root, published)
    table = pd.read_csv(os.path.join(DOC_DATA_DIR, f"{dataset}.tsv"), sep="\t", dtype=str)
    table = table[table["Dataset"] != "test"]
    missing = set(val) - set(table["Original"])
    if missing:
        raise ValueError(f"Validation crops of {dataset} are missing in the table: {sorted(missing)}")

    # Fail on an existing split folder, so that a second run cannot leave stale crops behind.
    for split in ("train", "val"):
        os.makedirs(os.path.join(output_dir, split))
        if dataset.startswith("synapses"):
            for sub in ("images", "labels"):
                os.makedirs(os.path.join(output_dir, split, sub))

    rows = []
    for _, row in table.iterrows():
        cochlea_id, crop = split_name(row["Standardized"])
        alias = PAPER_ALIASES.get(cochlea_id) or NEW_ALIASES[cochlea_id]
        name = f"{alias}_{crop}"
        split = "val" if row["Original"] in val else "train"
        source_dir = os.path.join(input_root, DATASETS[dataset]["source"], row["Dataset"])
        copy_crop(dataset, source_dir, row["Original"], os.path.join(output_dir, split), name)
        rows.append({
            "Dataset": dataset, "Published_dataset": published, "Split": split,
            "Original": row["Original"], "Standardized": row["Standardized"],
            "CochleaNet_alias": PAPER_ALIASES.get(cochlea_id, ""), "Publication_alias": alias, "Name": name,
            "Species": SPECIES[alias.removeprefix("LV_").split("_")[0]], "Crop_center": row["Crop_center"],
            "Shape_zyx": row["dim"], "n_objects": int(float(row["n_samples"])),
        })

    df = pd.DataFrame(rows).sort_values(["Split", "Name"])
    if not df["Name"].is_unique:
        raise ValueError(f"Duplicate publication names in {dataset}: {df['Name'][df['Name'].duplicated()].tolist()}")
    crop_columns = ["Name", "Split", "Publication_alias", "Species", "Crop_center", "Shape_zyx", "n_objects"]
    df[crop_columns].rename(columns={"Publication_alias": "Alias"}).to_csv(
        os.path.join(output_dir, "crops.tsv"), sep="\t", index=False
    )
    return df[["Dataset", "Published_dataset", "Split", "Original", "Standardized", "CochleaNet_alias",
               "Publication_alias", "Name"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True,
                        help="Root of the original training data, with one folder per dataset.")
    parser.add_argument("-o", "--output", required=True, help="Root of the exported training data.")
    args = parser.parse_args()

    overview = pd.concat([export_dataset(dataset, args.input, args.output) for dataset in DATASETS])
    overview.to_csv(os.path.join(args.output, "overview.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
