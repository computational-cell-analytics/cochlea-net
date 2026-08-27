"""Select matched synapse-prediction blocks for the normalization experiment.

The selection is deterministic and prediction-free. It chooses four blocks from a
low-contrast / low-count regime and four from a good-contrast / typical-count regime for
each cochlea. The output is a JSON manifest consumed by ``normalization_experiment.py``.
"""

import argparse
import json
import math
import os

import bioimage_cpp as bic
import numpy as np
import pandas as pd
import zarr
from elf.io import open_file


DATA_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"
WS = "/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools"
BLOCK_SHAPE = (64, 256, 256)
VOXEL_SIZE = 0.38


CONFIG = {
    "G_EK_000233_L": {
        "table": f"{DATA_ROOT}/mobie_project/cochlea-lightsheet/G_EK_000233_L/tables/IHC_v11/default.tsv",
        "shape_path": f"{DATA_ROOT}/predictions/G_EK_000233_L/IHC_v11/segmentation.zarr",
        "shape_key": "segmentation",
    },
    "G_LR_000301_R": {
        "table": f"{DATA_ROOT}/mobie_project/cochlea-lightsheet/G_LR_000301_R/tables/IHC_v11/default.tsv",
        "shape_path": f"{DATA_ROOT}/predictions/G_LR_000301_R/IHC_v11/segmentation.zarr",
        "shape_key": "segmentation",
    },
    "G_LR_000302_R": {
        "table": f"{DATA_ROOT}/mobie_project/cochlea-lightsheet/G_LR_000302_R/tables/IHC_v11/default.tsv",
        "shape_path": f"{DATA_ROOT}/predictions/G_LR_000302_R/IHC_v11/segmentation.zarr",
        "shape_key": "segmentation",
    },
    "G_LR_000301_L": {
        "table": f"{WS}/prediction/G301L/IHC_v11_dilated_mask1/default_components.tsv",
        "shape_path": f"{WS}/prediction/G301L/IHC_v11_dilated_mask1/segmentation.zarr",
        "shape_key": "segmentation",
        "raw_s4_path": f"{WS}/G301L/GLR_301L_CTBP2_fused.n5",
        "raw_s4_key": "setup0/timepoint0/s4",
    },
}


def _selected_ihcs(cochlea):
    table = pd.read_csv(CONFIG[cochlea]["table"], sep="\t")
    table = table[table["component_labels"] > 0].copy()
    if "syn_per_IHC" in table.columns:
        table = table[table["syn_per_IHC"] >= 0].copy()
    return table


def _g301l_contrast(table):
    """Measure p99.9 in a small s4 ROI around every finalized G301L IHC."""
    cfg = CONFIG["G_LR_000301_L"]
    with open_file(cfg["raw_s4_path"], "r") as f:
        raw = f[cfg["raw_s4_key"]]
        result = []
        scale = VOXEL_SIZE * 16
        for row in table.itertuples():
            lo = np.floor(np.array([row.bb_min_z, row.bb_min_y, row.bb_min_x]) / scale).astype(int) - 2
            hi = np.ceil(np.array([row.bb_max_z, row.bb_max_y, row.bb_max_x]) / scale).astype(int) + 3
            lo = np.maximum(0, lo)
            hi = np.minimum(raw.shape, hi)
            values = np.asarray(raw[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))])
            result.append(float(np.quantile(values, 0.999)))
    table["raw_roi_p999"] = result
    return table


def _block_margin(coord, block):
    coord = np.asarray(coord)
    begin = np.asarray(block.begin)
    end = np.asarray(block.end)
    return int(np.min(np.minimum(coord - begin, end - 1 - coord)))


def _choose_blocks(table, blocking, regime, n_blocks, used, prefer="spread"):
    """Choose distinct blocks, preferring target IHCs away from block boundaries."""
    table = table.copy()
    table["coord_zyx"] = table.apply(
        lambda row: np.rint(
            np.asarray([row.anchor_z, row.anchor_y, row.anchor_x], dtype=float) / VOXEL_SIZE
        ).astype(int),
        axis=1,
    )
    table["block_id"] = table["coord_zyx"].apply(
        lambda x: int(blocking.coordinates_to_block_id(x.tolist()))
    )
    table["block_margin"] = table.apply(
        lambda row: _block_margin(row.coord_zyx, blocking.get_block(int(row.block_id))), axis=1
    )

    # Avoid cells right on a block face; their ribbons are more likely to lie in an
    # unselected neighbor. Fall back progressively if a regime is very small.
    eligible = table[table["block_margin"] >= 12]
    if len(eligible[~eligible["block_id"].isin(used)].drop_duplicates("block_id")) < n_blocks:
        eligible = table[table["block_margin"] >= 6]
    if len(eligible[~eligible["block_id"].isin(used)].drop_duplicates("block_id")) < n_blocks:
        eligible = table

    eligible = eligible[~eligible["block_id"].isin(used)].drop_duplicates("block_id")
    if len(eligible) < n_blocks:
        raise RuntimeError(f"Only {len(eligible)} distinct eligible blocks for {regime}; need {n_blocks}.")

    if prefer == "low_contrast":
        chosen = eligible.sort_values(["raw_roi_p999", "block_margin"], ascending=[True, False]).head(n_blocks)
    elif prefer == "high_contrast":
        chosen = eligible.sort_values(["raw_roi_p999", "block_margin"], ascending=[False, False]).head(n_blocks)
    else:
        # Spread the blocks over the requested tonotopic interval rather than taking four
        # neighboring cells from the same local patch.
        eligible = eligible.sort_values("length_fraction")
        positions = np.linspace(0, len(eligible) - 1, n_blocks).round().astype(int)
        chosen = eligible.iloc[positions]

    records = []
    for row in chosen.itertuples():
        block_id = int(row.block_id)
        used.add(block_id)
        record = {
            "block_id": block_id,
            "grid_position": list(map(int, blocking.block_grid_position(block_id))),
            "regime": regime,
            "target_label": int(row.label_id),
            "target_anchor_zyx_voxels": list(map(int, row.coord_zyx)),
            "target_block_margin_voxels": int(row.block_margin),
        }
        for column in ("length_fraction", "syn_per_IHC", "raw_roi_p999"):
            if hasattr(row, column):
                value = getattr(row, column)
                if not (isinstance(value, float) and math.isnan(value)):
                    record[column] = float(value)
        records.append(record)
    return records


def select_for_cochlea(cochlea, n_per_regime=4):
    cfg = CONFIG[cochlea]
    shape = tuple(zarr.open(cfg["shape_path"], mode="r")[cfg["shape_key"]].shape)
    blocking = bic.utils.Blocking([0, 0, 0], list(shape), list(BLOCK_SHAPE))
    table = _selected_ihcs(cochlea)
    used = set()

    if cochlea == "G_EK_000233_L":
        good = table[table["length_fraction"].between(0.2, 0.5) & (table["syn_per_IHC"] >= 18)]
        poor = table[table["length_fraction"].between(0.8, 1.0) & table["syn_per_IHC"].between(5, 16)]
        records = _choose_blocks(good, blocking, "good_contrast_high_count", n_per_regime, used)
        records += _choose_blocks(poor, blocking, "lower_count_basal", n_per_regime, used)
    elif cochlea == "G_LR_000301_R":
        poor = table[table["length_fraction"].between(0.6, 0.7) & (table["syn_per_IHC"] <= 2)]
        good = table[table["length_fraction"].between(0.9, 1.0) & table["syn_per_IHC"].between(25, 45)]
        records = _choose_blocks(poor, blocking, "poor_contrast_middle", n_per_regime, used)
        records += _choose_blocks(good, blocking, "good_contrast_basal", n_per_regime, used)
    elif cochlea == "G_LR_000302_R":
        # Exclude zero-count endpoint cells: the terminal block can be fully blank in the
        # CtBP2 input, for which block-local normalization is undefined and cannot test
        # whether weak-but-recoverable signal benefits from local statistics.
        poor = table[
            table["length_fraction"].between(0.8, 1.0)
            & table["syn_per_IHC"].between(1, 8)
        ]
        good = table[table["length_fraction"].between(0.2, 0.7) & table["syn_per_IHC"].between(10, 20)]
        records = _choose_blocks(poor, blocking, "lower_count_basal", n_per_regime, used)
        records += _choose_blocks(good, blocking, "typical_count_middle", n_per_regime, used)
    else:
        table = _g301l_contrast(table)
        records = _choose_blocks(table, blocking, "poor_contrast", n_per_regime, used, "low_contrast")
        records += _choose_blocks(table, blocking, "good_contrast", n_per_regime, used, "high_contrast")

    if len(records) != 2 * n_per_regime or len({r["block_id"] for r in records}) != len(records):
        raise RuntimeError(f"Invalid selection for {cochlea}")
    return {"shape": list(shape), "blocks": records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--blocks-per-regime", type=int, default=4)
    args = parser.parse_args()

    manifest = {
        "block_shape": list(BLOCK_SHAPE),
        "halo": [16, 64, 64],
        "voxel_size": [VOXEL_SIZE] * 3,
        "selection": {
            cochlea: select_for_cochlea(cochlea, args.blocks_per_regime)
            for cochlea in CONFIG
        },
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    for cochlea, info in manifest["selection"].items():
        print(cochlea)
        for record in info["blocks"]:
            extras = []
            for key in ("length_fraction", "syn_per_IHC", "raw_roi_p999"):
                if key in record:
                    extras.append(f"{key}={record[key]:.2f}")
            print(f"  block {record['block_id']:5d} {record['regime']:<27} " + " ".join(extras))


if __name__ == "__main__":
    main()
