"""Compare two synapse stains at their spots: match status, intensities, threshold sweep and crop selection.

The reference stain (e.g. CtBP2) takes the role of the ground truth and the test stain (e.g. Homer1) the role
of the detections, as in `scripts/measurements/synapse_colocalization.py`. The input tables are the outputs of
`flamingo_tools.synapse_detection.spot_intensity`, measured in both channels.

Run `python -m flamingo_tools.synapse_detection.stain_comparison -h` for the command line.
"""

import argparse
import json
import os
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from flamingo_tools.json_util import export_dictionary_as_json
from flamingo_tools.s3_utils import read_table
from flamingo_tools.validation import compute_scores_from_counts, match_detections

COORDINATES = ["x", "y", "z"]
SUMMARY_MEASURES = ("peak", "background", "ratio", "snr", "integrated")
SUMMARY_QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)

# Blue and orange, the first two colors of `stain_summary.COCHLEA_COLORS`.
_MATCHED_COLOR = "#2a78d6"
_UNMATCHED_COLOR = "#eb6834"
_TEXT_COLOR = "#52514e"


def _spot_ids(table: pd.DataFrame) -> np.ndarray:
    return table["spot_id"].to_numpy() if "spot_id" in table else table.index.to_numpy()


def match_stains(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    max_dist: float = 2.0,
    valid_ihcs: Optional[Sequence[int]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Match the spots of two stains one to one and mark the match status in both tables.

    Args:
        reference: The spots of the reference stain, with 'x', 'y', 'z' in micrometer.
        test: The spots of the test stain.
        max_dist: The maximum distance in micrometer for a match.
        valid_ihcs: Keep only the spots with a 'matched_ihc' in this list.

    Returns:
        Copies of both tables, filtered by the valid IHCs, with the columns 'matched' (bool) and
        'partner_id' (the 'spot_id' of the partner in the other table, NaN without a partner).
    """
    if valid_ihcs is not None:
        reference = reference[reference.matched_ihc.isin(valid_ihcs)]
        test = test[test.matched_ihc.isin(valid_ihcs)]
    reference = reference.reset_index(drop=True)
    test = test.reset_index(drop=True)

    tp_test, tp_reference, _, _ = match_detections(
        test[COORDINATES].to_numpy(), reference[COORDINATES].to_numpy(), max_dist=max_dist
    )
    for table, own, other, other_table in (
        (reference, tp_reference, tp_test, test), (test, tp_test, tp_reference, reference)
    ):
        matched = np.zeros(len(table), dtype=bool)
        matched[own] = True
        partner = np.full(len(table), np.nan)
        partner[own] = _spot_ids(other_table)[other]
        table["matched"] = matched
        table["partner_id"] = partner
    return reference, test


def intensity_summary(
    table: pd.DataFrame,
    channels: Sequence[str],
    measures: Sequence[str] = SUMMARY_MEASURES,
    quantiles: Sequence[float] = SUMMARY_QUANTILES,
) -> pd.DataFrame:
    """Summarize the intensity measures of a spot table, for all spots and split by the match status.

    Args:
        table: A spot table from `match_stains`, with the columns `<channel>_<measure>`.
        channels: The channels to summarize.
        measures: The measures to summarize.
        quantiles: The quantiles in the summary.

    Returns:
        One row per group ('all', 'matched', 'unmatched'), channel and measure, with the number of
        spots with a value, the mean and the quantiles.
    """
    groups = {"all": np.ones(len(table), dtype=bool)}
    if "matched" in table:
        groups.update({"matched": table.matched.to_numpy(), "unmatched": ~table.matched.to_numpy()})
    rows = []
    for group, mask in groups.items():
        for channel in channels:
            for measure in measures:
                values = table.loc[mask, f"{channel}_{measure}"].dropna()
                row = {"group": group, "channel": channel, "measure": measure, "n": len(values),
                       "mean": values.mean() if len(values) else np.nan}
                for q in quantiles:
                    row[f"q{round(100 * q):02d}"] = values.quantile(q) if len(values) else np.nan
                rows.append(row)
    return pd.DataFrame(rows)


def threshold_sweep(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    column: str,
    thresholds: Optional[Sequence[float]] = None,
    max_dist: float = 2.0,
    n_ihcs: Optional[int] = None,
) -> pd.DataFrame:
    """Remove the test spots below a threshold of one column and match the rest against the reference again.

    Spots without a value in the column are kept.

    Args:
        reference: The spots of the reference stain.
        test: The spots of the test stain, with the column.
        column: The column for the threshold, e.g. 'Homer1_snr'.
        thresholds: The thresholds. Default: the 0 % to 95 % quantiles of the column in steps of 5 %.
        max_dist: The maximum distance in micrometer for a match.
        n_ihcs: The number of IHCs, for the columns 'test_per_ihc' and 'reference_per_ihc'.

    Returns:
        One row per threshold with the spot counts, the matching counts and scores, and the fraction of
        removed test spots.
    """
    values = test[column].to_numpy(dtype="float64")
    if thresholds is None:
        thresholds = np.unique(np.nanquantile(values, np.linspace(0, 0.95, 20)))
    reference_coords = reference[COORDINATES].to_numpy()
    rows = []
    for threshold in thresholds:
        keep = ~(values < threshold)
        tp, _, fp, fn = match_detections(test.loc[keep, COORDINATES].to_numpy(), reference_coords, max_dist)
        row = {
            "threshold": threshold, "n_test": int(keep.sum()), "n_reference": len(reference),
            "removed_fraction": 1 - keep.mean() if len(keep) else np.nan,
            "tp": len(tp), "fp": len(fp), "fn": len(fn),
            **compute_scores_from_counts(len(tp), len(fp), len(fn)),
        }
        if n_ihcs:
            row["test_per_ihc"] = row["n_test"] / n_ihcs
            row["reference_per_ihc"] = len(reference) / n_ihcs
        rows.append(row)
    return pd.DataFrame(rows)


def select_excess_ihcs(
    test: pd.DataFrame,
    ihc_table: pd.DataFrame,
    n_ihcs: int = 3,
    crop_extent: Optional[Sequence[float]] = None,
) -> pd.DataFrame:
    """Select the IHCs with the most unmatched test spots, for example as crop centers.

    The selection is greedy. With `crop_extent`, an IHC is skipped if a crop of this size around its anchor
    would overlap with the crop of an IHC already selected.

    Args:
        test: The test spots from `match_stains`, with 'matched' and 'matched_ihc'.
        ihc_table: The IHC table with 'label_id' and 'anchor_x', 'anchor_y', 'anchor_z' in micrometer.
        n_ihcs: The number of IHCs to select.
        crop_extent: The full crop size in micrometer, (x, y, z).

    Returns:
        The selected IHCs with 'label_id', 'n_unmatched', 'n_test' and the anchor 'x', 'y', 'z'.
    """
    counts = pd.DataFrame({
        "n_unmatched": test.loc[~test.matched, "matched_ihc"].value_counts(),
        "n_test": test.matched_ihc.value_counts(),
    }).fillna(0).astype(int)
    counts.index = counts.index.astype(int)
    counts = counts.rename_axis("label_id").reset_index()
    anchors = ihc_table[["label_id", "anchor_x", "anchor_y", "anchor_z"]].rename(
        columns={"anchor_x": "x", "anchor_y": "y", "anchor_z": "z"}
    )
    candidates = counts.merge(anchors, on="label_id").sort_values(
        ["n_unmatched", "label_id"], ascending=[False, True]
    )

    positions = candidates[COORDINATES].to_numpy(dtype="float64")
    selected = []
    for index, position in enumerate(positions):
        if len(selected) == n_ihcs:
            break
        if crop_extent is not None and any(
            np.all(np.abs(position - positions[other]) < np.asarray(crop_extent)) for other in selected
        ):
            continue
        selected.append(index)
    return candidates.iloc[selected].reset_index(drop=True)


def plot_stain_comparison(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    reference_channel: str,
    test_channel: str,
    sweep: pd.DataFrame,
    output_path: str,
    measure: str = "snr",
) -> None:
    """Plot the intensity measure of both stains by match status and the threshold sweep in one figure.

    Args:
        reference: The reference spots from `match_stains`.
        test: The test spots from `match_stains`.
        reference_channel: The channel of the reference stain.
        test_channel: The channel of the test stain.
        sweep: The output of `threshold_sweep`.
        output_path: The output image file.
        measure: The intensity measure for the histograms and the scatter plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def by_match(ax, table, column, title):
        values = table[column].dropna()
        lo, hi = np.nanquantile(values, [0.005, 0.995]) if len(values) else (0, 1)
        bins = np.linspace(lo, hi, 60)
        for matched, color, label in ((True, _MATCHED_COLOR, "matched"), (False, _UNMATCHED_COLOR, "unmatched")):
            subset = table.loc[table.matched == matched, column].dropna()
            ax.hist(subset, bins=bins, histtype="step", linewidth=2, color=color, label=f"{label} (n={len(subset)})")
        ax.set_xlabel(f"{column} (0.5 - 99.5 % range)")
        ax.set_ylabel("spots")
        ax.set_title(title, loc="left")
        ax.legend(frameon=False)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    by_match(axes[0, 0], test, f"{test_channel}_{measure}", f"{test_channel} spots, {test_channel} stain")
    by_match(axes[0, 1], reference, f"{reference_channel}_{measure}",
             f"{reference_channel} spots, {reference_channel} stain")

    ax = axes[0, 2]
    x_column, y_column = f"{test_channel}_{measure}", f"{reference_channel}_{measure}"
    if y_column in test:
        for matched, color, label in ((False, _UNMATCHED_COLOR, "unmatched"), (True, _MATCHED_COLOR, "matched")):
            subset = test[test.matched == matched]
            ax.scatter(subset[x_column], subset[y_column], s=8, alpha=0.4, color=color, linewidths=0, label=label)
        ax.set_xlim(*np.nanquantile(test[x_column], [0.005, 0.995]))
        ax.set_ylim(*np.nanquantile(test[y_column], [0.005, 0.995]))
        ax.set_xlabel(f"{x_column} (0.5 - 99.5 % range)")
        ax.set_ylabel(f"{y_column} (0.5 - 99.5 % range)")
        ax.legend(frameon=False, markerscale=2)
    else:
        ax.text(0.5, 0.5, f"{y_column} not measured", ha="center", va="center", color=_TEXT_COLOR)
    ax.set_title(f"{test_channel} spots, both stains", loc="left")

    sweep_column = sweep.attrs.get("column", f"{test_channel}_{measure}")
    ax = axes[1, 0]
    ax.plot(sweep.threshold, sweep.n_test, color=_UNMATCHED_COLOR, linewidth=2, marker="o", markersize=4,
            label=f"{test_channel} spots kept")
    ax.axhline(sweep.n_reference.iloc[0], color=_MATCHED_COLOR, linewidth=2,
               label=f"{reference_channel} spots ({sweep.n_reference.iloc[0]})")
    ax.set_xlabel(f"threshold on {sweep_column}")
    ax.set_ylabel("spots")
    ax.set_ylim(bottom=0)
    ax.set_title("Spot count after the threshold", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    ax.plot(sweep.threshold, sweep.precision, color=_UNMATCHED_COLOR, linewidth=2, marker="o", markersize=4,
            label=f"precision ({test_channel} with partner)")
    ax.plot(sweep.threshold, sweep.recall, color=_MATCHED_COLOR, linewidth=2, marker="o", markersize=4,
            label=f"recall ({reference_channel} with partner)")
    ax.set_xlabel(f"threshold on {sweep_column}")
    ax.set_ylabel("fraction")
    ax.set_ylim(0, 1)
    ax.set_title("Colocalization after the threshold", loc="left")
    ax.legend(frameon=False)

    ax = axes[1, 2]
    for table, channel, color in ((reference, reference_channel, _MATCHED_COLOR),
                                  (test, test_channel, _UNMATCHED_COLOR)):
        values = table[f"{channel}_integrated"].dropna()
        values = values / values.median()
        ax.hist(values, bins=np.linspace(-1, 6, 71), histtype="step", linewidth=2, color=color,
                label=f"{channel} spots (n={len(values)})")
    ax.set_xlabel("integrated intensity / median of the stain (-1 to 6 shown)")
    ax.set_ylabel("spots")
    ax.set_title("Integrated intensity, own stain", loc="left")
    ax.legend(frameon=False)

    for ax in axes.ravel():
        ax.grid(True, color="#e4e3df", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def write_crop_json(
    output_path: str,
    crop_centers: Sequence[Sequence[float]],
    dataset_name: str,
    image_channel: Sequence[str],
    roi_halo: Sequence[int],
    segmentation_channel: Optional[str] = None,
    force_overwrite: bool = False,
) -> dict:
    """Write a block-extraction JSON file for `flamingo_tools.extract_block --json_info`.

    Args:
        output_path: The output JSON file.
        crop_centers: The crop centers in micrometer, (x, y, z).
        dataset_name: The name of the cochlea.
        image_channel: The image and segmentation channels to crop.
        roi_halo: The halo around the crop center in voxels, (x, y, z).
        segmentation_channel: The segmentation channel.
        force_overwrite: Overwrite an existing file.

    Returns:
        The parameter dictionary.
    """
    params = {"dataset_name": dataset_name, "image_channel": list(image_channel)}
    if segmentation_channel is not None:
        params["segmentation_channel"] = segmentation_channel
    params["roi_halo"] = [int(h) for h in roi_halo]
    params["crop_centers"] = [[int(round(float(c))) for c in center] for center in crop_centers]
    export_dictionary_as_json(params, output_path, force_overwrite=force_overwrite)
    return params


def compare_stains(
    reference: pd.DataFrame,
    test: pd.DataFrame,
    reference_channel: str,
    test_channel: str,
    output_folder: str,
    reference_name: str = "reference",
    test_name: str = "test",
    valid_ihcs: Optional[Sequence[int]] = None,
    max_dist: float = 2.0,
    sweep_measure: str = "snr",
    thresholds: Optional[Sequence[float]] = None,
) -> Dict:
    """Match two stains, summarize the intensities, sweep a threshold on the test stain, and save the results.

    The output folder receives `<name>_matched.tsv` for both stains, 'intensity_summary.tsv',
    'sweep.tsv', 'summary.json' and 'overview.png'. 'summary.json' records the names and the channels,
    so that `flamingo_tools.synapse_detection.stain_summary` can read the folder.

    Args:
        reference: The spot table of the reference stain from `spot_intensity`.
        test: The spot table of the test stain from `spot_intensity`.
        reference_channel: The channel of the reference stain, e.g. 'CTBP2'.
        test_channel: The channel of the test stain, e.g. 'Homer1'.
        output_folder: The output folder.
        reference_name: The file prefix for the matched reference table.
        test_name: The file prefix for the matched test table.
        valid_ihcs: Keep only the spots of these IHCs.
        max_dist: The maximum distance in micrometer for a match.
        sweep_measure: The measure of the test channel for the threshold sweep.
        thresholds: The thresholds of the sweep. Default: see `threshold_sweep`.

    Returns:
        The summary and the matched tables under the keys 'summary', 'reference' and 'test'.
    """
    os.makedirs(output_folder, exist_ok=True)
    reference, test = match_stains(reference, test, max_dist=max_dist, valid_ihcs=valid_ihcs)
    channels = [channel for channel in (reference_channel, test_channel)
                if f"{channel}_peak" in reference and f"{channel}_peak" in test]
    n_ihcs = len(valid_ihcs) if valid_ihcs is not None else None

    summaries = []
    for name, table in ((reference_name, reference), (test_name, test)):
        table.to_csv(os.path.join(output_folder, f"{name}_matched.tsv"), sep="\t", index=False)
        summaries.append(intensity_summary(table, channels).assign(stain=name))
    intensity = pd.concat(summaries, ignore_index=True)
    intensity.to_csv(os.path.join(output_folder, "intensity_summary.tsv"), sep="\t", index=False)

    sweep_column = f"{test_channel}_{sweep_measure}"
    sweep = threshold_sweep(reference, test, sweep_column, thresholds=thresholds, max_dist=max_dist, n_ihcs=n_ihcs)
    sweep.attrs["column"] = sweep_column
    sweep.to_csv(os.path.join(output_folder, "sweep.tsv"), sep="\t", index=False)

    tp = int(test.matched.sum())
    summary = {
        "reference_name": reference_name,
        "test_name": test_name,
        "reference_channel": reference_channel,
        "test_channel": test_channel,
        "max_dist": max_dist,
        "n_ihcs": n_ihcs,
        "n_reference": len(reference),
        "n_test": len(test),
        "tp": tp, "fp": len(test) - tp, "fn": len(reference) - tp,
        **compute_scores_from_counts(tp, len(test) - tp, len(reference) - tp),
        "saturated_fraction": {
            name: {channel: round(float(np.mean(table[f"{channel}_n_saturated"] > 0)), 4)
                   for channel in channels if table[f"{channel}_n_saturated"].notna().any()}
            for name, table in ((reference_name, reference), (test_name, test))
        },
        "median": {
            name: {
                group: {
                    f"{row.channel}_{row.measure}": round(float(row.q50), 3)
                    for row in intensity[(intensity.stain == name) & (intensity.group == group)].itertuples()
                }
                for group in ("matched", "unmatched")
            }
            for name in (reference_name, test_name)
        },
    }
    with open(os.path.join(output_folder, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    plot_stain_comparison(
        reference, test, reference_channel, test_channel, sweep,
        os.path.join(output_folder, "overview.png"), measure=sweep_measure,
    )
    return {"summary": summary, "reference": reference, "test": test, "sweep": sweep}


def _file_prefix(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def main():
    parser = argparse.ArgumentParser(
        description="Compare the spots of a test stain (e.g. Homer1) with a reference stain (e.g. CtBP2): match "
        "status, intensity summary, a threshold sweep on the test stain, and an overview figure. The input tables "
        "come from flamingo_tools.synapse_detection.spot_intensity."
    )
    parser.add_argument("-r", "--reference", required=True, help="Spot intensity table of the reference stain.")
    parser.add_argument("-t", "--test", required=True, help="Spot intensity table of the test stain.")
    parser.add_argument("--reference_channel", required=True, help="Channel of the reference stain, e.g. CTBP2.")
    parser.add_argument("--test_channel", required=True, help="Channel of the test stain, e.g. Homer1.")
    parser.add_argument("-o", "--output_folder", required=True, help="Output folder.")
    parser.add_argument("--max_dist", type=float, default=2.0,
                        help="Maximum distance in micrometer for a match between two spots. Default: 2")
    parser.add_argument("--sweep_measure", default="snr",
                        help="Measure of the test channel for the threshold sweep. Default: snr")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="Thresholds of the sweep. Default: the 0 to 95 %% quantiles in steps of 5 %%.")
    parser.add_argument("--ihc_table", default=None,
                        help="IHC table for the selection of valid IHCs. Without it, all spots are compared.")
    parser.add_argument("--ihc_column", default="component_labels",
                        help="Column of the IHC table for the selection. Default: component_labels")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1],
                        help="Values of '--ihc_column' that select the valid IHCs. Default: 1")
    parser.add_argument("--s3", action="store_true", help="Read the IHC table from the S3 bucket.")
    parser.add_argument("--s3_credentials", default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", default=None, help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    crops = parser.add_argument_group(
        "crops", "Write a block-extraction JSON file with crops at the IHCs with the most unmatched test spots."
    )
    crops.add_argument("--crop_json", default=None, help="Output JSON file for flamingo_tools.extract_block.")
    crops.add_argument("--n_crops", type=int, default=3, help="Number of crops. Default: 3")
    crops.add_argument("--dataset_name", default=None, help="Name of the cochlea.")
    crops.add_argument("--image_channel", nargs="+", default=None, help="Channels to crop.")
    crops.add_argument("--segmentation_channel", default=None, help="Segmentation channel.")
    crops.add_argument("--roi_halo", type=int, nargs=3, default=[128, 128, 128],
                       help="Halo around the crop center in voxels (x, y, z). Default: 128 128 128")
    crops.add_argument("--voxel_size", type=float, nargs=3, default=[0.38, 0.38, 0.38],
                       help="Voxel size in micrometer (x, y, z).")
    crops.add_argument("-f", "--force", action="store_true", help="Overwrite an existing crop JSON file.")
    args = parser.parse_args()

    if args.crop_json is not None and (args.ihc_table is None or args.dataset_name is None or not args.image_channel):
        parser.error("--crop_json requires --ihc_table, --dataset_name and --image_channel.")

    ihc_table, valid_ihcs = None, None
    if args.ihc_table is not None:
        ihc_table = read_table(
            args.ihc_table, args.s3, bucket_name=args.s3_bucket_name,
            service_endpoint=args.s3_service_endpoint, credential_file=args.s3_credentials,
        )
        valid_ihcs = ihc_table.label_id[ihc_table[args.ihc_column].isin(args.components)].astype(int).tolist()

    result = compare_stains(
        reference=pd.read_csv(args.reference, sep="\t"),
        test=pd.read_csv(args.test, sep="\t"),
        reference_channel=args.reference_channel,
        test_channel=args.test_channel,
        output_folder=args.output_folder,
        reference_name=_file_prefix(args.reference),
        test_name=_file_prefix(args.test),
        valid_ihcs=valid_ihcs,
        max_dist=args.max_dist,
        sweep_measure=args.sweep_measure,
        thresholds=args.thresholds,
    )
    print(json.dumps({key: value for key, value in result["summary"].items() if key != "median"}, indent=2))

    if args.crop_json is not None:
        crop_extent = 2 * np.asarray(args.roi_halo) * np.asarray(args.voxel_size)
        selected = select_excess_ihcs(result["test"], ihc_table, n_ihcs=args.n_crops, crop_extent=crop_extent)
        print("Crops at the IHCs with the most unmatched test spots:")
        print(selected.to_string(index=False))
        write_crop_json(
            args.crop_json, selected[COORDINATES].to_numpy(), dataset_name=args.dataset_name,
            image_channel=args.image_channel, roi_halo=args.roi_halo,
            segmentation_channel=args.segmentation_channel, force_overwrite=args.force,
        )


if __name__ == "__main__":
    main()
