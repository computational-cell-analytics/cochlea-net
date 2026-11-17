"""Evaluate the IHC row correction on the annotated training crops with the real IHC_v11 output.

The input is the S3 IHC_v11 segmentation of the training cochleae in the box of each annotated crop. The
reference is the refined annotation (common.annotation_cells). The variants of the IHC row are the variants of
evaluate_planes.py:
- 'objects': all IHC_v11 instances, without the row filter;
- 'components': the IHC_v11 row components of the processing file of the cochlea in reproducibility/processing,
  replayed with `label_components_ihc`. For a file with 'refine_row', these are the seed components of the
  refinement ('component_list_path');
- 'refined_5000': `refine_ihc_row`, seeded with the same components, with min_size 5000, max_path_deviation
  15 um and the row merged into one component. For a file with 'refine_row', this is the archived row;
- '<variant>+correct': `correct_ihc_row` with its defaults, in 3D windows around the row crossings of the crop
  (`common.correct_region`), along the row path of `common.row_path`.

The count rule is the rule of the annotation: an output label counts if it has MIN_IHC_SIZE pixels or more in the
crop. The labels below this size are removed before the matching (F1 at IoU 0.5).

IHC_v11 trained on the train crops of its split, so only the IHC_v11 val crops are fair for the model output.
"""
import argparse
import os

import numpy as np
import pandas as pd
from elf.evaluation import matching
from skimage.segmentation import relabel_sequential

from flamingo_tools.json_util import load_processing_params
from flamingo_tools.postprocessing.label_components import label_components_ihc
from flamingo_tools.s3_utils import default_table_path, read_table

from common import (
    MAX_PATH_DEVIATION, MIN_IHC_SIZE, REPOSITORY, animal_ci, annotation_cells, correct_region, crop_info, crop_start,
    gt_crops, read_volume, row_variants, volume_shape,
)

SEGMENTATION = "IHC_v11"
PROCESSING = os.path.join(REPOSITORY, "reproducibility", "processing")
REFINED_MIN_SIZE = 5000
CORRECTED = ["components", "refined_5000"]


def row_labeling(cochlea):
    """The IHC_v11 components of the processing file of a cochlea, with the parameters of `label_components_single`.

    Returns the table with the component labels and the row components, or None without an IHC_v11 entry.
    """
    path = os.path.join(PROCESSING, f"{cochlea}_IHC.json")
    if not os.path.exists(path):
        return None
    entries = [e for e in load_processing_params(path, "label_components") if e["segmentation_channel"] == SEGMENTATION]
    if not entries:
        return None
    entry = entries[0]
    params = {"min_size": entry.get("min_size", 1000), "min_component_length": entry.get("min_component_length", 50),
              "max_edge_distance": entry.get("max_edge_distance", 30)}
    table = label_components_ihc(read_table(default_table_path(cochlea, SEGMENTATION, s3=True), s3=True), **params)
    if not entry.get("refine_row"):
        if entry.get("max_path_deviation") is not None:
            raise NotImplementedError(f"{cochlea}: the path deviation filter without refine_row is not replayed.")
        return table, entry.get("component_list", [1])
    refinement = (params["min_size"], entry.get("max_path_deviation"), entry.get("merge_components"))
    if refinement != (REFINED_MIN_SIZE, MAX_PATH_DEVIATION, True):
        raise NotImplementedError(f"{cochlea}: the archived refinement {refinement} is not 'refined_5000'.")
    return table, entry.get("component_list_path") or entry.get("component_list", [1])


def read_crop(cochlea, start, shape, volume, cache_folder):
    """The S3 segmentation in the crop box. The parts of the box outside the volume are 0."""
    low = np.maximum(start, 0)
    high = np.minimum(start + np.array(shape), volume)
    key = "_".join(f"{a}-{b}" for a, b in zip(low, high))
    data = read_volume(cochlea, SEGMENTATION, tuple(slice(a, b) for a, b in zip(low, high)),
                       os.path.join(cache_folder, f"{cochlea}_{SEGMENTATION}_{key}.tif"))
    crop = np.zeros(shape, dtype="uint32")
    crop[tuple(slice(a - s, b - s) for a, b, s in zip(low, high, start))] = data
    return crop


def score(labels, gt):
    """F1 at IoU 0.5 and the count of the output labels with MIN_IHC_SIZE pixels or more."""
    ids, sizes = np.unique(labels[labels > 0], return_counts=True)
    labels = relabel_sequential(np.where(np.isin(labels, ids[sizes >= MIN_IHC_SIZE]), labels, 0))[0]
    n_gt, n_out = int(gt.max()), int(labels.max())
    tp = int(round(matching(labels, gt, threshold=0.5)["recall"] * n_gt)) if n_out and n_gt else 0
    return {"n_gt": n_gt, "n_out": n_out, "tp": tp, "fp": n_out - tp, "fn": n_gt - tp,
            "f1": 2 * tp / max(n_out + n_gt, 1), "count_error": n_out - n_gt}


def summarize(results):
    """The scores per variant, species and IHC_v11 split.

    'f1' is the F1 of the summed counts. 'count_error' is relative to the annotated cells, 'count_mae' the mean
    absolute count error per crop, 'count_exact' the fraction of crops with an exact count. 'delta_mae' is the
    change of the absolute count error per crop by the correction, with a 95% CI from resampling the animals.
    """
    rng = np.random.default_rng(0)
    splits = [("all", results), ("train", results[~results["val"]]), ("val", results[results["val"]])]
    rows = []
    for split, table in splits:
        for species in ["all", "gerbil", "mouse"]:
            group = table if species == "all" else table[table["species"] == species]
            for variant, scores in group.groupby("variant", sort=False):
                tp, fp, fn = scores["tp"].sum(), scores["fp"].sum(), scores["fn"].sum()
                row = {"variant": variant, "split": split, "species": species, "crops": len(scores),
                       "animals": scores["animal"].nunique(), "tp": tp, "fp": fp, "fn": fn,
                       "f1": 2 * tp / max(2 * tp + fp + fn, 1), "n_gt": scores["n_gt"].sum(),
                       "n_out": scores["n_out"].sum(),
                       "count_error": scores["count_error"].sum() / max(scores["n_gt"].sum(), 1),
                       "count_mae": scores["count_error"].abs().mean(),
                       "count_exact": (scores["count_error"] == 0).mean()}
                if variant.endswith("+correct"):
                    base = group[group["variant"] == variant[:-len("+correct")]].set_index("crop")
                    delta = (scores.set_index("crop")["count_error"].abs()
                             - base["count_error"].abs().reindex(scores["crop"]))
                    low, high = animal_ci(delta.values, scores["animal"].values, rng)
                    row.update({"delta_mae": delta.mean(), "delta_ci_low": low, "delta_ci_high": high})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-o", "--output", required=True, help="Output folder.")
    parser.add_argument("--cache", default=None, help="Folder of the cached S3 crops and windows. "
                        "Default: <output>/cache.")
    parser.add_argument("--crops", nargs="+", default=None, help="Only these crops, for a pilot run.")
    args = parser.parse_args()

    cache_folder = args.cache or os.path.join(args.output, "cache")
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(cache_folder, exist_ok=True)
    crops = gt_crops()
    if args.crops is not None:
        crops = [crop for crop in crops if crop["name"] in args.crops]
    rows, skipped = {}, []
    results = []
    for crop in crops:
        cochlea = crop["cochlea"]
        if cochlea not in rows:
            labeling = row_labeling(cochlea)
            rows[cochlea] = None if labeling is None else (
                labeling[0], row_variants(*labeling, [REFINED_MIN_SIZE]), volume_shape(cochlea, SEGMENTATION))
        if rows[cochlea] is None:
            skipped.append(crop["name"])
            continue
        table, variants, volume = rows[cochlea]
        print(f"{crop['name']}: {cochlea}")
        gt = annotation_cells(crop)
        start = crop_start(crop, gt.shape)
        region = read_crop(cochlea, start, gt.shape, volume, cache_folder)
        info = {**crop_info(crop), "cochlea": cochlea}
        results.append({**info, "variant": "objects", **score(region, gt)})
        for variant, (row_ids, path) in variants.items():
            labels = np.where(np.isin(region, row_ids), region, 0)
            results.append({**info, "variant": variant, **score(labels, gt)})
            if variant in CORRECTED:
                corrected, actions = correct_region(region, start, table, row_ids, path, cochlea, SEGMENTATION,
                                                    cache_folder, volume)
                results.append({**info, "variant": f"{variant}+correct", **score(corrected, gt), **actions})
        print(pd.DataFrame(results[-len(variants) - len(CORRECTED) - 1:])[
            ["variant", "n_gt", "n_out", "tp", "fp", "fn", "f1"]].to_string(index=False))

    print(f"Skipped {len(skipped)} crops without an IHC_v11 row labeling: {skipped}")
    results = pd.DataFrame(results)
    results.to_csv(os.path.join(args.output, "evaluation_crops.tsv"), sep="\t", index=False)
    summary = summarize(results)
    summary.to_csv(os.path.join(args.output, "evaluation_crops_summary.tsv"), sep="\t", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
