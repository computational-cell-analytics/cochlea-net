"""Build the dataset list of the mask analysis and its frozen train/validation split.

One dataset is one (cochlea, class) pair with a reference segmentation on S3. The relevant
objects are the objects in the component list of that segmentation. The split is made per
animal and stratified by preparation protocol or cohort, so L and R ears and the SGN and IHC
datasets of one animal are always in the same split.
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import pandas as pd

from mask_common import (
    CLASSES, DATASETS_JSON, MASK_JOBS_CSV, REPO_ROOT, VOXEL_SIZE_S0,
    load_figure_util, read_s3_json, read_s3_table,
)

STAIN_CANDIDATES = {"SGN": ["PV", "CR"], "IHC": ["Vglut3", "CR", "MYO"]}
# Below these numbers of relevant objects a table is treated as a failed run without a usable
# reference. A complete cochlea holds about 10,000-20,000 SGNs and 600-1,200 IHCs.
MIN_RELEVANT = {"SGN": 2000, "IHC": 200}
SEED = 20261113
# The MWfLS IHC_v11 tables on S3 hold the refined IHC row as component 1 (2026-11-09). Their mask
# runs are documented (OME-Zarr input, absolute threshold 150, logged kept fraction), so they
# replace the IHC_v9 entries of the processing JSONs, which predate the refinement.
OVERRIDES = {
    (cochlea, "IHC"): ("IHC_v11", [1], "refined_row")
    for cochlea in ("M_AMD_000126_L", "M_AMD_000126_R", "M_AMD_000127_L", "M_AMD_000127_R")
}


def version_key(name):
    match = re.search(r"_v(\d+)([a-z]?)", name)
    return (int(match.group(1)), match.group(2)) if match else (-1, "")


def is_standard_seg(name, cls):
    return re.fullmatch(fr"{cls}_v\d+[a-z]?", name) is not None


def curated_components(cochlea, cls, figure_util):
    """Return {segmentation: (component list, source)} from the processing JSONs and COCHLEA_DICT."""
    curated = {}
    entry = figure_util.COCHLEA_DICT.get(cochlea, {}).get(cls, {})
    for seg, info in entry.items():
        if "component" in info:
            curated[seg] = (list(info["component"]), "cochlea_dict")
    path = os.path.join(REPO_ROOT, "reproducibility", "processing", f"{cochlea}_{cls}.json")
    if os.path.exists(path):
        info = json.load(open(path))
        if "component_list" in info:
            curated[info["segmentation_channel"]] = (list(info["component_list"]), "processing_json")
    return curated


def stain_for(seg, cls, images):
    for stain in images:
        if seg.startswith(f"{stain}_") or seg.endswith(f"_{stain}"):
            return stain
    for stain in STAIN_CANDIDATES[cls]:
        if stain in images:
            return stain
    return None


def cohort_lookup():
    cohorts = json.load(open(os.path.join(REPO_ROOT, "reproducibility", "cohorts.json")))
    lookup = defaultdict(dict)
    for axis, members in cohorts["axes"].items():
        for name, info in members.items():
            for cochlea in info["cochleae"]:
                lookup[cochlea][axis] = name
                lookup[cochlea]["animal_species"] = info.get("animal")
    return lookup, set(cohorts.get("rerun_of", {}))


def reference_run(mask_jobs, cochlea, seg):
    """Return the job that wrote the mask of the reference run, or None.

    The preprocess step keeps an existing mask.zarr, so the earliest completed mask job of an
    output folder decides the mask of every later prediction into that folder.
    """
    jobs = mask_jobs[(mask_jobs.cochlea == cochlea) & (mask_jobs.output_name == seg) & ~mask_jobs.sweep_only]
    if len(jobs) == 0:
        return None, 0
    done = jobs[jobs.state == "COMPLETED"]
    job = (done if len(done) else jobs).sort_values("date").iloc[0]
    rules = jobs[["rule_kind", "rule_percentile", "rule_threshold", "rule_min", "rule_max"]].astype(str)
    return job, len(rules.drop_duplicates())


def describe_rule(job):
    if job is None:
        return {"rule_kind": "unknown"}
    out = {key: (None if pd.isna(job[key]) else job[key]) for key in (
        "rule_kind", "rule_percentile", "rule_threshold", "rule_min", "rule_max", "rule_relative",
        "seg_class", "input", "input_key", "date", "commit", "commit_status", "log_kept_blocks",
        "log_total_blocks", "log_min_intensity",
    ) if key in job}
    input_path = str(job.get("input") or "")
    out["input_format"] = "n5" if input_path.endswith(".n5") else ("ome.zarr" if "ome.zarr" in input_path else "other")
    return out


def s0_voxel_size(fs, cochlea, source):
    attrs = read_s3_json(fs, f"{cochlea}/images/ome-zarr/{source}.ome.zarr/.zattrs")
    for dataset in attrs["multiscales"][0]["datasets"]:
        if dataset["path"] == "s0":
            for transform in dataset.get("coordinateTransformations", []):
                if transform["type"] == "scale":
                    return transform["scale"]
    return None


def evaluate_candidate(fs, cochlea, cls, seg, seg_info, images, curated):
    """Return (dataset entry, None) for a usable reference segmentation, else (None, reason)."""
    table_folder = seg_info.get("tableData", {}).get("tsv", {}).get("relativePath")
    if table_folder is None:
        return None, f"{seg}: no table"
    try:
        table = read_s3_table(fs, cochlea, table_folder)
    except FileNotFoundError:
        return None, f"{seg}: table missing"
    if "component_labels" not in table.columns:
        return None, f"{seg}: no component_labels"
    stain = stain_for(seg, cls, images)
    if stain is None:
        return None, f"{seg}: no {cls} stain"
    scale = s0_voxel_size(fs, cochlea, stain)
    if scale is None or not np.allclose(scale, VOXEL_SIZE_S0, rtol=0.01):
        return None, f"{seg}: s0 voxel size {scale} of {stain}"
    components, component_source = curated.get(seg, ([1], "default"))
    n_relevant = int(table.component_labels.isin(components).sum())
    labeled = sorted(int(c) for c in table.component_labels.unique() if c > 0)
    n_labeled = int((table.component_labels > 0).sum())
    if n_relevant < MIN_RELEVANT[cls] and component_source != "default" and n_labeled >= MIN_RELEVANT[cls]:
        # The curated list does not match the components of the table on S3, e.g. after a
        # relabeling. All labeled components are the closest valid reference.
        components, component_source, n_relevant = labeled, f"all_labeled ({component_source} stale)", n_labeled
    return {
        "stain": stain, "segmentation": seg, "table_folder": table_folder,
        "components": components, "component_source": component_source,
        "n_objects": int(len(table)), "n_relevant": n_relevant,
        "status": "ok" if n_relevant >= MIN_RELEVANT[cls] else "failure",
    }, None


def build_datasets(fs, mask_jobs):
    figure_util = load_figure_util()
    cohorts, reruns = cohort_lookup()
    project = read_s3_json(fs, "project.json")
    datasets, excluded = [], []
    for cochlea in sorted(project["datasets"]):
        if cochlea.startswith("LaVision") or cochlea in reruns or "resized" in cochlea:
            excluded.append({"cochlea": cochlea, "reason": "LaVision, rerun or resized dataset"})
            continue
        try:
            sources = read_s3_json(fs, f"{cochlea}/dataset.json")["sources"]
        except FileNotFoundError:
            excluded.append({"cochlea": cochlea, "reason": "no dataset.json"})
            continue
        images = [name for name, info in sources.items() if "image" in info]
        segs = {name: info["segmentation"] for name, info in sources.items() if "segmentation" in info}
        for cls in CLASSES:
            curated = curated_components(cochlea, cls, figure_util)
            if (cochlea, cls) in OVERRIDES:
                seg, components, source = OVERRIDES[(cochlea, cls)]
                curated = {seg: (components, source)}
            candidates = sorted([s for s in segs if s in curated], key=version_key, reverse=True)
            candidates += sorted([s for s in segs if is_standard_seg(s, cls) and s not in curated],
                                 key=version_key, reverse=True)
            if not candidates:
                continue
            reasons, chosen = [], None
            for seg in candidates:
                entry, reason = evaluate_candidate(fs, cochlea, cls, seg, segs[seg], images, curated)
                if entry is None:
                    reasons.append(reason)
                    continue
                if chosen is None or (chosen["status"] != "ok" and entry["status"] == "ok"):
                    chosen = entry
                if chosen["status"] == "ok":
                    break
            if chosen is None:
                excluded.append({"cochlea": cochlea, "class": cls, "reason": "; ".join(reasons)})
                continue
            job, n_rules = reference_run(mask_jobs, cochlea, chosen["segmentation"])
            info = cohorts.get(cochlea, {})
            chosen.update({
                "key": f"{cochlea}_{cls}", "cochlea": cochlea, "animal": re.sub(r"_[LR]$", "", cochlea),
                "class": cls, "species": {"M": "mouse", "G": "gerbil"}.get(cochlea[0], "other"),
                "protocol": info.get("protocol"), "group": info.get("group"),
                "reference_mask": describe_rule(job), "reference_mask_rules_in_folder": n_rules,
                "skipped_candidates": reasons,
            })
            datasets.append(chosen)
    return datasets, excluded


def stratum_of(ds):
    stratum = ds["protocol"] or ds["group"] or "undefined"
    # The cochleae without a cohort are a mixed set of mice and gerbils.
    return f"undefined_{ds['species']}" if stratum == "undefined" else stratum


def split_animals(datasets):
    """Assign every animal to train or val, stratified by protocol, else cohort group."""
    strata = defaultdict(set)
    for ds in datasets:
        if ds["status"] != "ok":
            continue
        strata[stratum_of(ds)].add(ds["animal"])
    rng = np.random.default_rng(SEED)
    assignment, singleton_turn = {}, 0
    for stratum in sorted(strata):
        animals = sorted(strata[stratum])
        rng.shuffle(animals)
        if len(animals) == 1:
            assignment[animals[0]] = ("train", "val")[singleton_turn % 2]
            singleton_turn += 1
            continue
        n_train = len(animals) // 2 + (rng.random() < 0.5 if len(animals) % 2 else 0)
        for i, animal in enumerate(animals):
            assignment[animal] = "train" if i < n_train else "val"
    for ds in datasets:
        ds["stratum"] = stratum_of(ds)
        ds["split"] = assignment.get(ds["animal"], "failure") if ds["status"] == "ok" else "failure"
    return datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DATASETS_JSON)
    parser.add_argument("--force", action="store_true", help="Overwrite an existing (frozen) split.")
    args = parser.parse_args()
    if os.path.exists(args.output) and not args.force:
        raise FileExistsError(f"{args.output} exists. The split is frozen; pass --force to rebuild it.")

    from flamingo_tools.s3_utils import create_s3_target
    fs = create_s3_target()
    mask_jobs = pd.read_csv(MASK_JOBS_CSV)
    datasets, excluded = build_datasets(fs, mask_jobs)
    datasets = split_animals(datasets)
    with open(args.output, "w") as f:
        json.dump({"seed": SEED, "datasets": datasets, "excluded": excluded}, f, indent=1, default=str)

    table = pd.DataFrame(datasets)
    print(f"{len(table)} datasets, {len(excluded)} exclusions -> {args.output}")
    print(table.groupby(["class", "split"]).size().unstack(fill_value=0).to_string())
    print(table[table.split != "failure"].groupby(["stratum", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
