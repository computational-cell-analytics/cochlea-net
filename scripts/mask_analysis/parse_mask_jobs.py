"""Collect the mask rule of every archived mask job into mask_jobs.csv.

The mask of a prediction run is written once by `run_unet_prediction_preprocess_slurm` (the mask,
mean_std and mask_ratio jobs) and reused by later runs into the same output folder. The rule
depends on the code version of the job: fixed per-class thresholds until 2026-06, an adaptive
threshold until 2026-07-24, and an absolute threshold from the template after that. The rule is
read from the `find_mask` source at the job's commit, so dirty working trees can deviate.
"""
import argparse
import ast
import functools
import glob
import json
import os
import re
import subprocess

import pandas as pd

from mask_common import MASK_JOBS_CSV, REPO_ROOT

ARCHIVES = [
    os.path.expanduser("~/Documents/job_archive/job_archive"),
    os.path.expanduser("~/Documents/job_archive_NHR/job_archive"),
    os.path.expanduser("~/Documents/job_archive_NHR_Test"),
]
SLURM_OUTPUTS = [os.path.expanduser("~/Documents/output_slurm_NHR")]
MASK_CALLS = ("run_unet_prediction_preprocess_slurm", "sweep_mask_thresholds")
UNET_FILE = "flamingo_tools/segmentation/unet_prediction.py"


def _expand(value, env):
    value = value.strip().strip('"').strip("'")

    def repl(match):
        return env.get(match.group(1) or match.group(2), match.group(0))

    for _ in range(3):
        value = re.sub(r"\$\{(\w+)\}|\$(\w+)", repl, value)
        value = value.replace('"', "")
    return value


def parse_sbatch(path):
    env = {}
    for line in open(path):
        match = re.match(r"\s*(?:export\s+)?([A-Z_][A-Z0-9_]*)=(.*)$", line.rstrip("\n"))
        if match:
            env[match.group(1)] = _expand(match.group(2), env)
    return env


@functools.lru_cache(maxsize=None)
def _find_mask_rules(commit):
    """Return the find_mask parameters per seg_class at a commit, or None if the commit is unknown."""
    try:
        source = subprocess.run(
            ["git", "show", f"{commit}:{UNET_FILE}"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return None
    tree = ast.parse(source)
    func = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "find_mask")
    arg_names = [a.arg for a in func.args.args]
    defaults = dict(zip(arg_names[len(arg_names) - len(func.args.defaults):], func.args.defaults))
    relative = ast.literal_eval(defaults["relative_threshold"]) if "relative_threshold" in defaults else None
    has_absolute = "absolute_threshold" in arg_names

    def assigns(body):
        values = {}
        for node in body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    values[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        return values

    rules = {}
    if_node = next((node for node in func.body if isinstance(node, ast.If)
                    and "seg_class" in ast.unparse(node.test)), None)
    while if_node is not None:
        match = re.search(r"seg_class == ['\"](\w+)['\"]", ast.unparse(if_node.test))
        rules[match.group(1)] = assigns(if_node.body)
        orelse = if_node.orelse
        if len(orelse) == 1 and isinstance(orelse[0], ast.If):
            if_node = orelse[0]
        else:
            rules["_default"] = assigns(orelse)
            if_node = None
    if not rules:
        # Before the per-class rules: one percentile and threshold for every class.
        text = ast.unparse(func)
        perc = re.search(r"np\.percentile\(\w+, (\d+(?:\.\d+)?)\)", text)
        thr = re.search(r"> (\d+)", text)
        rules["_default"] = {"upper_percentile": float(perc.group(1)), "min_intensity": float(thr.group(1))}
    return {"rules": rules, "relative": relative, "has_absolute": has_absolute}


@functools.lru_cache(maxsize=None)
def _commit_before(date):
    result = subprocess.run(["git", "rev-list", "-1", f"--before={date} 23:59", "master"],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() or None


def mask_rule(commit, seg_class, absolute_threshold, date):
    info = _find_mask_rules(commit) if commit else None
    source = "job_commit"
    if info is None:
        commit = _commit_before(date)
        info = _find_mask_rules(commit) if commit else None
        source = "date"
    if info is None:
        return {"rule_kind": "unknown"}
    params = info["rules"].get(seg_class, info["rules"]["_default"])
    out = {"rule_percentile": params.get("upper_percentile"), "rule_source": source,
           "rule_class": seg_class if seg_class in info["rules"] else "_default"}
    if info["has_absolute"] and absolute_threshold not in (None, "", "None"):
        out.update(rule_kind="absolute", rule_threshold=float(absolute_threshold))
    elif "min_intensity" in params:
        out.update(rule_kind="fixed", rule_threshold=float(params["min_intensity"]))
    else:
        out.update(rule_kind="adaptive", rule_min=params.get("absolute_min", float("nan")),
                   rule_max=params["absolute_max"], rule_relative=info["relative"])
    return out


def slurm_log_info(jobid):
    info = {}
    for root in SLURM_OUTPUTS:
        for path in glob.glob(os.path.join(root, f"slurm-{jobid}*.out")):
            text = open(path, errors="replace").read()
            match = re.search(r"(\d+)/(\d+) \(([\d.]+) %\) used for segmentation", text)
            if match:
                info.update(log_kept_blocks=int(match.group(1)), log_total_blocks=int(match.group(2)))
            match = re.search(r"min_intensity: ([\d.]+)(?: \(global_high=([\d.]+))?", text)
            if match:
                info["log_min_intensity"] = float(match.group(1))
                if match.group(2):
                    info["log_global_high"] = float(match.group(2))
    return info


def parse_job(job_dir):
    sbatch = os.path.join(job_dir, "sbatch.sbatch")
    if not os.path.exists(sbatch):
        candidates = glob.glob(os.path.join(job_dir, "*.sbatch"))
        if not candidates:
            return None
        sbatch = candidates[0]
    text = open(sbatch, errors="replace").read()
    if not any(call in text for call in MASK_CALLS):
        return None
    env = parse_sbatch(sbatch)
    meta_path = os.path.join(job_dir, "metadata.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    repos = {r["repo_name"]: r for r in meta.get("repositories", [])}
    repo = repos.get("cochlea-net") or repos.get("flamingo-tools") or {}
    states = [r.get("state") for r in meta.get("reportseff", [])]

    seg_class = env.get("SEG_MASKING") or env.get("SEG_CLASS")
    match = re.search(r"seg_class=['\"](\w+)['\"]", text)
    if seg_class is None and match:
        seg_class = match.group(1)
    if seg_class is None and "run_unet_prediction_preprocess_slurm" in text:
        seg_class = "sgn"  # The default argument of find_mask.
    output_folder = env.get("OUTPUT_FOLDER", "")
    parts = output_folder.rstrip("/").split("/")
    row = {
        "job": os.path.basename(job_dir),
        "date": meta.get("date", os.path.basename(job_dir)[:10]),
        "jobid": meta.get("jobid"),
        "state": states[-1] if states else None,
        "sweep_only": "sweep_mask_thresholds" in text and "run_unet_prediction_preprocess_slurm" not in text,
        "input": env.get("INPUT"),
        "input_key": env.get("INPUT_KEY"),
        "output_folder": output_folder,
        "cochlea": parts[-2] if len(parts) >= 2 else None,
        "output_name": parts[-1] if parts else None,
        "seg_class": seg_class,
        "absolute_threshold": env.get("ABSOLUTE_THRESHOLD"),
        "commit": repo.get("repo_version"),
        "commit_status": repo.get("status"),
    }
    row.update(mask_rule(row["commit"], seg_class, row["absolute_threshold"], row["date"]))
    if row["jobid"]:
        row.update(slurm_log_info(row["jobid"]))
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=MASK_JOBS_CSV)
    args = parser.parse_args()

    rows, seen = [], set()
    for archive in ARCHIVES:
        for job_dir in sorted(glob.glob(os.path.join(archive, "*"))):
            row = parse_job(job_dir)
            if row is None or (row["jobid"] and row["jobid"] in seen):
                continue
            seen.add(row["jobid"])
            rows.append(row)
    table = pd.DataFrame(rows).sort_values(["cochlea", "output_name", "date"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"{len(table)} mask jobs -> {args.output}")
    print(table.groupby(["rule_kind", "seg_class"], dropna=False).size().to_string())


if __name__ == "__main__":
    main()
