"""Collect exact s0 block statistics for the current mask rule (HPC job, one dataset per task).

`find_mask` keeps a block if its p-th percentile is above T. That is the same as
count(I > T) > (1 - p / 100) * N for the N voxels of the block. The script stores count(I > t)
per 128^3 block for a grid of thresholds t, so the analysis can evaluate the current rule exactly
for every percentile, every threshold of the grid, and for 256^3 blocks (sum of 8 blocks).

The script also harvests the block occupancy of existing mask.zarr folders of the reference
runs (the stored chunk files are the kept blocks) and the chunk shape of the reference input.

Usage on the cluster, with mask_datasets.json and mask_jobs.csv copied next to this script:
    python collect_s0_block_counts.py --task_id $SLURM_ARRAY_TASK_ID --output_folder <out>
"""
import argparse
import json
import multiprocessing as mp
import os
import time
from concurrent import futures

import numpy as np
import pandas as pd

BLOCK = 128
MOBIE_ROOT = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/mobie_project/cochlea-lightsheet"
THRESHOLDS = np.unique(np.concatenate([
    np.round(np.geomspace(80, 5000, 60)), [100, 120, 150, 175, 200, 220, 250, 300, 400],
])).astype("uint16")
PERCENTILES = (95, 99, 99.9)


def open_s0(cochlea, stain, mobie_root):
    import zarr
    local = os.path.join(mobie_root, cochlea, "images", "ome-zarr", f"{stain}.ome.zarr")
    if os.path.isdir(local):
        return zarr.open(local, mode="r")["s0"], local
    from flamingo_tools.s3_utils import get_s3_path
    store, _ = get_s3_path(os.path.join(cochlea, "images", "ome-zarr", f"{stain}.ome.zarr"))
    return zarr.open(store, mode="r")["s0"], "s3"


def block_statistics(volume, n_threads):
    grid = tuple(int(np.ceil(s / BLOCK)) for s in volume.shape)
    n_blocks = int(np.prod(grid))
    counts = np.zeros((n_blocks, len(THRESHOLDS)), dtype="int64")
    n_voxels = np.zeros(n_blocks, dtype="int64")
    vmax = np.zeros(n_blocks, dtype="uint16")
    mean = np.zeros(n_blocks, dtype="float32")
    percentiles = np.zeros((n_blocks, len(PERCENTILES)), dtype="float32")

    def process(block_id):
        index = np.unravel_index(block_id, grid)
        bb = tuple(slice(i * BLOCK, min((i + 1) * BLOCK, s)) for i, s in zip(index, volume.shape))
        data = volume[bb].ravel()
        # Value v is above threshold k exactly if k < (number of thresholds below v).
        below = np.searchsorted(THRESHOLDS, data, side="left")
        hist = np.bincount(below, minlength=len(THRESHOLDS) + 1)
        counts[block_id] = data.size - np.cumsum(hist)[:-1]
        n_voxels[block_id] = data.size
        vmax[block_id] = data.max()
        mean[block_id] = data.mean()
        percentiles[block_id] = np.percentile(data, PERCENTILES)

    with futures.ThreadPoolExecutor(n_threads) as tp:
        list(tp.map(process, range(n_blocks)))
    return grid, counts, n_voxels, vmax, mean, percentiles


def harvest_mask(output_folder):
    """Return the kept-block occupancy and the chunk shape of <output_folder>/mask.zarr, or None."""
    mask_dir = os.path.join(output_folder, "mask.zarr", "mask")
    meta_path = os.path.join(mask_dir, ".zarray")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(mask_dir, "attributes.json")
    if not os.path.exists(meta_path):
        return None
    meta = json.load(open(meta_path))
    shape = meta.get("shape") or meta.get("dimensions")[::-1]
    chunks = meta.get("chunks") or meta.get("blockSize")[::-1]
    grid = tuple(int(np.ceil(s / c)) for s, c in zip(shape, chunks))
    occupancy = np.zeros(grid, dtype=bool)
    for root, _, files in os.walk(mask_dir):
        for name in files:
            if name.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(root, name), mask_dir)
            parts = rel.replace(os.sep, ".").split(".")
            try:
                index = tuple(int(p) for p in parts)
            except ValueError:
                continue
            if len(index) == 3:
                if "dimensions" in meta:  # n5 chunk paths are in x/y/z order
                    index = index[::-1]
                occupancy[index] = True
    return {"shape": list(shape), "chunks": list(chunks), "occupancy": occupancy}


def input_chunks(path, key):
    if not path or not key or not os.path.exists(path):
        return None
    try:
        from elf.io import open_file
        with open_file(path, "r") as f:
            return list(f[key].chunks)
    except Exception as e:  # The reference input may have moved or been deleted.
        return f"error: {e}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--task_id", type=int, required=True, help="Index into the dataset list.")
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--datasets", default=os.path.join(here, "mask_datasets.json"))
    parser.add_argument("--mask_jobs", default=os.path.join(here, "mask_jobs.csv"))
    parser.add_argument("--mobie_root", default=MOBIE_ROOT)
    parser.add_argument("--n_threads", type=int, default=None)
    args = parser.parse_args()

    datasets = json.load(open(args.datasets))["datasets"]
    if args.task_id >= len(datasets):
        print(f"Task {args.task_id} is beyond the {len(datasets)} datasets.")
        return
    ds = datasets[args.task_id]
    out_npz = os.path.join(args.output_folder, f"{ds['key']}.npz")
    if os.path.exists(out_npz):
        print(f"{out_npz} exists.")
        return
    os.makedirs(args.output_folder, exist_ok=True)
    n_threads = args.n_threads or min(32, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                                      else mp.cpu_count())

    t0 = time.time()
    volume, source = open_s0(ds["cochlea"], ds["stain"], args.mobie_root)
    print(f"{ds['key']}: {ds['stain']} s0 {volume.shape} from {source}, {n_threads} threads", flush=True)
    grid, counts, n_voxels, vmax, mean, percentiles = block_statistics(volume, n_threads)
    print(f"block statistics in {time.time() - t0:.0f}s", flush=True)

    jobs = pd.read_csv(args.mask_jobs)
    jobs = jobs[(jobs.cochlea == ds["cochlea"]) & (jobs.output_name == ds["segmentation"])]
    masks, extra = {}, {"source": source, "shape": list(volume.shape), "masks": [], "inputs": []}
    for folder in sorted(set(jobs.output_folder.dropna())):
        harvested = harvest_mask(folder)
        extra["masks"].append({"output_folder": folder, "found": harvested is not None})
        if harvested is not None:
            name = f"mask_{len(masks)}"
            masks[name] = harvested["occupancy"]
            extra["masks"][-1].update(name=name, shape=harvested["shape"], chunks=harvested["chunks"],
                                      kept=int(harvested["occupancy"].sum()), total=int(harvested["occupancy"].size))
    for _, job in jobs.drop_duplicates(["input", "input_key"]).iterrows():
        extra["inputs"].append({"input": job.input, "input_key": job.input_key,
                                "chunks": input_chunks(job.input, job.input_key)})

    np.savez_compressed(out_npz, grid=np.array(grid), thresholds=THRESHOLDS, counts=counts, n_voxels=n_voxels,
                        block_max=vmax, block_mean=mean, percentile_levels=np.array(PERCENTILES),
                        percentiles=percentiles, **masks)
    with open(os.path.join(args.output_folder, f"{ds['key']}.json"), "w") as f:
        json.dump(extra, f, indent=1)
    print(f"done in {time.time() - t0:.0f}s: {json.dumps(extra)}", flush=True)


if __name__ == "__main__":
    main()
