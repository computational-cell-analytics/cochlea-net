"""Collect low-resolution intensity features of every dataset for the mask rule evaluation.

The stain is read once at a pyramid level (default s3 = 3.04 um, 8x per axis). The script
stores, per dataset:

* the exact histogram of the non-zero voxels (fusion padding is exactly 0);
* per 128^3 s0 block: the maximum of the volume after a box maximum filter of radius r, for
  several r. This is the maximum intensity within r of the block, the block score of a
  distance-limited dilation;
* per block: p95, p99 and the mean at the low level, as an approximation of the s0 rule;
* per relevant object: the bounding box in s0 voxels and the low-res maximum inside it.

It reads from S3 by default. Pass --local_root to read a local MoBIE copy, e.g. on the cluster.
"""
import argparse
import os
import time
import traceback
from concurrent import futures

import numpy as np
from scipy import ndimage

from mask_common import BLOCK, VOXEL_SIZE_S0, load_datasets, open_image, output_path, read_s3_table, table_bboxes_s0

RADII_UM = (0, 6, 12, 18, 24, 30)
QUANTILES = (50, 90, 95, 99, 99.9, 99.99, 99.999)


def histogram_quantiles(hist, quantiles):
    cdf = np.cumsum(hist)
    total = cdf[-1]
    return np.array([np.searchsorted(cdf, q / 100 * total, side="left") for q in quantiles], dtype="float64")


def block_view(volume, block):
    """Pad the volume to a multiple of the block and return (grid shape, (n_blocks, block^3) view)."""
    grid = tuple(int(np.ceil(s / block)) for s in volume.shape)
    padded = np.zeros(tuple(g * block for g in grid), dtype=volume.dtype)
    padded[tuple(slice(0, s) for s in volume.shape)] = volume
    view = padded.reshape(grid[0], block, grid[1], block, grid[2], block).transpose(0, 2, 4, 1, 3, 5)
    return grid, view.reshape(int(np.prod(grid)), block ** 3)


def collect(ds, level, local_root, out_path):
    scale = 2 ** int(level[1:])
    block = BLOCK // scale
    voxel_um = VOXEL_SIZE_S0 * scale
    t0 = time.time()
    volume = open_image(ds["cochlea"], ds["stain"], level, local_root)[:]
    t_read = time.time() - t0

    nonzero = volume[volume > 0]
    hist = np.bincount(nonzero, minlength=65536).astype("int64")
    quantiles = histogram_quantiles(hist, QUANTILES)
    median = quantiles[0]
    mad = histogram_quantiles(np.bincount(np.abs(nonzero.astype("int32") - int(median)), minlength=65536), (50,))[0]

    grid, view = block_view(volume, block)
    block_p95 = np.percentile(view, 95, axis=1).reshape(grid).astype("float32")
    block_p99 = np.percentile(view, 99, axis=1).reshape(grid).astype("float32")
    block_mean = view.mean(axis=1).reshape(grid).astype("float32")
    block_nonzero = (view > 0).mean(axis=1).reshape(grid).astype("float32")
    del view

    # Box maximum filters compose: two passes of size 2a+1 and 2b+1 equal one of size 2(a+b)+1.
    block_max = {}
    filtered, current = volume, 0
    for r_um in RADII_UM:
        r_vox = int(round(r_um / voxel_um))
        if r_vox > current:
            filtered = ndimage.maximum_filter(filtered, size=2 * (r_vox - current) + 1, mode="constant", cval=0)
            current = r_vox
        _, view = block_view(filtered, block)
        block_max[r_um] = view.max(axis=1).reshape(grid)
        del view

    from flamingo_tools.s3_utils import create_s3_target
    table = read_s3_table(create_s3_target(), ds["cochlea"], ds["table_folder"])
    relevant = table[table.component_labels.isin(ds["components"])]
    bb_min, bb_max = table_bboxes_s0(relevant)
    lo = np.clip(bb_min // scale, 0, np.array(volume.shape) - 1)
    hi = np.clip(-(-bb_max // scale), lo + 1, np.array(volume.shape))
    object_max = np.array([volume[a[0]:b[0], a[1]:b[1], a[2]:b[2]].max() for a, b in zip(lo, hi)], dtype="float32")
    anchors = np.stack([relevant[f"anchor_{ax}"].values for ax in "zyx"], axis=1) / (VOXEL_SIZE_S0 * scale)
    anchors = np.clip(np.round(anchors).astype("int64"), 0, np.array(volume.shape) - 1)
    object_anchor = volume[tuple(anchors.T)].astype("float32")

    np.savez_compressed(
        out_path, level=level, scale=scale, shape_lowres=np.array(volume.shape), grid=np.array(grid),
        hist=hist, quantile_levels=np.array(QUANTILES), quantiles=quantiles, median=median, mad=mad,
        zero_fraction=1 - nonzero.size / volume.size, block_p95=block_p95, block_p99=block_p99,
        block_mean=block_mean, block_nonzero=block_nonzero, radii_um=np.array(RADII_UM),
        block_max=np.stack([block_max[r] for r in RADII_UM]),
        label_id=relevant.label_id.values, component=relevant.component_labels.values,
        n_pixels=relevant.n_pixels.values, bb_min=bb_min, bb_max=bb_max,
        object_max=object_max, object_anchor=object_anchor,
    )
    return t_read, time.time() - t0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", default="s3")
    parser.add_argument("--keys", nargs="*", help="Dataset keys. Default: all datasets.")
    parser.add_argument("--local_root", default=None, help="Root of a local MoBIE project instead of S3.")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    datasets = load_datasets()["datasets"]
    if args.keys:
        datasets = [ds for ds in datasets if ds["key"] in args.keys]
    todo = []
    for ds in datasets:
        out_path = output_path(f"lowres_{args.level}", f"{ds['key']}.npz")
        if not os.path.exists(out_path):
            todo.append((ds, out_path))
    print(f"{len(todo)} of {len(datasets)} datasets to process at {args.level}", flush=True)

    def run(item):
        ds, out_path = item
        try:
            t_read, t_total = collect(ds, args.level, args.local_root, out_path)
            print(f"{ds['key']:28s} read {t_read:6.1f}s total {t_total:6.1f}s", flush=True)
        except Exception:
            print(f"{ds['key']:28s} FAILED\n{traceback.format_exc()}", flush=True)

    with futures.ThreadPoolExecutor(args.workers) as tp:
        list(tp.map(run, todo))


if __name__ == "__main__":
    main()
