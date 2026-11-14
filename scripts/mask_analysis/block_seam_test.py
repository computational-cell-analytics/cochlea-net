"""Real-data test: why do cells split at 128^3 block faces?

Predict an aligned crop (core of whole blocks plus a ring of one block for context) with the
production tiling and a given prediction halo, then run the production watershed with different
halos. Count label pairs that touch across block faces (splits) and compare with control planes
in the block interior. Runs on CPU; one crop takes about 10 minutes per prediction halo.

Example (report section 3):
    python block_seam_test.py --out <folder> --pred_halo 32 32 32
    python block_seam_test.py --cochlea M_AMD_000126_R --stain Vglut3 --seg IHC_v11 --model IHC \
        --min_size 5000 --core_start 384 2176 2560 --out <folder>
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from elf.io import open_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mask_common import open_image  # noqa: E402
from flamingo_tools.model_utils import get_model, get_default_segmentation_settings  # noqa: E402
from flamingo_tools.segmentation import unet_prediction as up  # noqa: E402


def contact_pairs(seg, axis, plane, min_contact):
    a = np.take(seg, plane - 1, axis=axis)
    b = np.take(seg, plane, axis=axis)
    both = (a > 0) & (b > 0) & (a != b)
    pairs, counts = np.unique(np.stack([a[both], b[both]]), axis=1, return_counts=True)
    return int(np.sum(counts >= min_contact)), int(len(np.unique(np.concatenate([a[a > 0], b[b > 0]]))))


def face_statistics(seg, core_lo, core_hi, min_contact=20):
    """Splits at the block faces around and inside the core, and at mid-block control planes.

    The region is the core plus 64 voxels on each side, inside the ring of context blocks.
    """
    rows = []
    lo, hi = [c - 64 for c in core_lo], [c + 64 for c in core_hi]
    sub = seg[tuple(slice(a, b) for a, b in zip(lo, hi))]
    for axis in range(3):
        extent = hi[axis] - lo[axis]
        for local in range(64, extent, 64):
            kind = "face" if (lo[axis] + local) % 128 == 0 else "control"
            splits, n_objects = contact_pairs(sub, axis, local, min_contact)
            rows.append({"axis": "zyx"[axis], "plane": lo[axis] + local, "kind": kind,
                         "splits": splits, "objects": n_objects})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cochlea", default="G_EK_000076_R")
    parser.add_argument("--stain", default="PV")
    parser.add_argument("--seg", default="SGN_v2")
    parser.add_argument("--model", default="SGN")
    parser.add_argument("--core_start", type=int, nargs=3, default=[3200, 3712, 1536])
    parser.add_argument("--core_blocks", type=int, nargs=3, default=[2, 1, 1])
    parser.add_argument("--min_size", type=int, default=1000)
    parser.add_argument("--pred_halo", type=int, nargs=3, default=[16, 32, 32])
    parser.add_argument("--ws_halos", default="2,8,8;8,8,8;8,16,16;16,32,32")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(8)

    core_start = np.array(args.core_start)
    core_stop = core_start + 128 * np.array(args.core_blocks)
    crop_start, crop_stop = core_start - 128, core_stop + 128
    crop = tuple(slice(a, b) for a, b in zip(crop_start, crop_stop))
    core_lo, core_hi = (core_start - crop_start).tolist(), (core_stop - crop_start).tolist()

    raw_path = os.path.join(args.out, "raw.zarr")
    if not os.path.exists(raw_path):
        t = time.time()
        raw = open_image(args.cochlea, args.stain, "s0")[crop]
        ref = open_image(args.cochlea, args.seg, "s0")[crop]
        with open_file(raw_path, "a") as f:
            f.create_dataset("raw", data=raw, chunks=(64, 64, 64), compression="gzip")
            f.create_dataset("reference", data=ref, chunks=(64, 64, 64), compression="gzip")
        print(f"read crop {raw.shape} in {time.time() - t:.0f}s", flush=True)

    model_file = os.path.join(args.out, "model.pt")
    if not os.path.exists(model_file):
        torch.save(get_model(args.model, device="cpu"), model_file)

    tag = "_".join(map(str, args.pred_halo))
    pred_folder = os.path.join(args.out, "pred" if tag == "16_32_32" else f"pred_{tag}")
    pred_path = os.path.join(pred_folder, "predictions.zarr")
    done = os.path.join(args.out, "pred_done" if tag == "16_32_32" else f"pred_done_{tag}")
    if not os.path.exists(done):
        t = time.time()
        os.makedirs(pred_folder, exist_ok=True)
        up.prediction_impl(raw_path, "raw", pred_folder, model_file, None, (128, 128, 128), tuple(args.pred_halo),
                           output_channels=3, apply_postprocessing=True)
        open(done, "w").close()
        print(f"prediction in {time.time() - t:.0f}s", flush=True)

    settings = get_default_segmentation_settings(args.model)
    original = up.parallel.seeded_watershed
    with open_file(raw_path, "r") as f:
        results = {"reference": face_statistics(f["reference"][:], core_lo, core_hi)}
    halos = [tuple(int(v) for v in h.split(",")) for h in args.ws_halos.split(";")]
    for halo in halos:
        def patched(*a, halo_=halo, **kw):
            kw["halo"] = halo_
            return original(*a, **kw)
        up.parallel.seeded_watershed = patched
        folder = os.path.join(pred_folder, "ws_" + "_".join(map(str, halo)))
        os.makedirs(folder, exist_ok=True)
        t = time.time()
        up.distance_watershed_implementation(
            pred_path, output_folder=folder, min_size=args.min_size,
            center_distance_threshold=settings["center_distance_threshold"],
            boundary_distance_threshold=settings["boundary_distance_threshold"],
            fg_threshold=settings["fg_threshold"], distance_smoothing=settings["distance_smoothing"],
        )
        with open_file(os.path.join(folder, "segmentation.zarr"), "r") as f:
            seg = f["segmentation"]
            print(f"halo {halo}: watershed in {time.time() - t:.0f}s, chunks {seg.chunks}", flush=True)
            results[str(halo)] = face_statistics(seg[:], core_lo, core_hi)
    up.parallel.seeded_watershed = original

    with open(os.path.join(pred_folder, "face_statistics.json"), "w") as f:
        json.dump({"args": vars(args), "results": results}, f, indent=1)
    for name, rows in results.items():
        face = [r for r in rows if r["kind"] == "face"]
        control = [r for r in rows if r["kind"] == "control"]
        print(f"{name:>14}: face splits {sum(r['splits'] for r in face):4d} over {len(face)} planes, "
              f"control splits {sum(r['splits'] for r in control):4d} over {len(control)} planes")


if __name__ == "__main__":
    main()
