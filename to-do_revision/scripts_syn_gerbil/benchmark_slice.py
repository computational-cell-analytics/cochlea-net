"""Measure what GPU a synapse prediction task needs, to pick the MIG slice.

`1g.10gb` and `1g.20gb` are both one compute slice of an A100 and differ only in the
framebuffer, so the choice is not a speed trade-off: it is whether the forward pass fits in
10 GB. It is not obvious that it does. The synapse block is (64, 256, 256) with a
(16, 64, 64) halo, so the padded input is 96 x 384 x 384 = 14.2 M voxels, three and a half
times the 160^3 of the SGN model that the existing 1g.10gb measurement came from.

This runs the real prediction path on a slice of the real volume and reports the peak
allocation and the time per block. It writes to a scratch folder, symlinking the mask and the
normalization of the cochlea, so the production output folder is untouched.

Usage:
    python benchmark_slice.py --cochlea G_LR_000301_L --instances 50
"""

import argparse
import json
import os
import shutil
import threading
import time

import torch

from flamingo_tools.segmentation import synapse_detection as sd
from flamingo_tools.segmentation.unet_prediction import prediction_impl

# Imported for their paths only; keep in step with common.sh.
WS = "/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools"
OUT_ROOT = os.path.join(WS, "synapses-v3")
MODEL = ("/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/trained_models/"
         "Synapses/synapse_detection_model_v3.pt")
RAW = {
    "G_LR_000301_L": (os.path.join(WS, "G301L/GLR_301L_CTBP2_fused.n5"), "setup0/timepoint0/s0"),
    "G_LR_000302_R": (os.path.join(WS, "G_LR_000302_R/CTBP2.ome.zarr"), "s0"),
}


class DeviceMonitor:
    """Sample free memory on the whole device, not just this process's tensors.

    torch.cuda.max_memory_allocated only sees the main process. The prefetch workers each
    initialize their own CUDA context on the same slice -- the OOM on 1g.10gb listed nine
    processes holding ~188 MiB each -- so the number that decides whether a slice is big
    enough is device-level occupancy, which cuda.mem_get_info reports.
    """

    def __init__(self, interval: float = 0.2):
        self.interval = interval
        self.min_free = None
        self.total = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            free, total = torch.cuda.mem_get_info()
            self.total = total
            self.min_free = free if self.min_free is None else min(self.min_free, free)
            self._stop.wait(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=2)


def count_chunks(path: str) -> int:
    """Number of chunk files under a zarr array, i.e. blocks actually written."""
    total = 0
    for _, _, files in os.walk(path):
        total += sum(1 for f in files if not f.startswith(".") and not f.endswith(".json"))
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-c", "--cochlea", default="G_LR_000301_L", choices=sorted(RAW))
    parser.add_argument("--instances", type=int, default=50,
                        help="Split the volume into this many tasks and run one of them. Larger "
                             "means fewer blocks and a shorter test; 50 gives about 25 in-mask blocks.")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--scratch", default=None, help="Scratch output folder. Deleted and recreated.")
    args = parser.parse_args()

    source = os.path.join(OUT_ROOT, args.cochlea)
    for name in ("mask.zarr", "mean_std.json"):
        if not os.path.exists(os.path.join(source, name)):
            raise SystemExit(f"{source}/{name} is missing. Run the preprocessing job for {args.cochlea} first.")

    scratch = args.scratch or os.path.join(WS, "mig-benchmark", args.cochlea)
    if os.path.exists(scratch):
        shutil.rmtree(scratch)
    os.makedirs(scratch)
    # Symlink the inputs rather than copy: mask.zarr is small but the point is that this test
    # must not be able to write into the production folder.
    for name in ("mask.zarr", "mean_std.json"):
        os.symlink(os.path.join(source, name), os.path.join(scratch, name))

    with open(os.path.join(source, "mean_std.json")) as f:
        stats = json.load(f)

    if not torch.cuda.is_available():
        raise SystemExit("No GPU visible. Submit this with a --gpus request.")
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU: {name}, {total:.1f} GiB visible")
    print(f"Block shape {sd._PREDICTION_BLOCK_SHAPE}, halo {sd._PREDICTION_HALO}")

    input_path, input_key = RAW[args.cochlea]
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with DeviceMonitor() as monitor:
        prediction_impl(
            input_path, input_key, scratch, MODEL,
            scale=None,
            block_shape=sd._PREDICTION_BLOCK_SHAPE,
            halo=sd._PREDICTION_HALO,
            apply_postprocessing=False,
            output_channels=sd._get_model_out_channels(MODEL),
            prediction_instances=args.instances, slurm_task_id=args.task_id,
            mean=stats["mean"], std=stats["std"],
        )
    elapsed = time.perf_counter() - start

    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
    n_written = count_chunks(os.path.join(scratch, "predictions.zarr", "prediction"))

    print()
    print(f"blocks written        : {n_written}")
    print(f"wall time             : {elapsed:.1f} s")
    if n_written:
        print(f"time per block        : {elapsed / n_written:.2f} s")
    print(f"prefetch workers      : {os.environ.get('SLURM_CPUS_PER_TASK', '?')} cpus requested")
    print(f"peak torch allocated  : {peak_alloc:.2f} GiB   (main process tensors)")
    print(f"peak torch reserved   : {peak_reserved:.2f} GiB   (caching allocator, grows to fit)")
    print(f"visible framebuffer   : {total:.1f} GiB")
    if monitor.min_free is not None:
        used = (monitor.total - monitor.min_free) / 1024**3
        print(f"peak device occupancy : {used:.2f} GiB   (every process on the slice)")
        print()
        # Device occupancy is the number that decides whether a slice is big enough: it counts
        # the worker contexts, which the per-process torch counters do not see.
        print(f"VERDICT: {used:.2f} GiB of {total:.1f} GiB used at peak "
              f"({100 * used / total:.0f} %), {(total - used):.2f} GiB spare")


if __name__ == "__main__":
    main()
