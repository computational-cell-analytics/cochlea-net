"""Check that every task of a prediction array finished, and which ones to resubmit.

This is what makes the preemptible queue safe to use. `run_synapse_prediction_slurm` has no
resume: a task that is killed leaves its blocks empty, and the detection step then writes a
complete-looking table from a partial volume. Preemption makes that a routine event rather
than an accident, so it has to be detectable and cheaply repairable.

The authority is a per-task receipt. The apply job writes `tasks/task_<id>_of_<n>.json` after
`run_synapse_prediction_slurm` returns, so the receipt exists only if that task ran to
completion. A preempted task leaves none, and the missing ids are exactly the ones to
resubmit -- each redoes its own share, which the block assignment makes deterministic (a
permutation under seed 1234, split by `np.array_split`).

Deriving the same answer from the data instead was tried and abandoned. One block is one chunk
file, so missing chunks look like they should pinpoint missing blocks, and they nearly do: the
obstacle is knowing which blocks *should* have a chunk. That means replicating the skip rule
in `_prepare_block_input`, which tests the inner block of `ResizedVolume(mask, shape, order=0)`,
and a floor/ceil approximation of that nearest-neighbour rounding disagreed with the real thing
on 2 of 26 blocks in a test task -- in both directions. A receipt is exact and needs no
replication, so the chunk count is reported here for information only, never gated on.

Compare with check_coverage.py, which asks the biological question (is every IHC near a
detection) and belongs after the detection step. This asks the mechanical question (did every
task finish) and belongs after the array.

Usage:
    python verify_prediction.py G_LR_000301_L
    python verify_prediction.py G_LR_000301_L --instances 10
"""

import argparse
import json
import os
import re

WS = "/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools"
OUT_ROOT = os.path.join(WS, "synapses-v3")
COCHLEAE = ("G_LR_000301_L", "G_LR_000302_R")

RECEIPT_PATTERN = re.compile(r"^task_(\d+)_of_(\d+)\.json$")


def count_chunks(prediction_path: str) -> int:
    """Number of chunk files in the prediction array, i.e. blocks written."""
    total = 0
    for _, _, files in os.walk(prediction_path):
        total += sum(1 for f in files if not f.endswith(".json") and not f.startswith("."))
    return total


def read_receipts(tasks_dir: str):
    """Receipts on disk, as {task_id: payload}, plus the array sizes they claim."""
    receipts, sizes = {}, set()
    if not os.path.isdir(tasks_dir):
        return receipts, sizes
    for name in sorted(os.listdir(tasks_dir)):
        match = RECEIPT_PATTERN.match(name)
        if match is None:
            continue
        task_id, instances = int(match.group(1)), int(match.group(2))
        sizes.add(instances)
        with open(os.path.join(tasks_dir, name)) as f:
            try:
                receipts[task_id] = json.load(f)
            except json.JSONDecodeError:
                # A receipt is written in one go after the prediction returns, so a truncated
                # one means the job died mid-write. Treat it as absent.
                print(f"  warning: {name} is not valid JSON, treating the task as unfinished")
    return receipts, sizes


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cochlea", choices=COCHLEAE)
    parser.add_argument("-o", "--output_folder", default=None,
                        help="Folder holding predictions.zarr. Defaults to the workspace output.")
    parser.add_argument("-n", "--instances", type=int, default=None,
                        help="Expected array size. Defaults to what the receipts say, which is "
                             "what you want unless no receipt exists at all.")
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(OUT_ROOT, args.cochlea)
    prediction_path = os.path.join(output_folder, "predictions.zarr", "prediction")
    tasks_dir = os.path.join(output_folder, "tasks")

    print(f"=== {args.cochlea}")
    print(f"  {output_folder}")
    if not os.path.exists(prediction_path):
        print("  no predictions.zarr: nothing has run yet")
        return 1

    receipts, sizes = read_receipts(tasks_dir)
    print(f"  chunks written: {count_chunks(prediction_path)}   (information only)")

    if len(sizes) > 1:
        print(f"\n  Receipts disagree on the array size: {sorted(sizes)}.")
        print("  The folder holds output from two different array sizes, whose block")
        print("  assignments do not match. Delete predictions.zarr and start over.")
        return 1

    instances = args.instances or (sizes.pop() if sizes else None)
    if instances is None:
        print("\n  No receipts at all. Either the array never ran, or it predates the receipts.")
        print("  Pass --instances to say what size to expect.")
        return 1

    missing = [task_id for task_id in range(instances) if task_id not in receipts]
    print(f"  receipts: {len(receipts)} of {instances}")

    elapsed = [r["elapsed_s"] for r in receipts.values() if "elapsed_s" in r]
    if elapsed:
        print(f"  task run time: {min(elapsed) / 60:.1f} to {max(elapsed) / 60:.1f} min")
    requeued = sorted(task_id for task_id, r in receipts.items() if r.get("restarts"))
    if requeued:
        print(f"  tasks that were requeued at least once: {requeued}")

    if not missing:
        print("\nEvery task finished. The prediction is complete.")
        return 0

    print(f"\n{len(missing)} task(s) did not finish: {missing}")
    print("Resubmit exactly those; each redoes its own share, which is a few minutes:")
    print(f"  sbatch --array={','.join(str(t) for t in missing)} \\")
    print(f"    2026-08-24_sbatch_apply_syn-v3.sbatch {args.cochlea}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
