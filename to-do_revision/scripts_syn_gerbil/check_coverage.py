"""Check that a synapse detection actually covers the whole IHC helix.

The detection pipeline fails silently. If the prediction array does not finish, the maxima
are still detected in whatever blocks were written and the tables look perfectly well formed
-- they are just missing every synapse in the unpredicted regions. The published
G_LR_000302_R result had 6,068 detections spread over a third of its helix and nothing about
the files said so; it took comparing it against the v5 run on the same mask to see it.

The test is deliberately independent of the synapse counts: for every IHC, how far away is
the nearest detection of any kind? Around an IHC with ribbons that distance is a few
micrometer, because the model fires on them. Where there are none it is hundreds of micrometer,
because the nearest detection belongs to a different turn of the helix.

A large distance has two causes and this script cannot tell them apart on its own: the blocks
were never predicted, or they were predicted and hold no detectable ribbons. It therefore reads
the per-task receipts of the prediction array before drawing a conclusion. G_LR_000301_L is the
case that forced this: it came out with a third of its helix uncovered and a complete set of
receipts, and the raw CTBP2 there turned out to be flat -- background around 144 with a maximum
of 303, against 134 and 531 where the detections are.

That distinction is the whole point, and it is why the verdict keys on the *median* distance
rather than on the fraction of IHCs with a detection nearby. A region with genuinely few
synapses has a low covered fraction but a small median distance -- G_LR_000301_R has a stretch
where only half the IHCs carry a detection within 20 um, yet the median there is 10 um, so it
was predicted and simply has fewer ribbons. A region that was never predicted has a median in
the hundreds. Only the second is a pipeline failure.

Usage:
    python check_coverage.py G_LR_000301_L
    python check_coverage.py G_LR_000302_R --compare_old
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

WS = "/mnt/lustre-rzg/workspaces/ws/nim00007/u12086-flamingo-tools"
OUT_ROOT = os.path.join(WS, "synapses-v3")
VAST_PREDICTIONS = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet/predictions"

# An IHC counts as covered when a detection sits within this distance of its anchor. Reported
# for information; the verdict does not depend on it.
COVERED_DISTANCE = 20.0

# Where the per-task receipts of the prediction array live, relative to the output folder.
# If they show a complete array then an uncovered region cannot be an unpredicted one, which
# changes the conclusion completely -- see the verdict at the bottom.
RECEIPT_DIRNAME = "tasks"

# A region whose median nearest detection is further away than this was not predicted. The
# two scales it separates are ~10 um (a predicted IHC, ribbons included) and ~200 um (the gap
# to the next turn of the helix), so anything between them works.
TRUNCATED_DISTANCE = 50.0

# Components smaller than this are helix fragments and stray false-positive IHCs off the
# organ of Corti. They routinely sit outside the dilated mask and carry no detections, which
# says nothing about whether the prediction finished.
MIN_COMPONENT_SIZE = 30

# How many neighbouring bins of the profile have to look unpredicted before it counts as a
# hole. An array task owns a contiguous range of blocks, so an unfinished one always shows up
# as a run of bins in the hundreds of micrometer -- the truncated G_LR_000302_R has a run of
# twelve. A single bin just over the threshold is a patch of weak signal instead:
# G_LR_000301_R has one bin at 51 um whose neighbours are at 12 and 9.
MIN_HOLE_BINS = 2

# Where to read the IHC segmentation table for each cochlea. G_LR_000301_L exists only in the
# workspace; use the finalized dilated-mask re-prediction and its selected components.
IHC_TABLES = {
    "G_EK_000233_L": {"s3": "G_EK_000233_L/tables/IHC_v11/default.tsv"},
    "G_LR_000301_L": {
        "local": os.path.join(WS, "prediction/G301L/IHC_v11_dilated_mask1/default_components.tsv"),
    },
    "G_LR_000301_R": {"s3": "G_LR_000301_R/tables/IHC_v11/default.tsv"},
    "G_LR_000302_R": {"s3": "G_LR_000302_R/tables/IHC_v11/default.tsv"},
}

# The existing result on vast, for --compare_old. For the two finished cochleae this is the
# reference the new runs are judged against; for G_LR_000302_R it is the truncated one being
# replaced. G_EK_000233_L has two copies of its table and only the one under
# synapses_v3_ihc_v11 is in micrometer -- the one under synapses_v3 is the older run in voxels,
# and comparing that against the IHC anchors gives nonsense.
OLD_DETECTIONS = {
    "G_EK_000233_L": os.path.join(VAST_PREDICTIONS, "G_EK_000233_L/synapses_v3_ihc_v11/synapse_detection.tsv"),
    "G_LR_000301_R": os.path.join(VAST_PREDICTIONS, "G_LR_000301_R/synapses_v3/synapse_detection.tsv"),
    "G_LR_000302_R": os.path.join(VAST_PREDICTIONS, "G_LR_000302_R/synapses_v3/synapse_detection.tsv"),
}


def load_ihc_table(cochlea: str) -> pd.DataFrame:
    """Read the IHC_v11 table of one cochlea, from the workspace or from S3."""
    if cochlea not in IHC_TABLES:
        raise KeyError(f"No IHC table configured for {cochlea}, expected one of {list(IHC_TABLES)}.")
    source = IHC_TABLES[cochlea]

    if "local" in source:
        table = pd.read_csv(source["local"], sep="\t")
    else:
        from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target
        s3 = create_s3_target()
        with s3.open(os.path.join(BUCKET_NAME, source["s3"]), mode="rb") as f:
            table = pd.read_csv(f, sep="\t")

    if "component_labels" not in table:
        raise ValueError(f"The IHC table of {cochlea} has no component_labels column.")
    return table


def helix_order(anchors: np.ndarray) -> np.ndarray:
    """Order IHCs along the first principal axis of their anchors.

    A stand-in for ordering them along the helix. It needs no tonotopic mapping, which
    G_LR_000301_L does not have yet, and it is enough to tell a contiguous hole from evenly
    scattered misses.
    """
    centered = anchors - anchors.mean(axis=0)
    # The first right singular vector is the direction of largest variance.
    axis = np.linalg.svd(centered, full_matrices=False)[2][0]
    return np.argsort(centered @ axis)


def prediction_was_complete(output_folder: str):
    """True if the per-task receipts show a complete prediction array, None if unknown.

    A gap in the coverage has two very different causes and this is what separates them. If
    the array did not finish, the blocks are empty and the fix is to resubmit the missing
    tasks. If it did finish, the blocks were predicted and the model found nothing in them,
    which is a property of the image and no amount of re-running will change it.
    """
    tasks_dir = os.path.join(output_folder, RECEIPT_DIRNAME)
    if not os.path.isdir(tasks_dir):
        return None
    found, sizes = set(), set()
    for name in os.listdir(tasks_dir):
        match = re.match(r"^task_(\d+)_of_(\d+)\.json$", name)
        if match:
            found.add(int(match.group(1)))
            sizes.add(int(match.group(2)))
    if len(sizes) != 1:
        return None
    instances = sizes.pop()
    return len(found) == instances and found == set(range(instances))


def check(cochlea: str, detections_path: str, label: str, n_bins: int) -> bool:
    """Report the coverage of one detection table. Returns True if nothing looks truncated."""
    detections = pd.read_csv(detections_path, sep="\t")
    tree = cKDTree(detections[["x", "y", "z"]].values)
    ihc = load_ihc_table(cochlea)
    sizes = ihc.component_labels.value_counts()

    print(f"  {label}: {len(detections)} detections, {len(ihc)} IHCs in the table")

    # Report per component, largest first. A component is a connected run of IHCs, so a hole
    # in the prediction shows up as a component with a large median distance. Component 0 is
    # the unassigned remainder and is reported for information only.
    truncated = []
    for component, size in sizes.items():
        anchors = ihc.loc[ihc.component_labels == component, ["anchor_x", "anchor_y", "anchor_z"]].values
        distances = tree.query(anchors)[0]
        median = float(np.median(distances))
        judged = component != 0 and size >= MIN_COMPONENT_SIZE
        if judged and median > TRUNCATED_DISTANCE:
            truncated.append(component)
        if size < MIN_COMPONENT_SIZE and component != 0:
            continue
        note = "   (unassigned remainder)" if component == 0 else ("" if judged else "   (too small to judge)")
        print(f"    component {component:3d}: {size:5d} IHCs, covered {(distances <= COVERED_DISTANCE).mean():.2f}, "
              f"median nearest detection {median:7.1f} um{note}")
    small = int((sizes.drop(index=0, errors="ignore") < MIN_COMPONENT_SIZE).sum())
    if small:
        print(f"    ({small} components below {MIN_COMPONENT_SIZE} IHCs not listed)")

    # The profile catches a hole inside one large component, which its overall median can hide.
    largest = sizes.drop(index=0, errors="ignore").idxmax()
    anchors = ihc.loc[ihc.component_labels == largest, ["anchor_x", "anchor_y", "anchor_z"]].values
    distances = tree.query(anchors)[0][helix_order(anchors)]
    medians = [float(np.median(b)) for b in np.array_split(distances, n_bins)]
    print(f"    component {largest} along its main axis, median nearest detection per bin "
          f"('#' has detections, '!' above {TRUNCATED_DISTANCE:.0f} um):")
    print("      [" + "".join("!" if m > TRUNCATED_DISTANCE else "#" for m in medians) + "]")
    print("      " + " ".join(f"{m:5.0f}" for m in medians))
    # The longest run of neighbouring bins above the threshold, see MIN_HOLE_BINS.
    longest_run, run = 0, 0
    for median in medians:
        run = run + 1 if median > TRUNCATED_DISTANCE else 0
        longest_run = max(longest_run, run)
    isolated = sum(m > TRUNCATED_DISTANCE for m in medians) - longest_run

    # Phrased as an observation, not a cause: whether this means "not predicted" or "predicted
    # and empty" is decided by the receipts, in the verdict at the end of main().
    if truncated:
        print(f"    -> components {truncated} have no detections near their IHCs")
    if longest_run >= MIN_HOLE_BINS:
        print(f"    -> {longest_run} neighbouring bins of component {largest} have none either")
    elif longest_run or isolated:
        print(f"    -> {longest_run + isolated} isolated bin(s) of component {largest} above "
              f"{TRUNCATED_DISTANCE:.0f} um, but no run of {MIN_HOLE_BINS}: weak signal, not a hole")
    return not (truncated or longest_run >= MIN_HOLE_BINS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("cochlea", help="Cochlea to check.")
    parser.add_argument("-o", "--output_folder", default=None,
                        help="Folder holding synapse_detection.tsv. Defaults to the workspace output.")
    parser.add_argument("--compare_old", action="store_true",
                        help="Also report the coverage of the previous result on vast, for comparison.")
    parser.add_argument("--n_bins", type=int, default=20, help="Number of bins of the coverage profile.")
    args = parser.parse_args()

    output_folder = args.output_folder or os.path.join(OUT_ROOT, args.cochlea)
    detections_path = os.path.join(output_folder, "synapse_detection.tsv")

    print(f"=== {args.cochlea}")
    if os.path.isfile(detections_path):
        ok = check(args.cochlea, detections_path, "new", args.n_bins)
    else:
        # Checking one of the two finished cochleae, which have no run in the workspace.
        print(f"  no new detection at {detections_path}, only checking the result on vast")
        ok = True
        args.compare_old = True

    if args.compare_old:
        old = OLD_DETECTIONS.get(args.cochlea)
        if old is None:
            print(f"  no previous result configured for {args.cochlea}")
        elif not os.path.isfile(old):
            print(f"  previous result not found at {old}")
        else:
            print()
            ok = check(args.cochlea, old, "on vast", args.n_bins) and ok

    print()
    if ok:
        print("Every component and every bin has detections nearby. The result looks complete.")
        return 0

    # Do not blame the pipeline before checking whether it actually finished. The receipts are
    # the authority on that, and if they say the array completed then these regions were
    # predicted and simply hold no detectable ribbons.
    complete = prediction_was_complete(output_folder)
    if complete:
        print("The regions above have no detections nearby, but the prediction array finished:")
        print("every task receipt is present, so those blocks were predicted and the model")
        print("found nothing in them. Re-running changes nothing. Look at the image instead --")
        print("check the raw CTBP2 for punctate signal in those regions, and check that the IHC")
        print("segmentation there is real. Report a partial result, do not resubmit.")
    elif complete is None:
        print("Coverage is incomplete and there are no usable task receipts, so whether the")
        print("prediction finished is unknown. Run verify_prediction.py first.")
    else:
        print("Coverage is incomplete and the task receipts show the prediction array did not")
        print("finish. Run verify_prediction.py and resubmit the tasks it names; do not")
        print("resubmit the whole array on top of a partial one.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
