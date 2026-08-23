"""Sanity-check the accuracy of the four SGN v2 seed variants in SGN_3D.json.

None of these checks is about the models being good; they are about the pipeline not being broken.
The decisive one is that `tp + fn` per crop must equal the number of annotations in that crop, which
holds for any segmentation regardless of quality -- so a mismatch means a wrong slice index, a wrong
component or a coordinate frame error rather than a weak network.
"""

import argparse
import json

# The published reference, measured with the same 12 consensus slices and the same watershed
# parameters. The completed IHC seed-variance experiment spread over +-0.01 F1 around its own
# reference, so a variant outside the tolerance below points at a broken stage.
REFERENCE_KEY = "v2"
VARIANT_KEYS = ["v2-1", "v2-2", "v2-3", "v2-4"]
F1_TOLERANCE = 0.03
PRECISION_RECALL_TOLERANCE = 0.05
N_CROPS = 12


def check_results(accuracy_file: str) -> None:
    with open(accuracy_file) as f:
        data = json.load(f)

    problems = []
    if REFERENCE_KEY not in data:
        raise SystemExit(f"{accuracy_file} has no '{REFERENCE_KEY}' entry to compare against.")
    reference = data[REFERENCE_KEY]
    reference_counts = [tp + fn for tp, fn in zip(reference["tp"], reference["fn"])]

    missing = [key for key in VARIANT_KEYS if key not in data]
    if missing:
        raise SystemExit(f"Missing entries in {accuracy_file}: {missing}")

    print(f"{'key':6s} {'precision':>10s} {'recall':>8s} {'f1':>8s} {'crops':>6s}")
    for key in [REFERENCE_KEY] + VARIANT_KEYS:
        entry = data[key]
        print(f"{key:6s} {entry['precision']:10.3f} {entry['recall']:8.3f} "
              f"{entry['f1-score']:8.3f} {len(entry['crops']):6d}")

    for key in VARIANT_KEYS:
        entry = data[key]
        if len(entry["crops"]) != N_CROPS:
            problems.append(f"{key}: {len(entry['crops'])} crops, expected {N_CROPS}")
        if entry["crops"] != reference["crops"]:
            problems.append(f"{key}: evaluated different crops than '{REFERENCE_KEY}'")
        counts = [tp + fn for tp, fn in zip(entry["tp"], entry["fn"])]
        if counts != reference_counts:
            problems.append(
                f"{key}: tp+fn per crop is {counts}, expected {reference_counts}. This is the "
                "number of annotations per crop and must match for any segmentation, so the slice "
                "index, the component filter or the coordinate frame is wrong."
            )
        for name, tolerance in (
            ("f1-score", F1_TOLERANCE),
            ("precision", PRECISION_RECALL_TOLERANCE),
            ("recall", PRECISION_RECALL_TOLERANCE),
        ):
            deviation = abs(entry[name] - reference[name])
            if deviation > tolerance:
                problems.append(
                    f"{key}: {name} {entry[name]} deviates by {deviation:.3f} from the reference "
                    f"{reference[name]} (tolerance {tolerance})"
                )

    # Identical scores across all four variants mean the same segmentation was scored four times,
    # which is what a shared evaluation cache folder does.
    f1_scores = {data[key]["f1-score"] for key in VARIANT_KEYS}
    if len(f1_scores) == 1:
        problems.append(
            "All four variants have the same F1-score. The most likely cause is a shared "
            "--cache_folder: the cache file names do not contain the segmentation name."
        )

    if problems:
        print("\nProblems:")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit("Result check FAILED.")

    spread = max(data[k]["f1-score"] for k in VARIANT_KEYS) - min(data[k]["f1-score"] for k in VARIANT_KEYS)
    print(f"\nF1 spread across the four seeds: {spread:.3f}")
    print("Result check passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--accuracy_file", required=True, help="Path to SGN_3D.json.")
    args = parser.parse_args()
    check_results(args.accuracy_file)


if __name__ == "__main__":
    main()
