import argparse
import json

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import read_table
from flamingo_tools.validation import compute_scores_from_counts, match_detections


def evaluate_colocalization(reference, test, max_dist, valid_ihcs=None):
    """Match the synapses of a test stain against the synapses of a reference stain.

    The test stain takes the role of the detections and the reference stain the role of the ground truth.
    Point matching has no true negatives, so accuracy is TP / (TP + FP + FN).
    With `valid_ihcs`, only the synapses of these IHCs are evaluated.

    Returns the scores and a table with the synapse counts per IHC.
    A matched pair counts for the IHC of its reference synapse.
    """
    if valid_ihcs is not None:
        reference = reference[reference.matched_ihc.isin(valid_ihcs)]
        test = test[test.matched_ihc.isin(valid_ihcs)]

    coords = ["x", "y", "z"]
    tp, tp_reference, fp, fn = match_detections(test[coords].values, reference[coords].values, max_dist=max_dist)
    tp, fp, fn = len(tp), len(fp), len(fn)
    n_total = tp + fp + fn
    ihcs_reference, ihcs_test = set(reference.matched_ihc), set(test.matched_ihc)

    if valid_ihcs is None:
        ihc_ids = np.union1d(reference.matched_ihc, test.matched_ihc)
    else:
        ihc_ids = np.unique(valid_ihcs)
    ihc_counts = {"label_id": ihc_ids}
    for name, ihcs in [
        ("n_reference", reference.matched_ihc),
        ("n_test", test.matched_ihc),
        ("n_matched", reference.matched_ihc.iloc[tp_reference]),
    ]:
        ihc_counts[name] = ihcs.value_counts().reindex(ihc_ids, fill_value=0).values

    scores = {
        "n_synapses_reference": len(reference),
        "n_synapses_test": len(test),
        "n_ihcs": len(ihc_ids),
        "n_ihcs_reference": len(ihcs_reference),
        "n_ihcs_test": len(ihcs_test),
        "n_ihcs_shared": len(ihcs_reference & ihcs_test),
        "tp": tp, "fp": fp, "fn": fn,
        **compute_scores_from_counts(tp, fp, fn),
        "accuracy": round(tp / n_total, 3) if n_total else None,
    }
    return scores, pd.DataFrame(ihc_counts)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the co-staining of a synapse stain against a reference stain, e.g. CTBP2."
    )
    parser.add_argument("-r", "--reference", required=True, help="Filtered synapse table of the reference stain.")
    parser.add_argument("-t", "--test", required=True, help="Filtered synapse table of the test stain.")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file for the evaluation.")
    parser.add_argument("--max_dist", type=float, default=2.0,
                        help="Maximum distance in micrometer for a match between two synapses.")
    parser.add_argument("--ihc_table", default=None,
                        help="IHC table for the selection of valid IHCs. Without it, all synapses are evaluated.")
    parser.add_argument("--ihc_column", default="component_labels",
                        help="Column of the IHC table for the selection, e.g. 'is_ihc' for older tables.")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1],
                        help="Values of '--ihc_column' that select the valid IHCs.")
    parser.add_argument("--ihc_counts", default=None,
                        help="Optional output TSV file with the synapse counts per IHC.")
    parser.add_argument("--s3", action="store_true", help="Read the tables from the S3 bucket.")
    parser.add_argument("--s3_credentials", default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")
    args = parser.parse_args()

    s3_kwargs = dict(
        s3=args.s3, bucket_name=args.s3_bucket_name,
        service_endpoint=args.s3_service_endpoint, credential_file=args.s3_credentials,
    )
    reference = read_table(args.reference, **s3_kwargs)
    test = read_table(args.test, **s3_kwargs)

    result = {"reference": args.reference, "test": args.test, "max_dist": args.max_dist}
    valid_ihcs = None
    if args.ihc_table is not None:
        ihc_table = read_table(args.ihc_table, **s3_kwargs)
        valid_ihcs = ihc_table.label_id[ihc_table[args.ihc_column].isin(args.components)].values
        result.update({"ihc_table": args.ihc_table, "ihc_column": args.ihc_column, "components": args.components})

    scores, ihc_counts = evaluate_colocalization(reference, test, args.max_dist, valid_ihcs=valid_ihcs)
    result.update(scores)
    print(json.dumps(result, indent=2))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    if args.ihc_counts is not None:
        ihc_counts.to_csv(args.ihc_counts, sep="\t", index=False)


if __name__ == "__main__":
    main()
