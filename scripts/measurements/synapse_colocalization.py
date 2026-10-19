import argparse
import json

from flamingo_tools.s3_utils import read_table
from flamingo_tools.validation import compute_scores_from_counts, match_detections


def evaluate_colocalization(reference, test, max_dist):
    """Match the synapses of a test stain against the synapses of a reference stain.

    The test stain takes the role of the detections and the reference stain the role of the ground truth.
    Point matching has no true negatives, so accuracy is TP / (TP + FP + FN).
    """
    coords = ["x", "y", "z"]
    tp, _, fp, fn = match_detections(test[coords].values, reference[coords].values, max_dist=max_dist)
    tp, fp, fn = len(tp), len(fp), len(fn)
    n_total = tp + fp + fn
    ihcs_reference, ihcs_test = set(reference.matched_ihc), set(test.matched_ihc)
    return {
        "n_synapses_reference": len(reference),
        "n_synapses_test": len(test),
        "n_ihcs_reference": len(ihcs_reference),
        "n_ihcs_test": len(ihcs_test),
        "n_ihcs_shared": len(ihcs_reference & ihcs_test),
        "tp": tp, "fp": fp, "fn": fn,
        **compute_scores_from_counts(tp, fp, fn),
        "accuracy": round(tp / n_total, 3) if n_total else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the co-staining of a synapse stain against a reference stain, e.g. CTBP2."
    )
    parser.add_argument("-r", "--reference", required=True, help="Filtered synapse table of the reference stain.")
    parser.add_argument("-t", "--test", required=True, help="Filtered synapse table of the test stain.")
    parser.add_argument("-o", "--output", required=True, help="Output JSON file for the evaluation.")
    parser.add_argument("--max_dist", type=float, default=2.0,
                        help="Maximum distance in micrometer for a match between two synapses.")
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
    result.update(evaluate_colocalization(reference, test, args.max_dist))
    print(json.dumps(result, indent=2))
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
