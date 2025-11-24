import argparse
import json
from typing import Optional, List

import pandas as pd

import flamingo_tools.postprocessing.label_components as postprocessing
import flamingo_tools.s3_utils as s3_utils


def main(
    in_path: str,
    out_path: str,
    n_neighbors: Optional[List[int]] = None,
    local_ripley_radius: Optional[List[int]] = None,
    r_neighbors: Optional[List[int]] = None,
    s3: Optional[bool] = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Expand TSV table with additional parameters for postprocessing.

    Args:
        in_path: Path to table in TSV format.
        out_path: Path to save output.
        n_neighbors: Value(s) for nearest neighbor distance.
        local_ripley_radius: Value(s) for calculating local Ripley's K function.
        r_neighbors: Value(s) for radii for calculating number of neighbors in range.
        s3: Flag for considering in_path fo S3 bucket.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        s3_credentials: File path to credentials for S3 bucket.
    """
    if s3:
        tsv_path, fs = s3_utils.get_s3_path(in_path, bucket_name=s3_bucket_name,
                                            service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, 'r') as f:
            tsv_table = pd.read_csv(f, sep="\t")
    else:
        with open(in_path, 'r') as f:
            tsv_table = pd.read_csv(f, sep="\t")

    if n_neighbors is not None:
        for n_neighbor in n_neighbors:
            if n_neighbor >= len(tsv_table):
                raise ValueError(f"Number of neighbors {n_neighbor} exceeds elements in dataframe: {len(tsv_table)}.")

            distance_avg = postprocessing.nearest_neighbor_distance(table=tsv_table, n_neighbors=n_neighbor)
            tsv_table['distance_nn'+str(n_neighbor)] = list(distance_avg)

    if local_ripley_radius is not None:
        for lr_radius in local_ripley_radius:
            local_k = postprocessing.local_ripleys_k(table=tsv_table, radius=lr_radius)
            tsv_table['local_ripley_radius'+str(lr_radius)] = list(local_k)

    if r_neighbors is not None:
        for r_neighbor in r_neighbors:
            neighbor_counts = postprocessing.neighbors_in_radius(table=tsv_table, radius=r_neighbor)
            neighbor_counts = list(neighbor_counts)
            neighbor_counts = [n[0] for n in neighbor_counts]
            tsv_table['neighbors_in_radius'+str(r_neighbor)] = neighbor_counts

    tsv_table.to_csv(out_path, sep="\t", index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Script for expanding the segmentation table of MoBIE with additonal parameters. "
        "Either locally or on an S3 bucket.")

    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)

    parser.add_argument("--n_neighbors", default=None,
                        help="Value(s) for calculating distance to 'n' nearest neighbors, json-encoded.")

    parser.add_argument("--local_ripley_radius", default=None,
                        help="Value(s) for radii for calculating local Ripley's K function, json-encoded.")

    parser.add_argument("--r_neighbors", default=None,
                        help="Value(s) for radii for calculating number of neighbors in range, json-encoded.")

    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    n_neighbors = json.loads(args.n_neighbors) if args.n_neighbors is not None else None
    local_ripley_radius = json.loads(args.local_ripley_radius) if args.local_ripley_radius is not None else None
    r_neighbors = json.loads(args.r_neighbors) if args.r_neighbors is not None else None

    main(
        args.input, args.output, n_neighbors, local_ripley_radius, r_neighbors,
        args.s3, args.s3_credentials, args.s3_bucket_name, args.s3_service_endpoint,
    )
