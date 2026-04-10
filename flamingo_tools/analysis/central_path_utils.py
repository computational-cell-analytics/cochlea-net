import math
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from flamingo_tools.s3_utils import get_s3_path


def insert_coordinate(
    df: pd.DataFrame,
    optimal_pos: int,
    new_point: Tuple[int],
) -> pd.DataFrame:
    print(f"Inserting {new_point} at {optimal_pos}.")
    # start of DataFrame
    if optimal_pos == 0:
        entry_dic = {
            "spot_id": len(df),
            "x": new_point[0],
            "y": new_point[1],
            "z": new_point[2],
        }
        insert_df = pd.DataFrame([entry_dic])
        result = pd.concat([
            insert_df,
            df.iloc[:]
        ], ignore_index=True)
    # end of DataFrame
    elif optimal_pos == len(df):
        insert_df = pd.DataFrame([entry_dic])
        result = pd.concat([
            df.iloc[:],
            insert_df
        ], ignore_index=True)

    else:
        # Insert the point at the optimal position
        coord_columns = ["x", "y", "z"]

        extrapolate_columns = [c for c in list(df.columns)[1:] if c not in coord_columns]
        entry_dic = {
            "spot_id": len(df),
            "x": new_point[0],
            "y": new_point[1],
            "z": new_point[2],
        }
        coord_before = [df.iloc[optimal_pos - 1]["x"], df.iloc[optimal_pos - 1]["y"], df.iloc[optimal_pos - 1]["z"]]
        coord_after = [df.iloc[optimal_pos + 1]["x"], df.iloc[optimal_pos + 1]["y"], df.iloc[optimal_pos + 1]["z"]]
        dist_before = math.dist(new_point, coord_before)
        dist_after = math.dist(new_point, coord_after)
        factor_before = dist_before / (dist_before + dist_after)
        factor_after = dist_after / (dist_before + dist_after)
        print(factor_before, factor_after)
        for col in extrapolate_columns:
            value_before = df.iloc[optimal_pos - 1][col]
            value_after = df.iloc[optimal_pos + 1][col]
            entry_dic[col] = factor_before * value_before + factor_after * value_after

        insert_df = pd.DataFrame([entry_dic])
        print(insert_df)

        result = pd.concat([
            df.iloc[:optimal_pos],
            insert_df,
            df.iloc[optimal_pos:]
        ], ignore_index=True)

    return result


def find_optimal_point_for_insert(
    df: pd.DataFrame,
    new_point: Tuple[int],
    tolerance: str = 1e-10,
) -> Tuple[int]:
    """Insert a single point into the path to minimize total path length.

    Args:
        df: pandas DataFrame with 'x' and 'y' columns (ordered path)
        new_point: tuple or array-like (x, y) of the point to insert

    Returns:
        Optimal position for point insertion
    """
    # Convert new point to numpy array
    new_point = np.array(new_point)
    coord_columns = ["x", "y", "z"]
    # Check if point already exists in the dataframe (with tolerance)
    # Check for existing point using numeric columns only
    df_numeric = df[coord_columns]
    distances = np.sqrt(((df_numeric.values - new_point) ** 2).sum(axis=1))
    if (distances < tolerance).any():
        print(f"Point {new_point} already exists in the path. Skipping insertion.")
        return None

    # Calculate distances between consecutive points in the original path
    coords = df[coord_columns].values
    n = len(coords)

    # If path has only one point, insert after it
    if n == 1:
        return 1

    # Calculate the cost of inserting at each position
    # Cost = distance from previous point to new point + distance from new point to next point
    # minus the original distance between previous and next point
    costs = []
    for i in range(n + 1):
        if i == 0:
            # Insert at beginning
            cost = math.dist(new_point, coords[0])
        elif i == n:
            # Insert at end
            cost = math.dist(new_point, coords[n - 1])
        else:
            # Insert between i and i+1
            cost = (math.dist(coords[i - 1], new_point) +
                    math.dist(new_point, coords[i]) -
                    math.dist(coords[i - 1], coords[i]))

        costs.append(cost)

    # Find the position with minimum cost (most reduction in total length)
    optimal_pos = np.argmin(costs)
    print(optimal_pos)

    return int(optimal_pos)


def insert_coordinates_to_central_path(
    table_path: str,
    output_path: str,
    coordinates: List[List[int]],
    s3: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
):
    """Insert coordinates into a central path of an SGN or IHC segmentation.

    Args:
        table_path: Path to table featuring central path as spots.
        output_path: Output path.
        coordinates: List of a single or multiple coordinates to insert into the path.
        s3: Use S3 bucket.
        s3_credentials:
        s3_bucket_name:
        s3_service_endpoint:
    """
    if s3:
        tsv_path, fs = get_s3_path(table_path, bucket_name=s3_bucket_name,
                                   service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
        with fs.open(tsv_path, "r") as f:
            table = pd.read_csv(f, sep="\t")
    else:
        table = pd.read_csv(table_path, sep="\t")

    for coord in coordinates:
        optimal_pos = find_optimal_point_for_insert(table, coord)
        if optimal_pos is not None:
            table = insert_coordinate(table, optimal_pos, coord)

    table.to_csv(output_path, sep="\t", index=False)
