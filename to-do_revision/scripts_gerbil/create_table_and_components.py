"""Create a MoBIE compatible segmentation table for a local segmentation and label its components.

This is a light-weight replacement for adding the segmentation to MoBIE, which is the usual way of
obtaining the segmentation table. The table is computed block-wise with 'bioimage_py.morphology',
so that it also works for full cochlea volumes, and it is written next to the segmentation.
The columns are the same as in a MoBIE table (see 'compute_table_on_the_fly' in
flamingo_tools.postprocessing.label_components), i.e. the anchor (= centroid) and the bounding box
are given in micrometer, and the bounding box maximum is exclusive.
"""

import argparse
import multiprocessing as mp
import os
from typing import Optional, Sequence, Union

import pandas as pd

from bioimage_py import open_source
from bioimage_py.morphology import morphology

from flamingo_tools.postprocessing.label_components import label_components_single

TABLE_NAME = "default.tsv"
"""The name of the segmentation table, matching the MoBIE table name."""

COMPONENT_TABLE_NAME = "default_components.tsv"
"""The name of the segmentation table with the additional 'component_labels' column."""


def morphology_to_mobie_table(
    morphology_table: pd.DataFrame,
    voxel_size: Union[float, Sequence[float]] = 0.38,
) -> pd.DataFrame:
    """Convert a morphology table computed by 'bioimage_py.morphology' into a MoBIE table.

    Args:
        morphology_table: The morphology table, with coordinates in pixels.
        voxel_size: The physical voxel spacing of the data in (x, y, z) order.

    Returns:
        The segmentation table, with the anchor and bounding box in micrometer.
    """
    if isinstance(voxel_size, float):
        voxel_size = 3 * (voxel_size,)
    voxel_size = tuple(voxel_size)
    if len(voxel_size) == 1:
        voxel_size = 3 * voxel_size
    assert len(voxel_size) == 3
    voxel_size = {axis: size for axis, size in zip("xyz", voxel_size)}

    table = pd.DataFrame({"label_id": morphology_table["label"].astype("int64")})
    # Transform the pixel distances to physical units. The bounding box maximum of the morphology
    # table is already the exclusive stop, matching the MoBIE table.
    for axis in "xyz":
        table[f"anchor_{axis}"] = morphology_table[f"com_{axis}"] * voxel_size[axis]
    for prefix in ("bb_min", "bb_max"):
        for axis in "xyz":
            table[f"{prefix}_{axis}"] = morphology_table[f"{prefix}_{axis}"] * voxel_size[axis]
    table["n_pixels"] = morphology_table["size"].astype("int64")

    # Reorder to the MoBIE column order.
    columns = ["label_id"] + [f"{name}_{axis}" for name in ("anchor", "bb_min", "bb_max") for axis in "xyz"]
    return table[columns + ["n_pixels"]]


def create_table_and_components(
    segmentation_folder: str,
    cell_type: str,
    segmentation_key: str = "segmentation",
    voxel_size: Union[float, Sequence[float]] = 0.38,
    n_threads: Optional[int] = None,
    force: bool = False,
    **component_kwargs,
) -> None:
    """Create the segmentation table for a local segmentation and label its connected components.

    Both tables are written next to the segmentation, as 'default.tsv' and 'default_components.tsv'.

    Args:
        segmentation_folder: The folder with the segmentation, i.e. the output folder of the U-Net pipeline.
        cell_type: The cell type of the segmentation. Either 'sgn' or 'ihc'.
        segmentation_key: The key of the segmentation in 'segmentation.zarr'.
        voxel_size: The physical voxel spacing of the data in (x, y, z) order.
        n_threads: The number of threads. By default all available cores, capped at 16, are used.
        force: Recompute the segmentation table even if it already exists.
        component_kwargs: Parameters for the component labeling, see 'label_components_single'.
    """
    segmentation_path = os.path.join(segmentation_folder, "segmentation.zarr")
    table_path = os.path.join(segmentation_folder, TABLE_NAME)

    if os.path.exists(table_path) and not force:
        print(f"The segmentation table {table_path} already exists and is used as is. Pass --force to recompute it.")
    else:
        if n_threads is None:
            n_threads = min(16, mp.cpu_count())
        segmentation = open_source(segmentation_path, segmentation_key)
        print(f"Computing the segmentation table for shape {tuple(segmentation.shape)} with {n_threads} threads.")
        # The morphology is computed block-wise, using the chunks of the segmentation as blocks.
        table = morphology_to_mobie_table(
            morphology(segmentation, num_workers=n_threads), voxel_size=voxel_size
        )
        table.to_csv(table_path, sep="\t", index=False)
        print(f"Wrote the segmentation table with {len(table)} objects to {table_path}.")

    # The component table is always rewritten, as the component parameters are what gets tuned.
    label_components_single(
        table_path=table_path, out_path=os.path.join(segmentation_folder, COMPONENT_TABLE_NAME),
        cell_type=cell_type, force_overwrite=True, **component_kwargs,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a MoBIE compatible segmentation table for a local segmentation "
        "and label its connected components."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="The folder with the segmentation, i.e. the output folder of the U-Net pipeline. "
                        f"The tables are written into it as '{TABLE_NAME}' and '{COMPONENT_TABLE_NAME}'.")
    parser.add_argument("--cell_type", default="sgn", help="Cell type of the segmentation. Either 'sgn' or 'ihc'.")
    parser.add_argument("--segmentation_key", default="segmentation",
                        help="The key of the segmentation in 'segmentation.zarr'.")
    parser.add_argument("--voxel_size", type=float, nargs="+", default=[0.38],
                        help="The voxel size of the segmentation in micrometer, in (x, y, z) order.")
    parser.add_argument("--n_threads", type=int, default=None, help="The number of threads.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Recompute the segmentation table even if it already exists.")

    # Options for the component labeling.
    parser.add_argument("--min_size", type=int, default=1000,
                        help="Minimal number of pixels for filtering small instances.")
    parser.add_argument("--min_component_length", type=int, default=50,
                        help="Minimal number of instances of a connected component. Filtered out if lower.")
    parser.add_argument("--max_edge_distance", type=float, default=30,
                        help="Maximal distance in micrometer between instances to create an edge between them.")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1],
                        help="List of connected components to count.")

    args = parser.parse_args()
    create_table_and_components(
        segmentation_folder=args.input,
        cell_type=args.cell_type,
        segmentation_key=args.segmentation_key,
        voxel_size=args.voxel_size,
        n_threads=args.n_threads,
        force=args.force,
        min_size=args.min_size,
        min_component_length=args.min_component_length,
        max_edge_distance=args.max_edge_distance,
        component_list=args.components,
    )


if __name__ == "__main__":
    main()
