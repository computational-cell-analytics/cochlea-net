import os
import warnings
from typing import List, Optional, Tuple

import pandas as pd

from flamingo_tools.s3_utils import get_s3_path


def filter_table(
    df: pd.DataFrame,
    column_subset: List[int] = [1],
    column: str = "component_labels",
) -> pd.DataFrame:
    """Filter a table based on a subset of column entries.
    """
    return df[df[column].isin(column_subset)]


def add_volume_column(
    df: pd.DataFrame,
    keyword: str = "volume[µm³]",
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
) -> pd.DataFrame:
    """Add column featuring volume in µm³ based on number of pixels and voxel size.
    """
    if keyword not in df.columns:
        n_pixels = df["n_pixels"].values
        volume = n_pixels * voxel_size[0] * voxel_size[1] * voxel_size[2]
        df[keyword] = volume
    return df


def dist_from_center(
    df: pd.DataFrame,
    old_name: str = "offset",
    new_name: str = "dist_from_center[µm]",
) -> pd.DataFrame:
    """Rename "offset" column for SGN segmentation table to "distance from center".
    """
    df.rename(columns={old_name: new_name}, inplace=True)
    return df


def add_stain_intensity(
    df: pd.DataFrame,
    table_meas: pd.DataFrame,
    stain: str,
    keyword: str = "median",
    bg_mask: bool = False,
) -> pd.DataFrame:
    """Add stain intensity to the main segmentation table from a table of object measures.
    """
    df = df.merge(table_meas[["label_id", keyword]], on="label_id", how="left")
    if bg_mask:
        df.rename(columns={keyword: f"{stain}_bg-mask_{keyword}"}, inplace=True)
    else:
        df.rename(columns={keyword: f"{stain}_{keyword}"}, inplace=True)
    return df


def add_object_measures_to_table(
    table_seg_path: str,
    table_meas_path: str,
    output_path: Optional[str] = None,
    s3_seg: bool = False,
    s3_meas: bool = False,
    bg_mask: bool = False,
    intensity_keyword: str = "median",
):
    """Add object measures to the main segmentation table.

    Args:
        table_seg_path: Path to segmentation table.
        table_meas_path: Path to table with object measures.
        output_path: Path to edited table. Default: Local segmentation table is overwritten. Required for S3.
        s3_seg: Flag for segmentation table on S3 bucket.
        s3_meas: Flag for measurement table on S3 bucket.
        bg_mask: Object measures using background mask. Implicit if meas table contains "bg-mask".
        intensity_keyword: Keyword for measurement table. Default: "median".
    """
    if output_path is None:
        if s3_seg:
            raise ValueError("Please provide a local output path for the segmentation table on S3.")
        else:
            output_path = table_seg_path

    if s3_seg:
        tsv_path, fs = get_s3_path(table_seg_path)
        with fs.open(tsv_path, "r") as f:
            table_seg = pd.read_csv(f, sep="\t")
    else:
        table_seg = pd.read_csv(table_seg_path, sep="\t")

    if s3_meas:
        tsv_path, fs = get_s3_path(table_meas_path)
        with fs.open(tsv_path, "r") as f:
            table_meas = pd.read_csv(f, sep="\t")
    else:
        table_meas = pd.read_csv(table_meas_path, sep="\t")

    base_name = os.path.basename(table_meas_path).split(".")[0]
    meas_content = base_name.split("_")
    if "object-measures" not in meas_content[-1]:
        warnings.warn("Change to default naming scheme for object measures: "
                      "<cochlea>_<stain>_<seg_name>_object-measures.tsv"
                      "<stain>_<seg_name>_object-measures.tsv")

    if "bg-mask" in base_name:
        bg_mask = True

    if len(meas_content) == 4:
        stain = meas_content[1]
    else:
        stain = meas_content[0]

    table_seg = add_stain_intensity(table_seg, table_meas, stain, bg_mask=bg_mask, keyword=intensity_keyword)
    table_seg.to_csv(output_path, sep="\t", index=False)


def create_main_table(
    input_path: str,
    output_path: str,
    component_list: List[int] = [1],
    cell_type: str = "sgn",
    voxel_size: Tuple[float, float, float] = (0.38, 0.38, 0.38),
    meas_tables: Optional[List[str]] = None,
    s3_seg: bool = False,
    s3_meas: bool = False,
    intensity_keyword: str = "median",
):
    """Create main table for analysis based on segmentation table.
    The segmentation instances are filtered by their component label.

    Args:
        input_path: Input path for segmentation table featuring all segmentation instances.
        output_path: Output path for main table.
        component_list: List of component labels of segmentation components.
        cell_type: Cell type. Either "ihc" or "sgn".
        voxel_size: Voxel size of data in micrometer.
        meas_tables: List of tables featuring object measures for stain/seg combinations.
        s3_seg: File path of segmentation table on S3 bucket.
        s3_meas: File path of measurement table on S3 bucket.
        intensity_keyword: Keyword for measurement table. Default: "median".
    """
    if os.path.realpath(input_path) == os.path.realpath(output_path):
        raise ValueError(f"Input path {input_path} and {output_path} are identical.")

    if s3_seg:
        tsv_path, fs = get_s3_path(input_path)
        with fs.open(tsv_path, "r") as f:
            df = pd.read_csv(f, sep="\t")
    else:
        df = pd.read_csv(input_path, sep="\t")

    df = filter_table(df, column_subset=component_list)

    volume_keyword = "volume[µm³]"
    if volume_keyword not in list(df.columns):
        df = add_volume_column(df, keyword=volume_keyword, voxel_size=voxel_size)

    if "offset" in list(df.columns) and cell_type == "sgn":
        df = dist_from_center(df)

    if meas_tables is not None:
        for table_meas_path in meas_tables:
            if s3_meas:
                tsv_path, fs = get_s3_path(table_meas_path)
                with fs.open(tsv_path, "r") as f:
                    table_meas = pd.read_csv(f, sep="\t")
            else:
                table_meas = pd.read_csv(table_meas_path, sep="\t")

            base_name = os.path.basename(table_meas_path).split(".")[0]
            meas_content = base_name.split("_")
            if "object-measures" not in meas_content[-1]:
                warnings.warn(f"Table {table_meas_path} does not fit default naming scheme for object measures: "
                              "<cochlea>_<stain>_<seg_name>_object-measures.tsv"
                              "<stain>_<seg_name>_object-measures.tsv")

            if "bg-mask" in base_name:
                bg_mask = True
            else:
                bg_mask = False

            if len(meas_content) == 4:
                stain = meas_content[1]
            else:
                stain = meas_content[0]

            df = add_stain_intensity(df, table_meas, stain, bg_mask=bg_mask, keyword=intensity_keyword)

    df.to_csv(output_path, sep="\t", index=False)
