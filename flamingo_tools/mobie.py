"""Functionality for exporting data to a MoBIE project.
"""

import os
import multiprocessing as mp
import tempfile
from typing import Optional, Tuple

from elf.io import open_file
from mobie import add_bdv_image, add_image, add_segmentation
from mobie.metadata.dataset_metadata import read_dataset_metadata

DEFAULT_RESOLUTION = (0.38, 0.38, 0.38)
DEFAULT_SCALE_FACTORS = [[2, 2, 2]] * 5
DEFAULT_CHUNKS = (128, 128, 128)
DEFAULT_UNIT = "micrometer"


def _source_exists(mobie_project, mobie_dataset, source_name):
    dataset_folder = os.path.join(mobie_project, mobie_dataset)
    metadata = read_dataset_metadata(dataset_folder)
    sources = metadata.get("sources", {})
    return source_name in sources


def _parse_spatial_args(
    resolution, scale_factors, chunks, input_path, input_key
):
    if resolution is None:
        resolution = DEFAULT_RESOLUTION
    if scale_factors is None:
        scale_factors = DEFAULT_SCALE_FACTORS
    if chunks is None:
        if input_path.endswith(".tif"):
            chunks = DEFAULT_CHUNKS
        else:
            with open_file(input_path, "r") as f:
                chunks = f[input_key].chunks
    return resolution, scale_factors, chunks


def add_raw_to_mobie(
    mobie_project: str,
    mobie_dataset: str,
    source_name: str,
    input_path: str,
    skip_existing: bool = True,
    input_key: Optional[str] = None,
    setup_id: int = 0,
    resolution: Optional[Tuple[float, float, float]] = None,
    scale_factors: Optional[Tuple[Tuple[int, int, int]]] = None,
    chunks: Optional[Tuple[int, int, int]] = None,
) -> None:
    """Add image data to a MoBIE project.

    The input may either be an xml file in BigDataViewer / BigStitcher format,
    a n5 / hdf5 / zarr file, or a tif file.

    Args:
        mobie_project: The MoBIE project directory.
        mobie_dataset The MoBIE dataset the image data will be added to.
        source_name: The name of the data to use in MoBIE.
        input_path: The path to the data.
        skip_existing: Whether to skip existing dataset.
            If this is set to false, then an exception will be thrown if the source already
            exists in the MoBIE dataset.
        input_key: The key of the input data. This only has to be specified if the input is
            a n5 / hdf5 / zarr file.
        setup_id: The setup_id that will be added to MoBIE. This is only used if the input data is an xml file.
        resolution: The resolution / voxel size of the data.
        scale_factors: The factors to use for downsampling the data when creating the multi-level image pyramid.
        chunks: The output chunks for writing the data.
    """
    # Check if we have converted this data already.
    have_source = _source_exists(mobie_project, mobie_dataset, source_name)
    if have_source and skip_existing:
        print(f"Source {source_name} already exists in {mobie_project}:{mobie_dataset}.")
        print("Conversion to mobie will be skipped.")
        return
    elif have_source:
        raise NotImplementedError

    max_jobs = min(16, mp.cpu_count())
    with tempfile.TemporaryDirectory() as tmpdir:
        if input_path.endswith(".xml"):
            add_bdv_image(
                xml_path=input_path,
                root=mobie_project,
                dataset_name=mobie_dataset,
                image_name=source_name,
                tmp_folder=tmpdir,
                file_format="bdv.n5",
                setup_ids=[setup_id],
            )
        else:
            use_memmap = False
            if input_path.endswith(".tif"):
                use_memmap = True
                assert input_key is None
            else:
                input_key = "setup0/timepoint0/s0" if input_key is None else input_key
            resolution, scale_factors, chunks = _parse_spatial_args(
                resolution, scale_factors, chunks, input_path, input_key
            )
            add_image(
                input_path=input_path,
                input_key=input_key,
                root=mobie_project,
                dataset_name=mobie_dataset,
                image_name=source_name,
                resolution=resolution,
                scale_factors=scale_factors,
                chunks=chunks,
                tmp_folder=tmpdir,
                use_memmap=use_memmap,
                unit=DEFAULT_UNIT,
                max_jobs=max_jobs,
            )


def add_segmentation_to_mobie(
    mobie_project: str,
    mobie_dataset: str,
    source_name: str,
    segmentation_path: str,
    segmentation_key: str,
    resolution: Optional[Tuple[float, float, float]] = None,
    scale_factors: Optional[Tuple[Tuple[int, int, int]]] = None,
    chunks: Optional[Tuple[int, int, int]] = None,
    skip_existing: bool = True,
) -> None:
    """Add a segmentation to a MoBIE dataset.

    Args:
        mobie_project: The MoBIE project directory.
        mobie_dataset The MoBIE dataset the segmentation will be added to.
        source_name: The name of the data to use in MoBIE.
        segmentation_path: The path to the data.
        segmentation_key: The key of the data.
        resolution: The resolution / voxel size of the data.
        scale_factors: The factors to use for downsampling the data when creating the multi-level image pyramid.
        chunks: The output chunks for writing the data.
        skip_existing: Whether to skip existing dataset.
            If this is set to false, then an exception will be thrown if the source already
            exists in the MoBIE dataset.
    """
    # Check if we have converted this data already.
    have_source = _source_exists(mobie_project, mobie_dataset, source_name)
    if have_source and skip_existing:
        print(f"Source {source_name} already exists in {mobie_project}:{mobie_dataset}.")
        print("Conversion to MoBIE format will be skipped.")
        return
    elif have_source:
        raise NotImplementedError

    resolution, scale_factors, chunks = _parse_spatial_args(
        resolution, scale_factors, chunks, segmentation_path, segmentation_key
    )

    max_jobs = min(16, mp.cpu_count())
    with tempfile.TemporaryDirectory() as tmpdir:
        add_segmentation(
            input_path=segmentation_path, input_key=segmentation_key,
            root=mobie_project, dataset_name=mobie_dataset,
            segmentation_name=source_name,
            resolution=resolution,
            scale_factors=scale_factors,
            chunks=chunks,
            tmp_folder=tmpdir,
            max_jobs=max_jobs,
        )
