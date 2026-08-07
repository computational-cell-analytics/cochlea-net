"""Functionality for converting data from flamingo microscopes to other data formats.
"""

import getpass
import json
import multiprocessing as mp
import os
import posixpath
import re
import warnings
import xml.etree.ElementTree as ET

from fnmatch import fnmatch
from glob import glob
from pathlib import Path
from shutil import rmtree
from typing import Optional, List, Dict, Tuple

import numpy as np
import pybdv
import tifffile

from elf.io import open_file
from pybdv.downsample import downsample
from pybdv.util import get_key
from skimage.transform import rescale

from .data_transfer_utils import (
    MAX_RETRIES,
    RETRY_DELAY,
    SMB_SERVER,
    SMB_TIMEOUT,
    RetryConfig,
    copy_file_resilient,
    read_into,
    remote_size_map_with_retry,
    resumable_download,
    retry_io,
    wait_for_path,
)
from .file_utils import _parse_shape, read_tif, read_raw

# The n5 chunk shape of the output. A slab write is aligned to the chunk depth.
CHUNKS = (128, 128, 128)
CHUNK_DEPTH = CHUNKS[0]
STATE_SUFFIX = ".convert_state.json"
DEFAULT_SLAB_MEMORY = 1.0  # GB
# The flamingo raw format has no header; see file_utils.read_raw.
RAW_DTYPE = "uint16"


def _read_voxel_size_and_unit_flamingo(mdata_path):
    voxel_size = None
    with open(mdata_path, "r") as f:
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            if line.startswith("Plane spacing"):
                voxel_size = float(line.split(" ")[-1])
                break
    if voxel_size is None:
        raise RuntimeError

    unit = "micrometer"

    # NOTE: The voxel size for the flamingo system is isotropic.
    # So we can just return the plane spacing value to get it.
    voxel_size = [voxel_size] * 3
    return voxel_size, unit


def _read_start_position_flamingo(path):
    at_start = False
    start_x, start_y, start_z = None, None, None

    with open(path, "r") as f:
        for line in f.readlines():
            line = line.strip().rstrip("\n")
            if line.startswith("<Start Position>"):
                at_start = True

            if at_start and line.startswith("X"):
                start_x = float(line.split(" ")[-1])
            if at_start and line.startswith("Y"):
                start_y = float(line.split(" ")[-1])
            if at_start and line.startswith("Z"):
                start_z = float(line.split(" ")[-1])

            if (start_x is not None) and (start_y is not None) and (start_z is not None):
                break

    assert (start_x is not None) and (start_y is not None) and (start_z is not None)
    start_position = [start_x, start_y, start_z]
    return start_position


def read_metadata_flamingo(
    metadata_path: str,
    offset: Optional[np.ndarray] = None,
    parse_affine: bool = False
) -> Tuple[List[float], str, List[float]]:
    """Read acquisition metadata from a flamingo metadata file.

    This will read the voxel size, the physical unit, and optionally the
    voxel grid transformation from the metadata file. The voxel grid transformation
    places tile at their correct tile position.

    Args:
        metadata_path: The path to the metadata file.
        offset: The spatial offset of this data.
        parse_affine: Whether to read the affine transformation from the metadata.

    Returns:
        The voxel size of the data.
        The physical unit of the voxel size.
        The affine voxel grid transformation of the data.
    """
    voxel_size, unit = None, None

    voxel_size, unit = _read_voxel_size_and_unit_flamingo(metadata_path)
    start_position = _read_start_position_flamingo(metadata_path)

    def _pos_to_trafo(pos):
        if offset is not None:
            pos -= offset

        # NOTE: the scale should be kept at 1.
        # This is only here for development purposes,
        # to support handling downsampled datasets.
        scale = 1

        # The calibration: scale factors on the diagonals.
        calib_trafo = [
            scale * voxel_size[0], 0.0, 0.0, 0.0,
            0.0, scale * voxel_size[1], 0.0, 0.0,
            0.0, 0.0, scale * voxel_size[2], 0.0,
        ]
        # The translation to the grid position.
        # Note that the translations are given in mm,
        # so they need to multiplied by a factor of 1000
        # to match the voxel_size given in microns.
        grid_trafo = [
            1.0, 0.0, 0.0, scale * pos[0] * 1000,
            0.0, 1.0, 0.0, scale * pos[1] * 1000,
            0.0, 0.0, 1.0, scale * pos[2] * 1000,
        ]
        trafo = {
            "Translation to Regular Grid": grid_trafo,
            "Calibration": calib_trafo,
        }
        return trafo

    if parse_affine:
        transformation = _pos_to_trafo(start_position)
    else:
        transformation = [
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
        ]
    # We have to reverse the voxel_size because pybdv expects ZYX.
    return voxel_size[::-1], unit, transformation


# TODO derive the scale factors from the shape rather than hard-coding it to 5 levels
def _derive_scale_factors(shape):
    scale_factors = [[2, 2, 2]] * 5
    return scale_factors


def _to_ome_zarr(data, out_path, scale_factors, timepoint, setup_id, attributes, unit, voxel_size):
    # Imported here so that the bdv/n5 conversion path does not require mobie.
    from mobie.import_data._format_metadata import write_format_metadata

    n_threads = mp.cpu_count()
    chunks = (128, 128, 128)

    # Write the base dataset.
    base_key = f"setup{setup_id}/timepoint{timepoint}"

    with open_file(out_path, "a") as f:
        ds = f.create_dataset(f"{base_key}/s0", shape=data.shape, compression="gzip",
                              chunks=chunks, dtype=data.dtype)
        ds.n_threads = n_threads
        ds[:] = data

        # TODO parallelized implementation.
        # Do downscaling.
        for level, scale_factor in enumerate(scale_factors, 1):
            inv_scale = [1.0 / sc for sc in scale_factor]
            data = rescale(data, inv_scale, preserve_range=True).astype(data.dtype)
            ds = f.create_dataset(f"{base_key}/s{level}", shape=data.shape, compression="gzip",
                                  chunks=chunks, dtype=data.dtype)
            ds.n_threads = n_threads
            ds[:] = data

        g = f[f"setup{setup_id}"]
        g.attrs.update(attributes)

    # Write the ome zarr metadata.
    metadata_dict = {"unit": unit, "resolution": voxel_size}
    write_format_metadata("ome.zarr", os.path.join(out_path, base_key), metadata_dict, scale_factors)


def flamingo_filename_parser(file_path: str, name_mapping: Optional[Dict]) -> Tuple[int, Dict[str, str], str]:
    """Parse the name of flamingo output files.

    This maps the filenames to the corresponding timepoint, the BigStitcher
    compatible attributes, and the id (name) of the attributes.

    Args:
        file_path: The path to the flamingo data.
        name_mapping: Optional mapping of parsed attributes to their actual names.

    Returns:
        The timepoint of this data.
        The dictionary mapping attribute names to their values.
        The normalized attribute names.
    """
    filename = os.path.basename(file_path)

    # Extract the timepoint.
    match = re.search(r"_t(\d+)_", filename)
    if match:
        timepoint = int(match.group(1))
    else:
        timepoint = 0

    # Extract the additional attributes.
    attributes = {}
    if name_mapping is None:
        name_mapping = {}

    # Extract the channel.
    match = re.search(r"_C(\d+)_", filename)
    channel = int(match.group(1)) if match else 0
    channel_mapping = name_mapping.get("channel", {})
    attributes["channel"] = {"id": channel, "name": channel_mapping.get(channel, str(channel))}

    # Extract the tile.
    match = re.search(r"_R(\d+)_", filename)
    tile = int(match.group(1)) if match else 0
    tile_mapping = name_mapping.get("tile", {})
    attributes["tile"] = {"id": tile, "name": tile_mapping.get(tile, str(tile))}

    # Extract the illumination.
    match = re.search(r"_I(\d+)_", filename)
    illumination = int(match.group(1)) if match else 0
    illumination_mapping = name_mapping.get("illumination", {})
    attributes["illumination"] = {"id": illumination, "name": illumination_mapping.get(illumination, str(illumination))}

    # Extract D. TODO what is this?
    match = re.search(r"_D(\d+)_", filename)
    D = int(match.group(1)) if match else 0
    D_mapping = name_mapping.get("D", {})
    attributes["D"] = {"id": D, "name": D_mapping.get(D, str(D))}

    # BDV also supports an angle attribute, but it does not seem to be stored in the filename
    # "angle": {"id": 0, "name": "0"}

    attribute_id = f"c{channel}-t{tile}-i{illumination}-d{D}"
    return timepoint, attributes, attribute_id


def _load_data(file_path, metadata_file):
    """Memory-map an input file. Only for the ome-zarr path; never use it on remote data."""
    if Path(file_path).suffix == ".raw":
        data = read_raw(file_path, metadata_file)
    else:
        data = read_tif(file_path)
    return data


def _write_missing_views(out_path):
    xml_path = Path(out_path).with_suffix(".xml")
    assert os.path.exists(xml_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    seqdesc = root.find("SequenceDescription")
    # A resumed run parses an xml that already has the element; a second one breaks BigStitcher.
    if seqdesc.find("MissingViews") is None:
        ET.SubElement(seqdesc, "MissingViews")

    pybdv.metadata.indent_xml(root)
    tree = ET.ElementTree(root)
    tree.write(xml_path)


class ConversionState:
    """Per-setup progress of a conversion, stored next to the output.

    The progress file is the only authority on what is complete. Dataset existence is not:
    pybdv keeps a partially written s0 and then skips the downscaling, which silently produces
    a truncated setup that claims to have a full resolution pyramid.
    """

    VERSION = 1

    def __init__(self, path: str):
        self.path = path
        self.data = {"version": self.VERSION, "setups": {}}

    @classmethod
    def load_or_create(cls, out_path: str, restart: bool = False) -> "ConversionState":
        """Load the progress file for an output, or create an empty one.

        Args:
            out_path: Output path of the conversion.
            restart: Delete an existing output and its progress file first.

        Returns:
            The conversion state.

        Raises:
            RuntimeError: The output exists without a progress file, so it cannot be resumed.
        """
        state_path = state_path_for(out_path)
        xml_path = str(Path(out_path).with_suffix(".xml"))
        state = cls(state_path)

        if restart:
            for path in (out_path, xml_path, state_path):
                if os.path.isdir(path):
                    rmtree(path)
                elif os.path.exists(path):
                    os.remove(path)
            return state

        if os.path.exists(state_path):
            with open(state_path) as f:
                loaded = json.load(f)
            if loaded.get("version") != cls.VERSION:
                raise RuntimeError(
                    f"The progress file {state_path} has version {loaded.get('version')}, "
                    f"but version {cls.VERSION} is expected. Pass restart=True to convert again."
                )
            state.data = loaded
        elif os.path.exists(out_path) or os.path.exists(xml_path):
            raise RuntimeError(
                f"The output {out_path} exists, but the progress file {state_path} does not. "
                "The output may contain a setup that was only written in part, which cannot be "
                "resumed safely. Pass --restart to discard the output and convert again, or move "
                "the output out of the way."
            )
        return state

    def save(self) -> None:
        """Write the progress file atomically."""
        tmp_path = f"{self.path}.tmp"
        with open(tmp_path, "w") as f:
            json.dump(self.data, f, indent=2)
        os.replace(tmp_path, self.path)

    def setup(self, attribute_id: str) -> dict:
        """Return the progress entry for one setup, creating an empty one if needed.

        Args:
            attribute_id: Attribute id of the setup, e.g. 'c0-t1-i0-d0'.

        Returns:
            The progress entry.
        """
        return self.data["setups"].setdefault(attribute_id, {})

    def update(self, attribute_id: str, **values) -> dict:
        """Update the progress entry for one setup and save the progress file.

        Args:
            attribute_id: Attribute id of the setup.
            values: Fields to set on the entry.

        Returns:
            The updated entry.
        """
        entry = self.setup(attribute_id)
        entry.update(values)
        self.save()
        return entry


def state_path_for(out_path: str) -> str:
    """Return the path of the progress file that belongs to an output.

    Args:
        out_path: Output path of the conversion.

    Returns:
        The path of the progress file.
    """
    return f"{out_path}{STATE_SUFFIX}"


def _slab_depth_for(shape, itemsize, slab_memory):
    """Derive a chunk-aligned slab depth from a memory budget in GB."""
    plane_bytes = int(np.prod(shape[1:])) * itemsize
    budget = int(slab_memory * (1 << 30))
    depth = (budget // max(plane_bytes, 1)) // CHUNK_DEPTH * CHUNK_DEPTH
    return max(CHUNK_DEPTH, min(int(depth), int(shape[0])))


def _slab_bounds(n_planes, slab_depth, start_z=0):
    return [
        (z, min(z + slab_depth, n_planes))
        for z in range(0, n_planes, slab_depth)
        if z + slab_depth > start_z
    ]


def _iter_raw_slabs(file_path, shape, slab_depth, config, start_z=0):
    """Yield (z_start, z_stop, slab) for a flamingo raw file, retrying every read."""
    plane_bytes = int(np.prod(shape[1:])) * np.dtype(RAW_DTYPE).itemsize
    buf = np.empty((slab_depth,) + tuple(shape[1:]), dtype=RAW_DTYPE)
    for z0, z1 in _slab_bounds(shape[0], slab_depth, start_z):
        view = buf[: z1 - z0]
        read_into(file_path, z0 * plane_bytes, view, config)
        yield z0, z1, view


def _tif_series_shape(file_path, config):
    """Read the shape and dtype of a tif from its header, retrying on an I/O error."""
    def _read_header():
        with tifffile.TiffFile(file_path) as f:
            series = f.series[0]
            return tuple(series.shape), np.dtype(series.dtype), len(f.pages)

    return retry_io(_read_header, f"read the header of {os.path.basename(file_path)}", config)


def _iter_tif_slabs(file_path, shape, n_pages, slab_depth, config, start_z=0):
    """Yield (z_start, z_stop, slab) for a tif, retrying every read.

    Reads one page range per slab. A tif whose pages do not map to the leading axis is read
    whole, because there is no page range that corresponds to a slab.
    """
    page_wise = len(shape) == 3 and n_pages == shape[0]

    if not page_wise:
        warnings.warn(
            f"The pages of {file_path} do not map to its first axis, so it is read as a whole. "
            "This needs enough memory for the complete image."
        )
        data = retry_io(lambda: read_tif(file_path), f"read {os.path.basename(file_path)}", config)
        for z0, z1 in _slab_bounds(shape[0], slab_depth, start_z):
            yield z0, z1, np.asarray(data[z0:z1])
        return

    for z0, z1 in _slab_bounds(shape[0], slab_depth, start_z):
        def _read(z0=z0, z1=z1):
            with tifffile.TiffFile(file_path) as f:
                return f.asarray(key=range(z0, z1))

        slab = retry_io(_read, f"read {os.path.basename(file_path)} planes {z0}-{z1}", config)
        yield z0, z1, slab


def _input_shape_and_dtype(file_path, metadata_file, config):
    """Return the shape and dtype of an input file without loading it."""
    if Path(file_path).suffix == ".raw":
        if metadata_file is None:
            raise RuntimeError(
                f"The shape of the raw file {file_path} can only be read from a metadata file, "
                "but no metadata pattern was given."
            )
        shape = _parse_shape(metadata_file)
        return tuple(shape), np.dtype(RAW_DTYPE), None

    shape, dtype, n_pages = _tif_series_shape(file_path, config)
    return shape, dtype, n_pages


def _iter_slabs(file_path, shape, n_pages, slab_depth, config, start_z=0):
    """Yield (z_start, z_stop, slab) for a raw or tif input file."""
    if Path(file_path).suffix == ".raw":
        return _iter_raw_slabs(file_path, shape, slab_depth, config, start_z=start_z)
    return _iter_tif_slabs(file_path, shape, n_pages, slab_depth, config, start_z=start_z)


def _check_raw_size(file_path, shape, config):
    """Check that a raw file holds exactly the number of bytes its metadata describes."""
    expected = int(np.prod(shape)) * np.dtype(RAW_DTYPE).itemsize
    actual = wait_for_path(file_path, config).st_size
    if actual != expected:
        raise RuntimeError(
            f"The raw file {file_path} has {actual} bytes, but its metadata describes a "
            f"{shape} volume of {expected} bytes. The file may be truncated or only partly synced."
        )


def _convert_tile(out_path, tile, state, scale_factors, n_threads, config):
    """Convert one tile into an existing bdv-n5 container, resuming from the recorded progress."""
    attribute_id = tile["attribute_id"]
    setup_id, timepoint = tile["setup_id"], tile["timepoint"]
    shape = tile["shape"]
    entry = state.setup(attribute_id)

    if entry.get("done"):
        print(f"Skipping setup {setup_id} (tile {attribute_id}): already converted.")
        return

    if not entry.get("initialized"):
        pybdv.initialize_bdv(
            out_path, shape, tile["dtype"],
            setup_id=setup_id, timepoint=timepoint,
            downscale_factors=scale_factors,
            resolution=tile["voxel_size"], unit=tile["unit"],
            affine=tile["transformation"], attributes=tile["attributes"],
            chunks=CHUNKS,
        )
        entry = state.update(
            attribute_id,
            setup_id=setup_id, timepoint=timepoint, file=tile["file_name"],
            shape=list(shape), slab_depth=tile["slab_depth"],
            initialized=True, z_done=0, pyramid_done=False, done=False,
        )

    # A resumed run keeps the slab depth of the first run, so that z_done stays on a slab border.
    slab_depth = entry.get("slab_depth", tile["slab_depth"])
    z_done = entry.get("z_done", 0)

    if z_done < shape[0]:
        base_key = get_key(False, timepoint=timepoint, setup_id=setup_id, scale=0)
        slabs = _iter_slabs(
            tile["local_path"], shape, tile["n_pages"], slab_depth, config, start_z=z_done,
        )
        with open_file(out_path, "a") as f:
            ds = f[base_key]
            ds.n_threads = n_threads
            for z0, z1, slab in slabs:
                ds[z0:z1] = slab
                state.update(attribute_id, z_done=z1)
                print(f"  setup {setup_id}: {z1} / {shape[0]} planes written")

    if not entry.get("pyramid_done"):
        for scale, factor in enumerate(scale_factors):
            in_key = get_key(False, timepoint=timepoint, setup_id=setup_id, scale=scale)
            out_key = get_key(False, timepoint=timepoint, setup_id=setup_id, scale=scale + 1)
            downsample(out_path, in_key, out_key, factor, "mean", n_threads=n_threads, overwrite=True)
        state.update(attribute_id, pyramid_done=True)

    state.update(attribute_id, done=True)


class SmbSource:
    """Download input files from the SMB share one at a time.

    This needs the `smbclient` binary. Use it where the share cannot be mounted, for example on
    a compute cluster. On a workstation that can mount the share, read the mounted path instead.
    """

    def __init__(self, username, password, remote_root, smb_server=SMB_SERVER, log_file=None,
                 retries=MAX_RETRIES, timeout=SMB_TIMEOUT):
        self.username = username
        self.password = password
        self.remote_root = remote_root.replace("\\", "/").rstrip("/")
        self.smb_server = smb_server
        self.log_file = log_file
        self.retries = retries
        self.timeout = timeout
        self.size_map = None

    def list_files(self) -> dict:
        """List every file below the remote root with its size.

        Returns:
            {relative_path: size_in_bytes}.

        Raises:
            RuntimeError: The listing failed after all retries.
        """
        if self.size_map is None:
            self.size_map = remote_size_map_with_retry(
                self.username, self.password, self.remote_root, os.getcwd(),
                retries=self.retries, smb_server=self.smb_server, timeout=self.timeout,
            )
            if self.size_map is None:
                raise RuntimeError(f"Could not list the remote directory {self.remote_root}.")
        return self.size_map

    def download(self, rel_path: str, stage_dir: str) -> str:
        """Download one remote file into a local directory, keeping its relative path.

        A file that is already present with the expected size is not downloaded again, and a
        transfer that stopped part way is continued where it stopped instead of started again.

        Args:
            rel_path: Path of the file relative to the remote root.
            stage_dir: Local directory that receives the file.

        Returns:
            The local path of the downloaded file.

        Raises:
            RuntimeError: The download did not reach the size the share reports.
        """
        rel_dir, name = posixpath.split(rel_path)
        local_dir = os.path.join(stage_dir, *rel_dir.split("/")) if rel_dir else stage_dir
        local_path = os.path.join(local_dir, name)
        expected = self.list_files().get(rel_path)

        if os.path.exists(local_path) and expected is not None and os.path.getsize(local_path) == expected:
            print(f"  [skip] {rel_path} is already staged ({expected} bytes)")
            return local_path

        remote_cd = f"{self.remote_root}/{rel_dir}" if rel_dir else self.remote_root
        ok = resumable_download(
            self.username, self.password, remote_cd=remote_cd, name=name, local_cwd=local_dir,
            expected_size=expected, retries=self.retries, log_file=self.log_file,
            smb_server=self.smb_server, timeout=self.timeout,
        )
        if not ok:
            got = os.path.getsize(local_path) if os.path.exists(local_path) else 0
            raise RuntimeError(
                f"Could not download {rel_path} from {self.remote_root}: got {got} bytes, "
                f"the share reports {expected} bytes. The partial file is kept at {local_path}, "
                "so running the same command again continues from there."
            )
        return local_path


def _collect_local_inputs(root, file_ext, metadata_file_name_pattern, metadata_root):
    """Return the image files and the matching metadata files below a local root."""
    files = sorted(glob(os.path.join(root, f"**/*{file_ext}"), recursive=True))
    if file_ext == ".tif":
        # We need to exlcude the max-projetion tifs that are saved alongside the tifs.
        files = [ff for ff in files if "_MP.tif" not in ff]
    if len(files) == 0:
        raise ValueError(f"Could not find any files in {root} with extension {file_ext}.")

    if metadata_file_name_pattern is None:
        return files, [None] * len(files)

    meta_root = root if metadata_root is None else metadata_root
    metadata_files = sorted(
        glob(os.path.join(meta_root, f"**/{metadata_file_name_pattern}"), recursive=True)
    )
    if len(metadata_files) != len(files):
        raise RuntimeError(
            f"Found {len(files)} image files matching '**/*{file_ext}' in {root}, but "
            f"{len(metadata_files)} metadata files matching '**/{metadata_file_name_pattern}' in "
            f"{meta_root}. Each image file needs exactly one metadata file."
        )
    return files, metadata_files


def _collect_smb_inputs(smb_source, file_ext, metadata_file_name_pattern, stage_dir):
    """Return the remote image files and the staged metadata files below the remote root.

    The metadata files are small, so all of them are downloaded up front. The image files stay
    on the share and are downloaded one at a time during the conversion.
    """
    size_map = smb_source.list_files()
    files = sorted(rel for rel in size_map if rel.endswith(file_ext))
    if file_ext == ".tif":
        files = [ff for ff in files if "_MP.tif" not in ff]
    if len(files) == 0:
        raise ValueError(
            f"Could not find any files in {smb_source.remote_root} with extension {file_ext}."
        )

    if metadata_file_name_pattern is None:
        return files, [None] * len(files)

    metadata_rel = sorted(
        rel for rel in size_map
        if fnmatch(posixpath.basename(rel), metadata_file_name_pattern)
    )
    if len(metadata_rel) != len(files):
        raise RuntimeError(
            f"Found {len(files)} image files with extension '{file_ext}' on "
            f"{smb_source.remote_root}, but {len(metadata_rel)} metadata files matching "
            f"'{metadata_file_name_pattern}'. Each image file needs exactly one metadata file."
        )

    print(f"Downloading {len(metadata_rel)} metadata files.")
    metadata_files = [smb_source.download(rel, stage_dir) for rel in metadata_rel]
    return files, metadata_files


def _verify_conversion(out_path, state, n_expected):
    """Report which setups are complete and check the xml against the number of input tiles."""
    setups = state.data["setups"]
    incomplete = sorted(aid for aid, entry in setups.items() if not entry.get("done"))

    print("\n=== Conversion summary ===")
    for attribute_id in sorted(setups):
        entry = setups[attribute_id]
        status = "done" if entry.get("done") else f"incomplete ({entry.get('z_done', 0)} planes)"
        print(f"  setup {entry.get('setup_id')} (tile {attribute_id}): {status}")

    xml_path = Path(out_path).with_suffix(".xml")
    n_setups = len(pybdv.metadata.get_setup_ids(str(xml_path))) if os.path.exists(xml_path) else 0
    print(f"  {n_setups} setup(s) in {xml_path.name}, {n_expected} input tile(s)")

    if incomplete or n_setups != n_expected:
        raise RuntimeError(
            f"The conversion is incomplete: {len(incomplete)} of {n_expected} tile(s) are not "
            f"finished and the xml lists {n_setups} setup(s). Run the same command again to resume."
        )
    print("  all tiles converted")


def convert_lightsheet_to_bdv(
    root: str,
    out_path: str,
    file_ext: str = ".tif",
    attribute_parser: callable = flamingo_filename_parser,
    attribute_names: Optional[Dict[str, Dict[int, str]]] = None,
    metadata_file_name_pattern: Optional[str] = "*_Settings.txt",
    metadata_root: Optional[str] = None,
    metadata_type: str = "flamingo",
    center_tiles: bool = False,
    voxel_size: Optional[List[float]] = None,
    unit: Optional[str] = None,
    scale_factors: Optional[List[List[int]]] = None,
    n_threads: Optional[int] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    smb_timeout: int = SMB_TIMEOUT,
    stage_dir: Optional[str] = None,
    slab_memory: float = DEFAULT_SLAB_MEMORY,
    retry_config: Optional[RetryConfig] = None,
    restart: bool = False,
    dry_run: bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Convert channels and tiles acquired with a lightsheet microscope.

    The data is converted to the bdv-n5 file format and can be opened with BigDataViewer
    or BigStitcher. This function is written with data layout and metadata of flamingo
    microscopes in mind, but could potentially be adapted to other data formats.

    The image data is read one z-slab at a time and every read is retried, so a conversion
    survives a connection that drops. Progress is recorded per slab in a file next to the output,
    so an interrupted run continues where it stopped. Never point this function at a
    memory-mapped copy of remote data: a failed page fault on a mapped network file kills the
    process without a Python traceback.

    There are three ways to reach the input data:

    - A local or mounted path. The default. Slabs are read straight from the mount.
    - A local or mounted path with `stage_dir`. Each file is copied to local storage first,
      with a resumable byte-offset copy, and removed after the tile is converted.
    - The SMB share, through `username`. Each file is downloaded with the `smbclient` binary
      into `stage_dir` and removed after the tile is converted. `smbclient` is not available on
      Windows, so this mode suits a compute cluster.

    TODO explain the attribute parsing.

    Args:
        root: Folder that contains the image data stored as tifs.
            This function will take into account all tif files in folders beneath this root directory.
            With `username` this is the directory on the SMB share instead.
        out_path: Output path where the converted data is saved.
        file_ext: The name of the file extension. By default assumes tif files (.tif).
            Change to '.raw' to read files stored in raw format instead.
        attribute_parser: TODO
        attribute_names: Optional mapping of parsed attributes to their actual names.
        metadata_file_name_pattern: The pattern for the names of files that contain the metadata.
            For flamingo metadata the following pattern should work: '*_Settings.txt'.
        metadata_root: Different root folder for the metadata. By default 'root' is used here as well.
        metadata_type: The type of the metadata (for now only 'flamingo' is supported).
        center_tiles: Whether to move the tiles to the origin.
        voxel_size: The physical size of one pixel. This is only used if the metadata is not read from file.
        unit: The unit of the given voxel size. This is only used if the metadata is not read from file.
        scale_factors: The scale factors for downsampling the image data.
            By default sensible factors will be determined based on the shape of the data.
            If you want to set the scale factors manually then you have to pass them as a list with the
            downsampling factors for each level. E.g.:
            - [[2, 2, 2], [2, 2, 2]] to downsample isotropically by a factor of 2 for two times.
            - [[1, 2, 2], [1, 2, 2]] to downsample anisotropically for two times.
            - [[1, 2, 2], [2, 2, 2]] to downsample anisotroically once and then isotropically.
        n_threads: The number of threads to use for parallelizing the data conversion.
            By default all available CPU cores will be used.
        username: GWDG username. Reads the input from the SMB share instead of from a path.
        password: GWDG password. Asked for interactively if it is not given.
        smb_server: The SMB server that holds the input data.
        smb_timeout: The per-operation timeout for smbclient in seconds. Raise it if a download
            stops with NT_STATUS_IO_TIMEOUT.
        stage_dir: Local directory that holds one input file at a time. Required with `username`.
        slab_memory: Memory budget for one z-slab, in GB. It sets how many planes are read at once.
        retry_config: Retry behavior for the image reads.
        restart: Delete an existing output and convert again, instead of resuming.
        dry_run: Only report the input files, their pairing and their shapes, then return.
        log_file: File that records failed reads and transfers.
    """
    if metadata_type != "flamingo":
        raise ValueError(f"Invalid metadata type: {metadata_type}.")
    if n_threads is None:
        n_threads = mp.cpu_count()

    # Make sure we convert to n5, in case no extension is passed.
    ext = os.path.splitext(out_path)[1]
    convert_to_ome_zarr = False
    if ext == "":
        out_path = str(Path(out_path).with_suffix(".n5"))
    elif ext == ".zarr":
        convert_to_ome_zarr = True
        warnings.warn(
            "The ome-zarr output holds the complete image and every pyramid level in memory and "
            "it has no resume. Convert to n5 for data on an unstable connection."
        )

    if log_file is None:
        log_file = f"{out_path}.convert_log.txt"
    if retry_config is None:
        retry_config = RetryConfig(log_file=log_file)
    elif retry_config.log_file is None:
        retry_config.log_file = log_file

    smb_source = None
    if username is not None:
        if stage_dir is None:
            raise ValueError("stage_dir is required when the input is read from the SMB share.")
        if password is None:
            password = getpass.getpass("Enter password: ")
        smb_source = SmbSource(
            username, password, remote_root=root, smb_server=smb_server, log_file=log_file,
            retries=retry_config.max_retries, timeout=smb_timeout,
        )

    if stage_dir is not None:
        os.makedirs(stage_dir, exist_ok=True)

    if smb_source is None:
        files, metadata_files = _collect_local_inputs(
            root, file_ext, metadata_file_name_pattern, metadata_root
        )
    else:
        files, metadata_files = _collect_smb_inputs(
            smb_source, file_ext, metadata_file_name_pattern, stage_dir
        )

    offset = None
    if center_tiles and metadata_file_name_pattern is not None:
        start_positions = [_read_start_position_flamingo(mpath) for mpath in metadata_files]
        offset = np.min(start_positions, axis=0)

    # The setup id follows the position in the sorted file list, so it is stable as long as the
    # same files are found. The progress file is keyed by the attribute id instead.
    next_setup_id = 0
    attrs_to_setups = {}
    tiles = []

    for file_path, metadata_file in zip(files, metadata_files):
        timepoint, attributes, attribute_id = attribute_parser(file_path, attribute_names)

        if attribute_id in attrs_to_setups:
            setup_id = attrs_to_setups[attribute_id]
        else:
            attrs_to_setups[attribute_id] = next_setup_id
            setup_id = next_setup_id
            next_setup_id += 1

        if metadata_file is None:  # No metadata given.
            # We don't use any tile transformation.
            tile_transformation = None
            # Set voxel_size and unit to their default values if they were not passed.
            tile_voxel_size = [1.0, 1.0, 1.0] if voxel_size is None else voxel_size
            tile_unit = "pixel" if unit is None else unit
        else:
            # NOTE: we don't add the calibration transformation here, as this
            # leads to issues with the BigStitcher export.
            tile_voxel_size, tile_unit, tile_transformation = read_metadata_flamingo(
                metadata_file, offset, parse_affine=False
            )

        tiles.append({
            "attribute_id": attribute_id,
            "setup_id": setup_id,
            "timepoint": timepoint,
            "attributes": attributes,
            "source": file_path,
            "file_name": os.path.basename(file_path),
            "metadata_file": metadata_file,
            "voxel_size": tile_voxel_size,
            "unit": tile_unit,
            "transformation": tile_transformation,
        })

    if dry_run:
        _report_dry_run(tiles, retry_config, remote=smb_source is not None)
        return

    state = None if convert_to_ome_zarr else ConversionState.load_or_create(out_path, restart=restart)

    for tile in tiles:
        print(
            f"Converting tp={tile['timepoint']}, setup={tile['setup_id']}, tile={tile['attribute_id']}"
        )
        if state is not None and state.setup(tile["attribute_id"]).get("done"):
            print("  already converted, skipping")
            continue

        staged_path = None
        if smb_source is not None:
            staged_path = smb_source.download(tile["source"], stage_dir)
        elif stage_dir is not None:
            staged_path = os.path.join(stage_dir, tile["file_name"])
            copy_file_resilient(tile["source"], staged_path, retry_config)
        tile["local_path"] = tile["source"] if staged_path is None else staged_path

        if convert_to_ome_zarr:
            data = _load_data(tile["local_path"], tile["metadata_file"])
            if scale_factors is None:
                scale_factors = _derive_scale_factors(data.shape)
            _to_ome_zarr(
                data, out_path, scale_factors, tile["timepoint"], tile["setup_id"],
                tile["attributes"], tile["unit"], tile["voxel_size"],
            )
            if staged_path is not None:
                os.remove(staged_path)
            continue

        shape, dtype, n_pages = _input_shape_and_dtype(
            tile["local_path"], tile["metadata_file"], retry_config
        )
        if Path(tile["local_path"]).suffix == ".raw":
            _check_raw_size(tile["local_path"], shape, retry_config)

        tile["shape"] = tuple(shape)
        tile["dtype"] = dtype
        tile["n_pages"] = n_pages
        tile["slab_depth"] = _slab_depth_for(shape, dtype.itemsize, slab_memory)

        if scale_factors is None:
            scale_factors = _derive_scale_factors(shape)

        slab_mb = tile["slab_depth"] * int(np.prod(shape[1:])) * dtype.itemsize / (1 << 20)
        print(f"  shape {tuple(shape)}, slab depth {tile['slab_depth']} ({slab_mb:.0f} MB per read)")

        _convert_tile(out_path, tile, state, scale_factors, n_threads, retry_config)

        if staged_path is not None:
            os.remove(staged_path)

    # We don't need to add additional xml metadata if we convert to ome-zarr.
    if convert_to_ome_zarr:
        return

    # Add an empty missing views field.
    # This is expected by BigStitcher.
    _write_missing_views(out_path)
    _verify_conversion(out_path, state, len(attrs_to_setups))


def _report_dry_run(tiles, config, remote):
    """Print the input files, their pairing and their shapes without converting anything."""
    print(f"Found {len(tiles)} input file(s).")
    for tile in tiles:
        print(f"\n  tile {tile['attribute_id']} -> setup {tile['setup_id']}, tp {tile['timepoint']}")
        print(f"    image:    {tile['source']}")
        print(f"    metadata: {tile['metadata_file']}")
        if remote:
            # The image is still on the share, so its header cannot be read without a download.
            continue
        try:
            shape, dtype, _ = _input_shape_and_dtype(tile["source"], tile["metadata_file"], config)
            print(f"    shape:    {tuple(shape)} of {dtype}")
            if Path(tile["source"]).suffix == ".raw":
                _check_raw_size(tile["source"], shape, config)
                print("    size:     matches the metadata")
        except Exception as error:
            print(f"    [error]   {error}")


def convert_lightsheet_to_bdv_cli():
    """@private
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert lightsheet data from a flamingo microscope to a format compatible with BigDataViewer / BigStitcher. "  # noqa
                    "For most flamingo data it should be sufficient to run the script like this: \n"
                    "python convert_flamingo_data.py -i /path/to/flamingo_data -o /path/to/output.n5 \n"
                    "Here, -i specifies the path to the input folder and -o specifies the path to the output data. \n"
                    "In order to process flamingo data stored in raw format you also need to pass the argument '-f .raw'."  # noqa
    )
    parser.add_argument(
        "--input_root", "-i", required=True,
        help="Folder that contains the data from the flamingo microscope. "
             "With --username this is the directory on the SMB share instead."
    )
    parser.add_argument(
        "--out_path", "-o", required=True, help="Output path where the converted data will be saved."
    )
    parser.add_argument(
        "--file_ext", "-f", default=".tif",
        help="The file extension of the image data. By default '.tif' is used, pass '.raw' if your data is stored as raw files."  # noqa
    )
    parser.add_argument(
        "--metadata_pattern", default="*_Settings.txt",
        help="The filepattern for finding metadata information. The default value works for flamingo data."
    )
    parser.add_argument(
        "--username", "-u", default=None,
        help="GWDG username. Reads the input from the SMB share with smbclient instead of from a path. "
             "Needs --stage_dir. smbclient is not available on Windows; mount the share there instead."
    )
    parser.add_argument(
        "--smb_server", "-s", default=SMB_SERVER, help="The SMB server that holds the input data."
    )
    parser.add_argument(
        "--smb_timeout", type=int, default=SMB_TIMEOUT,
        help="The per-operation timeout for smbclient in seconds. smbclient's own default is 20, "
             "which a multi-GB read exceeds. Raise it further if a download stops with "
             "NT_STATUS_IO_TIMEOUT."
    )
    parser.add_argument(
        "--stage_dir", default=None,
        help="Local directory that holds one input file at a time. Required with --username, "
             "optional for a mounted path where reading slabs directly is still too unreliable."
    )
    parser.add_argument(
        "--slab_memory", type=float, default=DEFAULT_SLAB_MEMORY,
        help="Memory budget for one z-slab in GB. It sets how many planes are read at once."
    )
    parser.add_argument(
        "--max_retries", type=int, default=MAX_RETRIES, help="Maximal number of attempts per read."
    )
    parser.add_argument(
        "--retry_delay", type=float, default=RETRY_DELAY,
        help="Delay before the second attempt in seconds. It grows for every further attempt."
    )
    parser.add_argument(
        "--max_retry_delay", type=float, default=60.0, help="Upper bound for the retry delay in seconds."
    )
    parser.add_argument(
        "--n_threads", type=int, default=None,
        help="Number of threads for writing and downscaling. All cores are used by default."
    )
    parser.add_argument(
        "--restart", action="store_true",
        help="Delete an existing output and convert again, instead of resuming it."
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only report the input files, their pairing and their shapes, then stop."
    )
    parser.add_argument(
        "--log_file", default=None,
        help="File that records failed reads and transfers. Defaults to <out_path>.convert_log.txt."
    )

    args = parser.parse_args()
    if args.metadata_pattern == "":
        metadata_pattern = None
    else:
        metadata_pattern = args.metadata_pattern

    retry_config = RetryConfig(
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        max_retry_delay=args.max_retry_delay,
        log_file=args.log_file,
    )

    convert_lightsheet_to_bdv(
        root=args.input_root,
        out_path=args.out_path,
        file_ext=args.file_ext,
        metadata_file_name_pattern=metadata_pattern,
        username=args.username,
        smb_server=args.smb_server,
        smb_timeout=args.smb_timeout,
        stage_dir=args.stage_dir,
        slab_memory=args.slab_memory,
        n_threads=args.n_threads,
        retry_config=retry_config,
        restart=args.restart,
        dry_run=args.dry_run,
        log_file=args.log_file,
    )
