"""Stain intensity at synapse detections, absolute and relative to the local background.

Each spot gets a cube of voxels for its signal, a core ellipsoid for its mean and integrated intensity,
and an ellipsoidal shell for its local background. The core and the shell are given as (x, y, z)
semi-axes in micrometer, because the spots of the light-sheet data are longer along z than along x
and y. The shell leaves out the cores of the other spots of the same table, so that dense synapses do
not raise the background.

Run `python -m flamingo_tools.synapse_detection.spot_intensity -h` for the command line.
"""

import argparse
import os
from concurrent import futures
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from flamingo_tools.cpu_utils import available_cpus
from flamingo_tools.file_utils import read_image_data
from flamingo_tools.s3_utils import read_table

Radius = Union[float, Sequence[float]]

# Validated on M_AMD_000117_L, 1550 CtBP2 and 2808 Homer1 spots in 30 blocks along the IHC row. The core
# semi-axes are the half width at half maximum of the median spot profiles: 0.7 - 0.9 um in x and y,
# 0.9 - 1.0 um in z. At the inner shell radius the profiles are within about 10 % of their plateau,
# and a change of the shell by +/- 0.5 um changes the median background by less than 4 %.
DEFAULT_CORE_RADIUS = (0.8, 0.8, 1.0)
DEFAULT_BACKGROUND_RADII = ((2.5, 2.5, 3.0), (4.5, 4.5, 5.0))
DEFAULT_SIGNAL_HALO = 1

INTENSITY_MEASURES = (
    "signal", "mean", "background", "noise", "net_signal", "ratio", "snr", "integrated", "n_saturated",
)
"""The per-channel columns `<channel>_<measure>` of `measure_spot_intensities`:
- 'signal': mean in the (2 x signal_halo + 1)^3 voxels around the spot voxel.
- 'mean': mean in the core ellipsoid.
- 'background': median in the background shell.
- 'noise': 1.4826 x the median absolute deviation in the shell, which estimates the standard deviation
  of one voxel.
- 'net_signal': signal - background.
- 'ratio': signal / background.
- 'snr': (signal - background) / noise.
- 'integrated': sum of (intensity - background) over the core, in intensity x voxels.
- 'n_saturated': voxels of the core or the signal cube at the maximum of an integer dtype. NaN for float data.
"""

# The scale factor from the median absolute deviation to the standard deviation of a normal distribution.
_MAD_TO_STD = 1.4826


def _semi_axes(radius: Radius, voxel_size: Sequence[float]) -> np.ndarray:
    """Convert a radius in micrometer, scalar or (x, y, z), to ZYX semi-axes in voxels."""
    radius = np.broadcast_to(np.asarray(radius, dtype="float64"), (3,))
    return radius[::-1] / np.asarray(voxel_size, dtype="float64")[::-1]


def _group_by_block(voxels: np.ndarray, block_shape: Sequence[int]) -> List[np.ndarray]:
    block_ids = voxels // np.asarray(block_shape)
    _, inverse = np.unique(block_ids, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    order = np.argsort(inverse, kind="stable")
    starts = np.flatnonzero(np.r_[True, inverse[order][1:] != inverse[order][:-1]])
    return np.split(order, starts[1:])


def _ellipsoid_distance(grid: Tuple[np.ndarray, ...], center: np.ndarray, semi_axes: np.ndarray) -> np.ndarray:
    """Squared normalized distance: values <= 1 lie inside the ellipsoid."""
    return sum(((g - c) / a) ** 2 for g, c, a in zip(grid, center, semi_axes))


def measure_spot_intensities(
    images: Dict[str, np.typing.ArrayLike],
    spots: pd.DataFrame,
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    signal_halo: int = DEFAULT_SIGNAL_HALO,
    core_radius: Radius = DEFAULT_CORE_RADIUS,
    background_radii: Tuple[Radius, Radius] = DEFAULT_BACKGROUND_RADII,
    exclusion_radius: Optional[Radius] = None,
    block_shape: Optional[Sequence[int]] = None,
    n_threads: Optional[int] = None,
) -> pd.DataFrame:
    """Measure the stain intensity of each image channel at each spot.

    Args:
        images: The image volumes in ZYX order, keyed by the channel name. All have the same shape.
        spots: The spot table, with the positions in micrometer in the columns 'x', 'y', 'z'.
        voxel_size: The voxel size in micrometer, in (x, y, z) order.
        signal_halo: The halo in voxels around the spot voxel for the 'signal' measure. 0 uses the spot
            voxel only, 1 the 3 x 3 x 3 voxels. The cube is clipped at the border of the volume.
        core_radius: The semi-axes of the core ellipsoid in micrometer, scalar or (x, y, z).
        background_radii: The inner and the outer semi-axes of the background shell in micrometer.
        exclusion_radius: The semi-axes of the region around every other spot that the shell leaves out.
            Default: the core radius.
        block_shape: The block shape in voxels (ZYX) for grouping the spots. Each block is read once,
            with a halo for the background shell. Default: twice the chunks of the first image.
        n_threads: The number of threads.

    Returns:
        The spot table with the columns 'nn_distance' (micrometer, to the nearest other spot),
        'n_core_voxels', 'n_background_voxels', and `<channel>_<measure>` for each measure in
        `INTENSITY_MEASURES`. Spots outside the volume get NaN.
    """
    if not images:
        raise ValueError("Pass at least one image.")
    shapes = {name: tuple(image.shape) for name, image in images.items()}
    shape = next(iter(shapes.values()))
    if any(sh != shape for sh in shapes.values()):
        raise ValueError(f"The images must have the same shape, got {shapes}.")
    shape = np.asarray(shape)
    if signal_halo < 0:
        raise ValueError(f"The signal halo must not be negative, got {signal_halo}.")

    if exclusion_radius is None:
        exclusion_radius = core_radius
    core_axes = _semi_axes(core_radius, voxel_size)
    inner_axes, outer_axes = (_semi_axes(radius, voxel_size) for radius in background_radii)
    exclusion_axes = _semi_axes(exclusion_radius, voxel_size)
    if np.any(inner_axes > outer_axes):
        raise ValueError("The inner background radius must not exceed the outer one.")
    window = np.maximum(np.ceil(np.maximum(outer_axes, core_axes)).astype(int), signal_halo)

    if block_shape is None:
        chunks = getattr(next(iter(images.values())), "chunks", None)
        block_shape = tuple(2 * c for c in chunks) if chunks is not None else (128, 128, 128)

    positions_um = spots[["x", "y", "z"]].to_numpy(dtype="float64")
    # Table coordinates are (x, y, z) in micrometer, the arrays are ZYX voxels.
    positions = positions_um[:, ::-1] / np.asarray(voxel_size, dtype="float64")[::-1]
    centers = np.round(positions).astype(int)
    inside = np.all((centers >= 0) & (centers < shape), axis=1)

    n_spots = len(spots)
    tree = cKDTree(positions_um) if n_spots > 0 else None
    nn_distance = np.full(n_spots, np.nan)
    if n_spots > 1:
        distances, _ = tree.query(positions_um, k=2)
        nn_distance = distances[:, 1]
    # Other spots within this distance can reach into the shell with their excluded core.
    exclusion_um = np.asarray(voxel_size)[::-1] * (outer_axes + exclusion_axes)
    neighbor_radius = float(np.linalg.norm(exclusion_um))

    geometry = {name: np.full(n_spots, np.nan) for name in ("n_core_voxels", "n_background_voxels")}
    measures = {
        f"{channel}_{measure}": np.full(n_spots, np.nan) for channel in images for measure in INTENSITY_MEASURES
    }
    saturation = {
        channel: np.iinfo(image.dtype).max if np.issubdtype(image.dtype, np.integer) else None
        for channel, image in images.items()
    }

    def measure_group(indices):
        begin = np.maximum(centers[indices].min(axis=0) - window, 0)
        end = np.minimum(centers[indices].max(axis=0) + window + 1, shape)
        bb = tuple(slice(b, e) for b, e in zip(begin, end))
        blocks = {channel: np.asarray(image[bb]) for channel, image in images.items()}

        for index in indices:
            center = centers[index]
            lo = np.maximum(center - window, begin)
            hi = np.minimum(center + window + 1, end)
            grid = np.ogrid[tuple(slice(lb, hb) for lb, hb in zip(lo, hi))]

            core = _ellipsoid_distance(grid, positions[index], core_axes) <= 1
            # A core smaller than one voxel still holds the voxel of the spot.
            core[tuple(center - lo)] = True
            # The grid is clipped to the volume, so the cube is clipped at the border, too.
            cube = np.ones(core.shape, dtype=bool)
            for g, c in zip(grid, center):
                cube &= np.abs(g - c) <= signal_halo
            saturation_region = core | cube
            shell = (_ellipsoid_distance(grid, positions[index], inner_axes) > 1) & (
                _ellipsoid_distance(grid, positions[index], outer_axes) <= 1
            )
            for neighbor in tree.query_ball_point(positions_um[index], neighbor_radius):
                if neighbor != index:
                    shell &= _ellipsoid_distance(grid, positions[neighbor], exclusion_axes) > 1

            geometry["n_core_voxels"][index] = core.sum()
            geometry["n_background_voxels"][index] = shell.sum()
            local = tuple(slice(lb - b, hb - b) for lb, hb, b in zip(lo, hi, begin))
            for channel, block in blocks.items():
                values = block[local]
                core_values = values[core].astype("float64")
                signal = float(values[cube].mean(dtype="float64"))
                result = {"signal": signal, "mean": core_values.mean()}
                if shell.any():
                    shell_values = values[shell].astype("float64")
                    background = np.median(shell_values)
                    noise = _MAD_TO_STD * np.median(np.abs(shell_values - background))
                    result.update({
                        "background": background,
                        "noise": noise,
                        "net_signal": signal - background,
                        "ratio": signal / background if background > 0 else np.nan,
                        "snr": (signal - background) / noise if noise > 0 else np.nan,
                        "integrated": float(np.sum(core_values - background)),
                    })
                if saturation[channel] is not None:
                    result["n_saturated"] = np.count_nonzero(values[saturation_region] == saturation[channel])
                for measure, value in result.items():
                    measures[f"{channel}_{measure}"][index] = value

    groups = _group_by_block(centers[inside], block_shape) if inside.any() else []
    inside_indices = np.flatnonzero(inside)
    n_threads = available_cpus() if n_threads is None else n_threads
    with futures.ThreadPoolExecutor(n_threads) as pool:
        list(pool.map(measure_group, [inside_indices[group] for group in groups]))

    result = spots.copy()
    result["nn_distance"] = nn_distance
    for name, values in {**geometry, **measures}.items():
        result[name] = values
    return result


def channel_name_from_path(path: str) -> str:
    """Derive the channel name from an image path, e.g. 'CTBP2' from '.../CTBP2.ome.zarr'."""
    return os.path.basename(path.rstrip("/")).split(".")[0]


def spot_intensity_table(
    table_path: str,
    image_paths: Sequence[str],
    output_path: Optional[str] = None,
    channel_names: Optional[Sequence[str]] = None,
    id_column: Optional[str] = None,
    input_key: Optional[str] = "s0",
    voxel_size: Sequence[float] = (0.38, 0.38, 0.38),
    s3_images: bool = False,
    s3_table: bool = False,
    s3_credentials: Optional[str] = None,
    s3_bucket_name: Optional[str] = None,
    s3_service_endpoint: Optional[str] = None,
    **measure_kwargs,
) -> pd.DataFrame:
    """Read a spot table and image volumes, measure the spot intensities, and optionally save a TSV.

    Args:
        table_path: The spot table with the positions in micrometer in the columns 'x', 'y', 'z'.
        image_paths: The image volumes, in zarr or n5 format, or as tif files with `input_key=None`.
        output_path: The output TSV file. Nothing is written if None.
        channel_names: The channel names for the output columns. Default: derived from the image paths.
        id_column: Keep only this column of the spot table, e.g. 'spot_id', next to the measures.
            The output then has the layout of an object-measures table. Default: keep all columns.
        input_key: The key of the image data in the containers.
        voxel_size: The voxel size in micrometer, in (x, y, z) order.
        s3_images: Read the images from the S3 bucket.
        s3_table: Read the spot table from the S3 bucket.
        s3_credentials: File with the S3 credentials.
        s3_bucket_name: S3 bucket name.
        s3_service_endpoint: S3 service endpoint.
        measure_kwargs: Further arguments for `measure_spot_intensities`.

    Returns:
        The spot table with the intensity measures, or only its id column with the measures.
    """
    if channel_names is None:
        channel_names = [channel_name_from_path(path) for path in image_paths]
    if len(channel_names) != len(image_paths):
        raise ValueError("Pass one channel name per image path.")
    if len(set(channel_names)) != len(channel_names):
        raise ValueError(f"The channel names must be unique, got {channel_names}.")

    s3_kwargs = dict(bucket_name=s3_bucket_name, service_endpoint=s3_service_endpoint, credential_file=s3_credentials)
    spots = read_table(table_path, s3_table, **s3_kwargs)
    if id_column is not None and id_column not in spots.columns:
        raise ValueError(f"The spot table has no column '{id_column}'.")
    images = {
        name: read_image_data(path, input_key, from_s3=s3_images, **s3_kwargs)
        for name, path in zip(channel_names, image_paths)
    }
    result = measure_spot_intensities(images, spots, voxel_size=voxel_size, **measure_kwargs)
    if id_column is not None:
        result = result[[id_column] + [column for column in result.columns if column not in spots.columns]]
    if output_path is not None:
        result.to_csv(output_path, sep="\t", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Measure the stain intensity at synapse detections: the signal, the core mean and the "
        "integrated intensity, and the local background with the ratio and the signal-to-noise ratio."
    )
    parser.add_argument("-t", "--table", required=True, help="Spot table with 'x', 'y', 'z' in micrometer.")
    parser.add_argument("-i", "--image_paths", required=True, nargs="+",
                        help="Image volumes, e.g. M_AMD_000117_L/images/ome-zarr/CTBP2.ome.zarr.")
    parser.add_argument("-o", "--output", required=True, help="Output TSV file.")
    parser.add_argument("--channel_names", nargs="+", default=None,
                        help="Column prefixes for the channels. Default: the image file names.")
    parser.add_argument("--id_column", default=None,
                        help="Write only this column of the spot table, e.g. spot_id, and the measures. "
                        "Default: write all columns of the spot table.")
    parser.add_argument("-k", "--input_key", default="s0", help="Key of the image data. Default: s0")
    parser.add_argument("--voxel_size", type=float, nargs=3, default=[0.38, 0.38, 0.38],
                        help="Voxel size in micrometer (x, y, z).")
    parser.add_argument("--signal_halo", type=int, default=DEFAULT_SIGNAL_HALO,
                        help="Halo in voxels around the spot voxel for the signal: 0 = spot voxel only, "
                        f"1 = 3 x 3 x 3 voxels. Default: {DEFAULT_SIGNAL_HALO}")
    parser.add_argument("--core_radius", type=float, nargs=3, default=list(DEFAULT_CORE_RADIUS),
                        help="Semi-axes of the core ellipsoid in micrometer (x, y, z).")
    parser.add_argument("--background_radii", type=float, nargs=6,
                        default=[r for radius in DEFAULT_BACKGROUND_RADII for r in radius],
                        help="Inner and outer semi-axes of the background shell in micrometer "
                        "(x_in, y_in, z_in, x_out, y_out, z_out).")
    parser.add_argument("--n_threads", type=int, default=None, help="Number of threads.")
    parser.add_argument("--s3_images", action="store_true", help="Read the images from the S3 bucket.")
    parser.add_argument("--s3_table", action="store_true", help="Read the spot table from the S3 bucket.")
    parser.add_argument("--s3_credentials", default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", default=None, help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")
    args = parser.parse_args()

    spot_intensity_table(
        table_path=args.table, image_paths=args.image_paths, output_path=args.output,
        channel_names=args.channel_names, id_column=args.id_column, input_key=args.input_key,
        voxel_size=args.voxel_size,
        s3_images=args.s3_images, s3_table=args.s3_table, s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name, s3_service_endpoint=args.s3_service_endpoint,
        signal_halo=args.signal_halo, core_radius=args.core_radius,
        background_radii=(args.background_radii[:3], args.background_radii[3:]),
        n_threads=args.n_threads,
    )


if __name__ == "__main__":
    main()
