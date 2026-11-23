"""@private
"""
import argparse
import os
from typing import Optional

import pandas as pd

from flamingo_tools.cli_utils import check_mode_options
from flamingo_tools.s3_utils import MOBIE_FOLDER, split_table_path
from .frequency import export_frequency, export_frequency_cochlea
from .lower_resolution import DEFAULT_CHANNELS, export_lower_resolution, export_lower_resolution_cochlea
from .marker import export_marker, export_marker_cochlea
from .subtypes import export_subtypes, export_subtypes_cochlea
from .synapses import export_synapses, export_synapses_cochlea

# The input modes, as named in the error messages.
_PATH = "--input"
_COCHLEA = "--cochlea"

# The options of each command that only one input mode uses.
_MODE_OPTIONS = {
    "lower_resolution": {"table": _PATH, "sgn_table": _PATH, "ihc_table": _PATH,
                         "channels": _COCHLEA, "filter_cochlea_channels": _COCHLEA},
    "marker": {"channels": _COCHLEA},
    "subtypes": {"table": _PATH, "segmentation": _COCHLEA},
    "synapses": {"synapse_table": _PATH, "ihc_table": _PATH, "synapse_name": _COCHLEA, "reference_ihcs": _COCHLEA},
    "frequency": {"table": _PATH, "segmentation": _COCHLEA},
}
_SHARED_MODE_OPTIONS = {"input_key": _PATH, "name": _PATH, "mobie_dir": _COCHLEA}

# The options of each command that one input mode requires.
_REQUIRED = {
    ("marker", _PATH): ["table"],
    ("marker", _COCHLEA): ["channels"],
    ("subtypes", _PATH): ["table", "stains"],
    ("synapses", _PATH): ["synapse_table"],
    ("frequency", _PATH): ["table"],
    ("frequency", _COCHLEA): ["segmentation"],
}

_MASK_OPTIONS = ("filter_sgn_components", "filter_ihc_components", "filter_dilation_iterations")


def _shared_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("-i", "--input", type=str, default=None,
                        help="Input path to the data in TIF or n5/zarr/OME-Zarr format. "
                        "With '--s3', the path is relative to the bucket.")
    source.add_argument("-c", "--cochlea", type=str, default=None,
                        help="Name of a cochlea in a MoBIE project, e.g. M_LR_000227_R. The data is read from "
                        "<cochlea>/images/ome-zarr/<source>.ome.zarr and the tables from "
                        "<cochlea>/tables/<source>/default.tsv, in '--mobie_dir' or with '--s3' in the bucket. "
                        "The output is written to <output_folder>/<cochlea>.")
    parser.add_argument("-o", "--output_folder", type=str, required=True,
                        help="Output folder. Each scale level gets a sub-folder 'scale<N>'.")
    parser.add_argument("-s", "--scale", type=int, nargs="+", required=True,
                        help="Scale level(s). Scale N has 2^N times the voxel size of the input. The command "
                        "uses the pyramid level N levels below the input key, e.g. s3 for s0 and scale 3, if the "
                        "container holds it. Otherwise, it downsamples the input.")

    # options for '--input'
    parser.add_argument("-k", "--input_key", type=str, default="s0",
                        help="Key of the input in an n5/zarr container, for '--input', e.g. setup0/timepoint0/s0. "
                        "'--voxel_size' is the voxel size of this key. Ignored for a TIF file. Default: s0")
    parser.add_argument("-n", "--name", type=str, default=None,
                        help="Name of the output, for '--input'. Default: the input file name without extension. "
                        "For 'synapses', the default is the name of the synapse table.")

    # options for '--cochlea'
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory of the MoBIE project, for '--cochlea' without '--s3'. "
                        "Default: the MoBIE project on the cluster, flamingo_tools.s3_utils.MOBIE_FOLDER.")

    # options for the crop
    parser.add_argument("--crop_center", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Crop center in micrometer. Requires '--roi_halo' or '--axis'.")
    parser.add_argument("--roi_halo", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                        help="Halo around the crop center in pixels at the target scale. "
                        "Optional with '--axis': the whole plane is cropped in that case.")
    parser.add_argument("--axis", type=int, choices=[0, 1, 2], default=None,
                        help="Axis (0=x, 1=y, 2=z) to crop as a single-pixel 2D slice at the crop center.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Label appended to the output name of a crop, e.g. a position name such as 'apex'.")

    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of the input in micrometer, one value or X Y Z. Default: 0.38 0.38 0.38")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true",
                        help="Flag for using S3 bucket. It applies to '--input' and to all data of '--cochlea'. "
                        "A table that is given as a path is always a local file.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")
    return parser


def _add_lower_resolution(parser):
    parser.add_argument("--channels", type=str, nargs="+", default=list(DEFAULT_CHANNELS),
                        help="Image or segmentation channels, for '--cochlea'. Default: PV Vglut3 CTBP2")
    parser.add_argument("--filter_by_components", type=int, nargs="+", default=None,
                        help="Keep only the objects of these components, for a segmentation. With '--input', "
                        "requires '--table'. With '--cochlea', the table of each channel is used.")
    parser.add_argument("--table", type=str, default=None,
                        help="Segmentation table (TSV) of the input, for '--input' with '--filter_by_components'.")
    parser.add_argument("--sgn_table", type=str, default=None,
                        help="SGN segmentation table (TSV) for a mask of the cochlea, for '--input'. "
                        "Everything outside of the mask is set to 0.")
    parser.add_argument("--ihc_table", type=str, default=None,
                        help="IHC segmentation table (TSV) for a mask of the cochlea, for '--input'. "
                        "With '--sgn_table', the mask also covers the region between the SGNs and the IHCs.")
    parser.add_argument("--filter_cochlea_channels", type=str, nargs="+", default=None,
                        help="Segmentation(s) for a mask of the cochlea, for '--cochlea'. A name with 'SGN' gives "
                        "the SGN table, a name with 'IHC' the IHC table.")
    parser.add_argument("--filter_sgn_components", type=int, nargs="+", default=[1],
                        help="Components of the SGN table that the cochlea mask covers. Default: 1")
    parser.add_argument("--filter_ihc_components", type=int, nargs="+", default=[1],
                        help="Components of the IHC table that the cochlea mask covers. Default: 1")
    parser.add_argument("--filter_dilation_iterations", type=int, default=8,
                        help="Iterations for the dilation of the cochlea mask. Default: 8")
    parser.add_argument("--max_intensity", type=float, default=None,
                        help="Clip the intensities of image data at this value. A segmentation is not clipped. "
                        "With '--cochlea', the default clips PV (except for LaVision cochleae) and CTBP2 at 1400.")
    parser.add_argument("--labels", action="store_true",
                        help="The input is a segmentation, e.g. one with 8 or 16 bit integers: downsample it by "
                        "nearest neighbor sampling and do not clip it. Default: a segmentation for "
                        "'--filter_by_components' and for 32 or 64 bit integer data.")
    parser.add_argument("--binarize", action="store_true", help="Write 1 for every non-zero voxel.")
    parser.add_argument("--ome_zarr", action="store_true", help="Write a zarr container instead of a TIF file.")


def _add_marker(parser):
    parser.add_argument("--channels", type=str, nargs="+", default=None,
                        help="Segmentation(s) of the cochlea, for '--cochlea'.")
    parser.add_argument("--table", type=str, default=None,
                        help="Segmentation table (TSV) with the column 'marker_labels'. Required for '--input'. "
                        "With '--cochlea', it replaces the default table of each channel, e.g. for an edited table.")


def _add_subtypes(parser):
    parser.add_argument("--segmentation", type=str, default=None,
                        help="SGN segmentation of the cochlea, for '--cochlea'. Default: the segmentation of the "
                        "cochlea in flamingo_tools.postprocessing.sgn_subtype_utils.COCHLEAE.")
    parser.add_argument("--table", type=str, default=None,
                        help="Segmentation table (TSV) with a column 'marker_<stain>' for each stain. "
                        "Required for '--input'.")
    parser.add_argument("--stains", type=str, nargs="+", default=None,
                        help="The subtype stains, one or two, e.g. CR Ntng1. Required for '--input'. "
                        "Default for '--cochlea': the stains of the cochlea in COCHLEAE.")
    parser.add_argument("--merge_masks", action="store_true",
                        help="Write one categorical subtype label volume instead of one binary mask per subtype.")


def _add_synapses(parser):
    parser.add_argument("--synapse_table", type=str, default=None,
                        help="Synapse table (TSV) with the columns x, y, z (in micrometer), spot_id, matched_ihc and "
                        "distance_to_ihc. Required for '--input', which then is the volume that sets the output "
                        "shape, e.g. the IHC segmentation. The default output name is the source name of a MoBIE "
                        "table <cochlea>/tables/<source>/default.tsv, otherwise the table file name.")
    parser.add_argument("--ihc_table", type=str, default=None,
                        help="IHC segmentation table (TSV), for '--input'. Only the synapses that are matched to the "
                        "IHCs of '--filter_ihc_components' are exported.")
    parser.add_argument("--synapse_name", type=str, default="synapse_v3_ihc_v11",
                        help="Synapse detection source, for '--cochlea'. Default: synapse_v3_ihc_v11")
    parser.add_argument("--reference_ihcs", type=str, default="IHC_v11",
                        help="IHC segmentation that the synapses are matched to, for '--cochlea'. Default: IHC_v11")
    parser.add_argument("--filter_ihc_components", type=int, nargs="+", default=[1],
                        help="Components of the IHCs. Default: 1")
    parser.add_argument("--max_dist", type=float, default=3.0,
                        help="Maximal distance of a synapse to its IHC in micrometer. Default: 3.0")
    parser.add_argument("--radius", type=int, default=3,
                        help="Radius of a synapse in the output volume in pixels. Default: 3")
    parser.add_argument("--id_offset", type=int, default=0,
                        help="Offset added to the IDs in the output volume, for other colors in a viewer. Default: 0")
    parser.add_argument("--use_syn_ids", action="store_true",
                        help="Write the synapse IDs instead of the IDs of the matched IHCs.")
    parser.add_argument("--points", action="store_true",
                        help="Write a CSV point table (z, y, x pixels at each scale) instead of the TIF volume. "
                        "napari opens it as a points layer.")


def _add_frequency(parser):
    parser.add_argument("--segmentation", type=str, default=None,
                        help="Segmentation with a tonotopic mapping, for '--cochlea', e.g. SGN_v2.")
    parser.add_argument("--table", type=str, default=None,
                        help="Segmentation table (TSV) with the columns label_id, component_labels and "
                        "frequency[kHz]. Required for '--input'.")
    parser.add_argument("--components", type=int, nargs="+", default=[1],
                        help="Export only the objects of these components. Default: 1")
    parser.add_argument("--colormap", type=str, default=None,
                        help="Matplotlib colormap for an RGBA output and a color bar. "
                        "Default: the frequency in kHz as value.")


_COMMANDS = {
    "lower_resolution": (_add_lower_resolution, "Export image data or a segmentation at a lower resolution."),
    "marker": (_add_marker, "Export the marker-positive (1) and the marker-negative (2) objects of a segmentation."),
    "subtypes": (_add_subtypes, "Export the SGN subtypes of a segmentation, from the marker labels of the stains."),
    "synapses": (_add_synapses, "Export synapse detections as a label volume or as a napari point table."),
    "frequency": (_add_frequency, "Export the tonotopic frequency of the objects of a segmentation."),
}


def _read_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    return None if path is None else pd.read_csv(path, sep="\t")


def _table_name(table_path: str) -> str:
    """Name an export after its table: the source of a MoBIE default table, otherwise the file name."""
    parts = split_table_path(table_path)
    if parts is not None:
        return parts[1]
    return os.path.splitext(os.path.basename(table_path))[0]


def _check_options(parser, args, mode):
    check_mode_options(parser, args, mode, {
        **{dest: {option_mode} for dest, option_mode in _SHARED_MODE_OPTIONS.items()},
        **{dest: {option_mode} for dest, option_mode in _MODE_OPTIONS[args.command].items()},
    })
    for dest in _REQUIRED.get((args.command, mode), []):
        if getattr(args, dest) is None:
            parser.error(f"the following arguments are required for {mode}: --{dest}")

    if any(scale < 0 for scale in args.scale):
        parser.error(f"--scale must be non-negative, got {args.scale}.")
    if len(args.voxel_size) not in (1, 3):
        parser.error(f"--voxel_size takes one value or X Y Z, got {args.voxel_size}.")
    if args.command == "subtypes" and args.stains is not None and len(args.stains) not in (1, 2):
        parser.error(f"--stains takes one or two stains, got {args.stains}.")
    # The old export scripts always read from S3. Without '--s3', the cochlea is read from the MoBIE project.
    if mode == _COCHLEA and not args.s3 and not os.path.isdir(os.path.join(args.mobie_dir, args.cochlea)):
        parser.error(f"The cochlea {args.cochlea} is not in the MoBIE project {args.mobie_dir}. "
                     "Give --s3 to read it from the S3 bucket, or give --mobie_dir.")

    if args.crop_center is None:
        for dest in ("roi_halo", "axis", "suffix"):
            if getattr(args, dest) is not None:
                parser.error(f"--{dest} requires --crop_center.")
    elif args.roi_halo is None and args.axis is None:
        parser.error("--crop_center requires --roi_halo, unless --axis is also given "
                     "(the whole plane is cropped in that case).")

    if args.command == "lower_resolution":
        if mode == _PATH and (args.table is None) != (args.filter_by_components is None):
            parser.error("--table and --filter_by_components require each other for --input.")
        has_mask = any(getattr(args, dest, None) is not None
                       for dest in ("sgn_table", "ihc_table", "filter_cochlea_channels"))
        for dest in _MASK_OPTIONS:
            if not has_mask and getattr(args, dest) != parser.get_default(dest):
                parser.error(f"--{dest} requires a cochlea mask: --sgn_table / --ihc_table or "
                             "--filter_cochlea_channels.")
    if args.command == "synapses" and mode == _PATH and args.ihc_table is None:
        if args.filter_ihc_components != parser.get_default("filter_ihc_components"):
            parser.error("--filter_ihc_components requires --ihc_table for --input.")


def _run(args, mode):
    common = dict(
        output_folder=args.output_folder,
        scale=args.scale,
        crop_center=args.crop_center,
        roi_halo=args.roi_halo,
        axis=args.axis,
        suffix=args.suffix,
        voxel_size=args.voxel_size,
        force=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )
    if mode == _PATH:
        source = dict(input_path=args.input, input_key=args.input_key, name=args.name)
    else:
        source = dict(cochlea=args.cochlea, mobie_dir=args.mobie_dir)
    command = args.command

    if command == "lower_resolution":
        options = dict(
            filter_by_components=args.filter_by_components,
            filter_sgn_components=args.filter_sgn_components,
            filter_ihc_components=args.filter_ihc_components,
            filter_dilation_iterations=args.filter_dilation_iterations,
            max_intensity=args.max_intensity,
            labels=True if args.labels else None,
            binarize=args.binarize,
            ome_zarr=args.ome_zarr,
        )
        if mode == _PATH:
            export_lower_resolution(
                table=_read_table(args.table), sgn_table=_read_table(args.sgn_table),
                ihc_table=_read_table(args.ihc_table), **source, **options, **common,
            )
        else:
            export_lower_resolution_cochlea(
                channels=args.channels, filter_cochlea_channels=args.filter_cochlea_channels,
                **source, **options, **common,
            )

    elif command == "marker":
        if mode == _PATH:
            export_marker(table=_read_table(args.table), **source, **common)
        else:
            export_marker_cochlea(channels=args.channels, table_path=args.table, **source, **common)

    elif command == "subtypes":
        options = dict(stains=args.stains, merge_masks=args.merge_masks)
        if mode == _PATH:
            export_subtypes(table=_read_table(args.table), **source, **options, **common)
        else:
            export_subtypes_cochlea(segmentation=args.segmentation, **source, **options, **common)

    elif command == "synapses":
        options = dict(
            max_dist=args.max_dist,
            radius=args.radius,
            id_offset=args.id_offset,
            filter_ihc_components=args.filter_ihc_components,
            use_syn_ids=args.use_syn_ids,
            points=args.points,
        )
        if mode == _PATH:
            if source["name"] is None:
                source["name"] = _table_name(args.synapse_table)
            export_synapses(
                synapse_table=_read_table(args.synapse_table), ihc_table=_read_table(args.ihc_table),
                **source, **options, **common,
            )
        else:
            export_synapses_cochlea(
                synapse_name=args.synapse_name, reference_ihcs=args.reference_ihcs, **source, **options, **common,
            )

    elif command == "frequency":
        options = dict(components=args.components, colormap=args.colormap)
        if mode == _PATH:
            export_frequency(table=_read_table(args.table), **source, **options, **common)
        else:
            export_frequency_cochlea(segmentation=args.segmentation, **source, **options, **common)


def export_data():
    parser = argparse.ArgumentParser(
        prog="flamingo_tools.export_data",
        description="Export data at a lower resolution, optionally as a crop, e.g. for overview images. "
        "Each command reads its input either from a path ('--input') or from a cochlea in a MoBIE project "
        "('--cochlea'). Run 'flamingo_tools.export_data <command> -h' for the options of a command.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="command")
    shared = _shared_parser()

    command_parsers = {}
    for command, (add_options, description) in _COMMANDS.items():
        command_parser = subparsers.add_parser(command, parents=[shared], help=description, description=description)
        add_options(command_parser)
        command_parsers[command] = command_parser

    args = parser.parse_args()
    command_parser = command_parsers[args.command]
    mode = _PATH if args.input is not None else _COCHLEA
    _check_options(command_parser, args, mode)
    _run(args, mode)


if __name__ == "__main__":
    export_data()
