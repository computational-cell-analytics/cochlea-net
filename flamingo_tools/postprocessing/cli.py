"""private
"""
import argparse

from .label_components import label_components_single
from .cochlea_mapping import tonotopic_mapping_json_wrapper, equidistant_centers_single
from flamingo_tools.json_util import export_dictionary_as_json
from flamingo_tools.measurements import object_measures_json_wrapper
from flamingo_tools.extract_block_util import extract_block_json_wrapper, extract_central_block_from_json
from flamingo_tools.s3_utils import MOBIE_FOLDER
from flamingo_tools.analysis.density_utils import calc_sgn_density


def equidistant_centers():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for JSON dictionary.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for equidistant centers
    parser.add_argument('-n', "--n_blocks", type=int, default=6,
                        help="Number of blocks to find equidistant centers for. Default: 6")
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'. Default: sgn")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1], help="List of connected components.")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    equidistant_centers_single(
        table_path=args.input,
        output_path=args.output,
        n_blocks=args.n_blocks,
        cell_type=args.cell_type,
        component_list=args.components,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


def extract_block():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument("-o", "--output", type=str, required=True, help="Output directory or file.")
    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to data in n5/ome-zarr/TIF format.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project. Only used for '--json_info'.")

    # options for file naming
    parser.add_argument("--dataset_name", type=str, default=None,
                        help="Name of dataset/cochlea, e.g. M_AMD_N75_L. Used as prefix in output name.")
    parser.add_argument("--image_channel", type=str, default=None,
                        help="Name of image channel/stain, e.g. PV. Used as suffix in output name.")

    # options for block extraction
    parser.add_argument("-c", "--coords", type=int, nargs="+", default=[],
                        help="3D coordinate as center of extracted block [µm].")
    parser.add_argument("--json_info", type=str, default=None,
                        help="JSON file with crop information.")
    parser.add_argument("-k", "--input_key", type=str, default="s0",
                        help="Input key for data in input file with n5/OME-ZARR format.")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Output key for data in output file with n5 format. Default: TIF file.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    parser.add_argument("--roi_halo", type=int, nargs="+", default=[128, 128, 64],
                        help="ROI halo around center coordinate [pixel]. Default: 128 128 64")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()
    args_dict = vars(args)
    extract_block_json_wrapper(
        input_path=args_dict.pop("input"),
        output_path=args_dict.pop("output"),
        json_file=args_dict.pop("json_info"),
        **args_dict,
    )


def extract_central_blocks():
    parser = argparse.ArgumentParser(
        description="Script to extract multiple blocks for intensity annotation based on a JSON file.")

    parser.add_argument("-i", "--input", type=str, default=None, help="Input JSON dictionary.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output directory for extracted blocks.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("-m", "--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project.")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    extract_central_block_from_json(
        json_file=args.input,
        output_path=args.output,
        force_overwrite=args.force,
        mobie_dir=args.mobie_dir,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


def json_block_extraction():
    parser = argparse.ArgumentParser(
        description="Script to create JSON dictionary used for extracting blocks for intensity annotation.")

    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for JSON dictionary.")
    parser.add_argument("-d", "--dataset_name", type=str, required=True, help="Dataset name.")
    parser.add_argument("-i", "--image_channel", type=str, required=True, nargs="+",
                        help="Input image channel(s), e.g. PV, SGN_v2.")
    parser.add_argument("-s", "--segmentation_channel", type=str, default="",
                        help="Segmentation channel as reference for finding equidistant centers.")
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'. Default: sgn")
    parser.add_argument("-c", "--component_list", type=int, nargs="+", default=[1],
                        help="List of connected components.")
    parser.add_argument('-n', "--n_blocks", type=int, default=6,
                        help="Number of blocks to extract. Default: 6")
    parser.add_argument("--roi_halo", type=int, nargs="+", default=[256, 256, 64],
                        help="ROI halo around center coordinate [pixel]. "
                        "Cropped mask has twice the size. Default: 256 256 64")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    args = parser.parse_args()
    args_dict = vars(args)

    export_dictionary_as_json(
        output_path=args_dict.pop("output"),
        force_overwrite=args_dict.pop("force"),
        param_dict=args_dict,
    )


def label_components():
    parser = argparse.ArgumentParser(
        description="Script to label segmentation using a segmentation table and graph connected components.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Input path to segmentation table.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for segmentation table. Default: Overwrite input table.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for post-processing
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'. Default: sgn")
    parser.add_argument("--min_size", type=int, default=1000,
                        help="Minimal number of pixels for filtering small instances. Default: 1000")
    parser.add_argument("--min_component_length", type=int, default=50,
                        help="Minimal length for filtering out connected components. Default: 50")
    parser.add_argument(
        "--max_edge_distance", type=float, default=30,
        help="Maximal distance in micrometer between points to create edges for connected components. Default: 30",
    )
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1], help="List of connected components.")
    parser.add_argument("--napari", action="store_true",
                        help="Use napari image viewer for visualizing labeled components.")
    parser.add_argument("--scale_factor", type=int, default=20,
                        help="Scale factor for down-scaling data for visualization in Napari. Default: 20")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    label_components_single(
        table_path=args.input,
        out_path=args.output,
        cell_type=args.cell_type,
        component_list=args.components,
        max_edge_distance=args.max_edge_distance,
        min_component_length=args.min_component_length,
        min_size=args.min_size,
        force_overwrite=args.force,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
        use_napari=args.napari,
        scale_factor=args.scale_factor,
    )


def object_measures():
    parser = argparse.ArgumentParser(
        description="Script to compute object measures for different stainings.")

    parser.add_argument("-o", "--output", type=str, nargs="+", default=[],
                        help="Output path(s). Either directory or specific file(s).")
    parser.add_argument("-i", "--image_paths", type=str, nargs="+", default=None,
                        help="Input path to one or multiple image channels in ome.zarr format.")
    parser.add_argument("--seg_table", type=str, default=None,
                        help="Input path to segmentation table.")
    parser.add_argument("--seg_path", type=str, default=None,
                        help="Input path to segmentation channel in ome.zarr format.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project. Only used for '--json_info'.")

    # options for object measures
    parser.add_argument("--json_info", type=str, default=None,
                        help="JSON file with parameters for object_measures.")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1], help="List of components.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    parser.add_argument("--bg_mask", action="store_true", help="Use background mask for calculating object measures.")
    parser.add_argument("--bg_cache_paths", type=str, nargs="+", default=[],
                        help="Cache path(s) for background mask in zarr format. Either directory or specific file(s).")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    object_measures_json_wrapper(
        out_paths=args.output,
        image_paths=args.image_paths,
        table_path=args.seg_table,
        seg_path=args.seg_path,
        force_overwrite=args.force,
        mobie_dir=args.mobie_dir,
        json_file=args.json_info,
        component_list=args.components,
        voxel_size=args.voxel_size,
        use_bg_mask=args.bg_mask,
        bg_cache_paths=args.bg_cache_paths,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


def tonotopic_mapping():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument("-i", "--input", type=str, default=None, help="Input path to segmentation table.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for segmentation table. Default: Overwrite input table.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for tonotopic mapping
    parser.add_argument("--json_info", type=str, default=None,
                        help="JSON file with dataset information.")
    parser.add_argument("--central_spots_path", type=str, default=None,
                        help="Dataframe containing spots of the central path of the segmentation.")
    parser.add_argument("--animal", type=str, default="mouse",
                        help="Animal type to be used for frequency mapping. Either 'mouse' or 'gerbil'.")
    parser.add_argument("--otof", action="store_true", help="Use frequency mapping for OTOF cochleae.")
    parser.add_argument("--apex_position", type=str, default="apex_higher",
                        help="Use frequency mapping for OTOF cochleae.")

    # options for post-processing
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'.")
    parser.add_argument(
        "-c", "--components", type=int, nargs="+", default=[1],
        help="List of connected components. The order has to match the order within the cochlear volume.",
    )
    parser.add_argument("--include_gap", action="store_true",
                        help="Include gaps between components for calculating the length of the central path.")

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials."
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    tonotopic_mapping_json_wrapper(
        table_path=args.input,
        out_path=args.output,
        json_file=args.json_info,
        central_spots_path=args.central_spots_path,
        force_overwrite=args.force,
        animal=args.animal,
        otof=args.otof,
        apex_position=args.apex_position,
        cell_type=args.cell_type,
        component_list=args.components,
        include_gap=args.include_gap,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )


def sgn_density():
    parser = argparse.ArgumentParser(
        description="Compute SGN density at one or more positions along the Rosenthal's Canal."
    )

    parser.add_argument("-i", "--input", type=str, default=None,
                        help="Input path to SGN segmentation table (TSV). Must contain length_fraction column "
                        "(produced by flamingo_tools.tonotopic_mapping). "
                        "Optional when --json_input is supplied.")
    parser.add_argument("-j", "--json_input", type=str, default=None,
                        help="Input JSON file with metadata information(dataset_name, image_channel, "
                        "segmentation_channel, …). When --input is absent the table path "
                        "is derived from dataset_name and segmentation_channel via --mobie_dir.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path for JSON file with density results.")
    parser.add_argument("--json_output", type=str, default=None,
                        help="Optional output path for block-extraction JSON compatible with "
                        "flamingo_tools.extract_block --json_info. "
                        "Contains crop_centers derived from the density bounding boxes.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument(
        "-p", "--positions", type=str, nargs="+", default=["apex", "mid", "base"],
        help="Cochlear positions to evaluate. Use preset names ('apex', 'mid', 'base') or "
        "floats in [0, 1]. Default: apex mid base",
    )
    parser.add_argument(
        "--slice_thickness", type=float, default=10.0,
        help="Total thickness of the horizontal slice in µm. Default: 10.0",
    )
    parser.add_argument(
        "--run_length_tolerance", type=float, default=0.1,
        help="Maximum allowed length_fraction difference to include an SGN instance. "
        "Reduces contamination from other cochlear turns. Default: 0.1",
    )
    parser.add_argument(
        "-c", "--component_label", type=int, nargs="+", default=[1],
        help="Component label(s) of the main Rosenthal's Canal component. Default: 1",
    )
    parser.add_argument(
        "--axis", type=str, default="z", choices=["x", "y", "z"],
        help="Volume axis perpendicular to the slice plane. Default: z",
    )
    parser.add_argument(
        "--length_fraction_column", type=str, default="length_fraction",
        help="Column name for the run-length fraction in the table. Default: length_fraction",
    )
    parser.add_argument(
        "--mode", type=str, default="2d", choices=["2d", "3d"],
        help="Density mode: '2d' computes density per cross-sectional area (SGN/µm²); "
        "'3d' computes density per convex-hull volume (SGN/µm³). Default: 2d",
    )
    parser.add_argument(
        "--roi_halo", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
        help="ROI halo in pixels [x y z] for the block-extraction JSON output, applied to all "
        "positions. Overrides the value from --json_input. "
        "If omitted, the halo is computed automatically from each position's bounding box.",
    )
    parser.add_argument(
        "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
        help="Voxel size in µm used to convert bounding box extents to pixels when "
        "computing the automatic roi_halo. Provide 1 value (isotropic) or 3 values (x y z). "
        "Default: 0.38 0.38 0.38",
    )
    parser.add_argument(
        "--mobie_dir", type=str, default=None,
        help="Local MoBIE project directory used to locate the table when --json_input is given "
        "and --input is absent. Default: Current directory",
    )
    parser.add_argument(
        "--seg_path", type=str, default=None,
        help="Path to the SGN segmentation volume (local TIF, N5/Zarr, or S3 OME-ZARR). "
        "When omitted and --json_input is given, the path is derived automatically as "
        "<dataset_name>/images/ome-zarr/<segmentation_channel>.ome.zarr. "
        "Only used when --min_overlap_fraction is set.",
    )
    parser.add_argument(
        "--seg_key", type=str, default="s0",
        help="Internal key for N5/Zarr/OME-ZARR segmentation (default: s0).",
    )
    parser.add_argument(
        "--min_overlap_fraction", type=float, default=None,
        help="Minimum fraction of an SGN's voxels (n_pixels) that must lie within the "
        "slice sub-volume to count the instance. Range (0, 1]. "
        "Default: None (no segmentation-based filtering). "
        "Whether --seg_path is a pre-extracted crop or a full volume is detected automatically.",
    )
    parser.add_argument(
        "--min_overlap_volume", type=float, default=None,
        help="Minimum volume[µm³] of an SGN's voxels (n_pixels) that must lie within the "
        "slice sub-volume to count the instance. "
        "Default: None (no segmentation-based filtering). "
        "Whether --seg_path is a pre-extracted crop or a full volume is detected automatically.",
    )

    # options for S3 bucket
    parser.add_argument("--s3", action="store_true", help="Flag for using S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    if args.input is None and args.json_input is None:
        parser.error("Provide --input or --json_input.")

    calc_sgn_density(
        output=args.output,
        seg_table_path=args.input,
        json_input=args.json_input,
        json_output=args.json_output,
        force_overwrite=args.force,
        positions=args.positions,
        slice_thickness=args.slice_thickness,
        run_length_tolerance=args.run_length_tolerance,
        component_list=args.component_label,
        axis=args.axis,
        length_fraction_column=args.length_fraction_column,
        density_mode=args.mode,
        roi_halo=args.roi_halo,
        voxel_size=args.voxel_size,
        mobie_dir=args.mobie_dir,
        seg_path=args.seg_path,
        seg_key=args.seg_key,
        min_overlap_fraction=args.min_overlap_fraction,
        min_overlap_volume=args.min_overlap_volume,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )