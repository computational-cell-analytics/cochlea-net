"""private
"""
import argparse

from .label_components import label_components_single
from .cochlea_mapping import tonotopic_mapping_single, equidistant_centers_single
from flamingo_tools.json_util import export_dictionary_as_json
from flamingo_tools.measurements import object_measures_json_wrapper
from flamingo_tools.extract_block_util import extract_block_json_wrapper, extract_central_block_from_json
from flamingo_tools.s3_utils import MOBIE_FOLDER


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

    parser.add_argument('-o', "--output", type=str, required=True, help="Output directory or file.")
    parser.add_argument('-i', '--input', type=str, default=None, help="Input path to data in n5/ome-zarr/TIF format.")
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
    parser.add_argument('-k', "--input_key", type=str, default=None,
                        help="Input key for data in input file with n5/OME-ZARR format.")
    parser.add_argument("--output_key", type=str, default=None,
                        help="Output key for data in output file with n5 format. Default: TIF file.")
    parser.add_argument('-r', "--resolution", type=float, default=0.38,
                        help="Resolution of input in micrometer [µm]. Default: 0.38")
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
    extract_block_json_wrapper(args_dict)


def extract_central_blocks():
    parser = argparse.ArgumentParser(
        description="Script to extract multiple blocks for intensity annotation based on a JSON file.")

    parser.add_argument('-i', '--input', type=str, default=None, help="Input JSON dictionary.")
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
    )


def object_measures():
    parser = argparse.ArgumentParser(
        description="Script to compute object measures for different stainings.")

    parser.add_argument("-o", "--output", type=str, nargs="+", default=[],
                        help="Output path(s). Either directory or specific file(s).")
    parser.add_argument("-i", "--image_paths", type=str, nargs="+", default=None,
                        help="Input path to one or multiple image channels in ome.zarr format.")
    parser.add_argument("-t", "--seg_table", type=str, default=None,
                        help="Input path to segmentation table.")
    parser.add_argument("-s", "--seg_path", type=str, default=None,
                        help="Input path to segmentation channel in ome.zarr format.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")
    parser.add_argument("--mobie_dir", type=str, default=MOBIE_FOLDER,
                        help="Directory containing MoBIE project. Only used for '--json_info'.")

    # options for object measures
    parser.add_argument("--json_info", type=str, default=None,
                        help="JSON file with parameters for object_measures.")
    parser.add_argument("-c", "--components", type=int, nargs="+", default=[1], help="List of components.")
    parser.add_argument("-r", "--resolution", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Resolution of input in micrometer.")
    parser.add_argument("--bg_mask", action="store_true", help="Use background mask for calculating object measures.")

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
        json_file=args.json_info,
        mobie_dir=args.mobie_dir,
        image_paths=args.image_paths,
        table_path=args.seg_table,
        seg_path=args.seg_path,
        force_overwrite=args.force,
        s3=args.s3,
    )


def tonotopic_mapping():
    parser = argparse.ArgumentParser(
        description="Script to extract region of interest (ROI) block around center coordinate.")

    parser.add_argument("-i", "--input", type=str, required=True, help="Input path to segmentation table.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for segmentation table. Default: Overwrite input table.")
    parser.add_argument("-f", "--force", action="store_true", help="Forcefully overwrite output.")

    # options for tonotopic mapping
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

    tonotopic_mapping_single(
        table_path=args.input,
        out_path=args.output,
        force_overwrite=args.force,
        animal=args.animal,
        otof=args.otof,
        apex_position=args.apex_position,
        cell_type=args.cell_type,
        component_list=args.components,
        s3=args.s3,
        s3_credentials=args.s3_credentials,
        s3_bucket_name=args.s3_bucket_name,
        s3_service_endpoint=args.s3_service_endpoint,
    )
