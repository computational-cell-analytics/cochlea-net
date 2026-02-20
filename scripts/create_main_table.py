import argparse
import json

from flamingo_tools.analysis.seg_table_utils import create_main_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Input path to segmentation table.")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path to main table.")
    parser.add_argument("-m", "--meas_tables", type=str, nargs="+", default=None,
                        help="File path(s) to tables with object measures.")
    parser.add_argument("-j", "--json_info", type=str, default=None,
                        help="Optional dictionary for component_list and cell_type information.")
    parser.add_argument("-c", "--component_list", type=int, nargs="+", default=[1],
                        help="List of connected components.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer.")
    parser.add_argument("--cell_type", type=str, default="sgn",
                        help="Cell type of segmentation. Either 'sgn' or 'ihc'. Default: sgn")
    parser.add_argument("--intensity_keyword", type=str, default="median",
                        help="Keyword for intensity information of object measures. Default: median")

    # options for data access on S3 bucket
    parser.add_argument("--s3_meas", action="store_true", help="Object measurement table path(s) on S3 bucket.")
    parser.add_argument("--s3_seg", action="store_true", help="Segmentation table path on S3 bucket.")

    args = parser.parse_args()
    args_dict = vars(args)

    if args.json_info is not None:
        with open(args.json_info, "r") as f:
            dic = json.loads(f.read())
        args_dict.update(dic)

    create_main_table(
        input_path=args_dict["input"],
        output_path=args_dict["output"],
        component_list=args_dict["component_list"],
        cell_type=args_dict["cell_type"],
        voxel_size=args_dict["voxel_size"],
        meas_tables=args_dict["meas_tables"],
        s3_seg=args_dict["s3_meas"],
        s3_meas=args_dict["s3_seg"],
        intensity_keyword=args_dict["intensity_keyword"],
    )


if __name__ == "__main__":
    main()
