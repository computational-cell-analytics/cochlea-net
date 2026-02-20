import argparse

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.segmentation.sgn_detection import sgn_detection


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Path to image data to be segmented.")
    parser.add_argument("-o", "--output_folder", required=True, help="Path to output folder.")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to SGN detection model.")
    parser.add_argument("-k", "--input_key", default=None,
                        help="The key / internal path to image data.")

    parser.add_argument("-d", "--extension_distance", type=float, default=12, help="Extension distance.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[3.0, 1.887779, 1.887779],
                        help="Voxel size of input in micrometer. Default: [3.0, 1.887779, 1.887779]")

    parser.add_argument("--s3", action="store_true", help="Use S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    block_shape = (12, 128, 128)
    halo = (10, 64, 64)

    if len(args.voxel_size) == 1:
        voxel_size = tuple(args.voxel_size, args.voxel_size, args.voxel_size)
    else:
        voxel_size = tuple(args.voxel_size)

    if args.s3:
        input_path, fs = s3_utils.get_s3_path(args.input, bucket_name=args.s3_bucket_name,
                                              service_endpoint=args.s3_service_endpoint,
                                              credential_file=args.s3_credentials)

    else:
        input_path = args.input

    sgn_detection(input_path=input_path, input_key=args.input_key, output_folder=args.output_folder,
                  model_path=args.model, block_shape=block_shape, halo=halo,
                  extension_distance=args.extension_distance, sampling=voxel_size,
                  min_distance=2, threshold_abs=0.4)


if __name__ == "__main__":
    main()
