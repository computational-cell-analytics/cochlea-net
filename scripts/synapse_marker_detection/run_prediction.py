import argparse

import flamingo_tools.s3_utils as s3_utils
from flamingo_tools.segmentation.synapse_detection import run_prediction


def main():

    parser = argparse.ArgumentParser(
        description="Run synapse prediction and peak detection. "
        "An optional IHC segmentation restricts the inference to the region around the IHCs. "
        "The detections are not matched to the IHCs here, use marker_detection.py for that."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to image data to be segmented.")
    parser.add_argument("-o", "--output_folder", required=True, help="Path to output folder.")
    parser.add_argument("-m", "--model", required=True,
                        help="Path to synapse detection model.")
    parser.add_argument("-k", "--input_key", default=None,
                        help="The key / internal path to image data.")
    parser.add_argument("-s", "--mask", default=None,
                        help="Path to an IHC segmentation used to restrict the inference to the region "
                        "around the IHCs. By default the prediction runs on the full volume.")
    parser.add_argument("--mask_input_key", default="s4",
                        help="The key to the downscaled IHC segmentation used for the mask.")
    parser.add_argument("-v", "--voxel_size", type=float, nargs="+", default=[0.38, 0.38, 0.38],
                        help="Voxel size of input in micrometer. Default: 0.38 0.38 0.38")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for peak detection.")

    parser.add_argument("--s3_input", action="store_true", help="Read the image data from the S3 bucket.")
    parser.add_argument("--s3_mask", action="store_true", help="Read the IHC segmentation from the S3 bucket.")
    parser.add_argument("--s3_credentials", type=str, default=None,
                        help="Input file containing S3 credentials. "
                        "Optional if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY were exported.")
    parser.add_argument("--s3_bucket_name", type=str, default=None,
                        help="S3 bucket name. Optional if BUCKET_NAME was exported.")
    parser.add_argument("--s3_service_endpoint", type=str, default=None,
                        help="S3 service endpoint. Optional if SERVICE_ENDPOINT was exported.")

    args = parser.parse_args()

    def resolve(path, from_s3):
        if path is None or not from_s3:
            return path
        s3_path, _ = s3_utils.get_s3_path(path, bucket_name=args.s3_bucket_name,
                                          service_endpoint=args.s3_service_endpoint,
                                          credential_file=args.s3_credentials)
        return s3_path

    run_prediction(
        input_path=resolve(args.input, args.s3_input), input_key=args.input_key,
        output_folder=args.output_folder, model_path=args.model,
        mask_path=resolve(args.mask, args.s3_mask), mask_input_key=args.mask_input_key,
        voxel_size=args.voxel_size, threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
