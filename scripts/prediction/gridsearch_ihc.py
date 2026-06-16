import argparse
import json
import os

from flamingo_tools.segmentation.gridsearch import gridsearch, DEFAULT_GRID_SEARCH_VALUES


def main():
    parser = argparse.ArgumentParser(
        description="Find watershed parameters that maximise symmetric best-Dice on a validation set for IHCs. "
                    "The parameters 'center_distance_threshold', 'boundary_distance_threshold', "
                    "and 'distance_smoothing' are optimized during the process."
    )
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-m", "--model", required=True)
    parser.add_argument("-o", "--output_folder", type=str, default=None)
    parser.add_argument("-b", "--block_shape", default=None, type=str)
    parser.add_argument("--halo", default=None, type=str)
    parser.add_argument("--center_distance_threshold", nargs="+", default=None, type=float,
                        help="The threshold applied to the distance center predictions to derive seeds.")
    parser.add_argument("--boundary_distance_threshold", nargs="+", default=None, type=float,
                        help="The threshold applied to the boundary predictions to derive seeds.")
    parser.add_argument("--distance_smoothing", default=None, type=float,
                        help="The sigma value for smoothing the distance predictions with a gaussian kernel.")
    parser.add_argument("--fg_threshold", default=0.5, type=float,
                        help="The threshold applied to the foreground prediction for deriving the watershed mask.")

    args = parser.parse_args()

    grid_search_values = DEFAULT_GRID_SEARCH_VALUES

    if args.center_distance_threshold is not None:
        grid_search_values["center_distance_threshold"] = args.center_distance_threshold

    if args.boundary_distance_threshold is not None:
        grid_search_values["boundary_distance_threshold"] = args.boundary_distance_threshold

    if args.distance_smoothing is not None:
        grid_search_values["distance_smoothing"] = args.distance_smoothing

    cache_path = os.path.splitext(args.model)[0] + "_best_params.json"

    if os.path.exists(cache_path):
        with open(cache_path) as fh:
            data = json.load(fh)
        print(f"Loaded cached best params from {cache_path}")
        return data["params"], data["score"]

    best_params, best_score = gridsearch(
        val_dir=args.input,
        model_path=args.model,
        result_dir=args.output_folder,
        grid_search_values=grid_search_values,
        block_shape=args.block_shape,
        halo=args.halo,
        fg_threshold=args.fg_threshold,
    )

    with open(cache_path, "w") as fh:
        json.dump({"params": best_params, "score": best_score}, fh, indent=2)
    print(f"Best params saved to {cache_path}")


if __name__ == "__main__":
    main()
