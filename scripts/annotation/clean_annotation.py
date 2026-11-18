"""Clean a corrected annotation as a post-processing step: stray pieces, holes and interleaved borders.

The cells that differ from the base (for example the S3 segmentation that the correction started from) are
cleaned with `cleanup.clean_labels`, the same step that the correction tool runs after each fill. All other
cells stay fixed. A separate piece with `--split_size` voxels or more gets a new id.

Usage:
    python clean_annotation.py -i corrected_annotations.tif --base s3_crop.tif -o cleaned_annotations.tif
"""
import argparse

import numpy as np
import tifffile

from flamingo_tools.annotation.cleanup import clean_labels


def clean_annotation(
    labels: np.ndarray, base: np.ndarray, voxel_size: float = 0.38, sigma: float = 1.0, band: float = 1.0,
    split_size: int = 1000,
):
    """The cleaned labels and the label that each new label of a split piece came from."""
    changed = labels != base
    editable = np.union1d(np.unique(labels[changed]), np.unique(base[changed]))
    editable = editable[editable > 0]
    next_id = [int(max(labels.max(), base.max())) + 1]

    def new_label():
        next_id[0] += 1
        return next_id[0] - 1

    return clean_labels(labels, base, editable.tolist(), voxel_size, sigma, band, split_size, new_label=new_label)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="The corrected annotation.")
    parser.add_argument("--base", required=True, help="The segmentation that the correction started from.")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--voxel_size", type=float, default=0.38)
    parser.add_argument("--sigma", type=float, default=1.0, help="Smoothing of the label masks in pixels.")
    parser.add_argument("--band", type=float, default=1.0, help="Smoothed band around the changes in um.")
    parser.add_argument("--split_size", type=int, default=1000)
    args = parser.parse_args()
    labels, base = tifffile.imread(args.input), tifffile.imread(args.base)
    cleaned, splits = clean_annotation(labels, base, args.voxel_size, args.sigma, args.band, args.split_size)
    for new_id, label_id in splits.items():
        print(f"A separate piece of label {label_id} became the label {new_id}.")
    print(f"{int((cleaned != labels).sum())} voxels changed.")
    tifffile.imwrite(args.output, cleaned.astype(labels.dtype), compression="zlib")


if __name__ == "__main__":
    main()
