import argparse
import json
import os
from concurrent import futures

import numpy as np
import zarr
from elf.evaluation.matching import label_overlap, intersection_over_union
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target, get_s3_path
from nifty.tools import blocking
from tqdm import tqdm

COCHLEA_DIR = "/mnt/vast-nhr/projects/nim00007/data/moser/cochlea-lightsheet"


def merge_segmentations(seg_a, seg_b, ids_b, offset, output_path):
    assert seg_a.shape == seg_b.shape

    output_file = zarr.open(output_path, mode="a")
    output = output_file.create_dataset("segmentation", shape=seg_a.shape, dtype=seg_a.dtype, chunks=seg_a.chunks)
    blocks = blocking([0, 0, 0], seg_a.shape, seg_a.chunks)

    def merge_block(block_id):
        block = blocks.getBlock(block_id)
        bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

        block_a = seg_a[bb]
        block_b = seg_b[bb]

        # Only insert non-background Ntng1 cells into pixels that are background
        # in seg_a, to avoid overwriting CR cells (can occur at full resolution
        # even when there was no overlap at the coarser scale used for filtering).
        insert_mask = np.isin(block_b, ids_b) & (block_a == 0)
        if insert_mask.sum() > 0:
            block_b[insert_mask] += offset
            block_a[insert_mask] = block_b[insert_mask]

        output[bb] = block_a

    n_blocks = blocks.numberOfBlocks
    with futures.ThreadPoolExecutor(12) as tp:
        list(tqdm(tp.map(merge_block, range(n_blocks)), total=n_blocks, desc="Merge segmentation"))


def get_segmentation(cochlea, seg_name, seg_key):
    print("Loading segmentation ...")
    s3 = create_s3_target()

    content = s3.open(f"{BUCKET_NAME}/{cochlea}/dataset.json", mode="r", encoding="utf-8")
    info = json.loads(content.read())
    sources = info["sources"]

    seg_source = sources[seg_name]
    seg_path = os.path.join(cochlea, seg_source["segmentation"]["imageData"]["ome.zarr"]["relativePath"])
    seg_store, _ = get_s3_path(seg_path)

    return zarr.open(seg_store, mode="r")[seg_key]


def merge_sgns(cochlea, name_a, name_b, overlap_threshold=0.25, output_folder=None):
    # Get the two segmentations at low resolution for computing the overlaps.
    seg_a = get_segmentation(cochlea, seg_name=name_a, seg_key="s2")[:]
    seg_b = get_segmentation(cochlea, seg_name=name_b, seg_key="s2")[:]

    # Compute the overlaps and determine which SGNs to add from SegB based on the overlap threshold.
    print("Compute label overlaps ...")
    overlap, ignore_label = label_overlap(seg_a, seg_b)
    overlap = intersection_over_union(overlap)
    cumulative_overlap = overlap[1:, :].sum(axis=0)
    all_ids_b = np.unique(seg_b)
    all_ids_b = all_ids_b[all_ids_b != 0]  # exclude background before threshold check
    ids_b = all_ids_b[cumulative_overlap[all_ids_b] < overlap_threshold]
    offset = seg_a.max()

    # Get the segmentations at full resolution to merge them.
    seg_a = get_segmentation(cochlea, seg_name=name_a, seg_key="s0")
    seg_b = get_segmentation(cochlea, seg_name=name_b, seg_key="s0")

    # Write out the merged segmentations.
    if output_folder is None:
        output_folder = f"./data/{cochlea}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "SGN_merged.zarr")
    merge_segmentations(seg_a, seg_b, ids_b, offset, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Merge SGN segmentation of two different stains.")

    parser.add_argument("-c", "--cochlea", type=str, required=True, help="Input cochlea, e.g. M_AMD_N180_L.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Output path for merged segemntation.")
    parser.add_argument("--stain_a", type=str, default="CR",
                        help="First stain for merging.")
    parser.add_argument("--stain_b", type=str, default="Ntng1",
                        help="Second stain for merging.")
    parser.add_argument("--sgn_version", type=str, default="SGN_v2",
                        help="SGN segmentation version.")

    args = parser.parse_args()

    # merge_sgns(cochlea="M_AMD_N180_L", name_a="CR_SGN_v2", name_b="Ntng1_SGN_v2")
    # merge_sgns(cochlea="M_AMD_N180_R", name_a="CR_SGN_v2", name_b="Ntng1_SGN_v2")
    cochlea = args.cochlea
    stain_a = args.stain_a
    stain_b = args.stain_b
    sgn_version = args.sgn_version
    if args.output is None:
        output_folder = os.path.join(COCHLEA_DIR, f"predictions/{cochlea}/{stain_a}_{stain_b}_{sgn_version}")
    else:
        output_folder = args.output
    merge_sgns(cochlea=cochlea, name_a=f"{stain_a}_{sgn_version}", name_b=f"{stain_b}_{sgn_version}",
               output_folder=output_folder)


if __name__ == "__main__":
    main()
