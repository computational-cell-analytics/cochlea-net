import json
import os
from concurrent import futures

import numpy as np
import zarr
import z5py

from elf.evaluation.matching import label_overlap, intersection_over_union
from flamingo_tools.s3_utils import BUCKET_NAME, create_s3_target, get_s3_path
from nifty.tools import blocking
from tqdm import tqdm


def merge_segmentations(seg_a, seg_b, ids_b, offset, output_path):
    assert seg_a.shape == seg_b.shape

    output_file = z5py.File(output_path, mode="a")
    output = output_file.create_dataset("segmentation", shape=seg_a.shape, dtype=seg_a.dtype, chunks=seg_a.chunks)
    blocks = blocking([0, 0, 0], seg_a.shape, seg_a.chunks)

    def merge_block(block_id):
        block = blocks.getBlock(block_id)
        bb = tuple(slice(begin, end) for begin, end in zip(block.begin, block.end))

        block_a = seg_a[bb]
        block_b = seg_b[bb]

        insert_mask = np.isin(block_b, ids_b)
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


def merge_sgns(cochlea, name_a, name_b, overlap_threshold=0.25):
    # Get the two segmentations at low resolution for computing the overlaps.
    seg_a = get_segmentation(cochlea, seg_name=name_a, seg_key="s2")[:]
    seg_b = get_segmentation(cochlea, seg_name=name_b, seg_key="s2")[:]

    # Compute the overlaps and determine which SGNs to add from SegB based on the overlap threshold.
    print("Compute label overlaps ...")
    overlap, ignore_label = label_overlap(seg_a, seg_b)
    overlap = intersection_over_union(overlap)
    cumulative_overlap = overlap[1:, :].sum(axis=0)
    all_ids_b = np.unique(seg_b)
    ids_b = all_ids_b[cumulative_overlap < overlap_threshold]
    if 0 in ids_b:  # Zero is likely in the ids due to the logic.
        ids_b = ids_b[1:]
    assert 0 not in ids_b
    offset = seg_a.max()

    # Get the segmentations at full resolution to merge them.
    seg_a = get_segmentation(cochlea, seg_name=name_a, seg_key="s0")
    seg_b = get_segmentation(cochlea, seg_name=name_b, seg_key="s0")

    # Write out the merged segmentations.
    output_folder = f"./data/{cochlea}"
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "SGN_merged.n5")
    merge_segmentations(seg_a, seg_b, ids_b, offset, output_path)


def main():
    merge_sgns(cochlea="M_AMD_N180_L", name_a="CR_SGN_v2", name_b="Ntng1_SGN_v2")
    merge_sgns(cochlea="M_AMD_N180_R", name_a="CR_SGN_v2", name_b="Ntng1_SGN_v2")


if __name__ == "__main__":
    main()
