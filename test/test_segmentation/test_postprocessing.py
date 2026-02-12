import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from elf.io import open_file
from skimage.data import binary_blobs
from skimage.measure import label, regionprops_table


class TestPostprocessing(unittest.TestCase):
    def _create_example_seg(self, tmp_dir):
        seg = binary_blobs(256, n_dim=3, volume_fraction=0.2)
        seg = label(seg)
        return seg

    def _test_postprocessing(self, spatial_statistics, threshold, **spatial_statistics_kwargs):
        from flamingo_tools.postprocessing.label_components import filter_segmentation

        with tempfile.TemporaryDirectory() as tmp_dir:
            example_seg = self._create_example_seg(tmp_dir)
            output_path = os.path.join(tmp_dir, "test-output.zarr")
            output_key = "seg-filtered"
            filter_segmentation(
                example_seg, output_path, spatial_statistics, threshold,
                output_key=output_key, **spatial_statistics_kwargs
            )
            self.assertTrue(os.path.exists(output_path))
            with open_file(output_path, "r") as f:
                filtered_seg = f[output_key][:]
            self.assertEqual(filtered_seg.shape, example_seg.shape)

    def test_nearest_neighbor_distance(self):
        from flamingo_tools.postprocessing.label_components import nearest_neighbor_distance

        self._test_postprocessing(nearest_neighbor_distance, threshold=5)

    def test_local_ripleys_k(self):
        from flamingo_tools.postprocessing.label_components import local_ripleys_k

        self._test_postprocessing(local_ripleys_k, threshold=0.5)

    def test_neighbors_in_radius(self):
        from flamingo_tools.postprocessing.label_components import neighbors_in_radius

        self._test_postprocessing(neighbors_in_radius, threshold=5)

    def test_compute_table_on_the_fly(self):
        from flamingo_tools.postprocessing.label_components import compute_table_on_the_fly
        from flamingo_tools.test_data import get_test_volume_and_segmentation

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, seg_path, _ = get_test_volume_and_segmentation(tmp_dir)
            segmentation = imageio.imread(seg_path)

        resolution = 0.38
        table = compute_table_on_the_fly(segmentation, resolution=resolution)

        properties = ("label", "bbox", "centroid")
        expected_table = regionprops_table(segmentation, properties=properties)
        expected_table = pd.DataFrame(expected_table)

        for (col, col_exp) in [
            ("label_id", "label"),
            ("anchor_x", "centroid-2"), ("anchor_y", "centroid-1"), ("anchor_z", "centroid-0"),
            ("bb_min_x", "bbox-2"), ("bb_min_y", "bbox-1"), ("bb_min_z", "bbox-0"),
            ("bb_max_x", "bbox-5"), ("bb_max_y", "bbox-4"), ("bb_max_z", "bbox-3"),
        ]:
            values = table[col].values.copy()
            if col != "label_id":
                values /= resolution
            self.assertTrue(np.allclose(values, expected_table[col_exp].values))


if __name__ == "__main__":
    unittest.main()
