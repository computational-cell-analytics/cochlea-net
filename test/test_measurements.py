import os
import unittest
from shutil import rmtree

import imageio.v3 as imageio
import pandas as pd
import numpy as np
from skimage.measure import regionprops_table


class TestMeasurements(unittest.TestCase):
    folder = "./tmp"

    def setUp(self):
        from flamingo_tools.test_data import get_test_volume_and_segmentation

        self.image_path, self.seg_path, self.table_path = get_test_volume_and_segmentation(self.folder)

    def tearDown(self):
        try:
            rmtree(self.folder)
        except Exception:
            pass

    def test_compute_object_measures(self):
        from flamingo_tools.measurements import compute_object_measures

        output_path = os.path.join(self.folder, "measurements.tsv")
        compute_object_measures(
            self.image_path, self.seg_path, self.table_path, output_path, n_threads=1
        )
        self.assertTrue(os.path.exists(output_path))

        table = pd.read_csv(output_path, sep="\t")
        self.assertTrue(len(table) >= 1)
        expected_columns = ["label_id", "mean", "stdev", "min", "max", "median"]
        expected_columns.extend([f"percentile-{p}" for p in (5, 10, 25, 75, 90, 95)])
        expected_columns.extend(["volume", "surface"])
        for col in expected_columns:
            self.assertIn(col, table.columns)

        n_objects = int(imageio.imread(self.seg_path).max())
        expected_shape = (n_objects, len(expected_columns))
        self.assertEqual(table.shape, expected_shape)

        image = imageio.imread(self.image_path)
        segmentation = imageio.imread(self.seg_path)
        properties = ("label", "intensity_mean", "intensity_std", "intensity_min", "intensity_max")
        expected_measures = regionprops_table(segmentation, intensity_image=image, properties=properties)
        expected_measures = pd.DataFrame(expected_measures)

        for (col, col_exp) in [
            ("label_id", "label"), ("mean", "intensity_mean"), ("stdev", "intensity_std"),
            ("min", "intensity_min"), ("max", "intensity_max"),
        ]:
            self.assertTrue(np.allclose(table[col].values, expected_measures[col_exp].values))

    # Test the object measurement functionality as it's used for the gfp intensity measurements:
    # - computing only median intensity
    # - with a dilation of 4
    # - with background subtraction
    # - and using a mask for the background subtraction
    def test_compute_object_measures_gfp(self):
        from flamingo_tools.measurements import compute_object_measures, compute_sgn_background_mask

        dilation = 4
        background_mask = compute_sgn_background_mask(self.image_path, self.seg_path, scale_factor=(2, 4, 4))

        output_path = os.path.join(self.folder, "measurements.tsv")
        compute_object_measures(
            self.image_path, self.seg_path, self.table_path, output_path, n_threads=1,
            dilation=dilation, median_only=True, feature_set="default_background_subtract",
            background_mask=background_mask,
        )
        self.assertTrue(os.path.exists(output_path))

        table = pd.read_csv(output_path, sep="\t")
        self.assertTrue(len(table) >= 1)
        expected_columns = ["label_id", "median"]
        for col in expected_columns:
            self.assertIn(col, table.columns)

    def test_normalize_background(self):
        """Every intensity measure drops by the background level, and the spread stays unchanged."""
        from flamingo_tools.measurements import _default_object_features, BACKGROUND_NORMALIZED_MEASURES

        background, voxel_size = 100, (0.38, 0.38, 0.38)
        rng = np.random.default_rng(0)
        shape = (80, 80, 80)
        bb = (slice(38, 44),) * 3

        image = np.full(shape, background, dtype="uint16")
        image[bb] = rng.integers(150, 400, size=(6, 6, 6)).astype("uint16")
        segmentation = np.zeros(shape, dtype="uint32")
        segmentation[bb] = 1
        table = pd.DataFrame({
            "label_id": [1],
            "bb_min_x": [38 * 0.38], "bb_min_y": [38 * 0.38], "bb_min_z": [38 * 0.38],
            "bb_max_x": [44 * 0.38], "bb_max_y": [44 * 0.38], "bb_max_z": [44 * 0.38],
            "anchor_x": [41 * 0.38], "anchor_y": [41 * 0.38], "anchor_z": [41 * 0.38],
        })

        raw = _default_object_features(1, table, image, segmentation, voxel_size=voxel_size)
        subtracted = _default_object_features(
            1, table, image, segmentation, voxel_size=voxel_size,
            background_radius=0.38 * 20, norm=np.subtract,
        )

        for measure in BACKGROUND_NORMALIZED_MEASURES:
            self.assertAlmostEqual(raw[measure] - subtracted[measure], background, places=6)
        # A constant offset does not change a spread.
        self.assertAlmostEqual(raw["stdev"], subtracted["stdev"], places=9)

        percentiles = [
            subtracted[name] for name in
            ("percentile-5", "percentile-10", "percentile-25", "median",
             "percentile-75", "percentile-90", "percentile-95")
        ]
        self.assertEqual(percentiles, sorted(percentiles))

        # An object darker than its background gives a negative value, not an unsigned wraparound.
        dark = np.full(shape, background, dtype="uint16")
        dark[bb] = 40
        measures = _default_object_features(
            1, table, dark, segmentation, voxel_size=voxel_size,
            background_radius=0.38 * 20, norm=np.subtract,
        )
        self.assertAlmostEqual(measures["median"], 40 - background, places=6)


if __name__ == "__main__":
    unittest.main()
