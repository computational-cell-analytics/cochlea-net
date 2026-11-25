import unittest

import numpy as np
import pandas as pd


class TestApplyMarkerLabels(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data.marker import apply_marker_labels
        self.fn = apply_marker_labels
        # label_id 5 is not in either list, so it must be dropped.
        self.segmentation = np.array([[[0, 1, 2, 3, 4, 5]]], dtype="float32")

    def test_class_values_are_not_confused_with_label_ids(self):
        # Regression: label_ids 1 and 2 collide with the class values written for positive and
        # negative. Both masks must come from the original label IDs.
        result = self.fn(self.segmentation.copy(), label_ids_positive=[3, 4], label_ids_negative=[1, 2])
        np.testing.assert_array_equal(result.ravel(), [0, 2, 2, 1, 1, 0])

    def test_positive_ids_containing_the_negative_class_value(self):
        result = self.fn(self.segmentation.copy(), label_ids_positive=[1, 2], label_ids_negative=[3, 4])
        np.testing.assert_array_equal(result.ravel(), [0, 1, 1, 2, 2, 0])

    def test_group_positive_drops_negatives(self):
        result = self.fn(self.segmentation.copy(), label_ids_positive=[3, 4], label_ids_negative=[1, 2],
                         group="positive")
        np.testing.assert_array_equal(result.ravel(), [0, 0, 0, 1, 1, 0])

    def test_group_negative_drops_positives(self):
        result = self.fn(self.segmentation.copy(), label_ids_positive=[3, 4], label_ids_negative=[1, 2],
                         group="negative")
        np.testing.assert_array_equal(result.ravel(), [0, 2, 2, 0, 0, 0])

    def test_empty_id_lists_clear_the_volume(self):
        result = self.fn(self.segmentation.copy(), label_ids_positive=[], label_ids_negative=[])
        self.assertEqual(result.sum(), 0)

    def test_invalid_group_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.segmentation.copy(), label_ids_positive=[1], label_ids_negative=[2], group="other")

    def test_output_is_float32(self):
        result = self.fn(self.segmentation.copy().astype("uint16"), label_ids_positive=[3], label_ids_negative=[1])
        self.assertEqual(result.dtype, np.dtype("float32"))


class TestMarkerLabelIds(unittest.TestCase):

    def test_ids_per_class(self):
        from flamingo_tools.export_data.marker import marker_label_ids

        table = pd.DataFrame({"label_id": [1, 2, 3, 4], "marker_labels": [2, 1, 0, 1]})
        self.assertEqual(marker_label_ids(table), ([2, 4], [1]))

    def test_rejects_table_without_marker_labels(self):
        from flamingo_tools.export_data.marker import marker_label_ids

        with self.assertRaisesRegex(ValueError, "missing required columns"):
            marker_label_ids(pd.DataFrame({"label_id": [1]}))


if __name__ == "__main__":
    unittest.main()
