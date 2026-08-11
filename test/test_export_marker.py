import os
import sys
import unittest

import numpy as np

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "export_data")
sys.path.insert(0, SCRIPTS_DIR)


class TestApplyMarkerLabels(unittest.TestCase):

    def setUp(self):
        import export_lower_resolution_marker
        self.fn = export_lower_resolution_marker.apply_marker_labels
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


if __name__ == "__main__":
    unittest.main()
