import os
import tempfile
import unittest

import pandas as pd


def _make_table():
    return pd.DataFrame({
        "label_id": [1, 2, 3, 4, 5],
        "anchor_x": [10.0, 20.0, 30.0, 40.0, 50.0],
        "anchor_y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "anchor_z": [100.0, 200.0, 300.0, 400.0, 500.0],
        "component_labels": [1, 1, 1, 2, 2],
        "length_fraction": [0.0, 0.25, 0.5, 0.75, 1.0],
        "frequency[kHz]": [80.0, 40.0, 20.0, 10.0, 5.0],
    })


class TestClosestRowToValue(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.analysis.seg_table_utils import closest_row_to_value
        self.fn = closest_row_to_value
        self.table = _make_table()

    def test_exact_match(self):
        row = self.fn(self.table, "length_fraction", 0.5)
        self.assertEqual(row["label_id"], 3)

    def test_nearest_neighbor(self):
        row = self.fn(self.table, "length_fraction", 0.6)
        self.assertEqual(row["label_id"], 3)

    def test_other_column(self):
        row = self.fn(self.table, "frequency[kHz]", 12.0)
        self.assertEqual(row["label_id"], 4)

    def test_missing_column_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.table, "not_a_column", 0.5)


class TestPrintTableInfo(unittest.TestCase):

    def setUp(self):
        self.table = _make_table()
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.table_path = os.path.join(self.tmp_dir.name, "default.tsv")
        self.table.to_csv(self.table_path, sep="\t", index=False)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_single_value(self):
        from flamingo_tools.analysis.seg_table_utils import print_table_info
        results = print_table_info(self.table_path, column="length_fraction", values=[0.5])
        self.assertEqual(results, [(3, (30.0, 3.0, 300.0))])

    def test_multiple_values(self):
        from flamingo_tools.analysis.seg_table_utils import print_table_info
        results = print_table_info(self.table_path, column="length_fraction", values=[0.0, 1.0])
        self.assertEqual(results, [(1, (10.0, 1.0, 100.0)), (5, (50.0, 5.0, 500.0))])

    def test_frequency_column(self):
        from flamingo_tools.analysis.seg_table_utils import print_table_info
        results = print_table_info(self.table_path, column="frequency[kHz]", values=[45.0])
        self.assertEqual(results, [(2, (20.0, 2.0, 200.0))])

    def test_component_filter(self):
        from flamingo_tools.analysis.seg_table_utils import print_table_info
        # Without filtering, label_id=5 (length_fraction=1.0) is closest to 0.9.
        results = print_table_info(self.table_path, column="length_fraction", values=[0.9])
        self.assertEqual(results[0][0], 5)
        # Restricting to component 1 excludes label_id 4 and 5.
        results = print_table_info(
            self.table_path, column="length_fraction", values=[0.9], component_list=[1]
        )
        self.assertEqual(results[0][0], 3)


if __name__ == "__main__":
    unittest.main()
