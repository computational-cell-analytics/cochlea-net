import unittest

import numpy as np
import pandas as pd


class TestRefineIhcRow(unittest.TestCase):

    def setUp(self):
        # 100 IHC positions, 9 µm apart on an arc with a radius of 300 µm. Cells 0-39 are component 1 and
        # cells 50-99 component 2. In the gap, cells 40-46 and 49 are missing, cell 47 is in no component
        # and cell 48 is small. Label 101 is a large object 40 µm outside the arc in component 1, and
        # labels 102-106 a component 200 µm away from the row.
        angles = np.arange(100) * 9.0 / 300.0
        present = [i for i in range(100) if not 40 <= i <= 46 and i != 49]
        rows = []
        for i in present:
            component = 1 if i < 40 else (2 if i >= 50 else 0)
            rows.append((i + 1, 300.0, angles[i], 500 if i == 48 else 20000, component))
        rows.append((101, 340.0, angles[20], 20000, 1))
        rows += [(102 + j, 100.0, angles[60] + 0.03 * j, 20000, 3) for j in range(5)]
        label_ids, radii, phis, sizes, components = map(np.array, zip(*rows))
        self.table = pd.DataFrame({
            "label_id": label_ids,
            "anchor_x": 500 + radii * np.cos(phis),
            "anchor_y": 500 + radii * np.sin(phis),
            "anchor_z": 100.0,
            "n_pixels": sizes,
            "component_labels": components,
        })
        self.expected_row = {i + 1 for i in present if i != 48}

    def test_refine_ihc_row(self):
        from flamingo_tools.postprocessing.label_components import refine_ihc_row

        table, path = refine_ihc_row(self.table.copy(), [1, 2], min_size=10000, max_path_deviation=15.0)
        self.assertEqual(set(table.loc[table["component_labels"] == 1, "label_id"]), self.expected_row)
        self.assertEqual(set(table["component_labels"]), {0, 1})
        self.assertEqual(path.shape[1], 3)

    def test_refine_ihc_row_too_few_instances(self):
        from flamingo_tools.postprocessing.label_components import refine_ihc_row

        with self.assertRaises(ValueError):
            refine_ihc_row(self.table.copy(), [1, 2], min_size=50000, max_path_deviation=15.0)

    def test_refine_row_requires_ihc(self):
        from flamingo_tools.postprocessing.label_components import label_components_single

        with self.assertRaises(ValueError):
            label_components_single("table.tsv", cell_type="sgn", refine_row=True)


if __name__ == "__main__":
    unittest.main()
