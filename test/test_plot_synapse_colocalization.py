import importlib.util
import os
import sys
import unittest

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(REPO_ROOT, "scripts", "figures")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in ("matplotlib", "sklearn")),
    "the figure scripts need matplotlib and scikit-learn",
)
class TestPlotSynapseColocalization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, FIGURE_DIR)
        try:
            import plot_synapse_colocalization
        finally:
            sys.path.remove(FIGURE_DIR)
        cls.module = plot_synapse_colocalization

    def test_colocalization_table(self):
        # IHC 3 has no synapse of any stain. It counts as 0 for the per-IHC values.
        ihc_counts = pd.DataFrame({
            "label_id": [1, 2, 3], "n_reference": [4, 2, 0], "n_test": [6, 1, 0], "n_matched": [3, 0, 0],
        })
        colocalization = {"M_AMD_000117_L": ({"max_dist": 2.0}, ihc_counts)}
        table = self.module.colocalization_table(colocalization).set_index("stain")

        self.assertEqual(table.index.tolist(), ["CtBP2", "Homer1", "Co-localized pairs"])
        self.assertEqual(table["n_ihcs"].tolist(), [3, 3, 3])
        self.assertEqual(table["n_ihcs_with_synapses"].tolist(), [2, 2, 1])
        self.assertEqual(table["n_synapses"].tolist(), [6, 7, 3])
        np.testing.assert_allclose(table["mean_per_ihc"], [2.0, 7 / 3, 1.0])
        np.testing.assert_allclose(table["sd_per_ihc"], [2.0, np.std([6, 1, 0], ddof=1), np.std([3, 0, 0], ddof=1)])
        self.assertEqual(table["median_per_ihc"].tolist(), [2.0, 1.0, 0.0])
        np.testing.assert_allclose(table["colocalized_share"].iloc[:2], [0.5, 3 / 7])
        self.assertTrue(np.isnan(table.loc["Co-localized pairs", "colocalized_share"]))

    def test_row_ihcs(self):
        # 100 IHC positions, 9 um apart on an arc with a radius of 300 um. Cells 41-47 are missing, so the
        # labeling gives component 1 = labels 48-100 and component 2 = labels 1-40. Label 21 has 2000 pixels:
        # the label_components default min_size of 1000 keeps it, the refine_ihc_row default of 10000 would not.
        present = np.array([i for i in range(100) if not 40 <= i <= 46])
        angles = present * 9.0 / 300.0
        table = pd.DataFrame({
            "label_id": present + 1,
            "anchor_x": 500 + 300 * np.cos(angles), "anchor_y": 500 + 300 * np.sin(angles), "anchor_z": 100.0,
            "n_pixels": np.where(present == 20, 2000, 20000),
        })
        plain = {"cell_type": "ihc", "component_list": [1], "min_component_length": 5, "max_edge_distance": 30}
        np.testing.assert_array_equal(np.sort(self.module.row_ihcs(table, plain)), np.arange(48, 101))

        refined = {**plain, "component_list_path": [2, 1], "refine_row": True, "merge_components": True}
        np.testing.assert_array_equal(np.sort(self.module.row_ihcs(table, refined)), present + 1)


if __name__ == "__main__":
    unittest.main()
