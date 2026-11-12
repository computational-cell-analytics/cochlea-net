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
class TestColocalizationTable(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, FIGURE_DIR)
        try:
            import plot_mwfls
        finally:
            sys.path.remove(FIGURE_DIR)
        cls.plot_mwfls = plot_mwfls

    def test_colocalization_table(self):
        # IHC 3 has no synapse of any stain. It counts as 0 for the per-IHC values.
        ihc_counts = pd.DataFrame({
            "label_id": [1, 2, 3], "n_reference": [4, 2, 0], "n_test": [6, 1, 0], "n_matched": [3, 0, 0],
        })
        colocalization = {"M_AMD_000117_L": ({"max_dist": 2.0}, ihc_counts)}
        table = self.plot_mwfls.colocalization_table(colocalization).set_index("stain")

        self.assertEqual(table.index.tolist(), ["CtBP2", "Homer1", "Co-localized pairs"])
        self.assertEqual(table["n_ihcs"].tolist(), [3, 3, 3])
        self.assertEqual(table["n_ihcs_with_synapses"].tolist(), [2, 2, 1])
        self.assertEqual(table["n_synapses"].tolist(), [6, 7, 3])
        np.testing.assert_allclose(table["mean_per_ihc"], [2.0, 7 / 3, 1.0])
        np.testing.assert_allclose(table["sd_per_ihc"], [2.0, np.std([6, 1, 0], ddof=1), np.std([3, 0, 0], ddof=1)])
        self.assertEqual(table["median_per_ihc"].tolist(), [2.0, 1.0, 0.0])
        np.testing.assert_allclose(table["colocalized_share"].iloc[:2], [0.5, 3 / 7])
        self.assertTrue(np.isnan(table.loc["Co-localized pairs", "colocalized_share"]))


if __name__ == "__main__":
    unittest.main()
