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
class TestPlotMwfls(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, FIGURE_DIR)
        try:
            import plot_mwfls
        finally:
            sys.path.remove(FIGURE_DIR)
        cls.plot_mwfls = plot_mwfls

    def test_cohort_values(self):
        plot_mwfls = self.plot_mwfls
        cochleae = plot_mwfls.cohort_cochleae("mwfls")
        outlier, = plot_mwfls.OUTLIER_DICT["synapse"]
        tables = {
            "SGN": {name: pd.DataFrame({"label_id": np.arange(100 + i)}) for i, name in enumerate(cochleae)},
            "IHC": {name: pd.DataFrame({"syn_per_IHC": [i, i + 2]}) for i, name in enumerate(cochleae)},
        }
        counts = [100 + i for i in range(len(cochleae))]
        self.assertEqual(plot_mwfls.cohort_values(tables, "SGN", "mwfls"), (counts, []))
        self.assertEqual(plot_mwfls.cohort_values(tables, "IHC", "mwfls"), ([2] * len(cochleae), []))

        means = [i + 1.0 for i in range(len(cochleae))]
        self.assertEqual(plot_mwfls.cohort_values(tables, "synapse", "mwfls"), (means, []))
        values, outliers = plot_mwfls.cohort_values(tables, "synapse", "mwfls", remove_outliers=True)
        outlier_mean = means[cochleae.index(outlier)]
        self.assertEqual(outliers, [outlier_mean])
        self.assertEqual(values, [mean for mean in means if mean != outlier_mean])

    def test_cohort_colors_are_panel_colors(self):
        # util.COHORT_DICT holds the hex values, because util does not import panel_style.
        plot_mwfls = self.plot_mwfls
        self.assertEqual(plot_mwfls.COHORT_DICT["mwfls"]["color"], plot_mwfls.COLORS["blue"])
        self.assertEqual(plot_mwfls.COHORT_DICT["idisco"]["color"], plot_mwfls.COLORS["green"])


if __name__ == "__main__":
    unittest.main()
