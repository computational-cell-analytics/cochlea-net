import contextlib
import importlib.util
import io
import os
import sys
import unittest
from unittest import mock

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
            import data
            import plot_mwfls
        finally:
            sys.path.remove(FIGURE_DIR)
        cls.data = data
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

    def test_counted_synapses(self):
        synapses = pd.DataFrame({
            "spot_id": [1, 2, 3, 4, 5], "matched_ihc": [1, 1, 2, 3, 2], "distance_to_ihc": [0.5, 3.0, 3.5, 1.0, 2.0],
        })
        counted = self.data.counted_synapses(synapses, ihc_ids=[1, 2])
        # Spot 3 is too far from its IHC, and spot 4 belongs to an IHC that is not in the table.
        self.assertEqual(counted["spot_id"].tolist(), [1, 2, 5])
        ihcs = self.data.add_synapses_per_ihc(pd.DataFrame({"label_id": [1, 2, 7]}), counted)
        self.assertEqual(ihcs["syn_per_IHC"].tolist(), [2, 1, 0])

    def test_get_synapse_table(self):
        synapses = pd.DataFrame({"spot_id": [1, 2, 3], "matched_ihc": [1, 1, 1], "distance_to_ihc": [1.0, 1.0, 1.0]})
        measures = pd.DataFrame({"spot_id": [1, 2]})
        for measure in self.data.SPOT_MEASURES:
            measures[f"CTBP2_{measure}"] = [10.0, 20.0]
        # Spot 2 has a signal, but no SNR. Spot 3 has no row in the spot-intensity table.
        measures.loc[1, "CTBP2_snr"] = np.nan
        tables = {"default.tsv": synapses, "CTBP2_synapse-v3-ihc-v11_spot-intensity.tsv": measures}

        def read_source_table(s3, cochlea, source_name, table_name="default.tsv"):
            return tables[table_name]

        with mock.patch.object(self.data, "read_source_table", side_effect=read_source_table), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = self.data.get_synapse_table(None, "M_test", "synapse_v3_ihc_v11", ihc_ids=[1], stain="CTBP2")
        self.assertEqual(result["spot_id"].tolist(), [1, 2, 3])
        self.assertEqual(result["signal"].iloc[:2].tolist(), [10.0, 20.0])
        self.assertTrue(np.isnan(result["signal"].iloc[2]))
        self.assertTrue(np.isnan(result["snr"].iloc[1]))
        # Only spot 3 has no signal.
        self.assertIn("1 / 3", output.getvalue())

    def test_cohort_colors_are_panel_colors(self):
        # util.COHORT_DICT holds the hex values, because util does not import panel_style.
        plot_mwfls = self.plot_mwfls
        self.assertEqual(plot_mwfls.COHORT_DICT["mwfls"]["color"], plot_mwfls.COLORS["blue"])
        self.assertEqual(plot_mwfls.COHORT_DICT["idisco"]["color"], plot_mwfls.COLORS["green"])


if __name__ == "__main__":
    unittest.main()
