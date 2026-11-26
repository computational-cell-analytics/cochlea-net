import contextlib
import importlib.util
import io
import os
import sys
import tempfile
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

    def _tables(self, cohorts):
        """Synthetic tables for the cochleae of the cohorts. Cochlea i has 100 + i SGNs and 10 + i IHCs.

        The SGN intensity of the first SGN is NaN. Each IHC has i synapses.
        """
        tables = {"SGN": {}, "IHC": {}, "synapse": {}}
        names = [name for cohort in cohorts for name in self.plot_mwfls.cohort_cochleae(cohort)]
        for i, name in enumerate(names):
            sgn_intensity = np.arange(100.0 + i)
            sgn_intensity[0] = np.nan
            tables["SGN"][name] = pd.DataFrame({"label_id": np.arange(100 + i), "intensity": sgn_intensity})
            tables["IHC"][name] = pd.DataFrame({
                "label_id": np.arange(10 + i), "intensity": np.full(10 + i, 50.0 + i),
                "syn_per_IHC": np.full(10 + i, i),
            })
            n_synapses = (10 + i) * i
            tables["synapse"][name] = pd.DataFrame({
                "intensity": np.full(n_synapses, 2.0 * i), "snr": np.full(n_synapses, 3.0),
            })
        return names, tables

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

    def test_synapse_checks(self):
        plot_mwfls = self.plot_mwfls
        # Each spot measure of get_synapse_table has one check, with its own axis title.
        self.assertEqual(set(plot_mwfls.SYNAPSE_CHECKS), set(self.data.SPOT_MEASURES))
        self.assertIn(plot_mwfls.SYNAPSE_INTENSITY, plot_mwfls.SYNAPSE_CHECKS)
        label = plot_mwfls._intensity_label
        self.assertEqual(label("synapse"), "Mean CtBP2 signal - bg (a.u.)")
        self.assertEqual(label("synapse", "net_signal"), label("synapse"))
        self.assertEqual(label("synapse", "ratio"), "Mean CtBP2 signal / bg")
        self.assertEqual(label("SGN"), "Mean PV intensity (a.u.)")

    def test_intensity_check_summary(self):
        plot_mwfls = self.plot_mwfls
        names, tables = self._tables(plot_mwfls.COHORTS)
        sgn = plot_mwfls.intensity_check_summary(tables, "SGN")
        self.assertEqual(sgn["cochlea"].tolist(), names)
        self.assertEqual(sgn["count"].tolist(), [100 + i for i in range(len(names))])
        # The SGN without an intensity counts, but stays out of the mean.
        self.assertEqual(sgn["mean_intensity"].tolist(), [np.arange(1, 100 + i).mean() for i in range(len(names))])
        np.testing.assert_allclose(sgn["measured_share"], [(99 + i) / (100 + i) for i in range(len(names))])

        ihc = plot_mwfls.intensity_check_summary(tables, "IHC")
        self.assertEqual(ihc["count"].tolist(), [10 + i for i in range(len(names))])
        self.assertEqual(ihc["mean_intensity"].tolist(), [50.0 + i for i in range(len(names))])

        synapse = plot_mwfls.intensity_check_summary(tables, "synapse", column="snr")
        self.assertEqual(synapse["count"].tolist(), [float(i) for i in range(len(names))])
        # The first cochlea has no synapses.
        self.assertTrue(np.isnan(synapse["mean_intensity"].iloc[0]))
        self.assertEqual(synapse["mean_intensity"].iloc[1:].tolist(), [3.0] * (len(names) - 1))

    def test_plot_intensity_check(self):
        plot_mwfls = self.plot_mwfls
        cohorts = ["mwfls", "idisco", "chreef_mouse"]
        names, tables = self._tables(cohorts)
        with tempfile.TemporaryDirectory() as tmp:
            for structure in ("SGN", "IHC", "synapse"):
                save_path = os.path.join(tmp, f"{structure}_intensity_check")
                summary = plot_mwfls.plot_intensity_check(tables, structure, save_path, cohorts=cohorts)
                self.assertEqual(summary["cochlea"].tolist(), names)
                self.assertTrue(os.path.exists(f"{save_path}.pdf"))
                self.assertTrue(os.path.exists(f"{save_path}.png"))
            save_path = os.path.join(tmp, "synapse_intensity_check_snr")
            plot_mwfls.plot_intensity_check(tables, "synapse", save_path, column="snr", cohorts=cohorts)
            self.assertTrue(os.path.exists(f"{save_path}.png"))

            # A cohort without synapse values shows n = 0. The other cohorts stay in the panels.
            idisco = plot_mwfls.cohort_cochleae("idisco")
            synapses = tables["synapse"]
            partial = {**tables, "synapse": {n: t.iloc[:0] if n in idisco else t for n, t in synapses.items()}}
            summary = plot_mwfls.plot_intensity_check(partial, "synapse", os.path.join(tmp, "partial"), cohorts=cohorts)
            self.assertTrue(summary.loc[summary["cohort"] == "idisco", "mean_intensity"].isna().all())
            self.assertTrue(summary.loc[summary["cohort"] == "chreef_mouse", "mean_intensity"].notna().all())
            empty = {**tables, "synapse": {n: t.iloc[:0] for n, t in synapses.items()}}
            with self.assertRaises(ValueError):
                plot_mwfls.plot_intensity_check(empty, "synapse", os.path.join(tmp, "empty"), cohorts=cohorts)
        # The literature band depends on the animal, so the cohorts must not mix animals.
        _, tables = self._tables(["mwfls", "fchrimson_gerbil"])
        with self.assertRaises(ValueError):
            plot_mwfls.plot_intensity_check(tables, "SGN", "unused", cohorts=["mwfls", "fchrimson_gerbil"])

    def test_cohort_colors_are_panel_colors(self):
        # util.COHORT_DICT holds the hex values, because util does not import panel_style.
        plot_mwfls = self.plot_mwfls
        self.assertEqual(plot_mwfls.COHORT_DICT["mwfls"]["color"], plot_mwfls.COLORS["blue"])
        self.assertEqual(plot_mwfls.COHORT_DICT["idisco"]["color"], plot_mwfls.COLORS["green"])


if __name__ == "__main__":
    unittest.main()
