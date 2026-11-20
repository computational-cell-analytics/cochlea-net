import os
import tempfile
import unittest

import numpy as np
import pandas as pd


def _tables(seed, n_reference=6, n_test=10):
    rng = np.random.default_rng(seed)
    reference = pd.DataFrame({
        "spot_id": np.arange(1, n_reference + 1),
        "x": 10.0 * np.arange(n_reference), "y": 0.0, "z": 0.0,
        "matched_ihc": np.repeat([1, 2], [n_reference // 2, n_reference - n_reference // 2]),
    })
    # The first half of the reference spots has a test partner 0.5 um away.
    n_pairs = n_reference // 2
    test_x = np.concatenate([reference.x.values[:n_pairs] + 0.5, 1000.0 + 10.0 * np.arange(n_test - n_pairs)])
    test = pd.DataFrame({
        "spot_id": np.arange(101, 101 + n_test), "x": test_x, "y": 0.0, "z": 0.0,
        "matched_ihc": np.where(np.arange(n_test) < n_test // 2, 1, 2),
    })
    for table in (reference, test):
        for channel in ("CTBP2", "Homer1"):
            for measure in ("peak", "background", "ratio", "snr", "integrated", "n_saturated"):
                table[f"{channel}_{measure}"] = rng.uniform(1, 10, size=len(table))
    return reference, test


class TestStainSummary(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.synapse_detection.stain_comparison import compare_stains

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.folders = {}
        for seed, name in enumerate(("M_A_L", "M_B_R")):
            folder = os.path.join(self.tmpdir.name, name, "comparison")
            reference, test = _tables(seed)
            compare_stains(reference, test, "CTBP2", "Homer1", folder, reference_name="CTBP2_spots",
                           test_name="Homer1_spots", valid_ihcs=[1, 2], thresholds=[-np.inf])
            self.folders[name] = folder

    def test_summary(self):
        from flamingo_tools.synapse_detection.stain_summary import (
            _default_name, load_comparison, summarize_comparisons,
        )

        comparisons = {_default_name(folder): load_comparison(folder) for folder in self.folders.values()}
        self.assertEqual(list(comparisons), ["M_A_L", "M_B_R"])
        output = os.path.join(self.tmpdir.name, "summary")
        table = summarize_comparisons(comparisons, output)

        self.assertEqual(table.cochlea.tolist(), ["M_A_L", "M_B_R"])
        np.testing.assert_allclose(table.reference_per_ihc, 3.0)
        np.testing.assert_allclose(table.test_per_ihc, 5.0)
        self.assertEqual(table.tp.tolist(), [3, 3])
        np.testing.assert_allclose(table.precision, 0.3)
        np.testing.assert_allclose(table.recall, 0.5)
        for name in ("summary.tsv", "intensity_summary.tsv", "summary_snr.png", "summary_ratio.png"):
            self.assertTrue(os.path.isfile(os.path.join(output, name)), name)
        intensity = pd.read_csv(os.path.join(output, "intensity_summary.tsv"), sep="\t")
        self.assertEqual(sorted(intensity.cochlea.unique()), ["M_A_L", "M_B_R"])
        self.assertEqual(sorted(intensity.stain.unique()), ["CTBP2_spots", "Homer1_spots"])

    def test_mismatched_channels(self):
        from flamingo_tools.synapse_detection.stain_summary import load_comparison, summarize_comparisons

        comparisons = {name: load_comparison(folder) for name, folder in self.folders.items()}
        comparisons["M_B_R"]["summary"]["test_channel"] = "Bsn"
        with self.assertRaises(ValueError):
            summarize_comparisons(comparisons, os.path.join(self.tmpdir.name, "summary"))


if __name__ == "__main__":
    unittest.main()
