import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "measurements")
sys.path.insert(0, SCRIPTS_DIR)


def _tables():
    # Reference spots 1 and 2 have a test partner within 2 micrometer, reference spot 3 has none.
    # Test spots 13 and 14 are unmatched. Both spots on IHC 9 match, but IHC 9 is not a valid IHC.
    reference = pd.DataFrame({
        "spot_id": [1, 2, 3, 4],
        "x": [0.0, 10.0, 20.0, 50.0], "y": 0.0, "z": 0.0, "matched_ihc": [1, 1, 2, 9],
    })
    test = pd.DataFrame({
        "spot_id": [11, 12, 13, 14, 15],
        "x": [0.5, 11.0, 30.0, 31.0, 50.5], "y": 0.0, "z": 0.0, "matched_ihc": [1, 1, 2, 2, 9],
    })
    for table in (reference, test):
        for channel in ("CTBP2", "Homer1"):
            for measure in ("peak", "background", "ratio", "snr", "integrated", "n_saturated"):
                table[f"{channel}_{measure}"] = np.arange(len(table), dtype="float64") + 1
    test["Homer1_snr"] = [10.0, 8.0, 1.0, 2.0, 9.0]
    return reference, test


class TestMatchStains(unittest.TestCase):

    def test_matches_colocalization_script(self):
        from flamingo_tools.synapse_detection.stain_comparison import match_stains
        from synapse_colocalization import evaluate_colocalization

        reference, test = _tables()
        for valid_ihcs in (None, [1, 2, 3]):
            matched_reference, matched_test = match_stains(reference, test, max_dist=2.0, valid_ihcs=valid_ihcs)
            scores, _ = evaluate_colocalization(reference, test, 2.0, valid_ihcs=valid_ihcs)
            self.assertEqual(matched_test.matched.sum(), scores["tp"])
            self.assertEqual(matched_reference.matched.sum(), scores["tp"])
            self.assertEqual((~matched_test.matched).sum(), scores["fp"])
            self.assertEqual((~matched_reference.matched).sum(), scores["fn"])

        matched_reference, matched_test = match_stains(reference, test, max_dist=2.0, valid_ihcs=[1, 2])
        self.assertEqual(matched_reference.partner_id.tolist()[:2], [11, 12])
        self.assertTrue(np.isnan(matched_reference.partner_id[2]))
        self.assertEqual(matched_test.partner_id.tolist()[:2], [1, 2])
        # The input tables stay unchanged.
        self.assertNotIn("matched", reference)


class TestThresholdSweep(unittest.TestCase):

    def test_sweep(self):
        from flamingo_tools.synapse_detection.stain_comparison import match_stains, threshold_sweep

        reference, test = match_stains(*_tables(), max_dist=2.0, valid_ihcs=[1, 2])
        sweep = threshold_sweep(reference, test, "Homer1_snr", thresholds=[-np.inf, 1.5, 5.0, 9.0], n_ihcs=2)
        self.assertEqual(sweep.n_test.tolist(), [4, 3, 2, 1])
        self.assertEqual(sweep.tp.tolist(), [2, 2, 2, 1])
        self.assertEqual(sweep.tp[0], test.matched.sum())
        self.assertEqual(sweep.precision.tolist(), [0.5, 0.667, 1.0, 1.0])
        self.assertEqual(sweep.test_per_ihc.tolist(), [2.0, 1.5, 1.0, 0.5])

    def test_nan_is_kept(self):
        from flamingo_tools.synapse_detection.stain_comparison import match_stains, threshold_sweep

        reference, test = match_stains(*_tables(), max_dist=2.0)
        test.loc[0, "Homer1_snr"] = np.nan
        sweep = threshold_sweep(reference, test, "Homer1_snr", thresholds=[100.0])
        self.assertEqual(sweep.n_test.tolist(), [1])


class TestSelectExcessIhcs(unittest.TestCase):

    def test_selection(self):
        from flamingo_tools.synapse_detection.stain_comparison import match_stains, select_excess_ihcs

        reference, test = match_stains(*_tables(), max_dist=2.0)
        # IHC 2 has two unmatched test spots, IHC 1 and IHC 9 none.
        ihc_table = pd.DataFrame({
            "label_id": [1, 2, 9], "anchor_x": [30.0, 25.0, 50.0], "anchor_y": 0.0, "anchor_z": 0.0,
        })
        selected = select_excess_ihcs(test, ihc_table, n_ihcs=2)
        self.assertEqual(selected.label_id.tolist(), [2, 1])
        self.assertEqual(selected.n_unmatched.tolist(), [2, 0])
        self.assertEqual(selected[["x", "y", "z"]].values.tolist()[0], [25.0, 0.0, 0.0])

        # A crop of 20 um around IHC 2 overlaps with the crop around IHC 1 (5 um apart), but not with IHC 9.
        selected = select_excess_ihcs(test, ihc_table, n_ihcs=2, crop_extent=(20.0, 20.0, 20.0))
        self.assertEqual(selected.label_id.tolist(), [2, 9])


class TestCompareStains(unittest.TestCase):

    def test_outputs(self):
        from flamingo_tools.synapse_detection.stain_comparison import compare_stains, write_crop_json

        reference, test = _tables()
        with tempfile.TemporaryDirectory() as tmp:
            result = compare_stains(
                reference, test, "CTBP2", "Homer1", tmp, reference_name="ctbp2", test_name="homer1",
                valid_ihcs=[1, 2], thresholds=[-np.inf, 5.0],
            )
            for name in ("ctbp2_matched.tsv", "homer1_matched.tsv", "intensity_summary.tsv", "sweep.tsv",
                         "summary.json", "overview.png"):
                self.assertTrue(os.path.isfile(os.path.join(tmp, name)), name)
            with open(os.path.join(tmp, "summary.json")) as f:
                summary = json.load(f)
            self.assertEqual((summary["tp"], summary["fp"], summary["fn"]), (2, 2, 1))
            self.assertEqual(summary["n_ihcs"], 2)
            self.assertEqual(
                [summary[key] for key in ("reference_name", "test_name", "reference_channel", "test_channel")],
                ["ctbp2", "homer1", "CTBP2", "Homer1"],
            )
            self.assertEqual(result["summary"], summary)

            json_path = os.path.join(tmp, "crops.json")
            write_crop_json(json_path, [[25.4, 0.0, 0.6]], "M_AMD_000117_L", ["CTBP2", "IHC_v11"], [8, 8, 8],
                            segmentation_channel="IHC_v11")
            with open(json_path) as f:
                params = json.load(f)
        self.assertEqual(params["crop_centers"], [[25, 0, 1]])
        self.assertEqual(params["image_channel"], ["CTBP2", "IHC_v11"])


if __name__ == "__main__":
    unittest.main()
