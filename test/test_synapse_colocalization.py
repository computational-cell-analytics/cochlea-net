import os
import sys
import unittest

import pandas as pd

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "measurements")
sys.path.insert(0, SCRIPTS_DIR)


class TestEvaluateColocalization(unittest.TestCase):

    def setUp(self):
        from synapse_colocalization import evaluate_colocalization
        self.fn = evaluate_colocalization
        # Reference synapses 0 and 1 have a test partner within 2 micrometer, reference synapse 2 has none.
        # Test synapse 2 is unmatched. Both synapses on IHC 9 match, but IHC 9 is not a valid IHC.
        self.reference = pd.DataFrame({
            "x": [0.0, 10.0, 20.0, 50.0], "y": 0.0, "z": 0.0, "matched_ihc": [1, 1, 2, 9],
        })
        self.test = pd.DataFrame({
            "x": [0.5, 11.0, 30.0, 50.5], "y": 0.0, "z": 0.0, "matched_ihc": [1, 1, 2, 9],
        })

    def test_all_ihcs(self):
        scores, ihc_counts = self.fn(self.reference, self.test, max_dist=2.0)
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (3, 1, 1))
        self.assertEqual(ihc_counts.label_id.tolist(), [1, 2, 9])

    def test_valid_ihcs(self):
        scores, ihc_counts = self.fn(self.reference, self.test, max_dist=2.0, valid_ihcs=[1, 2, 3])
        self.assertEqual((scores["tp"], scores["fp"], scores["fn"]), (2, 1, 1))
        self.assertEqual(scores["n_ihcs"], 3)
        expected = pd.DataFrame({
            "label_id": [1, 2, 3], "n_reference": [2, 1, 0], "n_test": [2, 1, 0], "n_matched": [2, 0, 0],
        })
        pd.testing.assert_frame_equal(ihc_counts, expected, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
