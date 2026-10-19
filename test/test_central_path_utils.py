import unittest

import pandas as pd


class TestMeanByOffset(unittest.TestCase):

    def test_bins(self):
        from flamingo_tools.analysis.central_path_utils import mean_by_offset

        table = pd.DataFrame({
            "length_fraction": [0.0, 0.01, 1.0, 0.0],
            "offset": [5.0, 9.0, 150.0, -1.0],
            "intensity": [1.0, 3.0, 7.0, 100.0],
        })
        result = mean_by_offset(table, ["intensity"])

        # 20 windows times 11 offset bins, of which the last is open.
        self.assertEqual(len(result), 220)
        self.assertEqual(result["n"].sum(), 3)

        first = result.iloc[0]
        self.assertEqual(first["n"], 2)
        self.assertEqual(first["intensity"], 2.0)
        self.assertEqual(first["offset_bin"].left, 0)

        last = result.iloc[-1]
        self.assertEqual(last["n"], 1)
        self.assertEqual(last["intensity"], 7.0)
        self.assertEqual(last["length_bin"].right, 1.0)
        self.assertEqual(last["offset_bin"].left, 100)

        empty = result[result["n"] == 0]
        self.assertTrue(empty["intensity"].isna().all())


if __name__ == "__main__":
    unittest.main()
