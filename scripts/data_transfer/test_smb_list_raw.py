"""Unit test for smb_list_raw.py. Run with `python -m unittest test_smb_list_raw` from this directory."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smb_list_raw  # noqa: E402


class TestRawFileList(unittest.TestCase):
    def test_groups_and_sorts_by_stain_folder(self):
        size_map = {
            "B_560/S000_t000000_V000_R0000_X001_Y000_C00_I0_D0_P00056.raw": 9,
            "notes.txt": 1,
            "A_488/sub/extra.txt": 3,
            "B_560/S000_t000000_V000_R0000_X000_Y000_C00_I0_D0_P00056.raw": 9,
            "A_488/Settings.txt": 2,
        }
        self.assertEqual(smb_list_raw.raw_file_list("raw", size_map), {
            "raw_data": "raw",
            "stain_folders": {
                "A_488": ["Settings.txt", "sub/extra.txt"],
                "B_560": [
                    "S000_t000000_V000_R0000_X000_Y000_C00_I0_D0_P00056.raw",
                    "S000_t000000_V000_R0000_X001_Y000_C00_I0_D0_P00056.raw",
                ],
            },
        })


if __name__ == "__main__":
    unittest.main()
