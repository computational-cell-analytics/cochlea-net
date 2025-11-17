import os
import unittest
from shutil import rmtree

import z5py

from subprocess import run


class TestCLI(unittest.TestCase):
    folder = "./tmp"

    def setUp(self):
        from flamingo_tools import create_test_data

        # TODO Create flamingo metadata.
        create_test_data(self.folder)

    def tearDown(self):
        rmtree(self.folder)

    def test_convert_data(self):
        out_path = os.path.join(self.folder, "converted_data.n5")
        cmd = ["flamingo_tools.convert_data", "-i", self.folder, "-o", out_path, "--metadata_pattern", ""]
        run(cmd)

        self.assertTrue(os.path.exists(out_path))
        xml_path = out_path.replace(".n5", ".xml")
        self.assertTrue(os.path.exists(xml_path))
        with z5py.File(out_path, "r") as f:
            self.assertTrue("setup0" in f)


if __name__ == "__main__":
    unittest.main()
