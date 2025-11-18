import os
import platform
import subprocess
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import z5py


@unittest.skipIf(platform.system() == "Windows", "CLI tests fail on windows.")
class TestSegmentationCLI(unittest.TestCase):
    shape = (64, 128, 128)

    def _create_data(self, tmp_dir):
        data = np.random.randint(0, 255, size=self.shape)
        path = os.path.join(tmp_dir, "data.tif")
        imageio.imwrite(path, data)
        return path

    def test_run_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = self._create_data(tmp_dir)
            output_folder = os.path.join(tmp_dir, "output")

            subprocess.run([
                "flamingo_tools.run_segmentation",
                "-i", data_path, "-o", output_folder, "-m", "SGN", "--min_size", "0"
            ])

            expected_path = os.path.join(output_folder, "segmentation.zarr")
            expected_key = "segmentation"

            self.assertTrue(os.path.exists(expected_path))
            with z5py.File(expected_path, "r") as f:
                self.assertTrue(expected_key in f)
                self.assertEqual(f[expected_key].shape, self.shape)

    def test_run_detection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = self._create_data(tmp_dir)
            output_folder = os.path.join(tmp_dir, "output")

            subprocess.run([
                "flamingo_tools.run_detection", "-i", data_path, "-o", output_folder
            ])

            expected_path = os.path.join(output_folder, "synapse_detection.tsv")
            self.assertTrue(os.path.exists(expected_path))


if __name__ == "__main__":
    unittest.main()
