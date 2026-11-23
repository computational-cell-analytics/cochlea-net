import os
import shutil
import subprocess
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import torch
import z5py
from torch_em.model import UNet3d


class TestSegmentationCLI(unittest.TestCase):
    shape = (64, 128, 128)

    def _executable(self, name):
        # Windows does not add ".exe" to a command name with a dot, but shutil.which does.
        executable = shutil.which(name)
        self.assertIsNotNone(executable, f"The console script {name} is not installed.")
        return executable

    def _run(self, name, argv, env=None):
        result = subprocess.run([self._executable(name)] + argv, capture_output=True, text=True, env=env)
        self.assertEqual(
            result.returncode, 0,
            msg=f"CLI failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def _create_data(self, tmp_dir):
        data = np.random.randint(0, 255, size=self.shape)
        path = os.path.join(tmp_dir, "data.tif")
        imageio.imwrite(path, data)
        return path

    def _create_model(self, tmp_dir, out_channels):
        """Save an untrained tiny model, so that the test does not fetch the released one.

        These tests cover the CLI, not the segmentation quality. The released models are 82 MB
        each and would be downloaded on every run.
        """
        model = UNet3d(in_channels=1, out_channels=out_channels, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)
        return model_path

    def test_run_segmentation(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = self._create_data(tmp_dir)
            model_path = self._create_model(tmp_dir, out_channels=3)
            output_folder = os.path.join(tmp_dir, "output")

            self._run("flamingo_tools.run_segmentation", [
                "-i", data_path, "-o", output_folder, "-m", "SGN", "--min_size", "0", "-c", model_path,
            ])
            self._check_segmentation(output_folder)

    def test_run_segmentation_stages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = self._create_data(tmp_dir)
            model_path = self._create_model(tmp_dir, out_channels=3)
            output_folder = os.path.join(tmp_dir, "output")

            env = {**os.environ, "SLURM_ARRAY_TASK_ID": "0"}
            for stage in (
                ["--stage", "mask", "-i", data_path],
                ["--stage", "predict", "-i", data_path, "-c", model_path],
                ["--stage", "segment", "--min_size", "0"],
            ):
                self._run("flamingo_tools.run_segmentation", ["-o", output_folder, "-m", "SGN"] + stage, env=env)
            self._check_segmentation(output_folder)

    def _check_segmentation(self, output_folder):
        expected_path = os.path.join(output_folder, "segmentation.zarr")
        expected_key = "segmentation"

        self.assertTrue(os.path.exists(expected_path))
        with z5py.File(expected_path, "r") as f:
            self.assertTrue(expected_key in f)
            self.assertEqual(f[expected_key].shape, self.shape)

    def test_run_detection(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = self._create_data(tmp_dir)
            model_path = self._create_model(tmp_dir, out_channels=1)
            output_folder = os.path.join(tmp_dir, "output")

            self._run("flamingo_tools.run_detection", [
                "-i", data_path, "-o", output_folder, "-c", model_path, "--threshold", "0.4",
                "--dilation_iterations", "4",
            ])
            self.assertTrue(os.path.exists(os.path.join(output_folder, "synapse_detection.tsv")))

    def test_run_detection_stages(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = self._create_data(tmp_dir)
            model_path = self._create_model(tmp_dir, out_channels=1)
            output_folder = os.path.join(tmp_dir, "output")

            env = {**os.environ, "SLURM_ARRAY_TASK_ID": "0"}
            for stage in (
                ["--stage", "mask", "-i", data_path],
                ["--stage", "predict", "-i", data_path, "-c", model_path],
                ["--stage", "detect", "--threshold", "0.4"],
            ):
                self._run("flamingo_tools.run_detection", ["-o", output_folder] + stage, env=env)
            self.assertTrue(os.path.exists(os.path.join(output_folder, "synapse_detection.tsv")))


if __name__ == "__main__":
    unittest.main()
