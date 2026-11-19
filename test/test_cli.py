import os
import shutil
import subprocess
import sys
import unittest
from tempfile import TemporaryDirectory
from unittest import mock

import z5py


class TestCLI(unittest.TestCase):
    def setUp(self):
        from flamingo_tools import create_test_data

        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.folder = tmp_dir.name
        create_test_data(self.folder)

    def test_convert_data(self):
        # Windows does not add ".exe" to a command name with a dot, but shutil.which does.
        executable = shutil.which("flamingo_tools.convert_data")
        self.assertIsNotNone(executable, "The console script flamingo_tools.convert_data is not installed.")

        out_path = os.path.join(self.folder, "converted_data.n5")
        cmd = [executable, "-i", self.folder, "-o", out_path, "--metadata_pattern", ""]
        result = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            result.returncode, 0,
            msg=f"CLI failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

        self.assertTrue(os.path.exists(out_path))
        xml_path = out_path.replace(".n5", ".xml")
        self.assertTrue(os.path.exists(xml_path))
        with z5py.File(out_path, "r") as f:
            self.assertTrue("setup0" in f)


class TestSlurmCLI(unittest.TestCase):
    """The Slurm commands forward their options to the function of the same name."""

    def check(self, module, command, argv, expected):
        from flamingo_tools.segmentation import cli

        with mock.patch.object(sys, "argv", [command] + argv), mock.patch.object(module, command) as function:
            getattr(cli, command)()
        function.assert_called_once_with(**expected)

    def test_unet_commands(self):
        from flamingo_tools.segmentation import unet_prediction

        s3 = {"s3": False, "s3_credentials": None, "s3_bucket_name": None, "s3_service_endpoint": None}
        io = ["-i", "in.n5", "-k", "setup0/timepoint0/s0", "-o", "out"]
        io_kwargs = {"input_path": "in.n5", "input_key": "setup0/timepoint0/s0", "output_folder": "out"}

        self.check(unet_prediction, "run_unet_prediction_preprocess_slurm",
                   io + ["--seg_class", "ihc", "--absolute_threshold", "None"],
                   {**io_kwargs, "seg_class": "ihc", "absolute_threshold": "None", "legacy_mask": False,
                    "background_factor": None, **s3})
        self.check(unet_prediction, "run_unet_prediction_preprocess_slurm",
                   io + ["--seg_class", "ihc", "--legacy_mask", "--absolute_threshold", "150"],
                   {**io_kwargs, "seg_class": "ihc", "absolute_threshold": "150", "legacy_mask": True,
                    "background_factor": None, **s3})
        self.check(unet_prediction, "run_unet_prediction_preprocess_slurm",
                   io + ["--seg_class", "sgn", "--background_factor", "1.2"],
                   {**io_kwargs, "seg_class": "sgn", "absolute_threshold": None, "legacy_mask": False,
                    "background_factor": 1.2, **s3})
        self.check(unet_prediction, "run_unet_prediction_slurm",
                   io + ["-m", "model", "--prediction_instances", "10"],
                   {**io_kwargs, "model_path": "model", "prediction_instances": 10, "seg_class": None, **s3})
        self.check(unet_prediction, "run_unet_segmentation_slurm",
                   ["-o", "out", "--min_size", "1000", "--center_distance_threshold", "0.5"],
                   {"output_folder": "out", "min_size": 1000, "center_distance_threshold": 0.5,
                    "boundary_distance_threshold": 0.5, "distance_smoothing": 0.0})

    def test_synapse_commands(self):
        from flamingo_tools.segmentation import synapse_detection

        s3 = {"s3_credentials": None, "s3_bucket_name": None, "s3_service_endpoint": None}
        io = ["-i", "in.ome.zarr", "-k", "s0", "-o", "out"]
        io_kwargs = {"input_path": "in.ome.zarr", "input_key": "s0", "output_folder": "out"}

        self.check(synapse_detection, "run_synapse_prediction_preprocess_slurm",
                   io + ["--mask_path", "ihc.ome.zarr", "--s3_mask"],
                   {**io_kwargs, "mask_path": "ihc.ome.zarr", "mask_input_key": "s4", "dilation_iterations": 4,
                    "max_distance": 3.0, "s3_input": False, "s3_mask": True, **s3})
        self.check(synapse_detection, "run_synapse_prediction_slurm",
                   io + ["-m", "model.pt", "--prediction_instances", "10"],
                   {**io_kwargs, "model_path": "model.pt", "prediction_instances": 10, "s3_input": False, **s3})
        self.check(synapse_detection, "run_synapse_detection_slurm",
                   ["-o", "out", "-v", "0.38", "0.38", "0.38", "--mask_path", "ihc.ome.zarr", "--s3_mask",
                    "--force_overwrite"],
                   {"output_folder": "out", "voxel_size": [0.38, 0.38, 0.38], "threshold": 0.5,
                    "mask_path": "ihc.ome.zarr", "mask_key": "s0", "max_distance": 3.0, "force_overwrite": True,
                    "s3_mask": True, **s3})


class TestSegmentationArguments(unittest.TestCase):
    """run_segmentation forwards the mask options and the default settings of the model."""

    def check(self, argv, expected):
        from flamingo_tools.model_utils import get_default_segmentation_settings
        from flamingo_tools.segmentation import cli

        argv = ["run_segmentation", "-i", "in.n5", "-k", "s0", "-o", "out", "-m", "SGN"] + argv
        with mock.patch.object(sys, "argv", argv), mock.patch.object(cli, "_get_model_path", return_value="model"), \
                mock.patch.object(cli, "run_unet_prediction") as function:
            cli.run_segmentation()
        function.assert_called_once_with(
            input_path="in.n5", input_key="s0", output_folder="out", model_path="model", min_size=250,
            **expected, **get_default_segmentation_settings("SGN"),
        )

    def test_mask_options(self):
        self.check([], {"use_mask": True, "legacy_mask": False, "background_factor": None})
        self.check(["--background_factor", "1.2"], {"use_mask": True, "legacy_mask": False, "background_factor": 1.2})
        self.check(["--legacy_mask"], {"use_mask": True, "legacy_mask": True, "background_factor": None})
        self.check(["--disable_masking"], {"use_mask": False, "legacy_mask": False, "background_factor": None})


if __name__ == "__main__":
    unittest.main()
