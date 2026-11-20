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


class TestStageArguments(unittest.TestCase):
    """'--stage' forwards the options to the stage function of slurm-workflows."""

    def call(self, command, argv):
        from flamingo_tools.segmentation import cli

        with mock.patch.object(sys, "argv", [command] + argv), \
                mock.patch.object(cli, "_get_model_path", side_effect=lambda model_type, path: path or "default"):
            getattr(cli, command)()

    def check(self, module, function, command, argv, expected):
        with mock.patch.object(module, function) as mocked:
            self.call(command, argv)
        mocked.assert_called_once_with(**expected)

    def check_rejected(self, command, argv):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.call(command, argv)

    def test_segmentation_stages(self):
        from flamingo_tools.segmentation import unet_prediction

        s3 = {"s3": False, "s3_credentials": None, "s3_bucket_name": None, "s3_service_endpoint": None}
        io = ["-i", "in.n5", "-k", "setup0/timepoint0/s0", "-o", "out"]
        io_kwargs = {"input_path": "in.n5", "input_key": "setup0/timepoint0/s0", "output_folder": "out"}

        self.check(unet_prediction, "run_unet_prediction_preprocess_slurm", "run_segmentation",
                   ["--stage", "mask", "-m", "IHC", "--absolute_threshold", "None"] + io,
                   {**io_kwargs, "seg_class": "ihc", "absolute_threshold": None, "legacy_mask": False,
                    "background_factor": None, **s3})
        self.check(unet_prediction, "run_unet_prediction_preprocess_slurm", "run_segmentation",
                   ["--stage", "mask", "-m", "IHC", "--legacy_mask", "--absolute_threshold", "150"] + io,
                   {**io_kwargs, "seg_class": "ihc", "absolute_threshold": 150.0, "legacy_mask": True,
                    "background_factor": None, **s3})
        self.check(unet_prediction, "run_unet_prediction_preprocess_slurm", "run_segmentation",
                   ["--stage", "mask", "-m", "SGN", "--background_factor", "1.2"] + io,
                   {**io_kwargs, "seg_class": "sgn", "absolute_threshold": None, "legacy_mask": False,
                    "background_factor": 1.2, **s3})
        self.check(unet_prediction, "run_unet_prediction_slurm", "run_segmentation",
                   ["--stage", "predict", "-m", "SGN", "-c", "model", "--prediction_instances", "10"] + io,
                   {**io_kwargs, "model_path": "model", "prediction_instances": 10, "seg_class": "sgn", **s3})
        self.check(unet_prediction, "run_unet_segmentation_slurm", "run_segmentation",
                   ["--stage", "segment", "-m", "SGN", "-o", "out", "--min_size", "1000",
                    "--center_distance_threshold", "0.5"],
                   {"output_folder": "out", "min_size": 1000, "center_distance_threshold": 0.5,
                    "boundary_distance_threshold": 0.5, "fg_threshold": 0.5, "distance_smoothing": 0.0})
        # The model gives the watershed values that are not given.
        self.check(unet_prediction, "run_unet_segmentation_slurm", "run_segmentation",
                   ["--stage", "segment", "-m", "IHC", "-o", "out", "--min_size", "1000"],
                   {"output_folder": "out", "min_size": 1000, "center_distance_threshold": 0.7,
                    "boundary_distance_threshold": 0.6, "fg_threshold": 0.5, "distance_smoothing": 1.0})
        self.check(unet_prediction, "run_unet_segmentation_slurm", "run_segmentation",
                   ["--stage", "segment", "-m", "SGN", "-o", "out", "--boundary_distance_threshold", "None"],
                   {"output_folder": "out", "min_size": 250, "center_distance_threshold": 0.4,
                    "boundary_distance_threshold": None, "fg_threshold": 0.5, "distance_smoothing": 0.0})

    def test_detection_stages(self):
        from flamingo_tools.segmentation import synapse_detection

        s3 = {"s3_credentials": None, "s3_bucket_name": None, "s3_service_endpoint": None}
        io = ["-i", "in.ome.zarr", "-k", "s0", "-o", "out"]
        io_kwargs = {"input_path": "in.ome.zarr", "input_key": "s0", "output_folder": "out"}

        self.check(synapse_detection, "run_synapse_prediction_preprocess_slurm", "run_detection",
                   ["--stage", "mask", "--mask_path", "ihc.ome.zarr", "--s3_mask"] + io,
                   {**io_kwargs, "mask_path": "ihc.ome.zarr", "mask_input_key": "s4", "dilation_iterations": 4,
                    "voxel_size": [0.38, 0.38, 0.38], "max_distance": 3.0, "s3_input": False, "s3_mask": True,
                    **s3})
        self.check(synapse_detection, "run_synapse_prediction_slurm", "run_detection",
                   ["--stage", "predict", "-c", "model.pt", "--prediction_instances", "10"] + io,
                   {**io_kwargs, "model_path": "model.pt", "prediction_instances": 10, "s3_input": False, **s3})
        self.check(synapse_detection, "run_synapse_detection_slurm", "run_detection",
                   ["--stage", "detect", "-o", "out", "-v", "0.38", "0.38", "0.38", "--mask_path", "ihc.ome.zarr",
                    "--s3_mask", "--force_overwrite"],
                   {"output_folder": "out", "voxel_size": [0.38, 0.38, 0.38], "threshold": 0.5,
                    "mask_path": "ihc.ome.zarr", "mask_key": "s0", "max_distance": 3.0, "force_overwrite": True,
                    "s3_mask": True, **s3})

    def test_options_outside_their_stage(self):
        io = ["-i", "in.n5", "-o", "out"]
        self.check_rejected("run_segmentation", ["-m", "SGN", "--prediction_instances", "10"] + io)
        self.check_rejected("run_segmentation", ["--stage", "mask", "-m", "SGN", "--min_size", "1000"] + io)
        self.check_rejected("run_segmentation", ["--stage", "segment", "-m", "SGN", "--legacy_mask", "-o", "out"])
        self.check_rejected("run_segmentation", ["--stage", "segment", "-m", "SGN"] + io)
        self.check_rejected("run_segmentation",
                            ["--stage", "segment", "-m", "SGN", "-o", "out", "--s3_bucket_name", "b"])
        self.check_rejected("run_detection", ["--force_overwrite"] + io)
        self.check_rejected("run_detection", ["--stage", "predict", "--mask_path", "ihc.ome.zarr"] + io)
        self.check_rejected("run_detection", ["--stage", "detect", "--dilation_iterations", "8", "-o", "out"])

    def test_seed_thresholds_not_both_none(self):
        # The center distance threshold of SGN-lowres is None.
        self.check_rejected("run_segmentation",
                            ["-m", "SGN-lowres", "--boundary_distance_threshold", "None", "-i", "in.n5", "-o", "out"])
        self.check_rejected("run_segmentation", ["--stage", "segment", "-m", "SGN", "-o", "out",
                                                 "--center_distance_threshold", "None",
                                                 "--boundary_distance_threshold", "None"])

    def test_input_path_required(self):
        self.check_rejected("run_segmentation", ["--stage", "predict", "-m", "SGN", "-o", "out"])
        self.check_rejected("run_segmentation", ["-m", "SGN", "-o", "out"])
        self.check_rejected("run_detection", ["--stage", "mask", "-o", "out"])


class TestSegmentationArguments(unittest.TestCase):
    """run_segmentation forwards the mask options and the default settings of the model."""

    def check(self, argv, expected, model_type="SGN"):
        from flamingo_tools.model_utils import get_default_segmentation_settings
        from flamingo_tools.segmentation import cli, unet_prediction

        argv = ["run_segmentation", "-i", "in.n5", "-k", "s0", "-o", "out", "-m", model_type] + argv
        with mock.patch.object(sys, "argv", argv), mock.patch.object(cli, "_get_model_path", return_value="model"), \
                mock.patch.object(unet_prediction, "run_unet_prediction") as function:
            cli.run_segmentation()
        defaults = {"legacy_mask": False, "absolute_threshold": None, "background_factor": None}
        function.assert_called_once_with(
            input_path="in.n5", input_key="s0", output_folder="out", model_path="model", min_size=250,
            **{**defaults, **get_default_segmentation_settings(model_type), **expected},
        )

    def test_mask_options(self):
        self.check([], {"use_mask": True})
        self.check(["--background_factor", "1.2"], {"use_mask": True, "background_factor": 1.2})
        self.check(["--legacy_mask"], {"use_mask": True, "legacy_mask": True})
        self.check(["--legacy_mask", "--absolute_threshold", "150"],
                   {"use_mask": True, "legacy_mask": True, "absolute_threshold": 150.0})
        self.check(["--disable_masking"], {"use_mask": False})

    def test_watershed_options(self):
        self.check(["--center_distance_threshold", "0.5", "--distance_smoothing", "1.0"],
                   {"use_mask": True, "center_distance_threshold": 0.5, "distance_smoothing": 1.0})
        # 'None' is a valid value, not the same as an option that is not given.
        self.check(["--center_distance_threshold", "None"], {"use_mask": True, "center_distance_threshold": None},
                   model_type="IHC")


if __name__ == "__main__":
    unittest.main()
