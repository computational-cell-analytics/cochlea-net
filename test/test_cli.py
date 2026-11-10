import os
import platform
import sys
import unittest
from shutil import rmtree
from unittest import mock

import z5py

from subprocess import run


@unittest.skipIf(platform.system() == "Windows", "CLI tests fail on windows.")
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
                   {**io_kwargs, "seg_class": "ihc", "absolute_threshold": "None", **s3})
        self.check(unet_prediction, "run_unet_prediction_slurm",
                   io + ["-m", "model", "--prediction_instances", "10"],
                   {**io_kwargs, "model_path": "model", "prediction_instances": 10, **s3})
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


if __name__ == "__main__":
    unittest.main()
