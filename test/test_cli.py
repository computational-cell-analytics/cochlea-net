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
                   {"output_folder": "out", "voxel_size": [0.38, 0.38, 0.38], "threshold": 0.5, "min_distance": 1,
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
        self.check_rejected("run_detection", ["--stage", "predict", "--min_distance", "2"] + io)

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


class TestExtractBlocksArguments(unittest.TestCase):
    """'flamingo_tools.extract_blocks' forwards the options of its mode to the block and center functions."""

    s3 = {"s3": False, "s3_credentials": None, "s3_bucket_name": None, "s3_service_endpoint": None}
    block = {"input_key": "s0", "output_key": None, "voxel_size": [0.38, 0.38, 0.38], "point_tables": None,
             "force": False}
    centers = {"n_blocks": 6, "cell_type": "sgn", "component_list": [1], "include_gap": False, "path_method": None}

    def call(self, argv):
        """Run the command with mocked functions and return the parent mock, which records the call order."""
        from flamingo_tools.postprocessing import cli

        parent = mock.Mock()
        with mock.patch.object(sys, "argv", ["extract_blocks"] + argv), \
                mock.patch.object(cli, "extract_block_json_wrapper", parent.extract), \
                mock.patch.object(cli, "equidistant_centers_json_wrapper", parent.find_centers):
            cli.extract_blocks()
        return parent

    def check_rejected(self, argv):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.call(argv)

    def test_one_block(self):
        parent = self.call(["-i", "in.zarr", "-c", "1", "2", "3.5", "-o", "out", "--image_channel", "PV"])
        self.assertEqual(parent.mock_calls, [mock.call.extract(
            output_path="out", input_path="in.zarr", coords=[1.0, 2.0, 3.5], roi_halo=[128, 128, 64],
            dataset_name=None, image_channel="PV", **self.block, **self.s3,
        )])

    def test_json_modes(self):
        from flamingo_tools.s3_utils import MOBIE_FOLDER

        extract = mock.call.extract(output_path="out", json_file="f.json", mobie_dir=MOBIE_FOLDER,
                                    **self.block, **self.s3)
        find_centers = mock.call.find_centers(json_file="f.json", mobie_dir=MOBIE_FOLDER, overrides={},
                                              **self.centers, **self.s3)
        self.assertEqual(self.call(["--json_info", "f.json", "-o", "out"]).mock_calls, [extract])
        self.assertEqual(self.call(["--json_info", "f.json", "--find_centers"]).mock_calls, [find_centers])
        self.assertEqual(self.call(["--json_info", "f.json", "--find_centers", "-o", "out"]).mock_calls,
                         [find_centers, extract])

    def test_only_typed_center_options_override_the_file(self):
        parent = self.call(["--json_info", "f.json", "--find_centers", "-n", "4", "--include_gap",
                            "--cell_type", "sgn"])
        # '--cell_type sgn' equals the default, so the file keeps its own cell type.
        self.assertEqual(parent.find_centers.call_args.kwargs["overrides"], {"n_blocks": 4, "include_gap": True})
        self.assertEqual(parent.find_centers.call_args.kwargs["n_blocks"], 4)

    def test_point_tables(self):
        import pandas as pd

        with TemporaryDirectory() as tmp:
            paths = [os.path.join(tmp, "CTBP2_spots.tsv"), os.path.join(tmp, "sub", "CTBP2_spots.tsv")]
            os.makedirs(os.path.join(tmp, "sub"))
            for path in paths:
                pd.DataFrame({"x": [1.0], "y": [2.0], "z": [3.0]}).to_csv(path, sep="\t", index=False)

            parent = self.call(["--json_info", "f.json", "-o", "out", "--point_tables", paths[0]])
            point_tables = parent.extract.call_args.kwargs["point_tables"]
            self.assertEqual(list(point_tables), ["CTBP2_spots"])
            self.assertEqual(point_tables["CTBP2_spots"]["z"].tolist(), [3.0])

            # The table name is the CSV file name, so two tables with one file name are rejected.
            self.check_rejected(["--json_info", "f.json", "-o", "out", "--point_tables", *paths])

    def test_options_outside_their_mode(self):
        one_block = ["-i", "in.zarr", "-c", "1", "2", "3", "-o", "out"]
        for argv in (
            ["--json_info", "f.json", "-o", "out", "--roi_halo", "4", "4", "2"],
            ["--json_info", "f.json", "-o", "out", "-c", "1", "2", "3"],
            ["--json_info", "f.json", "-o", "out", "--image_channel", "PV"],
            ["--json_info", "f.json", "-o", "out", "-n", "4"],
            ["--json_info", "f.json", "-o", "out", "--include_gap"],
            ["--json_info", "f.json", "--find_centers", "-f"],
            ["--json_info", "f.json", "--find_centers", "--point_tables", "spots.tsv"],
            one_block + ["--find_centers"],
            one_block + ["--mobie_dir", "mobie"],
            one_block + ["--path_method", "graph"],
        ):
            with self.subTest(argv=argv):
                self.check_rejected(argv)

    def test_required_options(self):
        for argv in (
            [],
            ["-i", "in.zarr", "-o", "out"],
            ["-i", "in.zarr", "-c", "1", "2", "3"],
            ["-i", "in.zarr", "-c", "1", "2", "-o", "out"],
            ["--json_info", "f.json"],
            ["-i", "in.zarr", "--json_info", "f.json", "-c", "1", "2", "3", "-o", "out"],
        ):
            with self.subTest(argv=argv):
                self.check_rejected(argv)


class TestJsonExtractBlocksArguments(unittest.TestCase):
    """'flamingo_tools.json_extract_blocks' writes the input of 'extract_blocks --find_centers'."""

    def call(self, argv):
        from flamingo_tools.postprocessing import cli

        with mock.patch.object(sys, "argv", ["json_extract_blocks"] + argv):
            cli.json_extract_blocks()

    def test_parameter_file(self):
        import json

        with TemporaryDirectory() as tmp:
            json_file = os.path.join(tmp, "M_A_L.json")
            self.call(["-o", json_file, "-d", "M_A_L", "-i", "PV", "SGN_v2", "-s", "SGN_v2", "--cell_type", "ihc",
                       "-c", "1", "2", "--roi_halo", "64", "64", "32"])
            with open(json_file) as f:
                params = json.load(f)
        self.assertEqual(params, {
            "dataset_name": "M_A_L", "image_channel": ["PV", "SGN_v2"], "segmentation_channel": "SGN_v2",
            "cell_type": "ihc", "component_list": [1, 2], "n_blocks": 6, "include_gap": False,
            "roi_halo": [64, 64, 32],
        })

    def test_rejected_options(self):
        required = ["-o", "out.json", "-d", "M_A_L", "-i", "PV"]
        for argv in (
            required,
            required + ["-s", "SGN_v2", "--cell_type", "ohc"],
            required + ["-s", "SGN_v2", "--roi_halo", "64", "64"],
        ):
            with self.subTest(argv=argv), mock.patch("sys.stderr"), self.assertRaises(SystemExit):
                self.call(argv)


class TestExtractBlocksEndToEnd(unittest.TestCase):
    """'--find_centers -o' finds the crop centers of a block parameter file and extracts a crop at each."""

    def setUp(self):
        import json
        import zarr

        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from test_cochlea_mapping import _make_arc_table

        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.folder = tmp_dir.name
        self.mobie_dir = os.path.join(self.folder, "mobie")

        # The arc lies around the origin. Shift it, so that the crops lie inside the volume.
        table, _, _ = _make_arc_table(comp_sizes=(120,))
        for axis, shift in (("x", 700.0), ("y", 700.0), ("z", 50.0)):
            for column in (f"anchor_{axis}", f"bb_min_{axis}", f"bb_max_{axis}"):
                table[column] += shift
        table_dir = os.path.join(self.mobie_dir, "M_T_L", "tables", "IHC")
        os.makedirs(table_dir)
        table.to_csv(os.path.join(table_dir, "default.tsv"), sep="\t", index=False)

        # At 10 micrometer per voxel, the shifted arc lies inside 140 voxels in x and y.
        image_path = os.path.join(self.mobie_dir, "M_T_L", "images", "ome-zarr", "PV.ome.zarr")
        zarr.open_group(image_path, mode="w").create_array("s0", shape=(10, 140, 140), dtype="uint16")

        self.json_file = os.path.join(self.folder, "M_T_L.json")
        with open(self.json_file, "w") as f:
            json.dump({
                "dataset_name": "M_T_L", "image_channel": ["PV"], "segmentation_channel": "IHC", "cell_type": "ihc",
                "component_list": [1], "n_blocks": 4, "roi_halo": [4, 4, 2], "voxel_size": [10, 10, 10],
            }, f)

    def test_find_centers_and_extract(self):
        import json
        from contextlib import redirect_stdout
        from io import StringIO

        import imageio.v3 as imageio
        from flamingo_tools.extract_block_util import _crop_file_name
        from flamingo_tools.postprocessing import cli

        output_dir = os.path.join(self.folder, "crops")
        argv = ["extract_blocks", "--json_info", self.json_file, "--find_centers", "-o", output_dir,
                "--mobie_dir", self.mobie_dir]
        with mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            cli.extract_blocks()

        with open(self.json_file) as f:
            params = json.load(f)
        self.assertEqual(len(params["crop_centers"]), 4)
        self.assertEqual(params["path_method"], "graph")

        expected = sorted(_crop_file_name(center, "M_T_L", "PV", ".tif") for center in params["crop_centers"])
        self.assertEqual(sorted(os.listdir(output_dir)), expected)
        crop = imageio.imread(os.path.join(output_dir, expected[0]))
        self.assertEqual(crop.shape, (4, 8, 8))


class TestExportDataArguments(unittest.TestCase):
    """'flamingo_tools.export_data' forwards the options of a command and an input mode to its export function."""

    common = dict(
        output_folder="out", scale=[2, 3], crop_center=None, roi_halo=None, axis=None, suffix=None,
        voxel_size=[0.38, 0.38, 0.38], force=False,
        s3=False, s3_credentials=None, s3_bucket_name=None, s3_service_endpoint=None,
    )
    path = dict(input_path="in.zarr", input_key="s0", name=None)
    path_argv = ["-i", "in.zarr", "-o", "out", "-s", "2", "3"]
    functions = [
        f"export_{name}{mode}" for name in ("lower_resolution", "marker", "subtypes", "synapses", "frequency")
        for mode in ("", "_cochlea")
    ]

    def setUp(self):
        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.folder = tmp_dir.name

        # Without '--s3', the command requires the cochlea folder in the MoBIE project.
        self.mobie_dir = os.path.join(self.folder, "mobie")
        os.makedirs(os.path.join(self.mobie_dir, "M_X"))
        self.cochlea = dict(cochlea="M_X", mobie_dir=self.mobie_dir)
        self.cochlea_argv = ["-c", "M_X", "--mobie_dir", self.mobie_dir, "-o", "out", "-s", "2", "3"]

    def call(self, argv):
        """Run the command with all export functions mocked, and return the mocks that were called."""
        from flamingo_tools.export_data import cli

        mocks = {function: mock.Mock() for function in self.functions}
        with mock.patch.object(sys, "argv", ["flamingo_tools.export_data"] + argv), \
                mock.patch.multiple(cli, **mocks):
            cli.export_data()
        return {function: mocked for function, mocked in mocks.items() if mocked.called}

    def check(self, argv, function, expected):
        called = self.call(argv)
        self.assertEqual(list(called), [function])
        called[function].assert_called_once_with(**expected)

    def check_rejected(self, argv):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            self.call(argv)

    def table(self, name, **columns):
        import pandas as pd

        path = os.path.join(self.folder, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pd.DataFrame(columns).to_csv(path, sep="\t", index=False)
        return path

    def test_lower_resolution(self):
        options = dict(
            filter_by_components=None, filter_sgn_components=[1], filter_ihc_components=[1],
            filter_dilation_iterations=8, max_intensity=None, labels=None, binarize=False, ome_zarr=False,
        )
        self.check(["lower_resolution"] + self.path_argv, "export_lower_resolution", dict(
            table=None, sgn_table=None, ihc_table=None, **self.path, **options, **self.common,
        ))

        argv = ["lower_resolution"] + self.cochlea_argv + [
            "--s3", "--channels", "Otof", "IHC_v11", "--filter_cochlea_channels", "IHC_v11",
            "--filter_dilation_iterations", "3", "--crop_center", "1", "2", "3", "--axis", "2", "--suffix", "apex",
        ]
        self.check(argv, "export_lower_resolution_cochlea", dict(
            channels=["Otof", "IHC_v11"], filter_cochlea_channels=["IHC_v11"], **self.cochlea,
            **{**options, "filter_dilation_iterations": 3},
            **{**self.common, "s3": True, "crop_center": [1.0, 2.0, 3.0], "axis": 2, "suffix": "apex"},
        ))

    def test_lower_resolution_reads_the_tables(self):
        table = self.table("seg.tsv", label_id=[1, 2], component_labels=[1, 2])
        sgn_table = self.table("sgn.tsv", label_id=[3])
        called = self.call(["lower_resolution"] + self.path_argv + [
            "--table", table, "--filter_by_components", "1", "--sgn_table", sgn_table, "--max_intensity", "1400",
            "--labels",
        ])
        kwargs = called["export_lower_resolution"].call_args.kwargs
        self.assertEqual(kwargs["table"].label_id.tolist(), [1, 2])
        self.assertEqual(kwargs["sgn_table"].label_id.tolist(), [3])
        self.assertIsNone(kwargs["ihc_table"])
        self.assertEqual(kwargs["filter_by_components"], [1])
        self.assertEqual(kwargs["max_intensity"], 1400.0)
        self.assertTrue(kwargs["labels"])

    def test_marker(self):
        table = self.table("edited.tsv", label_id=[1], marker_labels=[1])
        called = self.call(["marker"] + self.path_argv + ["--table", table])
        self.assertEqual(called["export_marker"].call_args.kwargs["table"].marker_labels.tolist(), [1])

        self.check(["marker"] + self.cochlea_argv + ["--channels", "IHC_v11", "--table", table],
                   "export_marker_cochlea", dict(channels=["IHC_v11"], table_path=table, **self.cochlea, **self.common))

    def test_subtypes(self):
        table = self.table("subtypes.tsv", label_id=[1], marker_CR=[1])
        called = self.call(["subtypes"] + self.path_argv + ["--table", table, "--stains", "CR", "--merge_masks"])
        kwargs = called["export_subtypes"].call_args.kwargs
        self.assertEqual((kwargs["stains"], kwargs["merge_masks"]), (["CR"], True))

        self.check(["subtypes"] + self.cochlea_argv, "export_subtypes_cochlea", dict(
            segmentation=None, stains=None, merge_masks=False, **self.cochlea, **self.common,
        ))

    def test_synapses(self):
        options = dict(max_dist=3.0, radius=3, id_offset=0, filter_ihc_components=[1], use_syn_ids=False,
                       points=False)
        # A MoBIE default table names the export after its source.
        synapse_table = self.table(os.path.join("M_X", "tables", "synapse_v3_ihc_v11", "default.tsv"), spot_id=[1])
        called = self.call(["synapses"] + self.path_argv + ["--synapse_table", synapse_table, "--points"])
        kwargs = called["export_synapses"].call_args.kwargs
        self.assertEqual(kwargs["name"], "synapse_v3_ihc_v11")
        self.assertIsNone(kwargs["ihc_table"])
        self.assertTrue(kwargs["points"])

        other_table = self.table("spots.tsv", spot_id=[1])
        called = self.call(["synapses"] + self.path_argv + ["--synapse_table", other_table])
        self.assertEqual(called["export_synapses"].call_args.kwargs["name"], "spots")

        self.check(["synapses"] + self.cochlea_argv + ["--id_offset", "1"], "export_synapses_cochlea", dict(
            synapse_name="synapse_v3_ihc_v11", reference_ihcs="IHC_v11", **self.cochlea,
            **{**options, "id_offset": 1}, **self.common,
        ))

    def test_frequency(self):
        table = self.table("sgn.tsv", label_id=[1])
        called = self.call(["frequency"] + self.path_argv + ["--table", table, "--colormap", "viridis"])
        self.assertEqual(called["export_frequency"].call_args.kwargs["colormap"], "viridis")

        self.check(["frequency"] + self.cochlea_argv + ["--segmentation", "SGN_v2"], "export_frequency_cochlea", dict(
            segmentation="SGN_v2", components=[1], colormap=None, **self.cochlea, **self.common,
        ))

    def test_rejected_options(self):
        table = self.table("t.tsv", label_id=[1])
        for argv in (
            [],
            ["lower_resolution", "-o", "out", "-s", "2"],
            ["lower_resolution"] + self.path_argv + ["-c", "M_X"],
            # options of the other input mode
            ["lower_resolution"] + self.path_argv + ["--channels", "PV"],
            ["lower_resolution"] + self.path_argv + ["--mobie_dir", "mobie"],
            ["lower_resolution"] + self.cochlea_argv + ["-k", "s1"],
            ["lower_resolution"] + self.cochlea_argv + ["--sgn_table", table],
            ["synapses"] + self.cochlea_argv + ["--synapse_table", table],
            ["frequency"] + self.path_argv + ["--table", table, "--segmentation", "SGN_v2"],
            # options that the mode requires
            ["marker"] + self.path_argv,
            ["marker"] + self.cochlea_argv,
            ["subtypes"] + self.path_argv + ["--table", table],
            ["synapses"] + self.path_argv,
            ["frequency"] + self.cochlea_argv,
            # options that require each other
            ["lower_resolution"] + self.path_argv + ["--filter_by_components", "1"],
            ["lower_resolution"] + self.path_argv + ["--table", table],
            ["lower_resolution"] + self.path_argv + ["--filter_dilation_iterations", "3"],
            ["synapses"] + self.path_argv + ["--synapse_table", table, "--filter_ihc_components", "2"],
            ["lower_resolution"] + self.path_argv + ["--crop_center", "1", "2", "3"],
            ["lower_resolution"] + self.path_argv + ["--axis", "2"],
            ["lower_resolution"] + self.path_argv + ["--roi_halo", "4", "4", "4"],
            ["lower_resolution"] + self.path_argv + ["--suffix", "apex"],
            # invalid values
            ["lower_resolution", "-i", "in.zarr", "-o", "out", "-s", "2", "-1"],
            ["lower_resolution"] + self.path_argv + ["-v", "0.38", "3.0"],
            ["subtypes"] + self.path_argv + ["--table", table, "--stains", "CR", "Ntng1", "Calb1"],
            # a cochlea that is not in the MoBIE project, without '--s3'
            ["lower_resolution", "-c", "M_Y", "--mobie_dir", self.mobie_dir, "-o", "out", "-s", "2"],
        ):
            with self.subTest(argv=argv):
                self.check_rejected(argv)


class TestExportDataEndToEnd(unittest.TestCase):

    def test_lower_resolution(self):
        import numpy as np
        import tifffile
        import zarr

        executable = shutil.which("flamingo_tools.export_data")
        self.assertIsNotNone(executable, "The console script flamingo_tools.export_data is not installed.")

        with TemporaryDirectory() as folder:
            input_path = os.path.join(folder, "PV.zarr")
            data = np.arange(8 * 8 * 8, dtype="uint16").reshape(8, 8, 8)
            zarr.open(input_path, mode="w").create_array("s0", data=data)

            output_folder = os.path.join(folder, "out")
            cmd = [executable, "lower_resolution", "-i", input_path, "-o", output_folder, "-s", "1",
                   "--max_intensity", "300"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(
                result.returncode, 0, msg=f"CLI failed.\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
            )
            exported = tifffile.imread(os.path.join(output_folder, "scale1", "PV.tif"))

        self.assertEqual(exported.shape, (4, 4, 4))
        self.assertEqual(exported.max(), 300)


if __name__ == "__main__":
    unittest.main()
