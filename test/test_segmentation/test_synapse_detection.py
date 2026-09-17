import os
import tempfile
import unittest
import warnings

import numpy as np
import pandas as pd
import zarr


def _reference_flow_correction(pred, peak_coords):
    """Per-peak reference implementation of the stereographic back-projection."""
    from flamingo_tools.segmentation.synapse_detection import _HEATMAP_FLOW_SIGMA

    s = _HEATMAP_FLOW_SIGMA
    adjusted = np.empty((len(peak_coords), 3), dtype=float)
    for i, (z, y, x) in enumerate(peak_coords):
        zi, yi, xi = int(z), int(y), int(x)
        w = float(pred[1, zi, yi, xi])
        vz = float(pred[2, zi, yi, xi])
        vy = float(pred[3, zi, yi, xi])
        vx = float(pred[4, zi, yi, xi])
        denom = 1.0 + w + 1e-8
        adjusted[i] = [z + s * vz / denom, y + s * vy / denom, x + s * vx / denom]
    return adjusted


class TestFlowCorrection(unittest.TestCase):
    shape = (5, 32, 64, 64)
    chunks = (1, 16, 32, 32)

    def _create_prediction(self, tmp_dir):
        rng = np.random.default_rng(0)
        data = np.zeros(self.shape, dtype="float32")
        peak_coords = np.stack(
            [rng.integers(0, sh, 200) for sh in self.shape[1:]], axis=1
        )
        # Duplicate coordinates would make the reference and the grouped result agree
        # trivially, so keep only unique peaks.
        peak_coords = np.unique(peak_coords, axis=0)
        data[0][tuple(peak_coords.T)] = rng.uniform(0.6, 4.0, len(peak_coords))
        for channel in range(1, 5):
            data[channel] = rng.uniform(-1, 1, self.shape[1:])

        path = os.path.join(tmp_dir, "predictions.zarr")
        f = zarr.open(path, mode="w")
        f.create_array("prediction", shape=self.shape, chunks=self.chunks, dtype="float32")[:] = data
        return zarr.open(path, mode="r")["prediction"], peak_coords, data

    def test_matches_reference(self):
        from flamingo_tools.segmentation.synapse_detection import _apply_flow_correction

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred, peak_coords, _ = self._create_prediction(tmp_dir)
            # The peaks must be spread over several chunks for the grouping to be exercised.
            n_chunks = len(np.unique(peak_coords // np.array(self.chunks[-3:]), axis=0))
            self.assertGreater(n_chunks, 1)

            expected = _reference_flow_correction(pred, peak_coords)
            for n_threads in (1, 4):
                result = _apply_flow_correction(pred, peak_coords, n_threads)
                np.testing.assert_array_equal(result, expected)

    def test_numpy_input(self):
        from flamingo_tools.segmentation.synapse_detection import _apply_flow_correction

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, peak_coords, data = self._create_prediction(tmp_dir)

            expected = _reference_flow_correction(data, peak_coords)
            result = _apply_flow_correction(data, peak_coords, 2)
            np.testing.assert_array_equal(result, expected)

    def test_single_peak(self):
        from flamingo_tools.segmentation.synapse_detection import _apply_flow_correction

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred, peak_coords, _ = self._create_prediction(tmp_dir)
            peak_coords = peak_coords[:1]

            expected = _reference_flow_correction(pred, peak_coords)
            result = _apply_flow_correction(pred, peak_coords, 2)
            np.testing.assert_array_equal(result, expected)

    def test_no_peaks(self):
        from flamingo_tools.segmentation.synapse_detection import _flow_corrected_detections

        with tempfile.TemporaryDirectory() as tmp_dir:
            pred, _, _ = self._create_prediction(tmp_dir)
            result, raw_result = _flow_corrected_detections(
                pred, min_distance=2, threshold_abs=100.0,
                block_shape=self.chunks[-3:], n_threads=2,
            )
            self.assertEqual(result.shape, (0, 3))
            self.assertEqual(raw_result.shape, (0, 3))

    def test_detection_from_prediction(self):
        from flamingo_tools.segmentation.synapse_detection import synapse_detection_from_prediction

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._create_prediction(tmp_dir)
            prediction_path = os.path.join(tmp_dir, "predictions.zarr")
            detection_path = os.path.join(tmp_dir, "synapse_detection.tsv")
            no_flow_path = os.path.join(tmp_dir, "synapse_detection_no-flow.tsv")

            detections = synapse_detection_from_prediction(prediction_path, detection_path, threshold=0.5,
                                                           save_no_flow=True)
            self.assertTrue(os.path.exists(detection_path))
            self.assertGreater(len(detections), 0)
            self.assertEqual(list(detections.columns), ["spot_id", "x", "y", "z"])

            # The no-flow sibling file must be written by default, in the same format.
            self.assertTrue(os.path.exists(no_flow_path))
            no_flow_detections = pd.read_csv(no_flow_path, sep="\t")
            self.assertEqual(list(no_flow_detections.columns), ["spot_id", "x", "y", "z"])
            self.assertEqual(len(no_flow_detections), len(detections))
            # Flow correction shifts coordinates, so the two outputs should differ.
            self.assertFalse(np.allclose(no_flow_detections.values, detections.values))

            # The second call must load the result that was written before.
            reloaded = synapse_detection_from_prediction(prediction_path, detection_path, threshold=0.5,
                                                         save_no_flow=True)
            np.testing.assert_allclose(reloaded.values, detections.values)

    def test_detection_from_prediction_no_flow_disabled(self):
        from flamingo_tools.segmentation.synapse_detection import synapse_detection_from_prediction

        with tempfile.TemporaryDirectory() as tmp_dir:
            self._create_prediction(tmp_dir)
            prediction_path = os.path.join(tmp_dir, "predictions.zarr")
            detection_path = os.path.join(tmp_dir, "synapse_detection.tsv")
            no_flow_path = os.path.join(tmp_dir, "synapse_detection_no-flow.tsv")

            synapse_detection_from_prediction(
                prediction_path, detection_path, threshold=0.5, save_no_flow=False,
            )
            self.assertTrue(os.path.exists(detection_path))
            self.assertFalse(os.path.exists(no_flow_path))


class TestDetectionBlockShape(unittest.TestCase):
    def test_block_shape(self):
        from flamingo_tools.segmentation.synapse_detection import (
            _DETECTION_BLOCK_VOXELS, _detection_block_shape,
        )

        for chunks in [(64, 256, 256), (32, 128, 128), (128, 128, 128), (256, 256, 256)]:
            block = _detection_block_shape(chunks)
            with self.subTest(chunks=chunks):
                # elf.parallel.common.get_blocking requires chunk-aligned blocks for n_threads > 1.
                for bl, ch in zip(block, chunks):
                    self.assertEqual(bl % ch, 0)
                    self.assertGreaterEqual(bl, ch)
                self.assertLessEqual(np.prod(block), _DETECTION_BLOCK_VOXELS)

    def test_block_shape_is_larger_than_chunks(self):
        from flamingo_tools.segmentation.synapse_detection import _detection_block_shape

        chunks = (64, 256, 256)
        self.assertGreater(np.prod(_detection_block_shape(chunks)), np.prod(chunks))


class TestVoxelSize(unittest.TestCase):
    def test_normalize(self):
        from flamingo_tools.segmentation.synapse_detection import _normalize_voxel_size

        self.assertEqual(_normalize_voxel_size(0.38), (0.38, 0.38, 0.38))
        self.assertEqual(_normalize_voxel_size("0.38"), (0.38, 0.38, 0.38))
        self.assertEqual(_normalize_voxel_size([0.38]), (0.38, 0.38, 0.38))
        self.assertEqual(_normalize_voxel_size((0.76, 0.76, 3.0)), (0.76, 0.76, 3.0))
        self.assertEqual(_normalize_voxel_size("0.76,0.76,3.0"), (0.76, 0.76, 3.0))
        self.assertEqual(_normalize_voxel_size("0.76 0.76 3.0"), (0.76, 0.76, 3.0))

        with self.assertRaises(ValueError):
            _normalize_voxel_size((0.38, 0.38))


class TestSynapseSlurmWorkflow(unittest.TestCase):
    """The three-stage slurm workflow must reproduce the single-job result."""

    shape = (32, 64, 64)
    prediction_instances = 3

    def setUp(self):
        self._task_id = os.environ.pop("SLURM_ARRAY_TASK_ID", None)

    def tearDown(self):
        os.environ.pop("SLURM_ARRAY_TASK_ID", None)
        if self._task_id is not None:
            os.environ["SLURM_ARRAY_TASK_ID"] = self._task_id

    def _create_input(self, tmp_dir):
        import torch
        import z5py
        from torch_em.model import UNet3d

        torch.manual_seed(0)
        model = UNet3d(in_channels=1, out_channels=5, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)

        data_path = os.path.join(tmp_dir, "data.n5")
        rng = np.random.default_rng(0)
        with z5py.File(data_path, "a") as f:
            f.create_dataset("data", data=rng.integers(0, 255, size=self.shape), chunks=(16, 16, 16))
        return data_path, "data", model_path

    def test_array_prediction_matches_single_job(self):
        from elf.io import open_file
        from flamingo_tools.segmentation.synapse_detection import (
            marker_detection, run_synapse_prediction_preprocess_slurm, run_synapse_prediction_slurm,
        )

        block_shape, halo = (16, 16, 16), (4, 4, 4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key, model_path = self._create_input(tmp_dir)

            single_folder = os.path.join(tmp_dir, "single")
            os.makedirs(single_folder)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                marker_detection(
                    data_path, data_key, None, single_folder, model_path,
                    block_shape=block_shape, halo=halo,
                )

            array_folder = os.path.join(tmp_dir, "array")
            run_synapse_prediction_preprocess_slurm(data_path, array_folder, input_key=data_key)
            self.assertTrue(os.path.isfile(os.path.join(array_folder, "mean_std.json")))

            for task_id in range(self.prediction_instances):
                os.environ["SLURM_ARRAY_TASK_ID"] = str(task_id)
                run_synapse_prediction_slurm(
                    data_path, array_folder, model_path, input_key=data_key,
                    block_shape=block_shape, halo=halo,
                    prediction_instances=self.prediction_instances,
                )

            with open_file(os.path.join(single_folder, "predictions.zarr"), "r") as f:
                expected = f["prediction"][:]
            with open_file(os.path.join(array_folder, "predictions.zarr"), "r") as f:
                actual = f["prediction"][:]

            self.assertEqual(expected.shape, (5,) + self.shape)
            self.assertGreater(np.abs(actual).sum(), 0)
            # Bit-identical: the tasks share the cached mean/std and cover disjoint blocks.
            self.assertTrue(np.array_equal(expected, actual))

    def test_requires_array_task_id(self):
        from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key, model_path = self._create_input(tmp_dir)
            with self.assertRaises(ValueError):
                run_synapse_prediction_slurm(
                    data_path, os.path.join(tmp_dir, "out"), model_path, input_key=data_key,
                    prediction_instances=self.prediction_instances,
                )

    def test_requires_preprocessing(self):
        """Without mean_std.json the tasks would normalize differently, so this must fail loudly."""
        from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key, model_path = self._create_input(tmp_dir)
            os.environ["SLURM_ARRAY_TASK_ID"] = "0"
            with self.assertRaises(ValueError):
                run_synapse_prediction_slurm(
                    data_path, os.path.join(tmp_dir, "out"), model_path, input_key=data_key,
                    prediction_instances=self.prediction_instances,
                )

    def test_rejects_task_id_beyond_instances(self):
        from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key, model_path = self._create_input(tmp_dir)
            os.environ["SLURM_ARRAY_TASK_ID"] = str(self.prediction_instances)
            with self.assertRaises(ValueError):
                run_synapse_prediction_slurm(
                    data_path, os.path.join(tmp_dir, "out"), model_path, input_key=data_key,
                    prediction_instances=self.prediction_instances,
                )


class TestMaskKeys(unittest.TestCase):
    """The image data and the IHC segmentation must use independent keys."""

    shape = (32, 64, 64)

    def _create_input(self, tmp_dir):
        import torch
        import z5py
        from torch_em.model import UNet3d

        torch.manual_seed(0)
        model = UNet3d(in_channels=1, out_channels=5, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)

        # The image data deliberately uses an n5 key that does not exist in the mask,
        # so reusing it for the mask would raise.
        data_path = os.path.join(tmp_dir, "data.n5")
        rng = np.random.default_rng(0)
        with z5py.File(data_path, "a") as f:
            f.create_dataset("setup2/timepoint0/s0", data=rng.integers(0, 255, size=self.shape), chunks=(16, 16, 16))
        return data_path, "setup2/timepoint0/s0", model_path

    def _create_mask(self, tmp_dir):
        """An IHC segmentation with an s0 and a 4x downscaled s4 level."""
        full = np.zeros(self.shape, dtype="uint16")
        full[12:20, 24:40, 24:40] = 7
        low = full[::4, ::4, ::4].copy()

        path = os.path.join(tmp_dir, "ihc.zarr")
        f = zarr.open(path, mode="w")
        f.create_array("s0", data=full)
        f.create_array("s4", data=low)
        self.assertNotEqual(full.shape, low.shape)
        return path

    def test_marker_detection_uses_separate_keys(self):
        from flamingo_tools.segmentation.synapse_detection import marker_detection

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key, model_path = self._create_input(tmp_dir)
            mask_path = self._create_mask(tmp_dir)
            output_folder = os.path.join(tmp_dir, "out")

            marker_detection(
                input_path=data_path, input_key=data_key, mask_path=mask_path,
                output_folder=output_folder, model_path=model_path,
                mask_input_key="s4", mask_key="s0", max_distance=8.0,
            )

            # The mask is built from the downscaled level.
            mask = zarr.open(os.path.join(output_folder, "mask.zarr"), mode="r")["mask"]
            self.assertEqual(tuple(mask.shape), tuple(s // 4 for s in self.shape))

            # The matching uses the full-resolution level.
            filtered_path = os.path.join(output_folder, "synapse_detection_filtered.tsv")
            self.assertTrue(os.path.exists(filtered_path))
            filtered = pd.read_csv(filtered_path, sep="\t")
            for column in ("matched_ihc", "distance_to_ihc"):
                self.assertIn(column, filtered.columns)

    def test_mask_key_defaults_do_not_reuse_input_key(self):
        """Reusing input_key for the segmentation must not silently come back."""
        import inspect
        from flamingo_tools.segmentation.synapse_detection import marker_detection

        params = inspect.signature(marker_detection).parameters
        self.assertEqual(params["mask_input_key"].default, "s4")
        self.assertEqual(params["mask_key"].default, "s0")


class TestPredictionMask(unittest.TestCase):
    """An optional IHC segmentation restricts the inference to the region around the IHCs."""

    shape = (32, 64, 64)

    def _create_input(self, tmp_dir):
        import torch
        import z5py
        from torch_em.model import UNet3d

        torch.manual_seed(0)
        model = UNet3d(in_channels=1, out_channels=5, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)

        data_path = os.path.join(tmp_dir, "data.n5")
        rng = np.random.default_rng(0)
        with z5py.File(data_path, "a") as f:
            f.create_dataset("data", data=rng.integers(0, 255, size=self.shape), chunks=(16, 16, 16))
        return data_path, "data", model_path

    def _create_mask(self, tmp_dir):
        """An IHC segmentation with a full-resolution s0 and a 4x downscaled s4 level."""
        full = np.zeros(self.shape, dtype="uint16")
        full[12:20, 24:40, 24:40] = 7
        mask_path = os.path.join(tmp_dir, "ihc.zarr")
        f = zarr.open(mask_path, mode="w")
        f.create_array("s0", data=full)
        f.create_array("s4", data=full[::4, ::4, ::4].copy())
        return mask_path

    def _run(self, tmp_dir, name, mask_path, input_):
        from elf.io import open_file
        from flamingo_tools.segmentation.synapse_detection import marker_detection

        data_path, data_key, model_path = input_
        output_folder = os.path.join(tmp_dir, name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            marker_detection(
                data_path, data_key, mask_path, output_folder, model_path,
                block_shape=(16, 16, 16), halo=(4, 4, 4),
                mask_input_key="s4", mask_key="s0", max_distance=8.0,
            )
        with open_file(os.path.join(output_folder, "predictions.zarr"), "r") as f:
            return output_folder, f["prediction"][:]

    def test_mask_restricts_the_prediction(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = self._create_mask(tmp_dir)
            input_ = self._create_input(tmp_dir)
            unmasked_folder, unmasked = self._run(tmp_dir, "unmasked", None, input_)
            masked_folder, masked = self._run(tmp_dir, "masked", mask_path, input_)

            self.assertFalse(os.path.exists(os.path.join(unmasked_folder, "mask.zarr")))
            self.assertTrue(os.path.exists(os.path.join(masked_folder, "mask.zarr")))

            # The masked run leaves the blocks outside the IHC region untouched.
            self.assertGreater(np.abs(masked).sum(), 0)
            self.assertLess((np.abs(masked) > 0).sum(), (np.abs(unmasked) > 0).sum())

    def test_stale_table_does_not_skip_the_prediction(self):
        """A leftover table from a partial run must not suppress the inference.

        This is the failure that let a truncated cochlea pass as finished: the prediction was
        skipped because the table existed, and the peaks were detected in a partial volume.
        """
        import shutil

        with tempfile.TemporaryDirectory() as tmp_dir:
            mask_path = self._create_mask(tmp_dir)
            input_ = self._create_input(tmp_dir)
            output_folder, _ = self._run(tmp_dir, "run", mask_path, input_)

            detection_path = os.path.join(output_folder, "synapse_detection.tsv")
            self.assertTrue(os.path.exists(detection_path))
            stale_id = 999999
            pd.DataFrame({"spot_id": [stale_id], "x": [-1.0], "y": [-1.0], "z": [-1.0]}).to_csv(
                detection_path, index=False, sep="\t"
            )
            shutil.rmtree(os.path.join(output_folder, "predictions.zarr"))

            self._run(tmp_dir, "run", mask_path, input_)
            self.assertTrue(os.path.exists(os.path.join(output_folder, "predictions.zarr")))
            # The table is derived from the new prediction, not read back from the stale file.
            self.assertNotIn(stale_id, pd.read_csv(detection_path, sep="\t").spot_id.values)


class TestMaskFallback(unittest.TestCase):
    """Prediction on the dilated IHC segmentation is the standard, the full volume a fallback."""

    def test_detection_stage_takes_mask_key(self):
        """The full-resolution key for matching must not be named after the inference mask."""
        import inspect
        from flamingo_tools.segmentation.synapse_detection import run_synapse_detection_slurm

        params = inspect.signature(run_synapse_detection_slurm).parameters
        self.assertEqual(params["mask_key"].default, "s0")
        self.assertNotIn("mask_input_key", params)

    def test_preprocess_warns_without_segmentation(self):
        import z5py
        from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_preprocess_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path = os.path.join(tmp_dir, "data.n5")
            rng = np.random.default_rng(0)
            with z5py.File(data_path, "a") as f:
                f.create_dataset("data", data=rng.integers(0, 255, size=(16, 16, 16)), chunks=(8, 8, 8))

            output_folder = os.path.join(tmp_dir, "out")
            with self.assertWarns(UserWarning):
                run_synapse_prediction_preprocess_slurm(data_path, output_folder, input_key="data")

            self.assertFalse(os.path.exists(os.path.join(output_folder, "mask.zarr")))
            self.assertTrue(os.path.isfile(os.path.join(output_folder, "mean_std.json")))

    def test_dilation_iterations_match_the_cube(self):
        """Four iterations of a 3x3x3 structure are the previous single 9x9x9 pass."""
        from scipy.ndimage import binary_dilation
        from flamingo_tools.segmentation.synapse_detection import build_ihc_mask

        seg = np.zeros((24, 24, 24), dtype="uint16")
        seg[10:14, 10:14, 10:14] = 3
        with tempfile.TemporaryDirectory() as tmp_dir:
            seg_path = os.path.join(tmp_dir, "ihc.zarr")
            zarr.open(seg_path, mode="w").create_array("s4", data=seg)

            output_folder = os.path.join(tmp_dir, "out")
            shape = build_ihc_mask(seg_path, output_folder, mask_input_key="s4", dilation_iterations=4)
            mask = zarr.open(os.path.join(output_folder, "mask.zarr"), mode="r")["mask"][:]

        self.assertEqual(shape, seg.shape)
        expected = binary_dilation(seg, structure=np.ones((9, 9, 9))).astype("uint8")
        self.assertTrue(np.array_equal(mask, expected))

    def test_no_dilation_keeps_the_segmentation_footprint(self):
        """scipy dilates to convergence for 'iterations' < 1, which would mask the whole volume."""
        from flamingo_tools.segmentation.synapse_detection import build_ihc_mask

        seg = np.zeros((24, 24, 24), dtype="uint16")
        seg[10:14, 10:14, 10:14] = 3
        with tempfile.TemporaryDirectory() as tmp_dir:
            seg_path = os.path.join(tmp_dir, "ihc.zarr")
            zarr.open(seg_path, mode="w").create_array("s4", data=seg)

            output_folder = os.path.join(tmp_dir, "out")
            build_ihc_mask(seg_path, output_folder, mask_input_key="s4", dilation_iterations=0)
            mask = zarr.open(os.path.join(output_folder, "mask.zarr"), mode="r")["mask"][:]

        self.assertTrue(np.array_equal(mask, (seg != 0).astype("uint8")))

    def test_warns_when_the_dilation_is_below_the_matching_distance(self):
        from flamingo_tools.segmentation.synapse_detection import _check_mask_dilation

        voxel_size, max_distance = (0.38, 0.38, 0.38), 3.0
        # A mask built at the full resolution: 4 iterations are 1.5 micrometer, too tight.
        with self.assertWarns(UserWarning):
            _check_mask_dilation((64, 64, 64), (64, 64, 64), 4, voxel_size, max_distance)
        # A mask built at s4: 4 iterations are 24 micrometer.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _check_mask_dilation((64, 64, 64), (4, 4, 4), 4, voxel_size, max_distance)


class TestS3Resolution(unittest.TestCase):
    """The stages resolve the image and the IHC segmentation against the bucket independently.

    There is no S3 fixture in the suite, so 'get_s3_path' is patched and only the plumbing is
    checked: which path reaches the bucket lookup, and how often.
    """

    def _create_input(self, tmp_dir):
        import z5py

        data_path = os.path.join(tmp_dir, "data.n5")
        rng = np.random.default_rng(0)
        with z5py.File(data_path, "a") as f:
            f.create_dataset("data", data=rng.integers(0, 255, size=(16, 16, 16)), chunks=(8, 8, 8))
        return data_path, "data"

    def _patch_get_s3_path(self, resolved):
        """Patch the bucket lookup and record the paths it is asked for."""
        from unittest import mock
        import flamingo_tools.s3_utils as s3_utils

        calls = []

        def fake(input_path, bucket_name=None, service_endpoint=None, credential_file=None):
            calls.append(input_path)
            return resolved, None

        return calls, mock.patch.object(s3_utils, "get_s3_path", fake)

    def test_resolve_path_passes_through(self):
        from flamingo_tools.s3_utils import resolve_path

        self.assertIsNone(resolve_path(None, True))
        self.assertEqual(resolve_path("p", False), "p")
        self.assertEqual(resolve_path("p", None), "p")
        # The slurm entry points read their flags from the environment as strings.
        for off in ("", "  ", "0", "false", "FALSE", "no"):
            self.assertEqual(resolve_path("p", off), "p", msg=off)

    def test_preprocess_resolves_the_mask_only(self):
        """The production case: a local image with the IHC segmentation on the bucket."""
        from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_preprocess_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key = self._create_input(tmp_dir)
            seg = np.zeros((8, 8, 8), dtype="uint16")
            seg[3:5, 3:5, 3:5] = 1
            mask_path = os.path.join(tmp_dir, "ihc.zarr")
            zarr.open(mask_path, mode="w").create_array("s4", data=seg)

            calls, patch = self._patch_get_s3_path(mask_path)
            with patch:
                run_synapse_prediction_preprocess_slurm(
                    data_path, os.path.join(tmp_dir, "out"), input_key=data_key,
                    mask_path="cochlea/images/ome-zarr/IHC.ome.zarr", s3_mask="1",
                )

        self.assertEqual(calls, ["cochlea/images/ome-zarr/IHC.ome.zarr"])

    def test_preprocess_resolves_nothing_without_the_flags(self):
        from flamingo_tools.segmentation.synapse_detection import run_synapse_prediction_preprocess_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key = self._create_input(tmp_dir)
            calls, patch = self._patch_get_s3_path(None)
            with patch, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                run_synapse_prediction_preprocess_slurm(
                    data_path, os.path.join(tmp_dir, "out"), input_key=data_key,
                )

        self.assertEqual(calls, [])

    def test_detection_stage_resolves_the_mask(self):
        from unittest import mock
        from flamingo_tools.segmentation.synapse_detection import run_synapse_detection_slurm

        with tempfile.TemporaryDirectory() as tmp_dir:
            calls, patch = self._patch_get_s3_path("resolved-store")
            # Stop after the resolution; the prediction and the matching are covered elsewhere.
            with patch, mock.patch(
                "flamingo_tools.segmentation.synapse_detection.synapse_detection_from_prediction",
                side_effect=RuntimeError("stop"),
            ):
                with self.assertRaises(RuntimeError):
                    run_synapse_detection_slurm(
                        tmp_dir, mask_path="cochlea/images/ome-zarr/IHC.ome.zarr",
                        mask_key="s0", s3_mask="1",
                    )

        self.assertEqual(calls, ["cochlea/images/ome-zarr/IHC.ome.zarr"])

    def test_marker_detection_resolves_each_path_once(self):
        """The stages must be handed the stores, never asked to resolve them again."""
        from unittest import mock
        from flamingo_tools.segmentation.synapse_detection import marker_detection

        with tempfile.TemporaryDirectory() as tmp_dir:
            calls, patch = self._patch_get_s3_path("resolved-store")
            with patch, mock.patch(
                "flamingo_tools.segmentation.synapse_detection.build_ihc_mask",
                side_effect=RuntimeError("stop"),
            ):
                with self.assertRaises(RuntimeError):
                    marker_detection(
                        "cochlea/images/ome-zarr/CTBP2.ome.zarr", "s0",
                        "cochlea/images/ome-zarr/IHC.ome.zarr",
                        os.path.join(tmp_dir, "out"), "model.pt",
                        s3_input="1", s3_mask="1",
                    )

        self.assertEqual(calls, [
            "cochlea/images/ome-zarr/CTBP2.ome.zarr",
            "cochlea/images/ome-zarr/IHC.ome.zarr",
        ])


if __name__ == "__main__":
    unittest.main()
