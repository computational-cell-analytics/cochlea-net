import os
import tempfile
import unittest

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

            detections = synapse_detection_from_prediction(prediction_path, detection_path, threshold=0.5)
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
            reloaded = synapse_detection_from_prediction(prediction_path, detection_path, threshold=0.5)
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


if __name__ == "__main__":
    unittest.main()
