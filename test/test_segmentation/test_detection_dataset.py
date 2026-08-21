import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import zarr

from flamingo_tools.synapse_detection.detection_dataset import (
    CsvHeatmapFlowTransform,
    CsvHeatmapTransform,
    DetectionDataset,
    MinPointSampler,
    load_labels,
    process_labels,
)

try:
    import spotiflow  # noqa
    _HAVE_SPOTIFLOW = True
except ImportError:
    _HAVE_SPOTIFLOW = False


class TestDetectionDataset(unittest.TestCase):
    shape = (64, 160, 160)
    patch_shape = [40, 112, 112]
    n_points = 40
    bb = tuple(slice(start, start + psh) for start, psh in zip((8, 16, 24), patch_shape))

    def _create_data(self, tmp_dir):
        rng = np.random.default_rng(0)

        raw_path = os.path.join(tmp_dir, "image.zarr")
        f = zarr.open(raw_path, mode="w")
        raw = rng.integers(0, 4000, self.shape).astype("uint16")
        f.create_array("raw", shape=self.shape, chunks=(32, 64, 64), dtype="uint16")[:] = raw

        points = np.stack([rng.uniform(0, sh - 1, self.n_points) for sh in self.shape], axis=1)
        label_path = os.path.join(tmp_dir, "image.csv")
        pd.DataFrame(points, columns=["axis-0", "axis-1", "axis-2"]).to_csv(label_path, index=False)

        return raw_path, label_path

    def _make_dataset(self, raw_path, label_path, label_transform, sampler=None, n_samples=4):
        return DetectionDataset(
            raw_path=raw_path, raw_key="raw", label_path=label_path,
            patch_shape=self.patch_shape, label_transform=label_transform,
            sampler=sampler, n_samples=n_samples,
        )

    def test_declared_halo(self):
        self.assertEqual(CsvHeatmapTransform.halo, 0)
        self.assertEqual(CsvHeatmapFlowTransform.halo, 10)

    def test_heatmap_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, label_path = self._create_data(tmp_dir)

            # process_labels normalizes with a hard-coded 1e-7, so use the same eps here.
            labels = CsvHeatmapTransform(sigma=1, eps=1e-7)(label_path, self.shape, self.bb, self.bb)

            self.assertEqual(labels.shape, (1, *self.patch_shape))
            self.assertEqual(labels.dtype, np.float32)
            self.assertAlmostEqual(float(labels.max()), 4.0, places=4)
            self.assertGreaterEqual(float(labels.min()), 0.0)

            # The transform must reproduce the label creation path of the v3 training.
            coords, n_points = load_labels(label_path, self.shape, self.bb)
            self.assertGreater(n_points, 1)
            expected = process_labels(coords, self.shape, sigma=1, eps=1e-7, bb=self.bb)
            self.assertTrue(np.allclose(labels[0], expected, atol=1e-6))

    def test_heatmap_transform_without_points(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = os.path.join(tmp_dir, "empty.csv")
            pd.DataFrame(columns=["axis-0", "axis-1", "axis-2"]).to_csv(label_path, index=False)

            labels = CsvHeatmapTransform(sigma=1, eps=1e-5)(label_path, self.shape, self.bb, self.bb)

            self.assertEqual(labels.shape, (1, *self.patch_shape))
            self.assertFalse(labels.any())

    def test_dataset_without_flow(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            ds = self._make_dataset(raw_path, label_path, CsvHeatmapTransform(sigma=1, eps=1e-5))

            self.assertEqual(ds.halo, 0)
            self.assertEqual(len(ds), 4)
            for index in range(len(ds)):
                raw, labels = ds[index]
                self.assertEqual(tuple(raw.shape), (1, *self.patch_shape))
                self.assertEqual(tuple(labels.shape), (1, *self.patch_shape))

    def test_dataset_halo_from_label_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)

            ds = self._make_dataset(raw_path, label_path, CsvHeatmapTransform(sigma=1, eps=1e-5))
            self.assertEqual(ds.halo, 0)

            # A label transform that declares no halo keeps the flow-compatible default.
            ds = self._make_dataset(raw_path, label_path, object())
            self.assertEqual(ds.halo, 10)

    def test_sampler_with_single_channel_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            labels = CsvHeatmapTransform(sigma=1, eps=1e-5)(label_path, self.shape, self.bb, self.bb)
            raw = np.zeros(self.patch_shape, dtype="float32")

            self.assertTrue(MinPointSampler(min_points=1, p_reject=1.0)(raw, labels))
            self.assertFalse(MinPointSampler(min_points=1000, p_reject=1.0)(raw, labels))

            # The sampler must not reject every patch when it is used in the dataset.
            ds = self._make_dataset(
                raw_path, label_path, CsvHeatmapTransform(sigma=1, eps=1e-5),
                sampler=MinPointSampler(min_points=1, p_reject=0.8),
            )
            _, labels = ds[0]
            self.assertEqual(tuple(labels.shape), (1, *self.patch_shape))

    def test_flow_transform_with_stub(self):
        """Exercise the flow branch where spotiflow is not installed."""
        module = "flamingo_tools.synapse_detection.detection_dataset"
        recorded = {}

        def _fake_points_to_flow3d(points, shape):
            recorded["points"], recorded["shape"] = points, shape
            flow = np.zeros((*shape, 4), dtype="float32")
            for channel in range(4):
                flow[..., channel] = channel + 1
            return flow

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, label_path = self._create_data(tmp_dir)

            with mock.patch(f"{module}._spotiflow_available", True), \
                 mock.patch(f"{module}.points_to_flow3d", _fake_points_to_flow3d, create=True):
                labels = CsvHeatmapFlowTransform(sigma=1, eps=1e-5)(
                    label_path, self.shape, self.bb, self.bb
                )

            heatmap = CsvHeatmapTransform(sigma=1, eps=1e-5)(
                label_path, self.shape, self.bb, self.bb
            )

            self.assertEqual(labels.shape, (5, *self.patch_shape))
            self.assertTrue(np.array_equal(labels[0], heatmap[0]))
            for channel in range(4):
                self.assertTrue(np.all(labels[channel + 1] == channel + 1))

            # The flow receives the points in patch-local, unrounded coordinates.
            self.assertEqual(recorded["shape"], tuple(self.patch_shape))
            self.assertEqual(recorded["points"].shape[1], 3)
            self.assertTrue(np.all(recorded["points"] >= 0))
            self.assertTrue(np.all(recorded["points"] < np.array(self.patch_shape)))

    @unittest.skipUnless(_HAVE_SPOTIFLOW, "spotiflow is required for the flow channels")
    def test_flow_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, label_path = self._create_data(tmp_dir)

            flow_labels = CsvHeatmapFlowTransform(sigma=1, eps=1e-5)(
                label_path, self.shape, self.bb, self.bb
            )
            heatmap_labels = CsvHeatmapTransform(sigma=1, eps=1e-5)(
                label_path, self.shape, self.bb, self.bb
            )

            self.assertEqual(flow_labels.shape, (5, *self.patch_shape))
            self.assertTrue(np.array_equal(flow_labels[0], heatmap_labels[0]))


if __name__ == "__main__":
    unittest.main()
