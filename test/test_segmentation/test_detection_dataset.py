import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import torch
import zarr
from skimage.filters import gaussian
from torch.nn.functional import mse_loss

from flamingo_tools.synapse_detection.detection_dataset import (
    CsvHeatmapFlowTransform,
    CsvHeatmapTransform,
    DetectionDataset,
    MinPointSampler,
)
from flamingo_tools.synapse_detection.training import DetectionLoss, _samples_per_dataset

try:
    import spotiflow  # noqa
    _HAVE_SPOTIFLOW = True
except ImportError:
    _HAVE_SPOTIFLOW = False


def _v3_reference_heatmap(label_path, bb, sigma):
    """Reimplementation of the v3 label creation, which CsvHeatmapTransform must reproduce."""
    shape = tuple(s.stop - s.start for s in bb)
    points = pd.read_csv(label_path)
    coords = [points[f"axis-{axis}"].to_numpy(copy=True) for axis in range(3)]
    for coord, bb_axis in zip(coords, bb):
        coord -= bb_axis.start

    mask = np.logical_and.reduce([
        np.logical_and(coord >= 0, coord < size) for coord, size in zip(coords, shape)
    ])
    coords = tuple(
        np.clip(np.round(coord[mask]).astype("int"), 0, size - 1)
        for coord, size in zip(coords, shape)
    )

    labels = np.zeros(shape, dtype="float32")
    labels[coords] = 1
    labels = gaussian(labels, sigma)
    labels /= (labels.max() + 1e-7)
    labels *= 4
    return labels, int(mask.sum())


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

            # The v3 code normalized with a hard-coded 1e-7, so use the same eps here.
            labels = CsvHeatmapTransform(sigma=1, eps=1e-7)(label_path, self.shape, self.bb, self.bb)

            self.assertEqual(labels.shape, (1, *self.patch_shape))
            self.assertEqual(labels.dtype, np.float32)
            self.assertAlmostEqual(float(labels.max()), 4.0, places=4)
            self.assertGreaterEqual(float(labels.min()), 0.0)

            # The transform must reproduce the label creation path of the v3 training.
            expected, n_points = _v3_reference_heatmap(label_path, self.bb, sigma=1)
            self.assertGreater(n_points, 1)
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

    def test_dataset_default_label_transform(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            ds = DetectionDataset(
                raw_path=raw_path, raw_key="raw", label_path=label_path,
                patch_shape=self.patch_shape, sigma=1, n_samples=1,
            )

            self.assertIs(type(ds.label_transform), CsvHeatmapTransform)
            self.assertEqual(ds.halo, 0)

            raw, labels = ds[0]
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

    def _seeded_dataset(self, raw_path, label_path, patch_seed, sampler=None):
        return DetectionDataset(
            raw_path=raw_path, raw_key="raw", label_path=label_path,
            patch_shape=self.patch_shape, label_transform=CsvHeatmapTransform(sigma=1, eps=1e-5),
            sampler=sampler, n_samples=4, patch_seed=patch_seed,
        )

    def test_patch_seed_fixes_the_patches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            ds = self._seeded_dataset(raw_path, label_path, patch_seed=42)

            first = [ds[index] for index in range(len(ds))]
            second = [ds[index] for index in range(len(ds))]
            for (raw_a, labels_a), (raw_b, labels_b) in zip(first, second):
                self.assertTrue(torch.equal(raw_a, raw_b))
                self.assertTrue(torch.equal(labels_a, labels_b))

            # A fixed patch per index, not one fixed patch for the whole dataset.
            self.assertFalse(torch.equal(first[0][0], first[1][0]))

    def test_patch_seed_is_reproducible_across_datasets(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            same = self._seeded_dataset(raw_path, label_path, patch_seed=42)
            other = self._seeded_dataset(raw_path, label_path, patch_seed=42)
            different = self._seeded_dataset(raw_path, label_path, patch_seed=7)

            self.assertTrue(torch.equal(same[0][0], other[0][0]))
            self.assertFalse(torch.equal(same[0][0], different[0][0]))

    def test_without_patch_seed_the_patches_are_redrawn(self):
        # Training must keep seeing a new patch on every access.
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            ds = self._seeded_dataset(raw_path, label_path, patch_seed=None)

            draws = [ds[0][0] for _ in range(4)]
            self.assertTrue(any(not torch.equal(draws[0], other) for other in draws[1:]))

    def test_patch_seed_still_honours_the_sampler(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            ds = self._seeded_dataset(
                raw_path, label_path, patch_seed=42,
                sampler=MinPointSampler(min_points=10 ** 6, p_reject=1.0),
            )
            ds.max_sampling_attempts = 3

            with self.assertRaises(RuntimeError):
                ds[0]

    def test_sampler_rejection_is_deterministic_with_a_seed(self):
        # An empty heatmap never meets min_points, so every call reaches the rejection draw.
        sampler = MinPointSampler(min_points=1, p_reject=0.5)
        raw = np.zeros(self.patch_shape, dtype="float32")
        labels = np.zeros((1, *self.patch_shape), dtype="float32")

        # The two argument call still works, and draws from the global generator.
        self.assertIn(sampler(raw, labels), (True, False))

        decisions = [sampler(raw, labels, np.random.default_rng((42, i))) for i in range(8)]
        repeated = [sampler(raw, labels, np.random.default_rng((42, i))) for i in range(8)]
        self.assertEqual(decisions, repeated)
        self.assertEqual(set(decisions), {True, False})

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

    def _single_point_labels(self, tmp_dir, point, mask_radius):
        label_path = os.path.join(tmp_dir, "single.csv")
        pd.DataFrame([point], columns=["axis-0", "axis-1", "axis-2"]).to_csv(label_path, index=False)
        transform = CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=mask_radius)
        return transform(label_path, self.shape, self.bb, self.bb)

    def test_heatmap_transform_with_mask(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, label_path = self._create_data(tmp_dir)

            unmasked = CsvHeatmapTransform(sigma=1, eps=1e-5)(label_path, self.shape, self.bb, self.bb)
            labels = CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=5)(
                label_path, self.shape, self.bb, self.bb
            )

            self.assertEqual(labels.shape, (2, *self.patch_shape))
            self.assertEqual(labels.dtype, np.float32)
            # The mask must not change the heatmap it is appended to.
            self.assertTrue(np.array_equal(labels[0], unmasked[0]))

            mask = labels[1]
            self.assertEqual(sorted(np.unique(mask)), [0.0, 1.0])
            self.assertTrue(np.all(mask[labels[0] > 0] == 1))
            # 40 points with cubes of 11 ** 3 voxels cover at most 10.6% of the patch.
            self.assertLess(mask.mean(), 0.15)

    def test_mask_covers_cube_around_point(self):
        radius = 4
        point = (20.0, 50.0, 60.0)
        with tempfile.TemporaryDirectory() as tmp_dir:
            mask = self._single_point_labels(tmp_dir, point, radius)[1]

            self.assertEqual(int(mask.sum()), (2 * radius + 1) ** 3)
            cube = tuple(
                slice(int(coord) - bb_axis.start - radius, int(coord) - bb_axis.start + radius + 1)
                for coord, bb_axis in zip(point, self.bb)
            )
            self.assertTrue(np.all(mask[cube] == 1))

    def test_mask_radius_raises_the_halo(self):
        # The class attribute stays the default; only the instance is raised.
        self.assertEqual(CsvHeatmapTransform(sigma=1).halo, 0)
        self.assertEqual(CsvHeatmapTransform(sigma=1, mask_radius=16).halo, 16)
        self.assertEqual(CsvHeatmapTransform.halo, 0)

    def test_mask_and_target_use_the_same_points(self):
        """An annotation in front of the patch must reach both the mask and the heatmap.

        If only the mask saw it, the cube around a real synapse would be supervised as background,
        which is the opposite of what the mask is for.
        """
        radius = 6
        transform = CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=radius)
        self.assertEqual(transform.halo, radius)

        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = os.path.join(tmp_dir, "border.csv")
            point = (self.bb[0].start - 3.0, 50.0, 60.0)
            pd.DataFrame([point], columns=["axis-0", "axis-1", "axis-2"]).to_csv(label_path, index=False)

            # DetectionDataset expands the patch by the halo, calls the transform and crops back.
            bb_for_loading = tuple(slice(axis.start - radius, axis.stop + radius) for axis in self.bb)
            labels = transform(label_path, self.shape, bb_for_loading, bb_for_loading)
            crop = (slice(None),) + tuple(slice(radius, radius + size) for size in self.patch_shape)
            heatmap, mask = labels[crop]

            self.assertTrue(mask[:4].any())
            self.assertGreater(float(heatmap[:4].max()), 0.0)
            self.assertTrue(np.all(mask[heatmap > 0] == 1))

    def test_dataset_with_mask(self):
        radius = 6
        with tempfile.TemporaryDirectory() as tmp_dir:
            raw_path, label_path = self._create_data(tmp_dir)
            ds = self._make_dataset(
                raw_path, label_path, CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=radius)
            )

            self.assertEqual(ds.halo, radius)
            for index in range(len(ds)):
                raw, labels = ds[index]
                self.assertEqual(tuple(raw.shape), (1, *self.patch_shape))
                self.assertEqual(tuple(labels.shape), (2, *self.patch_shape))
                self.assertTrue(bool((labels[1][labels[0] > 0] == 1).all()))

    def test_sampler_with_masked_target(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, label_path = self._create_data(tmp_dir)
            labels = CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=5)(
                label_path, self.shape, self.bb, self.bb
            )
            raw = np.zeros(self.patch_shape, dtype="float32")

            self.assertEqual(labels.shape[0], 2)
            self.assertTrue(MinPointSampler(min_points=0, p_reject=1.0)(raw, labels))
            self.assertFalse(MinPointSampler(min_points=1000, p_reject=1.0)(raw, labels))

    def test_mask_without_points(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            label_path = os.path.join(tmp_dir, "empty.csv")
            pd.DataFrame(columns=["axis-0", "axis-1", "axis-2"]).to_csv(label_path, index=False)

            labels = CsvHeatmapTransform(sigma=1, eps=1e-5, mask_radius=5)(
                label_path, self.shape, self.bb, self.bb
            )

            self.assertEqual(labels.shape, (2, *self.patch_shape))
            self.assertFalse(labels.any())

    def test_flow_transform_with_mask(self):
        module = "flamingo_tools.synapse_detection.detection_dataset"

        def _fake_points_to_flow3d(points, shape):
            return np.zeros((*shape, 4), dtype="float32")

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, label_path = self._create_data(tmp_dir)

            with mock.patch(f"{module}._spotiflow_available", True), \
                 mock.patch(f"{module}.points_to_flow3d", _fake_points_to_flow3d, create=True):
                transform = CsvHeatmapFlowTransform(sigma=1, eps=1e-5, mask_radius=5)
                # The flow halo already exceeds this mask radius.
                self.assertEqual(transform.halo, 10)
                self.assertEqual(CsvHeatmapFlowTransform(sigma=1, mask_radius=16).halo, 16)
                labels = transform(label_path, self.shape, self.bb, self.bb)

            self.assertEqual(labels.shape, (6, *self.patch_shape))
            self.assertTrue(np.all(labels[5][labels[0] > 0] == 1))

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


class TestDetectionLoss(unittest.TestCase):
    shape = (2, 4, 8, 8)

    def setUp(self):
        torch.manual_seed(0)

    def _sample(self, n_channels):
        return torch.rand(self.shape[0], n_channels, *self.shape[1:])

    def test_heatmap_only_matches_mse(self):
        prediction, target = self._sample(1), self._sample(1)
        # A single output channel must not touch the empty flow slice, which would give nan.
        self.assertAlmostEqual(
            float(DetectionLoss()(prediction, target)), float(mse_loss(prediction, target)), places=6
        )

    def test_flow_weighting(self):
        prediction, target = self._sample(5), self._sample(5)
        expected = (
            mse_loss(prediction[:, :1], target[:, :1])
            + 0.1 * mse_loss(prediction[:, 1:], target[:, 1:])
        )
        self.assertAlmostEqual(
            float(DetectionLoss(flow_weight=0.1)(prediction, target)), float(expected), places=6
        )

    def test_full_mask_matches_unmasked(self):
        prediction, target = self._sample(1), self._sample(1)
        masked_target = torch.cat([target, torch.ones_like(target)], dim=1)
        self.assertAlmostEqual(
            float(DetectionLoss(masked=True)(prediction, masked_target)),
            float(DetectionLoss()(prediction, target)), places=6,
        )

    def test_mask_restricts_the_loss(self):
        prediction, target = self._sample(1), self._sample(1)
        mask = torch.zeros_like(target)
        mask[:, :, :2] = 1

        loss = DetectionLoss(masked=True)(prediction, torch.cat([target, mask], dim=1))
        self.assertAlmostEqual(
            float(loss), float(mse_loss(prediction[:, :, :2], target[:, :, :2])), places=6
        )

        # Changing the target outside of the mask must not change the loss.
        target[:, :, 2:] += 10
        self.assertAlmostEqual(
            float(DetectionLoss(masked=True)(prediction, torch.cat([target, mask], dim=1))),
            float(loss), places=6,
        )

    def test_empty_mask(self):
        prediction = self._sample(1)
        target = torch.cat([self._sample(1), torch.zeros(self.shape[0], 1, *self.shape[1:])], dim=1)

        loss = DetectionLoss(masked=True)(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(loss), 0.0)

    def test_channel_mismatch_raises(self):
        prediction = self._sample(1)
        # A mask channel that the loss is not told about would otherwise be ignored silently.
        with self.assertRaises(ValueError):
            DetectionLoss()(prediction, self._sample(2))
        with self.assertRaises(ValueError):
            DetectionLoss(masked=True)(prediction, self._sample(1))

    def test_masked_flow(self):
        prediction, target = self._sample(5), self._sample(5)
        mask = torch.zeros(self.shape[0], 1, *self.shape[1:])
        mask[:, :, :2] = 1

        expected = (
            mse_loss(prediction[:, :1, :2], target[:, :1, :2])
            + 0.1 * mse_loss(prediction[:, 1:, :2], target[:, 1:, :2])
        )
        loss = DetectionLoss(flow_weight=0.1, masked=True)(prediction, torch.cat([target, mask], dim=1))
        self.assertAlmostEqual(float(loss), float(expected), places=6)


class TestSamplesPerDataset(unittest.TestCase):
    def test_distribution(self):
        self.assertEqual(_samples_per_dataset(None, 3), [None, None, None])

        for n_samples, n_datasets in [(3200, 26), (160, 3), (7, 3), (3, 3)]:
            split = _samples_per_dataset(n_samples, n_datasets)
            self.assertEqual(len(split), n_datasets)
            self.assertEqual(sum(split), n_samples)
            self.assertLessEqual(max(split) - min(split), 1)

    def test_fewer_samples_than_datasets(self):
        # The trailing crops get no sample at all and drop out of the epoch.
        self.assertEqual(_samples_per_dataset(2, 4), [1, 1, 0, 0])


if __name__ == "__main__":
    unittest.main()
