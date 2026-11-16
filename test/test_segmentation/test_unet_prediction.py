import json
import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import torch
import z5py
from elf.io import open_file
from scipy import ndimage
from torch_em.model import UNet3d


class TestUnetPrediction(unittest.TestCase):
    shape = (64, 128, 128)

    def _create_model(self, tmp_dir):
        model = UNet3d(in_channels=1, out_channels=3, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)
        return model_path

    def _create_data(self, tmp_dir, use_tif):
        data = np.random.randint(0, 255, size=self.shape)
        if use_tif:
            path = os.path.join(tmp_dir, "data.tif")
            key = None
            imageio.imwrite(path, data)
        else:
            path = os.path.join(tmp_dir, "data.n5")
            key = "data"
            with z5py.File(path, "a") as f:
                f.create_dataset(key, data=data, chunks=(32, 32, 32))
        return path, key

    def _test_run_unet_prediction(self, use_tif, use_mask, **extra_kwargs):
        from flamingo_tools.segmentation import run_unet_prediction

        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path, input_key = self._create_data(tmp_dir, use_tif)
            model_path = self._create_model(tmp_dir)
            output_folder = os.path.join(tmp_dir, "output")
            run_unet_prediction(
                input_path, input_key, output_folder, model_path,
                scale=None, min_size=100,
                block_shape=(64, 64, 64), halo=(16, 16, 16),
                **extra_kwargs
            )

            expected_path = os.path.join(output_folder, "segmentation.zarr")
            expected_key = "segmentation"

            self.assertTrue(os.path.exists(expected_path))
            with z5py.File(expected_path, "r") as f:
                self.assertTrue(expected_key in f)
                self.assertEqual(f[expected_key].shape, self.shape)

    def test_run_unet_prediction_n5(self):
        self._test_run_unet_prediction(use_tif=False, use_mask=False)

    def test_run_unet_prediction_n5_mask(self):
        self._test_run_unet_prediction(use_tif=False, use_mask=True)

    def test_run_unet_prediction_tif(self):
        self._test_run_unet_prediction(use_tif=True, use_mask=False)

    def test_run_unet_prediction_tif_mask(self):
        self._test_run_unet_prediction(use_tif=True, use_mask=True)

    def test_run_unet_prediction_complex_watershed(self):
        self._test_run_unet_prediction(
            use_tif=False, use_mask=True,
            center_distance_threshold=0.5, boundary_distance_threshold=0.5, distance_smoothing=1.0,
        )


class TestDistanceWatershedRerun(unittest.TestCase):
    """The segmentation stage must not inherit seeds from an earlier run in the same output folder."""

    shape = (32, 64, 64)
    chunks = (16, 16, 16)
    fg_threshold = 0.5
    center_distance_threshold = 0.4
    boundary_distance_threshold = 0.5

    # Two blobs in different blocks. Blob B drops out of the mask in the second run.
    blob_a = (slice(4, 12), slice(4, 12), slice(4, 12))
    blob_b = (slice(20, 28), slice(52, 60), slice(52, 60))

    def _seed_center(self, blob):
        return tuple(slice(s.start + 3, s.start + 5) for s in blob)

    def _write_prediction(self, path, blobs):
        prediction = np.zeros((3,) + self.shape, dtype="float32")
        # Distances above the thresholds everywhere, so that only the blob centers become seeds.
        prediction[1:] = 1.0
        for blob in blobs:
            prediction[(0,) + blob] = 1.0
            prediction[(slice(1, 3),) + self._seed_center(blob)] = 0.0

        with open_file(path, "a") as f:
            ds = f.create_dataset(
                "prediction", shape=prediction.shape, chunks=(1,) + self.chunks, dtype="float32", overwrite=True,
            )
            ds[:] = prediction
        return prediction

    def _run(self, output_folder):
        from flamingo_tools.segmentation.unet_prediction import distance_watershed_implementation

        distance_watershed_implementation(
            os.path.join(output_folder, "predictions.zarr"), output_folder,
            min_size=0,
            center_distance_threshold=self.center_distance_threshold,
            boundary_distance_threshold=self.boundary_distance_threshold,
            fg_threshold=self.fg_threshold,
            distance_smoothing=0.0,
        )

    def _read(self, output_folder, name, key):
        with open_file(os.path.join(output_folder, name), "r") as f:
            return f[key][:]

    def test_rerun_does_not_inherit_seeds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_folder = os.path.join(tmp_dir, "output")
            os.makedirs(output_folder)

            # First run: both blobs are foreground, so both get seeds.
            self._write_prediction(os.path.join(output_folder, "predictions.zarr"), [self.blob_a, self.blob_b])
            self._run(output_folder)
            seeds = self._read(output_folder, "seeds.zarr", "seeds")
            self.assertGreater(seeds[self.blob_b].max(), 0)

            # Second run: blob B is no longer foreground, so its block is excluded from the mask.
            prediction = self._write_prediction(os.path.join(output_folder, "predictions.zarr"), [self.blob_a])
            self._run(output_folder)

            mask = prediction[0] > self.fg_threshold
            seeds = self._read(output_folder, "seeds.zarr", "seeds")
            self.assertGreater(seeds[mask].max(), 0)
            self.assertEqual(seeds[~mask].max(), 0)

            # No label may cover two disconnected parts of the volume.
            seg = self._read(output_folder, "segmentation.zarr", "segmentation")
            label_ids = np.unique(seg)
            label_ids = label_ids[label_ids != 0]
            self.assertGreater(len(label_ids), 0)
            for label_id in label_ids:
                _, n_components = ndimage.label(seg == label_id)
                self.assertEqual(n_components, 1, f"label {label_id} is split into {n_components} parts")


class TestSelectChannel(unittest.TestCase):
    """The wrapper must select the channel in the store, without changing the returned data."""

    shape = (5, 16, 32, 32)

    def _check(self, volume, reference):
        from flamingo_tools.segmentation.unet_prediction import SelectChannel

        for channel in range(self.shape[0]):
            wrapper = SelectChannel(volume, channel)
            self.assertEqual(wrapper.shape, self.shape[1:])
            self.assertEqual(wrapper.ndim, 3)

            expected = reference[channel]
            indices = [
                np.s_[:],
                np.s_[:8, :16, :16],
                np.s_[2:10, 4:20, 8:24],
                np.s_[3],
                np.s_[3, 5],
                np.s_[3, 5, 7],
                np.s_[:, 4, :],
                np.s_[1:5, 6, 8:12],
            ]
            for index in indices:
                self.assertTrue(
                    np.array_equal(wrapper[index], expected[index]),
                    msg=f"channel {channel}, index {index}",
                )

    def test_numpy(self):
        data = np.random.rand(*self.shape).astype("float32")
        self._check(data, data)

    def test_zarr(self):
        import zarr

        data = np.random.rand(*self.shape).astype("float32")
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "pred.zarr")
            f = zarr.open(path, mode="w")
            f.create_array("prediction", data=data, chunks=(1, 8, 16, 16))
            self._check(zarr.open(path, mode="r")["prediction"], data)

    def test_reads_single_channel(self):
        """The store must never see a request for more than the selected channel."""
        from flamingo_tools.segmentation.unet_prediction import SelectChannel

        data = np.random.rand(*self.shape).astype("float32")

        class CountingVolume:
            def __init__(self, volume):
                self._volume = volume
                self.requested_channels = []

            def __getitem__(self, index):
                self.requested_channels.append(index[0])
                return self._volume[index]

            @property
            def shape(self):
                return self._volume.shape

            @property
            def ndim(self):
                return self._volume.ndim

            @property
            def chunks(self):
                return (1, 8, 16, 16)

        volume = CountingVolume(data)
        wrapper = SelectChannel(volume, 2)
        result = wrapper[:8, :16, :16]

        self.assertTrue(np.array_equal(result, data[2, :8, :16, :16]))
        self.assertEqual(volume.requested_channels, [2])


class TestPredictionInstances(unittest.TestCase):
    """Splitting the prediction across slurm array tasks must reproduce the single-job result."""

    shape = (64, 128, 128)
    block_shape = (32, 32, 32)
    halo = (8, 8, 8)
    prediction_instances = 4

    def _create_input(self, tmp_dir):
        model = UNet3d(in_channels=1, out_channels=3, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)

        data_path = os.path.join(tmp_dir, "data.n5")
        with z5py.File(data_path, "a") as f:
            f.create_dataset("data", data=np.random.randint(0, 255, size=self.shape), chunks=(32, 32, 32))
        return data_path, "data", model_path

    def _predict(self, data_path, data_key, model_path, output_folder, prediction_instances):
        from flamingo_tools.segmentation.unet_prediction import prediction_impl

        for task_id in range(prediction_instances):
            prediction_impl(
                data_path, data_key, output_folder, model_path, scale=None,
                block_shape=self.block_shape, halo=self.halo,
                prediction_instances=prediction_instances, slurm_task_id=task_id,
            )
        with open_file(os.path.join(output_folder, "predictions.zarr"), "r") as f:
            return f["prediction"][:]

    def test_split_matches_single_job(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            data_path, data_key, model_path = self._create_input(tmp_dir)

            single = self._predict(
                data_path, data_key, model_path, os.path.join(tmp_dir, "single"), prediction_instances=1
            )
            split = self._predict(
                data_path, data_key, model_path, os.path.join(tmp_dir, "split"),
                prediction_instances=self.prediction_instances,
            )

            self.assertEqual(single.shape, (3,) + self.shape)
            self.assertGreater(np.abs(single).sum(), 0)
            self.assertTrue(np.array_equal(single, split))

    def test_tasks_cover_all_blocks_exactly_once(self):
        """Every block must be predicted by exactly one task."""
        n_blocks = int(np.prod([sh // bs for sh, bs in zip(self.shape, self.block_shape)]))
        rng = np.random.default_rng(seed=1234)
        assignments = [x.tolist() for x in np.array_split(list(rng.permutation(n_blocks)), self.prediction_instances)]

        covered = sorted(block_id for task in assignments for block_id in task)
        self.assertEqual(covered, list(range(n_blocks)))


class TestClippedTiling(unittest.TestCase):
    def test_odd_slice_count(self):
        """A 61-slice volume clips the block to 61, and 61 + 2 * 16 is not divisible by 16."""
        from flamingo_tools.segmentation.unet_prediction import prediction_impl

        shape = (61, 64, 64)
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = os.path.join(tmp_dir, "model.pt")
            torch.save(UNet3d(in_channels=1, out_channels=3, initial_features=4, depth=4), model_path)
            data_path = os.path.join(tmp_dir, "data.tif")
            imageio.imwrite(data_path, np.random.randint(0, 255, size=shape).astype("uint8"))

            _, prediction = prediction_impl(
                data_path, None, None, model_path, scale=None, block_shape=None, halo=None,
            )
        self.assertEqual(prediction.shape, (3,) + shape)


class TestFindMask(unittest.TestCase):
    """The default mask rule, its low-res input, and the legacy rule."""

    shape = (256, 256, 256)

    def _volume(self, bright_blocks=(), fraction=0.125, seed=0):
        """Background noise around 100 with a bright cube in each block of `bright_blocks`.

        The cube covers `fraction` of the 128^3 block, at intensity 1000.
        """
        rng = np.random.default_rng(seed)
        volume = rng.normal(100, 5, size=self.shape).clip(1, None).astype("uint16")
        edge = int(round(128 * fraction ** (1 / 3)))
        for block in bright_blocks:
            volume[tuple(slice(b * 128, b * 128 + edge) for b in block)] = 1000
        return volume

    def _write(self, tmp_dir, volume, pyramid=True, name="data.ome.zarr", chunks=(64, 64, 64), subsampled=False):
        """Write the volume as 's0' with an 8x level 's3', or as 'raw' without a pyramid.

        The 's3' level is the 8^3 mean, or every 8th voxel if `subsampled` is set.
        """
        import zarr

        path = os.path.join(tmp_dir, name)
        f = zarr.open(path, mode="w")
        if pyramid:
            f.create_array("s0", data=volume, chunks=chunks)
            n = tuple(sh // 8 for sh in volume.shape)
            if subsampled:
                s3 = volume[::8, ::8, ::8]
            else:
                s3 = np.rint(volume.reshape(n[0], 8, n[1], 8, n[2], 8).mean(axis=(1, 3, 5)))
            f.create_array("s3", data=s3.astype(volume.dtype), chunks=chunks)
            return path, "s0"
        f.create_array("raw", data=volume, chunks=chunks)
        return path, "raw"

    def _blocks(self, *indices):
        keep = np.zeros(tuple(sh // 128 for sh in self.shape), dtype=bool)
        for index in indices:
            keep[index] = True
        return keep

    def test_pyramid_and_mean_pooling_agree(self):
        from flamingo_tools.segmentation.unet_prediction import compute_mask_blocks

        volume = self._volume(bright_blocks=[(1, 0, 1)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            keep_pyramid, stats_pyramid = compute_mask_blocks(*self._write(tmp_dir, volume), "sgn")
            keep_pooled, stats_pooled = compute_mask_blocks(
                *self._write(tmp_dir, volume, pyramid=False, name="raw.zarr"), "sgn"
            )
        self.assertEqual(stats_pyramid["lowres_source"], "s3")
        self.assertEqual(stats_pooled["lowres_source"], "mean pooling of the input")
        self.assertAlmostEqual(stats_pyramid["background"], stats_pooled["background"])
        self.assertTrue(np.array_equal(keep_pyramid, keep_pooled))
        self.assertTrue(np.array_equal(keep_pyramid, self._blocks((1, 0, 1))))

    def test_subsampled_pyramid_is_not_used(self):
        """A subsampled level keeps the noise of the input, so the rule pools the input instead."""
        from flamingo_tools.segmentation.unet_prediction import compute_mask_blocks

        volume = self._volume(bright_blocks=[(1, 0, 1)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            keep, stats = compute_mask_blocks(*self._write(tmp_dir, volume, subsampled=True), "sgn")
        self.assertEqual(stats["lowres_source"], "mean pooling of the input")
        self.assertTrue(np.array_equal(keep, self._blocks((1, 0, 1))))

    def test_sgn_keeps_bright_blocks_without_dilation(self):
        from flamingo_tools.segmentation.unet_prediction import compute_mask_blocks

        volume = self._volume(bright_blocks=[(0, 0, 0), (1, 1, 0)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            keep, stats = compute_mask_blocks(*self._write(tmp_dir, volume), "sgn")
        self.assertTrue(np.array_equal(keep, self._blocks((0, 0, 0), (1, 1, 0))))
        self.assertAlmostEqual(stats["background"], 100, delta=1)
        self.assertEqual(stats["kept_blocks"], 2)

    def test_ihc_adds_all_neighbours(self):
        from flamingo_tools.segmentation.unet_prediction import compute_mask_blocks

        # 2% bright voxels: above the IHC rule (p99), below the SGN rule (p95).
        self.shape = (256, 384, 384)
        volume = self._volume(bright_blocks=[(0, 0, 0)], fraction=0.02)
        with tempfile.TemporaryDirectory() as tmp_dir:
            path, key = self._write(tmp_dir, volume)
            keep_ihc, _ = compute_mask_blocks(path, key, "ihc")
            keep_sgn, _ = compute_mask_blocks(path, key, "sgn")
        expected = np.zeros_like(keep_ihc)
        expected[:2, :2, :2] = True
        self.assertTrue(np.array_equal(keep_ihc, expected))
        self.assertFalse(keep_sgn.any())

    def test_gain_and_zero_padding(self):
        """The rule is relative to the background, and fusion padding (0) does not shift it."""
        from flamingo_tools.segmentation.unet_prediction import compute_mask_blocks

        volume = self._volume(bright_blocks=[(0, 1, 1)])
        padded = np.zeros((256, 256, 512), dtype="uint16")
        padded[..., :256] = volume
        with tempfile.TemporaryDirectory() as tmp_dir:
            keep, stats = compute_mask_blocks(*self._write(tmp_dir, volume), "sgn")
            keep_gain, stats_gain = compute_mask_blocks(*self._write(tmp_dir, 2 * volume, name="gain.zarr"), "sgn")
            self.shape = padded.shape
            keep_pad, stats_pad = compute_mask_blocks(*self._write(tmp_dir, padded, name="pad.zarr"), "sgn")
        self.assertTrue(np.array_equal(keep, keep_gain))
        self.assertAlmostEqual(stats_gain["background"], 2 * stats["background"], delta=1)
        self.assertAlmostEqual(stats_pad["background"], stats["background"])
        self.assertTrue(np.array_equal(keep_pad[..., :2], keep))
        self.assertFalse(keep_pad[..., 2:].any())

    def test_find_mask_output(self):
        from flamingo_tools.segmentation.unet_prediction import find_mask

        volume = self._volume(bright_blocks=[(1, 1, 1)])
        with tempfile.TemporaryDirectory() as tmp_dir:
            path, key = self._write(tmp_dir, volume)
            in_memory = find_mask(path, key, None, seg_class="sgn")
            output_folder = os.path.join(tmp_dir, "output")
            os.makedirs(output_folder)
            self.assertIsNone(find_mask(path, key, output_folder, seg_class="sgn"))
            with z5py.File(os.path.join(output_folder, "mask.zarr"), "r") as f:
                stored = f["mask"]
                self.assertEqual(stored.chunks, (128, 128, 128))
                stored = stored[:]
            with open(os.path.join(output_folder, "mask_statistics.json")) as f:
                stats = json.load(f)
        self.assertEqual(in_memory.dtype, np.uint8)
        self.assertEqual(stored.dtype, np.uint8)
        self.assertTrue(np.array_equal(in_memory, stored))
        expected = np.zeros(self.shape, dtype="uint8")
        expected[128:, 128:, 128:] = 1
        self.assertTrue(np.array_equal(stored, expected))
        self.assertEqual(stats["rule"], "background")
        self.assertEqual(stats["kept_blocks"], 1)

    def test_legacy_rule(self):
        """The legacy rule keeps a block of 2 x chunks if its upper percentile exceeds the threshold."""
        from flamingo_tools.segmentation.unet_prediction import find_mask

        # The first 64^3 block has 1.6% bright voxels: p99 is above the threshold, p95 is not.
        volume = self._volume()
        volume[:16, :16, :16] = 1000
        volume[192:, 192:, 192:] = 1000
        chunks = (32, 32, 32)
        threshold = 150

        def reference(percentile):
            expected = np.zeros(self.shape, dtype="uint8")
            for index in np.ndindex(*(sh // 64 for sh in self.shape)):
                bb = tuple(slice(i * 64, (i + 1) * 64) for i in index)
                if np.percentile(volume[bb], percentile) > threshold:
                    expected[bb] = 1
            return expected

        with tempfile.TemporaryDirectory() as tmp_dir:
            path, key = self._write(tmp_dir, volume, pyramid=False, chunks=chunks)
            masks = {
                "sgn": find_mask(path, key, None, seg_class="sgn", legacy_mask=True, absolute_threshold=threshold),
                "sgn_low": find_mask(path, key, None, seg_class="sgn_low", absolute_threshold=threshold),
                "ihc": find_mask(path, key, None, seg_class="ihc", legacy_mask=True, absolute_threshold=threshold),
                "ihc_low": find_mask(path, key, None, seg_class="ihc_low", absolute_threshold=threshold),
            }
        for seg_class, mask in masks.items():
            expected = reference(99 if seg_class.startswith("ihc") else 95)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(np.array_equal(mask, expected), seg_class)
        self.assertGreater(masks["ihc"].sum(), masks["sgn"].sum())

    def test_absolute_threshold_needs_legacy_rule(self):
        from flamingo_tools.segmentation.unet_prediction import find_mask

        with self.assertRaises(ValueError):
            find_mask("unused.zarr", "s0", None, seg_class="ihc", absolute_threshold=150)
        with self.assertRaises(ValueError):
            find_mask("unused.zarr", "s0", None, seg_class="ihc", legacy_mask=True, background_factor=2.0)

    def test_lowres_key(self):
        from flamingo_tools.segmentation.unet_prediction import _lowres_key

        self.assertEqual(_lowres_key("s0"), "s3")
        self.assertEqual(_lowres_key("setup1/timepoint0/s0"), "setup1/timepoint0/s3")
        self.assertEqual(_lowres_key("s1"), "s4")
        self.assertIsNone(_lowres_key("raw"))
        self.assertIsNone(_lowres_key(None))


class TestPreprocessSlurmArguments(unittest.TestCase):
    """The slurm templates pass every argument of the mask step as a string."""

    def _call(self, **kwargs):
        from unittest import mock
        from flamingo_tools.segmentation import unet_prediction

        with tempfile.TemporaryDirectory() as tmp_dir, mock.patch.object(unet_prediction, "find_mask") as find_mask:
            unet_prediction.run_unet_prediction_preprocess_slurm("in.zarr", tmp_dir, input_key="s0", **kwargs)
        return find_mask.call_args.kwargs

    def test_default_rule(self):
        kwargs = self._call(seg_class="ihc", absolute_threshold="None", legacy_mask="false", background_factor="None")
        self.assertEqual(kwargs["absolute_threshold"], None)
        self.assertEqual(kwargs["legacy_mask"], False)
        self.assertEqual(kwargs["background_factor"], None)

    def test_legacy_rule(self):
        kwargs = self._call(seg_class="ihc", absolute_threshold="150", legacy_mask="true")
        self.assertEqual(kwargs["absolute_threshold"], 150.0)
        self.assertEqual(kwargs["legacy_mask"], True)

    def test_background_factor(self):
        kwargs = self._call(seg_class="sgn", legacy_mask="", background_factor="1.2")
        self.assertEqual(kwargs["background_factor"], 1.2)
        self.assertEqual(kwargs["legacy_mask"], False)


class TestWatershedHalo(unittest.TestCase):
    """A cell that crosses a block face must get one label, also in the block without its seed."""

    shape = (256, 128, 128)
    radii = (48.0, 12.0, 12.0)
    # The seeds (normalized center distance < 0.4) lie 9-47 voxels before the face at z = 128.
    centers = [(100, 32, 32), (100, 32, 96), (156, 96, 32), (156, 96, 96)]

    def _segment(self, tmp_dir, halo=None, distance_smoothing=0.0):
        from flamingo_tools.segmentation.unet_prediction import distance_watershed_implementation

        grid = np.meshgrid(*[np.arange(sh) for sh in self.shape], indexing="ij")
        cells = np.zeros(self.shape, dtype="uint32")
        distance = np.ones(self.shape, dtype="float32")
        for cell_id, center in enumerate(self.centers, 1):
            d = np.sqrt(sum(((g - c) / r) ** 2 for g, c, r in zip(grid, center, self.radii)))
            cells[d <= 1] = cell_id
            distance[d <= 1] = d[d <= 1]
        prediction = np.stack([(cells > 0).astype("float32"), distance, distance])

        path = os.path.join(tmp_dir, "predictions.zarr")
        with open_file(path, "a") as f:
            f.create_dataset("prediction", data=prediction, chunks=(1, 128, 128, 128))
        kwargs = {} if halo is None else {"halo": halo}
        distance_watershed_implementation(
            path, tmp_dir, min_size=0, center_distance_threshold=0.4, boundary_distance_threshold=0.5,
            distance_smoothing=distance_smoothing, **kwargs
        )
        with open_file(os.path.join(tmp_dir, "segmentation.zarr"), "r") as f:
            return cells, f["segmentation"][:]

    def test_default_halo_keeps_cells_complete(self):
        for distance_smoothing in (0.0, 1.0):
            with self.subTest(distance_smoothing=distance_smoothing), tempfile.TemporaryDirectory() as tmp_dir:
                cells, seg = self._segment(tmp_dir, distance_smoothing=distance_smoothing)
                for cell_id in range(1, len(self.centers) + 1):
                    labels = np.unique(seg[cells == cell_id])
                    self.assertEqual(len(labels), 1, f"cell {cell_id} has the labels {labels}")
                    self.assertGreater(labels[0], 0)

    def test_small_halo_cuts_cells(self):
        """The former halo (2, 8, 8) leaves the part of a cell in the block without its seed unlabeled.

        With smoothing, the halo of the smoothing must not replace the halo of the watershed.
        """
        for distance_smoothing in (0.0, 1.0):
            with self.subTest(distance_smoothing=distance_smoothing), tempfile.TemporaryDirectory() as tmp_dir:
                cells, seg = self._segment(tmp_dir, halo=(2, 8, 8), distance_smoothing=distance_smoothing)
                self.assertTrue(np.any(seg[cells > 0] == 0))


if __name__ == "__main__":
    unittest.main()
