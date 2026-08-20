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


if __name__ == "__main__":
    unittest.main()


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

        # A fixed mean and std, so that the tasks cannot disagree on the normalization.
        for task_id in range(prediction_instances):
            prediction_impl(
                data_path, data_key, output_folder, model_path, scale=None,
                block_shape=self.block_shape, halo=self.halo,
                prediction_instances=prediction_instances, slurm_task_id=task_id,
                mean=127.0, std=73.0,
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
