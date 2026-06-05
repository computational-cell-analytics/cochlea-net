import json
import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import torch
from torch_em.model import UNet3d


class TestIHCGridsearch(unittest.TestCase):
    shape = (32, 64, 64)

    GRID_SEARCH_VALUES = {
        "center_distance_threshold": [0.4, 0.6],
        "boundary_distance_threshold": [0.4, 0.6],
        "distance_smoothing": [0.0, 0.5],
    }

    def _create_model(self, tmp_dir):
        model = UNet3d(in_channels=1, out_channels=3, initial_features=4, depth=2)
        model_path = os.path.join(tmp_dir, "model.pt")
        torch.save(model, model_path)
        return model_path

    def _create_val_dir(self, tmp_dir, n_images=2):
        val_dir = os.path.join(tmp_dir, "val")
        os.makedirs(val_dir)
        rng = np.random.default_rng(0)
        for i in range(n_images):
            image = rng.integers(0, 255, size=self.shape, dtype=np.uint16)
            label = rng.integers(0, 3, size=self.shape, dtype=np.uint16)
            imageio.imwrite(os.path.join(val_dir, f"sample_{i:02d}.tif"), image)
            imageio.imwrite(os.path.join(val_dir, f"sample_{i:02d}_annotations.tif"), label)
        return val_dir

    def test_find_image_label_pairs(self):
        from flamingo_tools.segmentation.gridsearch import _find_image_label_pairs
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_dir = self._create_val_dir(tmp_dir, n_images=3)
            pairs = _find_image_label_pairs(val_dir)
            self.assertEqual(len(pairs), 3)
            for image_path, label_path in pairs:
                self.assertTrue(os.path.exists(image_path))
                self.assertTrue(os.path.exists(label_path))
                self.assertFalse(image_path.endswith("_annotations.tif"))
                self.assertTrue(label_path.endswith("_annotations.tif"))

    def test_gridsearch_returns_best_params(self):
        from flamingo_tools.segmentation.gridsearch import gridsearch
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_dir = self._create_val_dir(tmp_dir)
            model_path = self._create_model(tmp_dir)
            best_params, best_score = gridsearch(
                val_dir, model_path,
                grid_search_values=self.GRID_SEARCH_VALUES,
                block_shape=(32, 64, 64), halo=(4, 8, 8),
            )
            self.assertIsInstance(best_params, dict)
            self.assertIn("center_distance_threshold", best_params)
            self.assertIn("boundary_distance_threshold", best_params)
            self.assertIn("distance_smoothing", best_params)
            self.assertIsInstance(best_score, float)
            self.assertGreaterEqual(best_score, 0.0)
            self.assertLessEqual(best_score, 1.0)

    def test_gridsearch_writes_per_image_json(self):
        from flamingo_tools.segmentation.gridsearch import gridsearch
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_dir = self._create_val_dir(tmp_dir)
            model_path = self._create_model(tmp_dir)
            result_dir = os.path.join(tmp_dir, "results")
            gridsearch(
                val_dir, model_path,
                result_dir=result_dir,
                grid_search_values=self.GRID_SEARCH_VALUES,
                block_shape=(32, 64, 64), halo=(4, 8, 8),
            )
            json_files = [f for f in os.listdir(result_dir) if f.endswith(".json")]
            self.assertEqual(len(json_files), 2)
            for jf in json_files:
                with open(os.path.join(result_dir, jf)) as fh:
                    records = json.load(fh)
                # 2×2×2 combos = 8 records per image
                self.assertEqual(len(records), 8)
                for rec in records:
                    self.assertIn("dice", rec)
                    self.assertIn("center_distance_threshold", rec)

    def test_gridsearch_resumes_from_cache(self):
        from flamingo_tools.segmentation.gridsearch import gridsearch
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_dir = self._create_val_dir(tmp_dir, n_images=1)
            model_path = self._create_model(tmp_dir)
            result_dir = os.path.join(tmp_dir, "results")

            # First run: writes JSON
            params1, score1 = gridsearch(
                val_dir, model_path,
                result_dir=result_dir,
                grid_search_values=self.GRID_SEARCH_VALUES,
                block_shape=(32, 64, 64), halo=(4, 8, 8),
            )

            # Second run: must load from cache (model path is invalid to verify no re-prediction)
            params2, score2 = gridsearch(
                val_dir, "nonexistent_model.pt",
                result_dir=result_dir,
                grid_search_values=self.GRID_SEARCH_VALUES,
                block_shape=(32, 64, 64), halo=(4, 8, 8),
            )

            self.assertEqual(params1, params2)
            self.assertAlmostEqual(score1, score2, places=6)

    def test_get_or_compute_best_params_caches(self):
        from flamingo_tools.segmentation.gridsearch import get_or_compute_best_params
        with tempfile.TemporaryDirectory() as tmp_dir:
            val_dir = self._create_val_dir(tmp_dir)
            model_path = self._create_model(tmp_dir)
            result_dir = os.path.join(tmp_dir, "results")

            params1, score1 = get_or_compute_best_params(
                model_path, val_dir,
                result_dir=result_dir,
                grid_search_values=self.GRID_SEARCH_VALUES,
                block_shape=(32, 64, 64), halo=(4, 8, 8),
            )

            cache_path = os.path.splitext(model_path)[0] + "_best_params.json"
            self.assertTrue(os.path.exists(cache_path))

            params2, score2 = get_or_compute_best_params(
                model_path, val_dir,
                result_dir=result_dir,
                grid_search_values=self.GRID_SEARCH_VALUES,
                block_shape=(32, 64, 64), halo=(4, 8, 8),
            )

            self.assertEqual(params1, params2)
            self.assertAlmostEqual(score1, score2, places=6)


if __name__ == "__main__":
    unittest.main()
