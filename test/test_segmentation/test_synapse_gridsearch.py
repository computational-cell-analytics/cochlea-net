import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import zarr

from flamingo_tools.synapse_detection import gridsearch as gs


class TestDetectionGridsearch(unittest.TestCase):
    shape = (20, 128, 128)
    # Two adjacent synapses two voxels apart, and one isolated synapse.
    annotations = [(10, 20, 20), (10, 20, 22), (10, 60, 60)]
    # A spot far from every annotation, in the unannotated part of a training crop.
    unannotated = (10, 120, 120)

    def _run(self, **kwargs):
        heatmap = np.zeros((64, 256, 256), dtype="float32")
        for point, value in zip(self.annotations + [self.unannotated], (2.0, 1.5, 2.0, 2.0)):
            heatmap[point] = value

        with tempfile.TemporaryDirectory() as tmp_dir:
            image_dir, label_dir = os.path.join(tmp_dir, "images"), os.path.join(tmp_dir, "labels")
            os.makedirs(label_dir)
            zarr.open(store=os.path.join(image_dir, "crop.zarr"), mode="w").create_array(
                "raw", data=np.zeros(self.shape, dtype="uint16")
            )
            pd.DataFrame(self.annotations, columns=["axis-0", "axis-1", "axis-2"]).to_csv(
                os.path.join(label_dir, "crop.csv"), index=False
            )
            with mock.patch.object(gs, "prediction_impl", return_value=(None, heatmap)):
                return gs.gridsearch(
                    "model.pt", image_dir=image_dir, label_dir=label_dir, out_channels=1, **kwargs
                )

    def test_min_distance_one_separates_the_pair(self):
        threshold, min_distance, scores = self._run()
        self.assertEqual(min_distance, 1)
        self.assertLessEqual(threshold, 1.5)
        self.assertEqual(scores.loc[(1, 0.5), "recall"], 1.0)
        self.assertAlmostEqual(scores.loc[(2, 0.5), "recall"], 2 / 3)

    def test_unannotated_region_is_not_scored(self):
        _, _, scores = self._run()
        self.assertEqual(scores.loc[(1, 0.5), "precision"], 1.0)

        _, _, scores = self._run(annotated_radius=None)
        self.assertEqual(scores.loc[(1, 0.5), "precision"], 0.75)


if __name__ == "__main__":
    unittest.main()
