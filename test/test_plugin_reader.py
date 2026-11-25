import importlib.util
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


@unittest.skipIf(importlib.util.find_spec("napari_builtins") is None, "the napari reader needs napari")
class TestPluginReader(unittest.TestCase):
    def test_points_csv(self):
        from flamingo_tools.extract_block_util import extract_points_single
        from flamingo_tools.plugin.reader import POINT_SIZE, get_csv_reader

        points = pd.DataFrame({
            "spot_id": [1, 2], "x": [10.0, 12.0], "y": [5.0, 6.5], "z": [8.0, 9.0], "matched_ihc": [795, 793],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = extract_points_single(
                points, [10, 5, 8], tmp, (32, 32, 32), name="synapses", voxel_size=1.0, roi_halo=[4, 4, 4],
            )
            layers = get_csv_reader(path)(path)

        self.assertEqual(len(layers), 1)
        data, meta, layer_type = layers[0]
        self.assertEqual(layer_type, "points")
        self.assertEqual(meta["size"], POINT_SIZE)
        # The crop starts at the voxel (z, y, x) = (4, 1, 6).
        np.testing.assert_allclose(data, [[4, 4, 4], [5, 5.5, 6]])
        self.assertEqual(meta["properties"]["spot_id"].tolist(), [1, 2])
        self.assertEqual(meta["properties"]["matched_ihc"].tolist(), [795, 793])

    def test_shapes_csv(self):
        from flamingo_tools.plugin.reader import get_csv_reader

        table = pd.DataFrame({
            "index": [0, 0, 0], "shape-type": ["polygon"] * 3, "vertex-index": [0, 1, 2],
            "axis-0": [0.0, 0.0, 5.0], "axis-1": [0.0, 5.0, 5.0],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "shapes.csv")
            table.to_csv(path, index=False)
            layers = get_csv_reader(path)(path)

        _, meta, layer_type = layers[0]
        self.assertEqual(layer_type, "shapes")
        self.assertNotIn("size", meta)

    def test_other_path(self):
        from flamingo_tools.plugin.reader import get_csv_reader

        self.assertIsNone(get_csv_reader("crop.tif"))
        self.assertIsNone(get_csv_reader(["a.csv", "b.csv"]))


if __name__ == "__main__":
    unittest.main()
