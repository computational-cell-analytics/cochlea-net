import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile


class TestMeanByOffset(unittest.TestCase):

    def test_bins(self):
        from flamingo_tools.analysis.central_path_utils import mean_by_offset

        table = pd.DataFrame({
            "length_fraction": [0.0, 0.01, 1.0, 0.0],
            "offset": [5.0, 9.0, 150.0, -1.0],
            "intensity": [1.0, 3.0, 7.0, 100.0],
        })
        result = mean_by_offset(table, ["intensity"])

        # 20 windows times 11 offset bins, of which the last is open.
        self.assertEqual(len(result), 220)
        self.assertEqual(result["n"].sum(), 3)

        first = result.iloc[0]
        self.assertEqual(first["n"], 2)
        self.assertEqual(first["intensity"], 2.0)
        self.assertEqual(first["length_bin"], 0)
        self.assertEqual(first["offset_bin"], 0)

        last = result.iloc[-1]
        self.assertEqual(last["n"], 1)
        self.assertEqual(last["intensity"], 7.0)
        self.assertEqual(last["length_bin"], 0.95)
        self.assertEqual(last["offset_bin"], 100)

        empty = result[result["n"] == 0]
        self.assertTrue(empty["intensity"].isna().all())


class TestExportCentralPaths(unittest.TestCase):

    def test_output_prefix(self):
        from flamingo_tools.analysis.central_path_utils import _output_prefix

        cases = [
            ("M_AMD_000126_L/tables/SGN_v2/default.tsv", None, "M-AMD-000126-L_SGN-v2"),
            ("M_LR_000184_L/tables/PV_SGN_v2/default.tsv", None, "M-LR-000184-L_PV-SGN-v2"),
            ("M_AMD_000126_L/tables/SGN_merged/default.tsv", None, "M-AMD-000126-L"),
            ("rc_offset/M_AMD_000126_L_SGN_v2.tsv", None, "M-AMD-000126-L_SGN-v2"),
            ("M_AMD_000126_L_SGN_v2b.tsv", None, "M-AMD-000126-L_SGN-v2b"),
            ("M_LR_000184_L_PV_SGN_v2.tsv", "M_LR_000184_L", "M-LR-000184-L_PV-SGN-v2"),
            ("M_AMD_N163_L_v2.tsv", None, "M-AMD-N163-L-v2"),
            ("M_AMD_000126_L.tsv", None, "M-AMD-000126-L"),
            ("M_AMD_000126_L/tables/SGN_v2/default.tsv", "C_1", "C-1_SGN-v2"),
            # Without a 'tables' folder, the root '/' became the prefix and the output went to '/'.
            ("/scratch/out/default.tsv", None, "default"),
        ]
        for table_path, cochlea, expected in cases:
            with self.subTest(table_path=table_path, cochlea=cochlea):
                self.assertEqual(_output_prefix(table_path, cochlea), expected)

    def test_tifs(self):
        from flamingo_tools.analysis.central_path_utils import export_central_paths
        from flamingo_tools.postprocessing.cochlea_mapping import CENTRAL_PATH_METHODS

        # Straight tube along x with a radius of 40 µm and its axis at y = 200 µm, z = 300 µm.
        rng = np.random.default_rng(0)
        n_points = 2000
        radius = 40 * np.sqrt(rng.random(n_points))
        angle = rng.random(n_points) * 2 * np.pi
        table = pd.DataFrame({
            "label_id": np.arange(1, n_points + 1),
            "anchor_x": rng.uniform(100, 900, n_points),
            "anchor_y": 200 + radius * np.cos(angle),
            "anchor_z": 300 + radius * np.sin(angle),
            "component_labels": 1,
        })

        with tempfile.TemporaryDirectory() as tmp_dir:
            table_path = os.path.join(tmp_dir, "C_1", "tables", "SGN_v2", "default.tsv")
            os.makedirs(os.path.dirname(table_path))
            table.to_csv(table_path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, r"\[5\] are not in the table"):
                export_central_paths(table_path, tmp_dir, component_list=[5])
            export_central_paths(table_path, tmp_dir, scale_factor=10)
            seg, intensity, path_v1, path_v2 = (tifffile.imread(os.path.join(tmp_dir, f"C-1_SGN-v2_{name}.tif"))
                                                for name in ("seg", "intensity", "path-v1", "path-v2"))

        # (Z, Y, X) order puts the long x-axis of the tube last.
        self.assertEqual(seg.shape[2], max(seg.shape))
        self.assertEqual(path_v1.shape, seg.shape)
        self.assertEqual(path_v2.shape, seg.shape)
        self.assertEqual(set(np.unique(seg)), {0, 1})
        # Without the closing, only about 40 % of the voxels on the tube axis hold a centroid.
        self.assertTrue(seg[30, 20, 15:86].all())
        self.assertEqual(intensity.shape, seg.shape)
        self.assertAlmostEqual(intensity.sum() / n_points, 1, places=4)

        centroids = [list(zip(table["anchor_x"], table["anchor_y"], table["anchor_z"]))]
        for volume, method in ((path_v1, "edt"), (path_v2, "edt_refined")):
            nodes = np.asarray(CENTRAL_PATH_METHODS[method](centroids)[0]) // 10
            self.assertTrue(volume[tuple(nodes[:, ::-1].astype(int).T)].all())
            # Consecutive 'edt' nodes can be two voxels apart, so this checks the connecting lines.
            x_unique = np.unique(np.nonzero(volume)[2])
            self.assertGreater(x_unique.max() - x_unique.min(), 60)
            self.assertEqual(len(x_unique), x_unique.max() - x_unique.min() + 1)


if __name__ == "__main__":
    unittest.main()
