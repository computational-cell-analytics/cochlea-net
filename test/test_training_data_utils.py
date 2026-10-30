import unittest

import numpy as np
import pandas as pd


class TestFindCropCentersMissed(unittest.TestCase):

    def setUp(self):
        x = np.arange(30) * 9.0
        self.reference = pd.DataFrame({"anchor_x": x, "anchor_y": 100.0, "anchor_z": 100.0})
        # A run of 5 neighbouring misses (x = 90 to 126) and a single miss (x = 198).
        missed = list(range(10, 15)) + [22]
        self.table = self.reference.drop(index=missed)

    def test_largest_run_first(self):
        from flamingo_tools.analysis.training_data_utils import find_crop_centers_missed

        centers, n_missed = find_crop_centers_missed(self.reference, self.table, n_crops=3, crop_size=(40, 40, 40))
        self.assertEqual(centers, [[108, 100, 100], [198, 100, 100]])
        self.assertEqual(n_missed, [5, 1])

    def test_skips_overlapping_crops(self):
        from flamingo_tools.analysis.training_data_utils import find_crop_centers_missed

        centers, n_missed = find_crop_centers_missed(self.reference, self.table, n_crops=3, crop_size=(200, 40, 40))
        self.assertEqual(centers, [[108, 100, 100]])
        self.assertEqual(n_missed, [5])


class TestCorrectIhcRow(unittest.TestCase):

    def setUp(self):
        # A row of 11 parallel IHCs along y (long axis z), with voxel size 1 and touching neighbours.
        # Cell 0 is split into an upper and a lower half (labels 8, 9), cells 1-6 are correct
        # (labels 1-6), cells 7 and 8 share label 7, and cells 9 and 10 share label 11, which touches
        # the volume border. Label 10 is a small blob 22 px off the row, label 12 a small blob on the row
        # at the volume border, and label 13 a small blob off the row at the volume border. Label 3 has a
        # stray piece of 8 voxels away from its cell.
        shape = (60, 106, 40)
        z, y, x = np.indices(shape)
        centers_y = 12 + 9 * np.arange(11)
        nearest = np.argmin(np.abs(y[..., None] - centers_y), axis=-1)
        inside = ((z - 30) / 14) ** 2 + ((y - centers_y[nearest]) / 5) ** 2 + ((x - 20) / 4.5) ** 2 <= 1
        cell_labels = np.array([8, 1, 2, 3, 4, 5, 6, 7, 7, 11, 11])
        segmentation = np.where(inside, cell_labels[nearest], 0)
        segmentation[(segmentation == 8) & (z > 30)] = 9
        segmentation[(z - 52) ** 2 + (y - 50) ** 2 + (x - 20) ** 2 <= 9] = 10
        segmentation[28:33, 0:5, 18:23] = 12
        segmentation[50:55, 0:5, 18:23] = 13
        segmentation[55:57, 60:62, 5:7] = 3
        self.segmentation = segmentation.astype("uint32")
        self.path = np.array([[30.0, 0.0, 20.0], [30.0, 105.0, 20.0]])

    def test_correction(self):
        from flamingo_tools.analysis.training_data_utils import correct_ihc_row, ihc_row_reference

        seg = self.segmentation
        reference = ihc_row_reference([seg], self.path, voxel_size=1.0)
        np.testing.assert_allclose(np.abs(reference["axis"]), [1, 0, 0], atol=0.05)
        for tip_method in ("path", "height"):
            with self.subTest(tip_method=tip_method):
                labels, report, tips = correct_ihc_row(seg, self.path, reference, voxel_size=1.0, tip_method=tip_method)

                classes = dict(zip(report["label_id"], report["class"]))
                expected = ["normal"] * 6 + ["uncertain"] * 3 + ["off_row", "uncertain", "border", "off_row"]
                self.assertEqual([classes[i] for i in range(1, 14)], expected)
                self.assertTrue(np.all(labels[seg == 12] == 12))
                self.assertTrue(np.all(labels[seg == 13] == 0))
                self.assertTrue(np.all(labels[55:57, 60:62, 5:7] == 0))
                for label_id in range(1, 7):
                    self.assertGreaterEqual(np.mean(labels[seg == label_id] == label_id), 0.95)
                # The tips split the instances of two cells, also the one at the volume border.
                for label_id in (7, 11):
                    _, counts = np.unique(labels[seg == label_id], return_counts=True)
                    self.assertEqual(int((counts > 0.3 * counts.sum()).sum()), 2)
                # The two halves of cell 0 are short. They share the tips of the cell and become one cell.
                halves, counts = np.unique(labels[np.isin(seg, [8, 9])], return_counts=True)
                self.assertIn(halves[np.argmax(counts)], [8, 9])
                self.assertGreater(counts.max(), 0.95 * counts.sum())
                self.assertTrue(np.all(labels[seg == 10] == 0))
                # The small blob 12 can have a tip of its own in the height maps.
                self.assertEqual(tips.loc[tips["label_id"] != 12, "cell"].nunique(), 11)


class TestMergeParts(unittest.TestCase):

    def test_merge_parts(self):
        from flamingo_tools.analysis.training_data_utils import _merge_parts

        # The path runs along y, the long axis is z. Labels 1 (lower part) and 2 (upper part) are one cell at
        # y = 10. Label 3 is the neighbouring cell at y = 19 and touches label 2.
        segmentation = np.zeros((40, 30, 10), dtype="uint32")
        segmentation[5:20, 6:14, 2:8] = 1
        segmentation[20:35, 6:15, 2:8] = 2
        segmentation[20:35, 15:24, 2:8] = 3
        coords = {i: np.argwhere(segmentation == i) for i in (1, 2, 3)}
        features = {i: {"row": True, "border": False, "short": False, "n_pixels": len(coords[i])} for i in coords}
        points = np.stack([np.full(30, 20.0), np.arange(30.0), np.full(30, 5.0)], axis=1)
        axes = np.tile([1.0, 0.0, 0.0], (30, 1))

        merged, part_of = _merge_parts(segmentation, features, coords, points, axes, max_distance=4.0)
        self.assertEqual(part_of, {1: 2})
        self.assertTrue(np.all(merged[segmentation == 1] == 2))
        self.assertTrue(np.all(merged[segmentation == 3] == 3))


class TestFindIhcTips(unittest.TestCase):

    def setUp(self):
        # 5 parallel IHCs along y with the long axis z. The upper ends of the cells are cut by the volume border.
        shape = (40, 60, 30)
        z, y, x = np.indices(shape)
        centers_y = 10 + 9 * np.arange(5)
        nearest = np.argmin(np.abs(y[..., None] - centers_y), axis=-1)
        inside = ((z - 10) / 14) ** 2 + ((y - centers_y[nearest]) / 5) ** 2 + ((x - 15) / 4.5) ** 2 <= 1
        self.segmentation = np.where(inside, nearest + 1, 0)
        self.voxels = np.argwhere(self.segmentation > 0)
        self.labels = self.segmentation[tuple(self.voxels.T)]
        self.points = np.stack([np.full(60, 10.0), np.arange(60.0), np.full(60, 15.0)], axis=1)
        self.axis = np.array([1.0, 0.0, 0.0])

    def test_find_ihc_tips(self):
        from flamingo_tools.analysis.training_data_utils import find_ihc_tips_height, find_ihc_tips_path

        shape = self.segmentation.shape
        for name, tips in [
            ("height", find_ihc_tips_height(self.voxels, self.labels, self.axis, shape, voxel_size=1.0)),
            ("path", find_ihc_tips_path(self.voxels, self.labels, self.points, np.tile(self.axis, (60, 1)), shape,
                                        voxel_size=1.0)),
        ]:
            with self.subTest(method=name):
                # One tip per cell at its lower end (z = 10 + 14), none at the cut face z = 0.
                self.assertEqual(sorted(tips["label_id"]), [1, 2, 3, 4, 5])
                self.assertTrue(np.all(tips["z"] == 24))
                self.assertEqual(tips["cell"].nunique(), 5)


class TestIhcRowQuality(unittest.TestCase):

    def setUp(self):
        # 60 IHC positions, 9 µm apart on an arc with a radius of 300 µm, with parallel main axes along z.
        # Cells 20-22 are missing, cell 10 is a fragment, cells 40 and 41 are one merged instance, the
        # main axis of cell 30 is tilted by 60 degree, and cell 50 is round. Label 101 is an instance outside
        # the row next to cell 5, label 102 is far from the row, and label 103 is next to the row but small.
        angles = np.arange(60) * 9.0 / 300.0
        angles[40] = (angles[40] + angles[41]) / 2
        sizes = {10: 9000, 40: 60000}
        rows = [(i + 1, 300.0, angles[i], sizes.get(i, 30000), 1) for i in range(60) if i not in (20, 21, 22, 41)]
        rows += [(101, 308.0, angles[5], 5000, 0), (102, 400.0, angles[30], 30000, 0), (103, 306.0, angles[50], 500, 0)]
        label_ids, radii, phis, sizes, components = map(np.array, zip(*rows))
        self.table = pd.DataFrame({
            "label_id": label_ids,
            "anchor_x": 500 + radii * np.cos(phis),
            "anchor_y": 500 + radii * np.sin(phis),
            "anchor_z": 100.0,
            "n_pixels": sizes,
            "component_labels": components,
        })
        row_ids = label_ids[components == 1]
        tilt = np.radians(60)
        self.ellipsoids = pd.DataFrame({
            "label_id": row_ids,
            "a": 14.0, "b": np.where(row_ids == 51, 13.0, 7.0), "c": 5.0,
            "axis_x": np.where(row_ids == 31, np.sin(tilt), 0.0), "axis_y": 0.0,
            "axis_z": np.where(row_ids == 31, np.cos(tilt), 1.0),
        })
        self.gap_center = np.array([500 + 300 * np.cos(angles[21]), 500 + 300 * np.sin(angles[21]), 100.0])

    def test_ihc_row_quality(self):
        from flamingo_tools.analysis.training_data_utils import ihc_row_quality

        objects, windows = ihc_row_quality(self.table, [1], ellipsoids=self.ellipsoids)
        classes = objects.set_index("label_id")["class"]
        self.assertEqual(classes[11], "fragment")
        self.assertEqual(classes[41], "merged")
        self.assertEqual(classes[31], "shape")
        self.assertEqual(classes[51], "shape")
        self.assertEqual(classes[101], "extra")
        self.assertNotIn(102, classes.index)
        self.assertNotIn(103, classes.index)
        self.assertEqual((classes == "normal").sum(), 52)
        totals = windows[["missed", "merged", "fragment", "shape", "off_row", "extra"]].sum().to_dict()
        self.assertEqual(totals, {"missed": 3, "merged": 1, "fragment": 1, "shape": 2, "off_row": 0, "extra": 1})
        self.assertEqual(windows["n_row"].sum(), 56)

        # Without the ellipsoids, the shape errors are not found.
        _, windows = ihc_row_quality(self.table, [1])
        self.assertEqual(windows["shape"].sum(), 0)

    def test_find_crop_centers_quality(self):
        from flamingo_tools.analysis.training_data_utils import find_crop_centers_quality, ihc_row_quality

        _, windows = ihc_row_quality(self.table, [1], ellipsoids=self.ellipsoids)
        centers, n_errors = find_crop_centers_quality(windows, n_crops=2, crop_size=(50, 50, 50))
        self.assertEqual(n_errors[0], 3)
        self.assertLess(np.linalg.norm(np.array(centers[0]) - self.gap_center), 60)
        self.assertEqual(len(centers), 2)

        # An existing crop at the gap moves the first crop elsewhere.
        centers, _ = find_crop_centers_quality(windows, n_crops=1, crop_size=(50, 50, 50), exclude=[self.gap_center])
        self.assertGreater(np.linalg.norm(np.array(centers[0]) - self.gap_center), 50)


if __name__ == "__main__":
    unittest.main()
