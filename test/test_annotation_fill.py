import unittest

import numpy as np
from scipy import ndimage


def synthetic_row(long_axis="z", n_cells=7):
    """A row of touching ellipsoid cells along y, with voxel size 1, and an image with darker cell boundaries.

    With long_axis 'z', the cells are 28 px long along z. With 'x', they lie in the xy plane, 9 px thick in z.
    """
    shape = (60, 12 + 9 * n_cells, 40)
    z, y, x = np.indices(shape)
    centers = 10 + 9 * np.arange(n_cells)
    nearest = np.argmin(np.abs(y[..., None] - centers), axis=-1)
    if long_axis == "z":
        inside = ((z - 30) / 14) ** 2 + ((y - centers[nearest]) / 5) ** 2 + ((x - 20) / 4.5) ** 2 <= 1
    else:
        inside = ((z - 30) / 4.5) ** 2 + ((y - centers[nearest]) / 5) ** 2 + ((x - 20) / 14) ** 2 <= 1
    labels = np.where(inside, nearest + 1, 0).astype("uint32")
    rng = np.random.default_rng(0)
    image = ndimage.gaussian_filter((labels > 0) * 100.0 + rng.normal(0, 5, shape), 1)
    image[ndimage.morphological_gradient(labels, size=2) > 0] *= 0.7
    return labels, image.astype("float32")


def dice(a, b):
    return 2 * (a & b).sum() / (a.sum() + b.sum())


class TestIds(unittest.TestCase):

    def test_compact_restore(self):
        from flamingo_tools.annotation.ids import compact, restore

        labels = np.zeros((4, 5, 6), dtype="uint32")
        labels[0, :2] = 2_997_629_976
        labels[1, :3] = 17
        labels[3] = 4_000_000_000
        compact_labels, original = compact(labels)
        self.assertEqual(compact_labels.max(), 3)
        self.assertEqual(compact_labels.dtype, np.uint32)
        np.testing.assert_array_equal(restore(compact_labels, original), labels)
        # A new compact id gets the next free id after the largest original id.
        compact_labels[2, 0, 0] = 4
        self.assertEqual(restore(compact_labels, original)[2, 0, 0], 4_000_000_001)

    def test_restore_overflow(self):
        from flamingo_tools.annotation.ids import compact, restore

        labels = np.full((2, 2, 2), 2 ** 32 - 1, dtype="uint32")
        compact_labels, original = compact(labels)
        compact_labels[0, 0, 0] = 2
        with self.assertRaises(ValueError):
            restore(compact_labels, original)


class TestSelectSlices(unittest.TestCase):

    def test_slices(self):
        from flamingo_tools.annotation.fill import select_slices

        mask = np.zeros((60, 10, 10), dtype=bool)
        mask[20:41] = True
        z = select_slices(mask, 1.0, spacing=5.0, cap=2.0, inset=1.0)
        # The caps are 2 px beyond the cell ends, the end slices 1 px inside.
        self.assertEqual(z[0], 18)
        self.assertEqual(z[-1], 42)
        self.assertIn(21, z)
        self.assertIn(39, z)
        self.assertLessEqual(np.diff(z[1:-1]).max(), 5)

    def test_border_and_max_slices(self):
        from flamingo_tools.annotation.fill import select_slices

        mask = np.zeros((60, 10, 10), dtype=bool)
        mask[0:50] = True
        z = select_slices(mask, 1.0, spacing=1.0, cap=2.0, inset=1.0, max_slices=8)
        # At the volume border, the border slice is the end slice, and there is no cap below it.
        self.assertEqual(z[0], 0)
        self.assertEqual(z[-1], 51)
        self.assertLessEqual(len(z), 8)

    def test_visit_region(self):
        from flamingo_tools.annotation.fill import visit_region

        labels = np.zeros((50, 80, 70), dtype="uint32")
        labels[10:20, 20:30, 30:40] = 1
        labels[5:30, 33:45, 30:40] = 2  # a neighbour that reaches further in z and y
        labels[10:20, 50:79, 30:40] = 3  # a long label 20 px away
        roi = visit_region(labels, labels == 1, 1.0, slice_axis=0, margin=5.0, slice_margin=3.0, max_margin=15.0)
        # The neighbour 2 is complete; 3 is outside the start box and does not count.
        self.assertEqual(roi, (slice(5, 30), slice(15, 45), slice(25, 45)))
        # The region grows at most max_margin beyond the cell.
        roi = visit_region(labels, labels == 1, 1.0, max_margin=8.0)
        self.assertEqual(roi, (slice(5, 28), slice(15, 38), slice(25, 45)))
        # The slice margin applies along the slice axis only.
        roi = visit_region(labels, labels == 3, 1.0, slice_axis=2, margin=2.0, slice_margin=6.0)
        self.assertEqual(roi[2], slice(24, 46))


class TestFillSparse(unittest.TestCase):

    def _check_removed(self, long_axis):
        from flamingo_tools.annotation.fill import fill_sparse, select_slices

        gt, _ = synthetic_row(long_axis)
        cell = 4
        labels = gt.copy()
        labels[labels == cell] = 0
        z = select_slices(gt == cell, 1.0, spacing=2.0)
        edit = labels.copy()
        edit[z] = gt[z]
        result, shrunk = fill_sparse(labels, edit, z, 1.0)
        # The edited slices are exact, the other cells do not change, and the cell is recovered.
        np.testing.assert_array_equal(result[z], gt[z])
        others = (gt > 0) & (gt != cell)
        np.testing.assert_array_equal(result[others], gt[others])
        self.assertGreater(dice(result == cell, gt == cell), 0.9)
        self.assertEqual(shrunk, set())

    def test_removed_cell(self):
        for long_axis in ("z", "x"):
            with self.subTest(long_axis=long_axis):
                self._check_removed(long_axis)

    def test_unchanged_edit(self):
        from flamingo_tools.annotation.fill import fill_sparse

        gt, _ = synthetic_row()
        result, shrunk = fill_sparse(gt, gt.copy(), [10, 20, 30], 1.0)
        np.testing.assert_array_equal(result, gt)
        self.assertEqual(shrunk, set())

    def test_local_change(self):
        from flamingo_tools.annotation.fill import fill_sparse

        gt, _ = synthetic_row()
        cell = 4
        labels = gt.copy()
        # A cut tip at the upper end of the cell.
        labels[(gt == cell) & (np.indices(gt.shape)[0] >= 38)] = 0
        z = [20, 26, 32, 36, 41, 46]
        edit = labels.copy()
        edit[z] = gt[z]
        result, _ = fill_sparse(labels, edit, z, 1.0)
        # Only the intervals next to the changed slices 41 and 46 change.
        np.testing.assert_array_equal(result[:37], labels[:37])
        np.testing.assert_array_equal(result[47:], labels[47:])
        self.assertGreater(dice(result == cell, gt == cell), 0.97)

    def test_vanish_at_cap(self):
        from flamingo_tools.annotation.fill import fill_sparse

        gt, _ = synthetic_row()
        cell = 4
        labels = gt.copy()
        labels[labels == cell] = 0
        edit = labels.copy()
        # The cell is on slice 30 and absent on the cap slice 40: it vanishes halfway between them.
        edit[30] = gt[30]
        result, _ = fill_sparse(labels, edit, [20, 30, 40], 1.0, vanish=0.5)
        self.assertTrue((result[34] == cell).any())
        self.assertFalse((result[36:41] == cell).any())

    def test_boundary_moves(self):
        from flamingo_tools.annotation.fill import fill_sparse

        gt, _ = synthetic_row()
        cell, neighbour = 4, 5
        labels = gt.copy()
        # The neighbour holds the voxels of the cell next to it.
        near = ndimage.distance_transform_edt(gt != neighbour) <= 2
        labels[(gt == cell) & near] = neighbour
        z = [16, 20, 25, 30, 35, 40, 44]
        edit = labels.copy()
        edit[z] = gt[z]
        result, shrunk = fill_sparse(labels, edit, z, 1.0)
        self.assertGreater(dice(result == cell, gt == cell), dice(labels == cell, gt == cell))
        self.assertGreater(dice(result == cell, gt == cell), 0.97)
        self.assertEqual(shrunk, {neighbour})
        # A cell that the edit does not change keeps all its voxels.
        np.testing.assert_array_equal(result == 3, gt == 3)

    def test_moving_border_no_gap(self):
        from flamingo_tools.annotation.fill import fill_sparse

        # Two cells in one block share a border that moves by 12 pixels between two edited slices. Each cell on its
        # own would recede and advance with a gap of background between them midway.
        labels = np.zeros((11, 30, 20), dtype="uint32")
        labels[:, 4:20, 4:16] = 1
        labels[:, 20:26, 4:16] = 2
        edit = labels.copy()
        edit[10, 4:26, 4:16] = np.where(np.arange(4, 26) < 8, 1, 2)[:, None]
        result, _ = fill_sparse(labels, edit, [0, 10], 1.0)
        self.assertTrue((result[:, 4:26, 4:16] > 0).all())
        self.assertFalse((result[:, :4] > 0).any())
        # The border moves monotonically between the slices.
        border = [int(np.argmax(result[z, 4:26, 10] == 2)) + 4 for z in range(11)]
        self.assertEqual(border, sorted(border, reverse=True))
        self.assertTrue(8 < border[5] < 20)


if __name__ == "__main__":
    unittest.main()
