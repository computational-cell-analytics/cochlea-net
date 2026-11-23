import unittest

import numpy as np
from scipy import ndimage


def disc(shape, center, radius):
    y, x = np.indices(shape)
    return (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2


class TestCleanSlice(unittest.TestCase):

    def test_specks_and_holes(self):
        from flamingo_tools.annotation.cleanup import clean_slice

        before = np.zeros((40, 60), dtype="uint32")
        before[disc(before.shape, (20, 15), 8)] = 1
        before[disc(before.shape, (20, 30), 7) & (before == 0)] = 2
        before[disc(before.shape, (20, 48), 6)] = 3
        edit = before.copy()
        edit[disc(edit.shape, (20, 15), 9) & (edit == 0)] = 1
        edit[5, 5] = 1  # a stray pixel of the painted cell
        edit[20, 13] = 2  # a left-over pixel of the neighbour inside the cell
        edit[18:20, 16:18] = 0  # a gap of the brush
        edit[34, 48] = 3  # a pixel of a locked cell
        cleaned = clean_slice(edit, before, editable=[1, 2], voxel_size=1.0, min_area=4.0)
        self.assertEqual(cleaned[5, 5], 0)
        self.assertEqual(cleaned[20, 13], 1)
        self.assertTrue((cleaned[18:20, 16:18] == 1).all())
        # The locked cell is not cleaned, and the grown cell keeps its new voxels.
        self.assertEqual(cleaned[34, 48], 3)
        grown = (edit == 1) & (before == 0)
        grown[5, 5] = False
        self.assertTrue((cleaned[grown] == 1).all())

    def test_large_second_piece_stays(self):
        from flamingo_tools.annotation.cleanup import clean_slice

        before = np.zeros((30, 30), dtype="uint32")
        edit = before.copy()
        edit[5:10, 5:10] = 1
        edit[20:25, 20:25] = 1
        cleaned = clean_slice(edit, before, editable=[1], voxel_size=1.0, min_area=4.0)
        np.testing.assert_array_equal(cleaned, edit)


class TestCleanLabels(unittest.TestCase):

    def setUp(self):
        # Two touching blocks along x, and a locked block next to cell 1 along y.
        self.before = np.zeros((12, 30, 30), dtype="uint32")
        self.before[2:10, 5:15, 5:15] = 1
        self.before[2:10, 5:15, 15:25] = 2
        self.before[2:10, 15:22, 5:15] = 3

    def interleaved(self, labels, a, b):
        """The voxels of a or b whose 3x3x3 neighbourhood holds more of the other label."""
        fa = ndimage.uniform_filter((labels == a).astype(float), 3)
        fb = ndimage.uniform_filter((labels == b).astype(float), 3)
        return int(((labels == a) & (fb > fa)).sum() + ((labels == b) & (fa > fb)).sum())

    def test_smoothing_keeps_locked(self):
        from flamingo_tools.annotation.cleanup import clean_labels

        labels = self.before.copy()
        # The boundary between 1 and 2 moves by 2 voxels, with a jag of one voxel per row and stray voxels.
        rng = np.random.default_rng(0)
        for z in range(2, 10):
            for y in range(5, 15):
                labels[z, y, 15:17 + rng.integers(-1, 2)] = 1
        labels[4, 8, 19] = labels[6, 11, 20] = 1
        labels[5, 9, 13] = 2
        cleaned, splits = clean_labels(labels, self.before, [1, 2], voxel_size=1.0, sigma=1.0, band=2.0)
        self.assertEqual(splits, {})
        self.assertLess(self.interleaved(cleaned, 1, 2), self.interleaved(labels, 1, 2) / 2)
        np.testing.assert_array_equal(cleaned == 3, self.before == 3)
        # No gap opens between cell 1 and the locked cell 3.
        self.assertTrue((cleaned[2:10, 14, 6:14] == 1).all())

    def test_tip_stays(self):
        from flamingo_tools.annotation.cleanup import clean_labels

        # A thin tip of cell 2 grows into the background; the smoothing works on cell borders only.
        labels = self.before.copy()
        labels[5:7, 9:11, 25:29] = 2
        cleaned, _ = clean_labels(labels, self.before, [1, 2], voxel_size=1.0, sigma=1.0, band=2.0)
        np.testing.assert_array_equal(cleaned, labels)

    def test_pieces_and_holes(self):
        from flamingo_tools.annotation.cleanup import clean_labels

        labels = self.before.copy()
        labels[5, 8, 20] = 1  # a small piece of 1 inside cell 2
        labels[4:7, 8:11, 8:11] = 0  # a cavity in cell 1
        labels[5, 9, 9] = 1
        labels[1, 25:28, 1:10] = 2  # a large separate piece of 2, with a hole
        labels[1, 26, 5] = 0
        cleaned, splits = clean_labels(labels, self.before, [1, 2], voxel_size=1.0, sigma=0.0, band=2.0,
                                       split_size=20, new_label=iter([7]).__next__)
        self.assertEqual(cleaned[5, 8, 20], 2)
        self.assertTrue((cleaned[4:7, 8:11, 8:11] == 1).all())
        self.assertEqual(splits, {7: 2})
        self.assertTrue((cleaned[1, 25:28, 1:10] == 7).all())

    def test_plane_holes(self):
        from flamingo_tools.annotation.cleanup import clean_labels

        labels = self.before.copy()
        labels[5:7, 10:12, 17:25] = 0  # a tunnel along x through cell 2: a hole in the x planes only
        cleaned, _ = clean_labels(labels, self.before, [1, 2], voxel_size=1.0, sigma=0.0, band=2.0)
        np.testing.assert_array_equal(cleaned[5:7, 10:12, 17:24], 2)
        # A gap between two cells that opens to the outside is no hole of one of them, so it stays background.
        labels[4:8, 5:15, 14:16] = 0
        cleaned, _ = clean_labels(labels, self.before, [1, 2], voxel_size=1.0, sigma=0.0, band=2.0)
        self.assertTrue((cleaned[4:8, 5:15, 14:16] == 0).all())

    def test_edge_piece(self):
        from flamingo_tools.annotation.cleanup import clean_labels

        # A piece of cell 1 that touches cell 1 only at an edge, and cell 2 at a face.
        before = self.before.copy()
        # An annotated part of cell 1 with an edge contact, away from the change: it stays.
        before[10, 4, 6:9] = 1
        labels = before.copy()
        labels[3:5, 15:17, 15:18] = 1
        cleaned, splits = clean_labels(labels, before, [1, 2], voxel_size=1.0, sigma=0.0, band=2.0)
        self.assertEqual(splits, {})
        np.testing.assert_array_equal(cleaned[3:5, 15:17, 15:18], 2)
        np.testing.assert_array_equal(cleaned[10, 4, 6:9], 1)

    def test_no_change(self):
        from flamingo_tools.annotation.cleanup import clean_labels

        cleaned, splits = clean_labels(self.before, self.before, [1, 2], voxel_size=1.0)
        np.testing.assert_array_equal(cleaned, self.before)
        self.assertEqual(splits, {})


if __name__ == "__main__":
    unittest.main()
