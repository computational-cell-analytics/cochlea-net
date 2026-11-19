import unittest

import numpy as np


def two_cells():
    """Two touching blocks along x: 1 at x 5-14, 2 at x 15-24."""
    labels = np.zeros((14, 30, 30), dtype="uint32")
    labels[2:12, 5:15, 5:15] = 1
    labels[2:12, 5:15, 15:25] = 2
    return labels


class TestRemoveParticles(unittest.TestCase):

    def test_particles(self):
        from flamingo_tools.annotation.dataset_cleanup import remove_particles

        labels = two_cells()
        labels[5:7, 20:22, 8:10] = 3  # isolated inside the volume
        labels[5, 4, 6:9] = 7  # one slice on cell 1
        labels[5, 1:4, 7] = 4  # touches only the particle 7 and reaches beyond its box
        labels[0, 25:27, 25:27] = 5  # tiny at the border
        labels[0:2, 25:30, 0:5] = 6  # a cut cell at the border
        labels[12, 14, 14] = 2  # a piece of 2 with an edge contact to 2 and a face contact to 1
        cleaned, records = remove_particles(labels, min_size=100, min_border_size=10)
        expected = two_cells()
        expected[5, 4, 6:9] = 1
        expected[5, 1:4, 7] = 1
        expected[0:2, 25:30, 0:5] = 6
        expected[12, 14, 14] = 1
        np.testing.assert_array_equal(cleaned, expected)
        found = {(r["kind"], r["label_id"], r["target"], r["n_voxels"]) for r in records}
        self.assertEqual(found, {("label", 3, 0, 8), ("label", 4, 1, 3), ("label", 7, 1, 3), ("label", 5, 0, 4),
                                 ("piece", 2, 1, 1)})
        self.assertEqual({r["label_id"]: r["at_border"] for r in records},
                         {3: False, 4: False, 7: False, 5: True, 2: False})

    def test_high_ids(self):
        from flamingo_tools.annotation.dataset_cleanup import remove_particles

        labels = two_cells().astype("uint64") * 4000000000
        labels[5, 4, 6:9] = 7
        cleaned, records = remove_particles(labels, min_size=100)
        self.assertEqual(cleaned.dtype, labels.dtype)
        self.assertTrue((cleaned[5, 4, 6:9] == 4000000000).all())
        self.assertEqual(records[0]["target"], 4000000000)


class TestFillHoles(unittest.TestCase):

    def test_enclosed_background(self):
        from flamingo_tools.annotation.dataset_cleanup import fill_enclosed_background

        labels = two_cells()
        labels[4:7, 8:11, 8:11] = 0  # a cavity in cell 1, larger than max_island
        labels[5:7, 8:10, 14:16] = 0  # an island between 1 and 2
        labels[3:12, 12, 13:17] = 0  # a gap between 1 and 2 that opens to the outside
        filled, records = fill_enclosed_background(labels, max_island=10)
        self.assertTrue((filled[4:7, 8:11, 8:11] == 1).all())
        self.assertTrue((filled[5:7, 8:10, 14] == 1).all())
        self.assertTrue((filled[5:7, 8:10, 15] == 2).all())
        self.assertTrue((filled[3:12, 12, 13:17] == 0).all())
        self.assertEqual(sorted((r["n_voxels"], r["label_ids"], r["filled"]) for r in records),
                         [(8, [1, 2], True), (27, [1], True)])

        _, records = fill_enclosed_background(labels, max_island=5)
        self.assertIn((8, [1, 2], False), [(r["n_voxels"], r["label_ids"], r["filled"]) for r in records])

    def test_plane_holes(self):
        from flamingo_tools.annotation.dataset_cleanup import fill_plane_holes

        labels = two_cells()
        labels[6, 10, 17:25] = 0  # a tunnel into cell 2 that opens at x 24: a hole in the x planes only
        labels[3:6, 6:9, 18:25] = 0  # a wider tunnel: holes of 9 pixels
        labels[6:12, 8, 14:16] = 0  # a slit between 1 and 2 that opens at the top
        filled, records = fill_plane_holes(labels, max_area=4)
        self.assertTrue((filled[6, 10, 17:25] == 2).all())
        self.assertTrue((filled[3:6, 6:9, 18:25] == 0).all())
        self.assertTrue((filled[6:12, 8, 14:16] == 0).all())
        self.assertEqual({(r["axis"], r["n_voxels"], r["filled"]) for r in records}, {(2, 1, True), (2, 9, False)})
        self.assertIn((6, 10, 20), [r["center"] for r in records])


class TestRemapHighIds(unittest.TestCase):

    def test_high_ids(self):
        from flamingo_tools.annotation.dataset_cleanup import remap_high_ids

        labels = np.zeros((4, 6, 6), dtype="uint64")
        labels[0, :2, :2] = 7
        labels[1, :2, :2] = 300
        labels[2, :2, :2] = 111111111111
        labels[3, :2, :2] = 30993099
        remapped, mapping = remap_high_ids(labels, limit=1000)
        self.assertEqual(remapped.dtype, np.uint32)
        self.assertEqual(mapping, {30993099: 301, 111111111111: 302})
        np.testing.assert_array_equal(remapped[:2], labels[:2])
        self.assertTrue((remapped[2, :2, :2] == 302).all())
        self.assertTrue((remapped[3, :2, :2] == 301).all())
        np.testing.assert_array_equal(remapped == 0, labels == 0)

    def test_no_high_ids(self):
        from flamingo_tools.annotation.dataset_cleanup import remap_high_ids

        labels = np.zeros((2, 3, 3), dtype="uint64")
        labels[0] = 5
        remapped, mapping = remap_high_ids(labels, limit=10)
        self.assertEqual(mapping, {})
        np.testing.assert_array_equal(remapped, labels)

    def test_limit_exceeded(self):
        from flamingo_tools.annotation.dataset_cleanup import remap_high_ids

        labels = np.array([[0, 10, 11, 50]], dtype="uint64")
        with self.assertRaises(ValueError):
            remap_high_ids(labels, limit=11)


if __name__ == "__main__":
    unittest.main()
