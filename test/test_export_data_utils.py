import unittest

import numpy as np


class TestComputeCropBb(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import compute_crop_bb
        self.fn = compute_crop_bb
        self.crop_center = [100.0, 200.0, 300.0]  # x, y, z
        self.roi_halo = [10, 20, 30]  # halo_x, halo_y, halo_z
        self.shape = (1000, 1000, 1000)  # Z, Y, X

    def test_3d_crop(self):
        start, stop = self.fn(self.crop_center, self.roi_halo, voxel_size=1.0, scale=0, shape=self.shape)
        # ZYX order: center = [300, 200, 100], halo = [30, 20, 10]
        np.testing.assert_array_equal(start, [270, 180, 90])
        np.testing.assert_array_equal(stop, [330, 220, 110])

    def test_axis_0_collapses_x(self):
        # axis 0 (x) maps to array dim 2 (X in ZYX).
        start, stop = self.fn(
            self.crop_center, self.roi_halo, voxel_size=1.0, scale=0, shape=self.shape, axis=0
        )
        np.testing.assert_array_equal(start, [270, 180, 100])
        np.testing.assert_array_equal(stop, [330, 220, 101])

    def test_axis_1_collapses_y(self):
        # axis 1 (y) maps to array dim 1 (Y in ZYX).
        start, stop = self.fn(
            self.crop_center, self.roi_halo, voxel_size=1.0, scale=0, shape=self.shape, axis=1
        )
        np.testing.assert_array_equal(start, [270, 200, 90])
        np.testing.assert_array_equal(stop, [330, 201, 110])

    def test_axis_2_collapses_z(self):
        # axis 2 (z) maps to array dim 0 (Z in ZYX).
        start, stop = self.fn(
            self.crop_center, self.roi_halo, voxel_size=1.0, scale=0, shape=self.shape, axis=2
        )
        np.testing.assert_array_equal(start, [300, 180, 90])
        np.testing.assert_array_equal(stop, [301, 220, 110])

    def test_invalid_axis_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.crop_center, self.roi_halo, voxel_size=1.0, scale=0, shape=self.shape, axis=3)


class TestCropSuffix(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import crop_suffix
        self.fn = crop_suffix

    def test_without_axis(self):
        self.assertEqual(self.fn([100.4, 200.6, 300.0]), "_crop_0100-0201-0300")

    def test_with_axis(self):
        self.assertEqual(self.fn([100.4, 200.6, 300.0], axis=1), "_crop_0100-0201-0300_1")


if __name__ == "__main__":
    unittest.main()
