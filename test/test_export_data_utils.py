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

    def test_roi_halo_none_with_axis_crops_full_plane(self):
        # roi_halo omitted but axis=0 (x) given: full extent on Z, Y; single-pixel slice on X.
        start, stop = self.fn(
            self.crop_center, None, voxel_size=1.0, scale=0, shape=self.shape, axis=0
        )
        np.testing.assert_array_equal(start, [0, 0, 100])
        np.testing.assert_array_equal(stop, [1000, 1000, 101])

    def test_roi_halo_and_axis_none_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.crop_center, None, voxel_size=1.0, scale=0, shape=self.shape)

    def test_anisotropic_voxel_size(self):
        # voxel_size is (x, y, z) and must be reversed to ZYX before dividing the center.
        # center = [300 / 4, 200 / 2, 100 / 2] = [75, 100, 50], halo (ZYX) = [30, 20, 10].
        start, stop = self.fn(
            self.crop_center, self.roi_halo, voxel_size=[2.0, 2.0, 4.0], scale=0, shape=self.shape
        )
        np.testing.assert_array_equal(start, [45, 80, 40])
        np.testing.assert_array_equal(stop, [105, 120, 60])

    def test_single_value_voxel_size_is_isotropic(self):
        start, stop = self.fn(self.crop_center, self.roi_halo, voxel_size=[2.0], scale=0, shape=self.shape)
        expected = self.fn(self.crop_center, self.roi_halo, voxel_size=2.0, scale=0, shape=self.shape)
        np.testing.assert_array_equal(start, expected[0])
        np.testing.assert_array_equal(stop, expected[1])

    def test_anisotropic_voxel_size_at_higher_scale(self):
        # scale=1 doubles the effective voxel size on every axis:
        # center = [300 / 8, 200 / 4, 100 / 4] = [38 (37.5 rounds to even), 50, 25].
        start, stop = self.fn(
            self.crop_center, self.roi_halo, voxel_size=[2.0, 2.0, 4.0], scale=1, shape=self.shape
        )
        np.testing.assert_array_equal(start, [8, 30, 15])
        np.testing.assert_array_equal(stop, [68, 70, 35])

    def test_invalid_voxel_size_length_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.crop_center, self.roi_halo, voxel_size=[1.0, 2.0], scale=0, shape=self.shape)


class TestCropFilterVolume(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import crop_filter_volume
        self.fn = crop_filter_volume

    def test_shape_matches_crop_when_fully_covered(self):
        # filter_volume large enough to cover the requested crop after upscaling.
        filter_volume = np.ones((10, 10, 10), dtype=bool)
        start = np.array([5, 5, 5])
        stop = np.array([15, 15, 15])
        result = self.fn(filter_volume, start, stop, us_factor=2)
        self.assertEqual(result.shape, (10, 10, 10))
        self.assertTrue(result.all())

    def test_zero_pads_when_crop_exceeds_filter_volume_extent(self):
        # filter_volume only covers a small region (e.g. built from a segmentation table's
        # extent); a whole-plane crop (axis given, no roi_halo) can request a much larger
        # region -- the result must still have shape == stop - start, zero-padded outside
        # the covered extent, per Part B's roi_halo=None/whole-plane behavior.
        filter_volume = np.ones((5, 5, 5), dtype=bool)
        start = np.array([0, 0, 0])
        stop = np.array([100, 100, 100])
        result = self.fn(filter_volume, start, stop, us_factor=2)
        self.assertEqual(result.shape, (100, 100, 100))
        # Covered region (filter_volume upscaled by us_factor=2 -> 10x10x10) stays True...
        self.assertTrue(result[:10, :10, :10].all())
        # ...everything beyond it is zero-padded (False), not silently truncated.
        self.assertFalse(result[10:, :, :].any())
        self.assertFalse(result[:, 10:, :].any())
        self.assertFalse(result[:, :, 10:].any())

    def test_per_axis_us_factor(self):
        # One filter cell spans 4 pixels in Z, 2 in Y, 1 in X, so a single True cell at the origin
        # covers a 4 x 2 x 1 pixel block.
        filter_volume = np.zeros((3, 3, 3), dtype=bool)
        filter_volume[0, 0, 0] = True
        result = self.fn(filter_volume, np.array([0, 0, 0]), np.array([8, 8, 8]), us_factor=[4, 2, 1])
        self.assertEqual(result.shape, (8, 8, 8))
        self.assertTrue(result[:4, :2, :1].all())
        self.assertFalse(result[4:, :, :].any())
        self.assertFalse(result[:, 2:, :].any())
        self.assertFalse(result[:, :, 1:].any())

    def test_fractional_us_factor(self):
        # 2.5 pixels per filter cell: pixel i maps to cell floor(i / 2.5).
        filter_volume = np.zeros((4, 4, 4), dtype=bool)
        filter_volume[1, 1, 1] = True
        result = self.fn(filter_volume, np.array([0, 0, 0]), np.array([10, 10, 10]), us_factor=2.5)
        self.assertEqual(result.shape, (10, 10, 10))
        expected = np.array([int(i // 2.5) == 1 for i in range(10)])
        np.testing.assert_array_equal(result.any(axis=(1, 2)), expected)
        np.testing.assert_array_equal(result[3, 3, :], expected)

    def test_us_factor_below_one_downsamples(self):
        # us_factor < 1 happens when the export scale is coarser than the filter volume.
        filter_volume = np.zeros((8, 8, 8), dtype=bool)
        filter_volume[4, 4, 4] = True
        result = self.fn(filter_volume, np.array([0, 0, 0]), np.array([4, 4, 4]), us_factor=0.5)
        self.assertEqual(result.shape, (4, 4, 4))
        self.assertTrue(result[2, 2, 2])
        self.assertEqual(result.sum(), 1)

    def test_non_positive_us_factor_raises(self):
        with self.assertRaises(ValueError):
            self.fn(np.ones((2, 2, 2), dtype=bool), np.array([0, 0, 0]), np.array([2, 2, 2]), us_factor=0)


class TestFilterVolumeDownscaleFactors(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import filter_volume_downscale_factors
        self.fn = filter_volume_downscale_factors

    def test_isotropic_matches_historical_factor(self):
        self.assertEqual(self.fn(0.38), (48, 48, 48))
        self.assertEqual(self.fn([0.38, 0.38, 0.38]), (48, 48, 48))

    def test_anisotropic_la_vision(self):
        self.assertEqual(self.fn([1.887779, 1.887779, 3.0]), (10, 10, 6))

    def test_factor_is_at_least_one(self):
        self.assertEqual(self.fn(100.0), (1, 1, 1))


class TestCropSuffix(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import crop_suffix
        self.fn = crop_suffix

    def test_without_axis(self):
        self.assertEqual(self.fn([100.4, 200.6, 300.0]), "_crop_0100-0201-0300")

    def test_with_axis(self):
        self.assertEqual(self.fn([100.4, 200.6, 300.0], axis=1), "_crop_0100-0201-0300_axis-1")

    def test_with_suffix(self):
        self.assertEqual(self.fn([100.4, 200.6, 300.0], suffix="apex"), "_crop_0100-0201-0300_apex")

    def test_with_axis_and_suffix(self):
        self.assertEqual(
            self.fn([100.4, 200.6, 300.0], axis=2, suffix="apex"), "_crop_0100-0201-0300_axis-2_apex"
        )


if __name__ == "__main__":
    unittest.main()
