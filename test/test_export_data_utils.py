import os
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


class TestExportOutputPath(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import export_output_path
        self.fn = export_output_path
        self.crop_center = [100.0, 200.0, 300.0]
        self.out_folder = os.path.join("base", "out")

    def expected(self, file_name):
        return os.path.join(self.out_folder, file_name)

    def test_without_crop(self):
        self.assertEqual(self.fn(self.out_folder, "PV"), self.expected("PV.tif"))
        self.assertEqual(self.fn(self.out_folder, "PV", ome_zarr=True), self.expected("PV.ome.zarr"))

    def test_with_crop(self):
        self.assertEqual(
            self.fn(self.out_folder, "PV", crop_center=self.crop_center, axis=2, suffix="apex"),
            self.expected("PV_crop_0100-0200-0300_axis-2_apex.tif"),
        )

    def test_ome_zarr_keeps_the_crop_suffix(self):
        # Regression: the OME-Zarr path used to drop the suffix, so crops at different positions
        # overwrote each other.
        paths = {
            self.fn(self.out_folder, "PV", ome_zarr=True, crop_center=center, axis=2, suffix=label)
            for center, label in [([100.0, 200.0, 300.0], "apex"), ([400.0, 500.0, 600.0], "base")]
        }
        self.assertEqual(len(paths), 2)
        self.assertTrue(all(p.endswith(".ome.zarr") for p in paths))
        self.assertIn(self.expected("PV_crop_0100-0200-0300_axis-2_apex.ome.zarr"), paths)


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


class TestResolveSourceName(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import resolve_source_name
        self.fn = resolve_source_name
        self.sources = {
            "PV": "image",
            "PV_resized": "image",
            "VGlut3": "image",
            "SGN_v2": "segmentation",
            "IHC_v4c": "segmentation",
            "synapse_v3_ihc_v4b": "spots",
        }

    def test_exact_name_wins_over_prefix(self):
        self.assertEqual(self.fn(self.sources, "PV", "image"), "PV")

    def test_alias_resolves_the_version(self):
        self.assertEqual(self.fn(self.sources, "SGN"), "SGN_v2")
        self.assertEqual(self.fn(self.sources, "IHC"), "IHC_v4c")
        self.assertEqual(self.fn(self.sources, "synapses"), "synapse_v3_ihc_v4b")

    def test_different_capitalization(self):
        sources = {"sgn": "segmentation", "PV": "image"}
        self.assertEqual(self.fn(sources, "SGN", "segmentation"), "sgn")

    def test_alias_respects_the_kind(self):
        sources = {"IHC_v4c": "segmentation", "IHC_annotations": "spots"}
        self.assertEqual(self.fn(sources, "IHC"), "IHC_v4c")

    def test_ambiguous_prefix_raises(self):
        # Neither version is pinned in SOURCE_ALIASES, so the alias stays ambiguous.
        sources = {"SGN_v8": "segmentation", "SGN_v9": "segmentation"}
        with self.assertRaises(ValueError):
            self.fn(sources, "SGN")

    def test_wrong_kind_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.sources, "PV", "segmentation")

    def test_unknown_name_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.sources, "Myo7a")

    def test_preferred_version_resolves_ambiguity(self):
        sources = {"IHC_v4c": "segmentation", "IHC_v11": "segmentation", "IHC_v2": "segmentation"}
        self.assertEqual(self.fn(sources, "IHC"), "IHC_v11")

    def test_unique_version_without_the_preferred_one(self):
        sources = {"IHC_v9": "segmentation"}
        self.assertEqual(self.fn(sources, "IHC"), "IHC_v9")


class TestSynapseSourceForIhc(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import synapse_source_for_ihc
        self.fn = synapse_source_for_ihc
        self.sources = {
            "IHC_v4c": "segmentation",
            "IHC_v9": "segmentation",
            "IHC_v11": "segmentation",
            "synapse_v3": "spots",
            "synapse_v3_ihc_v4c": "spots",
            "synapse_v3_ihc_v9": "spots",
            "synapse_v5_ihc_v9": "spots",
            "synapse_v3_ihc_v11": "spots",
            "synapse_v5_ihc_v11": "spots",
        }

    def test_matched_source(self):
        self.assertEqual(self.fn(self.sources, "IHC_v4c"), "synapse_v3_ihc_v4c")

    def test_preferred_source_resolves_ambiguity(self):
        self.assertEqual(self.fn(self.sources, "IHC_v11"), "synapse_v3_ihc_v11")

    def test_several_matched_sources_raise(self):
        # Neither candidate for IHC_v9 is pinned in SOURCE_ALIASES.
        with self.assertRaises(ValueError):
            self.fn(self.sources, "IHC_v9")

    def test_no_matched_source_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.sources, "IHC_v9")

    def test_name_without_version_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.sources, "IHC")


class TestFindCropFiles(unittest.TestCase):

    def setUp(self):
        import tempfile
        from flamingo_tools.export_data_utils import find_crop_files
        self.fn = find_crop_files
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.crop_center = [10.0, 20.0, 30.0]

        self.files = {
            "scale4": ["PV_crop_0010-0020-0030_axis-0_apex.tif", "SGN_v2_crop_0010-0020-0030_axis-0_apex.tif",
                       "PV_crop_0040-0050-0060_axis-0_base.tif", "notes.txt"],
            "scale4_dilation8": ["PV_crop_0010-0020-0030_axis-0_apex.tif"],
            "scale2": ["PV_crop_0010-0020-0030_axis-0_apex.tif"],
        }
        for folder, names in self.files.items():
            os.makedirs(os.path.join(self.tmp_dir.name, folder))
            for name in names:
                open(os.path.join(self.tmp_dir.name, folder, name), "w").close()

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_grouped_by_folder(self):
        found = self.fn(self.tmp_dir.name, self.crop_center, axis=0, suffix="apex")

        self.assertEqual(
            [os.path.basename(folder) for folder in found], ["scale2", "scale4", "scale4_dilation8"]
        )
        self.assertEqual(
            [os.path.basename(path) for path in found[os.path.join(self.tmp_dir.name, "scale4")]],
            ["PV_crop_0010-0020-0030_axis-0_apex.tif", "SGN_v2_crop_0010-0020-0030_axis-0_apex.tif"],
        )

    def test_other_position_and_other_files_excluded(self):
        found = self.fn(self.tmp_dir.name, self.crop_center, axis=0, suffix="apex")
        names = [os.path.basename(path) for paths in found.values() for path in paths]

        self.assertNotIn("PV_crop_0040-0050-0060_axis-0_base.tif", names)
        self.assertNotIn("notes.txt", names)

    def test_no_match_returns_empty(self):
        self.assertEqual(self.fn(self.tmp_dir.name, [1.0, 2.0, 3.0], axis=0, suffix="apex"), {})


class TestLayerKind(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.export_data_utils import layer_kind
        self.fn = layer_kind
        self.sources = {
            "PV": "image",
            "IHC_v4c": "segmentation",
            "synapse_v3_ihc_v4c": "spots",
        }

    def test_image_source(self):
        self.assertEqual(self.fn(self.sources, "PV_crop_0010-0020-0030_apex.tif"), "image")

    def test_segmentation_source(self):
        self.assertEqual(self.fn(self.sources, "IHC_v4c_crop_0010-0020-0030_apex.tif"), "labels")

    def test_spots_source(self):
        self.assertEqual(self.fn(self.sources, "synapse_v3_ihc_v4c_crop_0010-0020-0030_apex.tif"), "labels")

    def test_derived_name_of_a_segmentation(self):
        # The marker and subtypes exports append to the segmentation name.
        self.assertEqual(self.fn(self.sources, "IHC_v4c_marker_positive_crop_0010-0020-0030_apex.tif"), "labels")

    def test_unknown_name_falls_back_to_image(self):
        self.assertEqual(self.fn(self.sources, "Myo7a_crop_0010-0020-0030_apex.tif"), "image")


if __name__ == "__main__":
    unittest.main()
