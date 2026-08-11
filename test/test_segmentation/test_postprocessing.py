import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from elf.io import open_file
from skimage.data import binary_blobs
from skimage.measure import label, regionprops_table


class TestPostprocessing(unittest.TestCase):
    def _create_example_seg(self, tmp_dir):
        seg = binary_blobs(256, n_dim=3, volume_fraction=0.2)
        seg = label(seg)
        return seg

    def _test_postprocessing(self, spatial_statistics, threshold, **spatial_statistics_kwargs):
        from flamingo_tools.postprocessing.label_components import filter_segmentation

        with tempfile.TemporaryDirectory() as tmp_dir:
            example_seg = self._create_example_seg(tmp_dir)
            output_path = os.path.join(tmp_dir, "test-output.zarr")
            output_key = "seg-filtered"
            filter_segmentation(
                example_seg, output_path, spatial_statistics, threshold,
                output_key=output_key, **spatial_statistics_kwargs
            )
            self.assertTrue(os.path.exists(output_path))
            with open_file(output_path, "r") as f:
                filtered_seg = f[output_key][:]
            self.assertEqual(filtered_seg.shape, example_seg.shape)

    def test_nearest_neighbor_distance(self):
        from flamingo_tools.postprocessing.label_components import nearest_neighbor_distance

        self._test_postprocessing(nearest_neighbor_distance, threshold=5)

    def test_local_ripleys_k(self):
        from flamingo_tools.postprocessing.label_components import local_ripleys_k

        self._test_postprocessing(local_ripleys_k, threshold=0.5)

    def test_neighbors_in_radius(self):
        from flamingo_tools.postprocessing.label_components import neighbors_in_radius

        self._test_postprocessing(neighbors_in_radius, threshold=5)

    def test_compute_table_on_the_fly(self):
        from flamingo_tools.postprocessing.label_components import compute_table_on_the_fly
        from flamingo_tools.test_data import get_test_volume_and_segmentation

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, seg_path, _ = get_test_volume_and_segmentation(tmp_dir)
            segmentation = imageio.imread(seg_path)

        voxel_size = 0.38
        table = compute_table_on_the_fly(segmentation, voxel_size=voxel_size)

        properties = ("label", "bbox", "centroid")
        expected_table = regionprops_table(segmentation, properties=properties)
        expected_table = pd.DataFrame(expected_table)

        for (col, col_exp) in [
            ("label_id", "label"),
            ("anchor_x", "centroid-2"), ("anchor_y", "centroid-1"), ("anchor_z", "centroid-0"),
            ("bb_min_x", "bbox-2"), ("bb_min_y", "bbox-1"), ("bb_min_z", "bbox-0"),
            ("bb_max_x", "bbox-5"), ("bb_max_y", "bbox-4"), ("bb_max_z", "bbox-3"),
        ]:
            values = table[col].values.copy()
            if col != "label_id":
                values /= voxel_size
            self.assertTrue(np.allclose(values, expected_table[col_exp].values))


class TestDownscaledCentroids(unittest.TestCase):
    def setUp(self):
        from flamingo_tools.postprocessing.label_components import downscaled_centroids

        self.fn = downscaled_centroids
        self.centroids = [(0.0, 0.0, 0.0), (10.0, 20.0, 30.0), (11.0, 21.0, 31.0)]

    def test_scalar_scale_factor(self):
        # Reference values computed by hand: coordinates are divided by 10 and truncated,
        # ref_dimensions // 10 + 1 gives the array shape.
        array = self.fn(self.centroids, scale_factor=10, ref_dimensions=(20.0, 30.0, 40.0),
                        downsample_mode="accumulated")
        self.assertEqual(array.shape, (3, 4, 5))
        self.assertEqual(array[0, 0, 0], 1)
        self.assertEqual(array[1, 2, 3], 2)
        self.assertEqual(array.sum(), 3)

    def test_scalar_and_uniform_sequence_agree(self):
        expected = self.fn(self.centroids, scale_factor=10, ref_dimensions=(20.0, 30.0, 40.0))
        result = self.fn(self.centroids, scale_factor=[10, 10, 10], ref_dimensions=(20.0, 30.0, 40.0))
        np.testing.assert_array_equal(result, expected)

    def test_per_axis_scale_factor(self):
        # Axis order of scale_factor matches the centroid axis order.
        array = self.fn(self.centroids, scale_factor=[2, 10, 20], ref_dimensions=(20.0, 30.0, 40.0),
                        downsample_mode="capped")
        self.assertEqual(array.shape, (11, 4, 3))
        self.assertEqual(array[0, 0, 0], 1)
        # The second and third centroid fall into the same cell, which "capped" writes once.
        self.assertEqual(array[5, 2, 1], 1)
        self.assertEqual(array.sum(), 2)

    def test_shape_without_ref_dimensions(self):
        array = self.fn(self.centroids, scale_factor=[2, 10, 20])
        self.assertEqual(array.shape, (6, 3, 2))


def _dilate_and_trim_reference(arr_orig, edt, iterations, offset):
    """Loop implementation of dilate_and_trim, kept as the reference for the vectorized version.

    Callers must keep the foreground away from the array border by at least `iterations` + 1
    voxels: the neighbour lookup is unguarded, so it raises IndexError at the upper faces.
    """
    from scipy.ndimage import binary_dilation

    border_coords = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    for _ in range(iterations):
        arr_dilated = binary_dilation(arr_orig)
        for x in range(arr_dilated.shape[0]):
            for y in range(arr_dilated.shape[1]):
                for z in range(arr_dilated.shape[2]):
                    if arr_dilated[x, y, z] != 0 and arr_orig[x, y, z] == 0:
                        min_dist = float("inf")
                        for dx, dy, dz in border_coords:
                            nx, ny, nz = x + dx, y + dy, z + dz
                            if arr_orig[nx, ny, nz] == 1:
                                min_dist = min([min_dist, edt[nx, ny, nz]])
                        if edt[x, y, z] >= min_dist - offset:
                            arr_dilated[x, y, z] = 0
        arr_orig = arr_dilated
    return arr_dilated


class TestDilateAndTrim(unittest.TestCase):
    def setUp(self):
        from flamingo_tools.postprocessing.label_components import dilate_and_trim

        self.fn = dilate_and_trim
        self.rng = np.random.default_rng(0)

    def _make_case(self, iterations, size=6, density=0.8, max_value=1):
        """Seed and target inside a central region, with an (iterations + 2) margin so the
        reference implementation cannot reach the border."""
        margin = iterations + 2
        shape = (size + 2 * margin,) * 3
        inner = (slice(margin, margin + size),) * 3
        seed = np.zeros(shape, dtype=int)
        target = np.zeros(shape, dtype=int)
        if max_value == 1:
            seed[inner] = (self.rng.random((size,) * 3) > density).astype(int)
        else:
            seed[inner] = self.rng.integers(0, max_value + 1, size=(size,) * 3)
        target[inner] = (self.rng.random((size,) * 3) > density).astype(int)
        return seed, target

    def _assert_matches_reference(self, seed, target, iterations, offset):
        from scipy.ndimage import distance_transform_edt

        edt = distance_transform_edt(~target.astype(bool))
        expected = _dilate_and_trim_reference(seed.copy(), edt, iterations, offset)
        result = self.fn(seed.copy(), edt, iterations=iterations, offset=offset)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, expected.dtype)

    def test_matches_reference(self):
        for iterations in (1, 2, 5):
            for offset in (0.0, 0.4, 0.45):
                seed, target = self._make_case(iterations)
                if seed.sum() == 0 or target.sum() == 0:
                    continue
                with self.subTest(iterations=iterations, offset=offset):
                    self._assert_matches_reference(seed, target, iterations, offset)

    def test_matches_reference_for_production_parameters(self):
        # The values used by filter_cochlea_volume.
        seed, target = self._make_case(20, size=8, density=0.75)
        self._assert_matches_reference(seed, target, iterations=20, offset=0.4)

    def test_values_above_one_dilate_but_are_no_distance_source(self):
        seed, target = self._make_case(3, max_value=2)
        self._assert_matches_reference(seed, target, iterations=3, offset=0.4)

    def test_grows_towards_the_target_only(self):
        # A single seed voxel and a single target voxel: the mask may only grow along the axis
        # that decreases the distance to the target.
        from scipy.ndimage import distance_transform_edt

        shape = (11, 11, 11)
        seed = np.zeros(shape, dtype=int)
        seed[5, 5, 3] = 1
        target = np.zeros(shape, dtype=bool)
        target[5, 5, 8] = True
        edt = distance_transform_edt(~target)

        result = self.fn(seed.copy(), edt, iterations=3, offset=0.4)
        np.testing.assert_array_equal(np.argwhere(result), [[5, 5, 3], [5, 5, 4], [5, 5, 5], [5, 5, 6]])

    def test_result_is_monotone(self):
        seed, target = self._make_case(4)
        from scipy.ndimage import distance_transform_edt

        edt = distance_transform_edt(~target.astype(bool))
        previous = seed.astype(bool)
        for iterations in range(1, 5):
            result = self.fn(seed.copy(), edt, iterations=iterations, offset=0.4)
            self.assertTrue(np.all(result[previous]))
            previous = result


class TestVoxelSizeOrdering(unittest.TestCase):
    """voxel_size is (x, y, z) per .claude/conventions.md, while the arrays are ZYX."""

    VOXEL_SIZE = (1.0, 2.0, 4.0)

    def setUp(self):
        from flamingo_tools.measurements import _get_bounding_box_and_center
        from flamingo_tools.postprocessing.label_components import compute_table_on_the_fly

        self.compute_table = compute_table_on_the_fly
        self.get_bb = _get_bounding_box_and_center

        # One object spanning z 2:4, y 3:7, x 5:11 in a ZYX volume.
        self.shape = (20, 20, 20)
        self.bb_px = ((2, 4), (3, 7), (5, 11))
        self.segmentation = np.zeros(self.shape, dtype="uint16")
        self.segmentation[2:4, 3:7, 5:11] = 1

    def test_compute_table_scales_each_axis_with_its_own_voxel_size(self):
        table = self.compute_table(self.segmentation, voxel_size=self.VOXEL_SIZE)
        row = table[table.label_id == 1].iloc[0]

        vx, vy, vz = self.VOXEL_SIZE
        (z0, z1), (y0, y1), (x0, x1) = self.bb_px
        self.assertAlmostEqual(row.bb_min_z, z0 * vz, places=4)
        self.assertAlmostEqual(row.bb_max_z, z1 * vz, places=4)
        self.assertAlmostEqual(row.bb_min_y, y0 * vy, places=4)
        self.assertAlmostEqual(row.bb_max_y, y1 * vy, places=4)
        self.assertAlmostEqual(row.bb_min_x, x0 * vx, places=4)
        self.assertAlmostEqual(row.bb_max_x, x1 * vx, places=4)
        # The anchor is the centroid of the object.
        self.assertAlmostEqual(row.anchor_z, 0.5 * (z0 + z1 - 1) * vz, places=4)
        self.assertAlmostEqual(row.anchor_x, 0.5 * (x0 + x1 - 1) * vx, places=4)

    def test_bounding_box_converts_back_to_the_original_pixels(self):
        # Round-trip: the table is written in µm, the bounding box is read back in ZYX pixels.
        table = self.compute_table(self.segmentation, voxel_size=self.VOXEL_SIZE)
        bb, center = self.get_bb(table, 1, self.VOXEL_SIZE, self.shape, dilation=0)

        # dilation=0 gives a bb_extension of 2 on every side.
        for axis, (start, stop) in enumerate(self.bb_px):
            self.assertEqual(bb[axis].start, start - 2)
            self.assertEqual(bb[axis].stop, stop + 2)

        # The center must land inside the object.
        self.assertEqual(self.segmentation[center], 1)

    def test_isotropic_voxel_size_is_order_independent(self):
        table_scalar = self.compute_table(self.segmentation, voxel_size=0.38)
        table_tuple = self.compute_table(self.segmentation, voxel_size=(0.38, 0.38, 0.38))
        for column in ("anchor_x", "anchor_z", "bb_min_z", "bb_max_x"):
            np.testing.assert_allclose(table_scalar[column].values, table_tuple[column].values)


if __name__ == "__main__":
    unittest.main()
