import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd
import tifffile
import zarr


def _write_zarr(path, arrays, chunks=None):
    f = zarr.open(path, mode="w")
    for key, data in arrays.items():
        f.create_array(key, data=data, chunks=data.shape if chunks is None else chunks)
    return path


def _imread(path):
    """Read an exported TIF. tifffile drops the singleton Z axis of an OME-TIFF, so restore it."""
    data = tifffile.imread(path)
    return data[None] if data.ndim == 2 else data


def _reference_mean(data, factor):
    """Mean over the factor^3 blocks, with the edge blocks averaged over the voxels inside the volume."""
    out_shape = tuple(-(-length // factor) for length in data.shape)
    out = np.zeros(out_shape, dtype="float64")
    for index in np.ndindex(out_shape):
        block = data[tuple(slice(i * factor, (i + 1) * factor) for i in index)]
        out[index] = block.mean()
    return out


class TestPyramidKey(unittest.TestCase):

    def test_pyramid_key(self):
        from flamingo_tools.file_utils import pyramid_key

        self.assertEqual(pyramid_key("s0", 3), "s3")
        self.assertEqual(pyramid_key("setup1/timepoint0/s0", 3), "setup1/timepoint0/s3")
        self.assertEqual(pyramid_key("s1", 2), "s3")
        self.assertEqual(pyramid_key("s0", 0), "s0")
        self.assertIsNone(pyramid_key("raw", 1))
        self.assertIsNone(pyramid_key(None, 1))

    def test_is_label_dtype(self):
        from flamingo_tools.file_utils import is_label_dtype

        self.assertTrue(is_label_dtype("uint64"))
        self.assertTrue(is_label_dtype("int32"))
        self.assertFalse(is_label_dtype("uint16"))
        self.assertFalse(is_label_dtype("float32"))


class TestOpenScaleLevel(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        rng = np.random.default_rng(0)
        # Odd sizes, so that the edge blocks of the downsampling are incomplete.
        self.image = rng.integers(0, 1000, size=(9, 10, 11), dtype="uint16")
        self.labels = rng.integers(0, 2 ** 40, size=(9, 10, 11), dtype="uint64")

    def tearDown(self):
        self.tmp.cleanup()

    def test_pyramid_level_is_used(self):
        from flamingo_tools.file_utils import open_scale_level

        level1 = np.full((5, 5, 6), 7, dtype="uint16")
        path = _write_zarr(os.path.join(self.folder, "data.ome.zarr"), {"s0": self.image, "s1": level1})
        level = open_scale_level(path, "s0", 1)

        self.assertEqual(level.factor, 1)
        self.assertEqual(level.shape, (5, 5, 6))
        np.testing.assert_array_equal(level.read(), level1)
        self.assertEqual(level.read().dtype, np.dtype("uint16"))

    def test_missing_level_is_downsampled(self):
        from flamingo_tools.file_utils import open_scale_level

        path = _write_zarr(os.path.join(self.folder, "data.zarr"), {"s0": self.image})
        level = open_scale_level(path, "s0", 1)

        self.assertEqual(level.factor, 2)
        self.assertEqual(level.shape, (5, 5, 6))
        result = level.read()
        self.assertEqual(result.dtype, np.dtype("float32"))
        np.testing.assert_allclose(result, _reference_mean(self.image, 2), rtol=1e-6)

    def test_level_with_wrong_shape_is_not_used(self):
        from flamingo_tools.file_utils import open_scale_level

        path = _write_zarr(
            os.path.join(self.folder, "data.zarr"), {"s0": self.image, "s1": np.zeros((9, 5, 6), dtype="uint16")},
        )
        level = open_scale_level(path, "s0", 1)
        self.assertEqual(level.factor, 2)

    def test_key_without_pyramid_is_downsampled(self):
        from flamingo_tools.file_utils import open_scale_level

        path = _write_zarr(os.path.join(self.folder, "segmentation.zarr"), {"segmentation": self.labels})
        level = open_scale_level(path, "segmentation", 2)

        self.assertEqual(level.shape, (3, 3, 3))
        result = level.read()
        self.assertEqual(result.dtype, np.dtype("uint64"))
        np.testing.assert_array_equal(result, self.labels[::4, ::4, ::4])

    def test_labels_option_overrides_the_data_type(self):
        from flamingo_tools.file_utils import open_scale_level

        path = _write_zarr(os.path.join(self.folder, "data.zarr"), {"s0": self.image})
        result = open_scale_level(path, "s0", 1).read(labels=True)
        np.testing.assert_array_equal(result, self.image[::2, ::2, ::2])

    def test_crop_matches_the_full_read_over_several_blocks(self):
        from flamingo_tools import file_utils

        path = _write_zarr(os.path.join(self.folder, "data.zarr"), {"s0": self.image}, chunks=(2, 2, 2))
        with mock.patch.object(file_utils, "_MIN_BLOCK_LENGTH", 2):
            level = file_utils.open_scale_level(path, "s0", 1)
            full = level.read()
            crop = level.read(start=[1, 2, 1], stop=[5, 5, 4])
        np.testing.assert_allclose(full, _reference_mean(self.image, 2), rtol=1e-6)
        np.testing.assert_array_equal(crop, full[1:5, 2:5, 1:4])

    def test_tif_ignores_the_key(self):
        from flamingo_tools.file_utils import open_scale_level

        path = os.path.join(self.folder, "data.tif")
        tifffile.imwrite(path, self.image)
        level = open_scale_level(path, "s0", 1)
        np.testing.assert_allclose(level.read(), _reference_mean(self.image, 2), rtol=1e-6)

        data = open_scale_level(path, "s0", 0).read()
        np.testing.assert_array_equal(data, self.image)
        # The tif file is memory-mapped with write access. A change of the data must not change the file.
        data[:] = 1
        del data
        np.testing.assert_array_equal(tifffile.imread(path), self.image)

    def test_n5_bdv_key(self):
        import z5py
        from flamingo_tools.file_utils import open_scale_level

        path = os.path.join(self.folder, "data.n5")
        level1 = np.full((5, 5, 6), 3, dtype="uint16")
        with z5py.File(path, mode="w") as f:
            f.create_dataset("setup0/timepoint0/s0", data=self.image, chunks=(4, 4, 4))
            f.create_dataset("setup0/timepoint0/s1", data=level1, chunks=(4, 4, 4))

        np.testing.assert_array_equal(open_scale_level(path, "setup0/timepoint0/s0", 1).read(), level1)
        np.testing.assert_allclose(
            open_scale_level(path, "setup0/timepoint0/s0", 2).read(), _reference_mean(self.image, 4), rtol=1e-6,
        )


class TestExportLowerResolution(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.output_folder = os.path.join(self.folder, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_component_filter_keeps_large_label_ids(self):
        from flamingo_tools.export_data import export_lower_resolution

        # float32 rounds 2^24 + 1 to 2^24, so filtering after a cast to float32 drops this object.
        keep_id, drop_id = 2 ** 24 + 1, 2 ** 24 + 2
        segmentation = np.zeros((4, 4, 4), dtype="uint64")
        segmentation[:2] = keep_id
        segmentation[2:] = drop_id
        path = _write_zarr(os.path.join(self.folder, "SGN.ome.zarr"), {"s0": segmentation})
        table = pd.DataFrame({"label_id": [keep_id, drop_id], "component_labels": [1, 2]})

        export_lower_resolution(path, self.output_folder, [0], table=table, filter_by_components=[1])

        result = _imread(os.path.join(self.output_folder, "scale0", "SGN.tif"))
        self.assertEqual(result.dtype, np.dtype("float32"))
        self.assertTrue(np.all(result[:2] > 0))
        self.assertTrue(np.all(result[2:] == 0))

    def test_clip_binarize_and_metadata(self):
        from flamingo_tools.export_data import export_lower_resolution

        image = np.arange(64, dtype="uint16").reshape(4, 4, 4) * 50
        labels = (np.arange(64, dtype="uint64").reshape(4, 4, 4) + 1) * 100
        image_path = _write_zarr(os.path.join(self.folder, "PV.ome.zarr"), {"s0": image})
        label_path = _write_zarr(os.path.join(self.folder, "IHC.ome.zarr"), {"s0": labels})

        export_lower_resolution(image_path, self.output_folder, 0, max_intensity=1400, voxel_size=0.5)
        export_lower_resolution(label_path, self.output_folder, 0, max_intensity=1400)
        export_lower_resolution(image_path, self.output_folder, 0, name="PV_binary", binarize=True)

        clipped = _imread(os.path.join(self.output_folder, "scale0", "PV.tif"))
        np.testing.assert_array_equal(clipped, np.minimum(image, 1400))
        not_clipped = _imread(os.path.join(self.output_folder, "scale0", "IHC.tif"))
        np.testing.assert_array_equal(not_clipped, labels.astype("float32"))
        binary = _imread(os.path.join(self.output_folder, "scale0", "PV_binary.tif"))
        np.testing.assert_array_equal(binary, (image > 0).astype("float32"))

        with tifffile.TiffFile(os.path.join(self.output_folder, "scale0", "PV.tif")) as f:
            self.assertIn('PhysicalSizeX="0.5"', f.ome_metadata)

    def test_crop_slice_and_existing_output(self):
        from flamingo_tools.export_data import export_lower_resolution

        image = np.random.default_rng(1).integers(0, 100, size=(16, 16, 16), dtype="uint16")
        path = _write_zarr(os.path.join(self.folder, "data.zarr"), {"s0": image})
        kwargs = dict(crop_center=[4.0, 4.0, 4.0], axis=2, suffix="apex", voxel_size=1.0)

        export_lower_resolution(path, self.output_folder, [1], **kwargs)
        out_path = os.path.join(self.output_folder, "scale1", "data_crop_0004-0004-0004_axis-2_apex.tif")
        result = _imread(out_path)
        np.testing.assert_allclose(result, _reference_mean(image, 2)[2:3], rtol=1e-6)

        tifffile.imwrite(out_path, np.zeros(1, dtype="float32"))
        export_lower_resolution(path, self.output_folder, [1], **kwargs)
        self.assertEqual(_imread(out_path).shape, (1,))
        export_lower_resolution(path, self.output_folder, [1], force=True, **kwargs)
        self.assertEqual(_imread(out_path).shape, (1, 8, 8))

    def test_cochlea_mask(self):
        from flamingo_tools.export_data import export_lower_resolution

        image = np.ones((64, 64, 64), dtype="uint16")
        path = _write_zarr(os.path.join(self.folder, "PV.ome.zarr"), {"s0": image})
        # Objects near the origin. With 10 µm voxels, one mask cell covers 2 voxels.
        anchors = np.array([[50.0, 50.0, 50.0], [60.0, 50.0, 50.0], [50.0, 60.0, 50.0], [50.0, 50.0, 60.0]])
        sgn_table = pd.DataFrame({
            "label_id": [1, 2, 3, 4], "component_labels": [1, 1, 1, 1],
            "anchor_x": anchors[:, 0], "anchor_y": anchors[:, 1], "anchor_z": anchors[:, 2],
        })

        export_lower_resolution(path, self.output_folder, [0], sgn_table=sgn_table, filter_dilation_iterations=1,
                                voxel_size=10.0)

        result = _imread(os.path.join(self.output_folder, "scale0_dilation1", "PV.tif"))
        self.assertEqual(result[5, 5, 5], 1)
        self.assertEqual(result[60, 60, 60], 0)

    def test_labels_of_a_16_bit_segmentation(self):
        from flamingo_tools.export_data import export_lower_resolution

        segmentation = np.repeat(np.arange(1, 9, dtype="uint16") * 1000, 64).reshape(8, 8, 8)
        path = _write_zarr(os.path.join(self.folder, "seg.zarr"), {"s0": segmentation})
        table = pd.DataFrame({"label_id": np.arange(1, 9) * 1000, "component_labels": [1] * 8})

        # The component filter marks the input as a segmentation: nearest neighbor sampling, no clipping.
        export_lower_resolution(path, self.output_folder, [1], table=table, filter_by_components=[1],
                                max_intensity=1400)
        export_lower_resolution(path, self.output_folder, [1], name="seg_labels", labels=True)
        export_lower_resolution(path, self.output_folder, [1], name="seg_mean")

        expected = segmentation[::2, ::2, ::2].astype("float32")
        np.testing.assert_array_equal(_imread(os.path.join(self.output_folder, "scale1", "seg.tif")), expected)
        np.testing.assert_array_equal(_imread(os.path.join(self.output_folder, "scale1", "seg_labels.tif")), expected)
        # Without the option, 16 bit data is image data: the mean of the slices 2 (3000) and 3 (4000).
        self.assertEqual(_imread(os.path.join(self.output_folder, "scale1", "seg_mean.tif"))[1, 0, 0], 3500)

    def test_filter_requires_the_table(self):
        from flamingo_tools.export_data import export_lower_resolution

        with self.assertRaises(ValueError):
            export_lower_resolution("unused.zarr", self.output_folder, [0], filter_by_components=[1])

    def test_crop_outside_the_volume(self):
        from flamingo_tools.export_data import export_lower_resolution

        image = np.ones((8, 8, 8), dtype="uint16")
        path = _write_zarr(os.path.join(self.folder, "data.zarr"), {"s0": image})
        for scale, crop in ((0, dict(axis=0)), (1, dict(axis=0)), (0, dict(roi_halo=[2, 2, 2]))):
            with self.subTest(scale=scale, crop=crop), self.assertRaisesRegex(ValueError, "outside the volume"):
                export_lower_resolution(path, self.output_folder, [scale], crop_center=[100.0, 1.0, 1.0],
                                        voxel_size=1.0, **crop)
            # A failed export leaves no file that a later run would skip as existing.
            self.assertEqual(os.listdir(os.path.join(self.output_folder, f"scale{scale}")), [])

    def test_negative_scale(self):
        from flamingo_tools.export_data import export_lower_resolution

        with self.assertRaisesRegex(ValueError, "non-negative"):
            export_lower_resolution("unused.zarr", self.output_folder, [1, -1])
        self.assertFalse(os.path.exists(self.output_folder))


class _MobieProject:
    """A local MoBIE project with one cochlea, in the layout of `default_image_path` / `default_table_path`."""

    def __init__(self, folder, cochlea):
        self.mobie_dir = os.path.join(folder, "mobie")
        self.cochlea = cochlea

    def add_image(self, name, data):
        folder = os.path.join(self.mobie_dir, self.cochlea, "images", "ome-zarr")
        os.makedirs(folder, exist_ok=True)
        _write_zarr(os.path.join(folder, f"{name}.ome.zarr"), {"s0": data})

    def add_table(self, name, table):
        folder = os.path.join(self.mobie_dir, self.cochlea, "tables", name)
        os.makedirs(folder, exist_ok=True)
        table.to_csv(os.path.join(folder, "default.tsv"), sep="\t", index=False)


class TestCochleaExports(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.output_folder = os.path.join(self.folder, "out")

    def tearDown(self):
        self.tmp.cleanup()

    def test_lower_resolution_clips_by_channel_name(self):
        from flamingo_tools.export_data import export_lower_resolution_cochlea

        image = np.full((4, 4, 4), 2000, dtype="uint16")
        for cochlea, expected in (("M_test", 1400), ("LaVision-test", 2000)):
            project = _MobieProject(self.folder, cochlea)
            project.add_image("PV", image)
            project.add_image("VGlut3", image)
            export_lower_resolution_cochlea(
                cochlea, self.output_folder, [0], channels=["PV", "VGlut3"], mobie_dir=project.mobie_dir,
            )
            out_folder = os.path.join(self.output_folder, cochlea, "scale0")
            self.assertTrue(np.all(_imread(os.path.join(out_folder, "PV.tif")) == expected))
            self.assertTrue(np.all(_imread(os.path.join(out_folder, "VGlut3.tif")) == 2000))

    def test_lower_resolution_reads_the_channel_table(self):
        from flamingo_tools.export_data import export_lower_resolution_cochlea

        project = _MobieProject(self.folder, "M_test")
        segmentation = np.repeat(np.arange(1, 5, dtype="uint64"), 16).reshape(4, 4, 4)
        project.add_image("SGN_v2", segmentation)
        project.add_table("SGN_v2", pd.DataFrame({"label_id": [1, 2, 3, 4], "component_labels": [1, 2, 1, 2]}))

        export_lower_resolution_cochlea(
            "M_test", self.output_folder, [0], channels=["SGN_v2"], filter_by_components=[1],
            mobie_dir=project.mobie_dir,
        )
        result = _imread(os.path.join(self.output_folder, "M_test", "scale0", "SGN_v2.tif"))
        np.testing.assert_array_equal(np.unique(result), [0, 1, 3])

    def test_marker_with_a_local_table(self):
        from flamingo_tools.export_data import export_marker_cochlea

        project = _MobieProject(self.folder, "M_test")
        project.add_image("IHC_v11", np.array([[[0, 1, 2, 3]]], dtype="uint64"))
        project.add_table("IHC_v11", pd.DataFrame({"label_id": [1, 2, 3], "marker_labels": [1, 1, 1]}))
        table_path = os.path.join(self.folder, "edited.tsv")
        pd.DataFrame({"label_id": [1, 2, 3], "marker_labels": [2, 1, 0]}).to_csv(table_path, sep="\t", index=False)

        export_marker_cochlea(
            "M_test", self.output_folder, [0], channels=["IHC_v11"], table_path=table_path,
            mobie_dir=project.mobie_dir,
        )
        out_folder = os.path.join(self.output_folder, "M_test", "scale0")
        np.testing.assert_array_equal(_imread(os.path.join(out_folder, "IHC_v11_marker_positive.tif")),
                                      [[[0, 0, 1, 0]]])
        np.testing.assert_array_equal(_imread(os.path.join(out_folder, "IHC_v11_marker_negative.tif")),
                                      [[[0, 2, 0, 0]]])

    def test_subtypes_require_segmentation_and_stains_for_an_unknown_cochlea(self):
        from flamingo_tools.export_data import export_subtypes_cochlea

        with self.assertRaisesRegex(ValueError, "no entry in COCHLEAE"):
            export_subtypes_cochlea("M_unknown", self.output_folder, [0])

    def test_synapses_use_the_ihc_segmentation(self):
        from flamingo_tools.export_data import export_synapses_cochlea

        project = _MobieProject(self.folder, "M_test")
        project.add_image("IHC_v11", np.zeros((10, 10, 10), dtype="uint64"))
        project.add_table("IHC_v11", pd.DataFrame({"label_id": [1, 2], "component_labels": [1, 2]}))
        project.add_table("synapse_v3_ihc_v11", pd.DataFrame({
            "spot_id": [1, 2], "x": [2.0, 6.0], "y": [2.0, 6.0], "z": [2.0, 6.0],
            "matched_ihc": [1, 2], "distance_to_ihc": [0.5, 0.5],
        }))

        export_synapses_cochlea("M_test", self.output_folder, [0], mobie_dir=project.mobie_dir, voxel_size=1.0,
                                points=True)
        points = pd.read_csv(os.path.join(self.output_folder, "M_test", "scale0", "synapse_v3_ihc_v11.csv"))
        self.assertEqual(points.spot_id.tolist(), [1])


class TestExportMarker(unittest.TestCase):

    def test_both_groups_from_one_read(self):
        from flamingo_tools.export_data import export_marker

        with tempfile.TemporaryDirectory() as folder:
            segmentation = np.array([[[0, 1, 2, 3, 4]]], dtype="uint64")
            path = _write_zarr(os.path.join(folder, "IHC.ome.zarr"), {"s0": segmentation})
            table = pd.DataFrame({"label_id": [1, 2, 3, 4], "marker_labels": [1, 2, 1, 0]})

            export_marker(path, folder, [0], table, crop_center=[1.0, 0.0, 0.0], roi_halo=[10, 1, 1],
                          suffix="mid", voxel_size=1.0)

            out_folder = os.path.join(folder, "scale0")
            positive = _imread(os.path.join(out_folder, "IHC_marker_positive_crop_0001-0000-0000_mid.tif"))
            negative = _imread(os.path.join(out_folder, "IHC_marker_negative_crop_0001-0000-0000_mid.tif"))
        np.testing.assert_array_equal(positive, [[[0, 1, 0, 1, 0]]])
        np.testing.assert_array_equal(negative, [[[0, 0, 2, 0, 0]]])


class TestExportSubtypes(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        # Calb1/Ntng1: Ib = Calb1+/Ntng1+, Ic = Calb1-/Ntng1+, inconclusive = Calb1+/Ntng1- or Calb1-/Ntng1-.
        self.table = pd.DataFrame({
            "label_id": [1, 2, 3, 4],
            "marker_Calb1": [1, 2, 1, 2],
            "marker_Ntng1": [1, 1, 2, 2],
        })
        self.path = _write_zarr(
            os.path.join(self.folder, "SGN.ome.zarr"), {"s0": np.array([[[0, 1, 2, 3, 4]]], dtype="uint64")},
        )

    def tearDown(self):
        self.tmp.cleanup()

    def _read(self, name):
        return _imread(os.path.join(self.folder, "scale0", f"SGN_{name}.tif"))

    def test_subtype_with_two_expressions(self):
        from flamingo_tools.export_data import export_subtypes

        export_subtypes(self.path, self.folder, [0], self.table, ["Calb1", "Ntng1"])

        np.testing.assert_array_equal(self._read("TypeIb"), [[[0, 1, 0, 0, 0]]])
        np.testing.assert_array_equal(self._read("TypeIc"), [[[0, 0, 1, 0, 0]]])
        np.testing.assert_array_equal(self._read("inconclusive"), [[[0, 0, 0, 1, 1]]])
        np.testing.assert_array_equal(self._read("TypeIbTypeIc"), [[[0, 1, 1, 0, 0]]])

    def test_merged_masks(self):
        from flamingo_tools.export_data import export_subtypes

        export_subtypes(self.path, self.folder, [0], self.table, ["Calb1", "Ntng1"], merge_masks=True)

        result = self._read("subtypes")
        self.assertEqual(result.dtype, np.dtype("uint8"))
        # Order of SUBTYPE_ID_ORDER: Type Ib = 1, Type Ic = 2, inconclusive = 3.
        np.testing.assert_array_equal(result, [[[0, 1, 2, 3, 3]]])

    def test_missing_stain_column(self):
        from flamingo_tools.export_data import export_subtypes

        with self.assertRaisesRegex(ValueError, "marker_CR"):
            export_subtypes(self.path, self.folder, [0], self.table, ["CR", "Ntng1"])

    def test_invalid_stains(self):
        from flamingo_tools.export_data import export_subtypes

        table = self.table.assign(marker_CR=1, marker_Foo=1)
        with self.assertRaisesRegex(ValueError, "one or two"):
            export_subtypes(self.path, self.folder, [0], table, ["Calb1", "Ntng1", "CR"])
        with self.assertRaisesRegex(ValueError, "No subtype is defined"):
            export_subtypes(self.path, self.folder, [0], table, ["Foo"], merge_masks=True)


class TestExportSynapses(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        self.path = _write_zarr(os.path.join(self.folder, "IHC.ome.zarr"), {"s0": np.zeros((20, 20, 20), "uint64")})
        self.table = pd.DataFrame({
            "spot_id": [1, 2, 3],
            "x": [4.0, 12.0, 16.0], "y": [4.0, 12.0, 16.0], "z": [4.0, 12.0, 16.0],
            "matched_ihc": [10, 20, 10],
            "distance_to_ihc": [0.5, 0.5, 5.0],
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_volume(self):
        from flamingo_tools.export_data import export_synapses

        ihc_table = pd.DataFrame({"label_id": [10, 20], "component_labels": [1, 2]})
        export_synapses(self.path, self.folder, [1], self.table, ihc_table=ihc_table, radius=1, id_offset=5,
                        voxel_size=1.0)

        result = _imread(os.path.join(self.folder, "scale1", "synapses_offset5.tif"))
        self.assertEqual(result.shape, (10, 10, 10))
        # Only synapse 1 is close to an IHC of component 1. The volume holds its IHC ID plus the offset.
        self.assertEqual(result[2, 2, 2], 15)
        self.assertEqual(set(np.unique(result)), {0, 15})

    def test_points_in_a_crop(self):
        from flamingo_tools.export_data import export_synapses

        export_synapses(self.path, self.folder, [0], self.table, name="syn", points=True, voxel_size=1.0,
                        crop_center=[12.0, 12.0, 12.0], roi_halo=[4, 4, 4])

        points = pd.read_csv(os.path.join(self.folder, "scale0", "syn_crop_0012-0012-0012.csv"))
        self.assertEqual(points.spot_id.tolist(), [2])
        self.assertEqual(points[["axis-0", "axis-1", "axis-2"]].values.tolist(), [[4.0, 4.0, 4.0]])


class TestExportFrequency(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = self.tmp.name
        # Label 5 is not in the table.
        self.path = _write_zarr(
            os.path.join(self.folder, "SGN.ome.zarr"), {"s0": np.array([[[0, 1, 2, 3, 5]]], dtype="uint64")},
        )
        self.table = pd.DataFrame({
            "label_id": [1, 2, 3], "component_labels": [1, 1, 2], "frequency[kHz]": [4.0, 8.0, 16.0],
        })

    def tearDown(self):
        self.tmp.cleanup()

    def test_frequencies(self):
        from flamingo_tools.export_data import export_frequency

        export_frequency(self.path, self.folder, [0], self.table)
        result = _imread(os.path.join(self.folder, "scale0", "frequencies_SGN.tif"))
        np.testing.assert_array_equal(result, [[[0, 4, 8, 0, 0]]])

    def test_float_segmentation(self):
        from flamingo_tools.export_data import export_frequency

        # `export_lower_resolution` writes a segmentation as float32.
        path = os.path.join(self.folder, "SGN_float.tif")
        tifffile.imwrite(path, np.array([[[0, 1, 2, 3, 5]]], dtype="float32"))
        export_frequency(path, self.folder, [0], self.table)
        result = _imread(os.path.join(self.folder, "scale0", "frequencies_SGN_float.tif"))
        np.testing.assert_array_equal(result, [[[0, 4, 8, 0, 0]]])

    def test_components_without_objects(self):
        from flamingo_tools.export_data import export_frequency

        with self.assertRaisesRegex(ValueError, "no objects of the components"):
            export_frequency(self.path, self.folder, [0], self.table, components=[9])

    def test_colormap(self):
        from flamingo_tools.export_data import export_frequency

        export_frequency(self.path, self.folder, [0], self.table, colormap="viridis")
        result = _imread(os.path.join(self.folder, "scale0", "frequencies_SGN_viridis.tif"))
        self.assertEqual(result.shape, (1, 1, 5, 4))
        self.assertEqual(result[0, 0, 0, 3], 0)
        self.assertEqual(result[0, 0, 1, 3], 1)
        self.assertTrue(os.path.exists(os.path.join(self.folder, "scale0", "frequencies_SGN_viridis.png")))


class TestExportName(unittest.TestCase):

    def test_extensions(self):
        from flamingo_tools.export_data_utils import export_name

        self.assertEqual(export_name("M_LR_000227_R/images/ome-zarr/PV.ome.zarr"), "PV")
        self.assertEqual(export_name("/data/segmentation.zarr/"), "segmentation")
        self.assertEqual(export_name("fused.n5"), "fused")
        self.assertEqual(export_name("crop.TIF"), "crop")


if __name__ == "__main__":
    unittest.main()
