import os
import tempfile
import unittest

import imageio.v3 as imageio
import numpy as np


class TestExtractDensityCrops(unittest.TestCase):

    shape = (40, 40, 40)  # Z, Y, X

    def setUp(self):
        from flamingo_tools.extract_block_util import extract_density_crops
        self.extract_density_crops = extract_density_crops

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        seg_data = np.zeros(self.shape, dtype="uint32")
        seg_data[10:30, 10:30, 10:30] = 1
        self.seg_path = os.path.join(self.tmpdir.name, "seg.tif")
        imageio.imwrite(self.seg_path, seg_data)

        img_data = np.random.randint(0, 255, size=self.shape).astype("uint16")
        self.img_path = os.path.join(self.tmpdir.name, "PV.tif")
        imageio.imwrite(self.img_path, img_data)

        self.output_dir = os.path.join(self.tmpdir.name, "crops")

        self.block_list = [
            {
                "position_label": "mid",
                "dataset_name": "G_EK_000076_L",
                "segmentation_channel": "SGN_v2",
                "crop_centers": [[20, 20, 20]],
                "roi_halo": [5, 5, 5],
            },
        ]

    def test_creates_segmentation_and_img_path_crops(self):
        self.extract_density_crops(
            self.block_list,
            output_path=self.output_dir,
            seg_path=self.seg_path,
            img_paths=[self.img_path],
            input_key=None,
            voxel_size=1.0,
        )
        seg_crop = os.path.join(self.output_dir, "G-EK-000076-L_crop_0020-0020-0020_SGN-v2.tif")
        img_crop = os.path.join(self.output_dir, "G-EK-000076-L_crop_0020-0020-0020_PV.tif")
        self.assertTrue(os.path.isfile(seg_crop))
        self.assertTrue(os.path.isfile(img_crop))
        self.assertEqual(imageio.imread(seg_crop).shape, (10, 10, 10))

    def test_skips_segmentation_when_seg_path_none(self):
        self.extract_density_crops(
            self.block_list,
            output_path=self.output_dir,
            seg_path=None,
            img_paths=[self.img_path],
            input_key=None,
            voxel_size=1.0,
        )
        files = os.listdir(self.output_dir)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith("_PV.tif"))

    def test_channel_name_and_dataset_name_fallbacks(self):
        # Entry has neither 'segmentation_channel' nor 'dataset_name' -> fall back to the
        # basename of seg_path and the explicit dataset_name argument, respectively.
        block_list = [
            {
                "position_label": "mid",
                "crop_centers": [[20, 20, 20]],
                "roi_halo": [5, 5, 5],
            },
        ]
        self.extract_density_crops(
            block_list,
            output_path=self.output_dir,
            seg_path=self.seg_path,
            dataset_name="G_EK_000076_L",
            input_key=None,
            voxel_size=1.0,
        )
        expected = os.path.join(self.output_dir, "G-EK-000076-L_crop_0020-0020-0020_seg.tif")
        self.assertTrue(os.path.isfile(expected))


class TestExtractBlockSingle(unittest.TestCase):

    shape = (20, 30, 40)  # Z, Y, X
    voxel_size = (0.5, 0.5, 1.0)  # x, y, z

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.data = np.arange(np.prod(self.shape), dtype="uint32").reshape(self.shape)
        self.input_path = os.path.join(self.tmpdir.name, "PV.tif")
        imageio.imwrite(self.input_path, self.data)
        self.output_dir = os.path.join(self.tmpdir.name, "crops")
        os.makedirs(self.output_dir)

    def _extract(self, coords, roi_halo, **kwargs):
        from flamingo_tools.extract_block_util import extract_block_single

        extract_block_single(
            self.input_path, coords, output_path=self.output_dir, channel_name="PV", input_key=None,
            voxel_size=self.voxel_size, roi_halo=roi_halo, **kwargs,
        )
        files = os.listdir(self.output_dir)
        return os.path.join(self.output_dir, files[0])

    def test_center_crop(self):
        # (x, y, z) = (10, 5, 8) um is the voxel (z, y, x) = (8, 10, 20).
        crop = imageio.imread(self._extract([10, 5, 8], [4, 3, 3]))
        np.testing.assert_array_equal(crop, self.data[5:11, 7:13, 16:24])

    def test_border_crop_is_clipped(self):
        # The crop around the voxel (2, 1, 1) reaches past the volume start on every axis.
        crop = imageio.imread(self._extract([0.5, 0.5, 2], [4, 3, 3]))
        np.testing.assert_array_equal(crop, self.data[0:5, 0:4, 0:5])

    def test_existing_output_is_skipped(self):
        path = self._extract([10, 5, 8], [4, 3, 3])
        imageio.imwrite(path, np.zeros((1, 1, 1), dtype="uint32"))
        self._extract([10, 5, 8], [4, 3, 3])
        self.assertEqual(imageio.imread(path).shape, (1, 1, 1))
        self._extract([10, 5, 8], [4, 3, 3], force_overwrite=True)
        self.assertEqual(imageio.imread(path).shape, (6, 6, 8))

    def test_hull_mask_is_clipped_like_crop(self):
        from flamingo_tools.analysis.density_utils import hull_to_mask
        from flamingo_tools.extract_block_util import save_mask_from_hull_vertices

        coords, roi_halo = [0.5, 0.5, 2], [4, 3, 3]
        hull_vertices = [[[0.0, 0.0], [3.0, 0.0], [3.0, 2.0], [0.0, 2.0]]]
        crop = imageio.imread(self._extract(coords, roi_halo))
        save_mask_from_hull_vertices(coords, hull_vertices, self.output_dir, voxel_size=self.voxel_size,
                                     roi_halo=roi_halo, shape=self.shape)
        mask = imageio.imread(os.path.join(self.output_dir, "crop_0000-0000-0002_hull.tif"))
        self.assertEqual(mask.shape, crop.shape)
        # The unclipped crop starts at the voxel (-1, -2, -3).
        full_mask = hull_to_mask(hull_vertices[0], bb_center=coords, roi_halo=roi_halo, voxel_size=self.voxel_size)
        np.testing.assert_array_equal(mask, full_mask[1:, 2:, 3:])
        self.assertTrue(mask.any())

    def test_hull_mask_uses_crop_voxel_grid(self):
        """A crop center between voxels gives a mask on the voxel grid of the crop."""
        from flamingo_tools.analysis.density_utils import hull_to_mask

        # The center (10.2, 5.1, 8.3) um lies in the voxel (z, y, x) = (8, 10, 20). The crop starts at x = 8 um,
        # so its voxel centers in x are 8.0, 8.5, ... um, and 9.5, 10.0 and 10.5 um lie inside the hull.
        hull = [[9.15, 3.0], [10.65, 3.0], [10.65, 8.0], [9.15, 8.0]]
        mask = hull_to_mask(hull, bb_center=[10.2, 5.1, 8.3], roi_halo=[4, 3, 2], voxel_size=self.voxel_size)
        self.assertEqual(mask.shape, (4, 6, 8))
        expected = np.zeros(8, dtype=bool)
        expected[3:6] = True
        np.testing.assert_array_equal(mask, np.broadcast_to(expected, mask.shape))

    def test_json_entries_do_not_share_keys(self):
        import json
        import zarr
        from flamingo_tools.extract_block_util import extract_block_json_wrapper

        mobie_dir = os.path.join(self.tmpdir.name, "mobie")
        for cochlea in ("M_A_L", "M_B_R"):
            path = os.path.join(mobie_dir, cochlea, "images", "ome-zarr", "PV.ome.zarr")
            zarr.open_group(path, mode="w").create_array("s0", data=self.data)
        json_file = os.path.join(self.tmpdir.name, "crops.json")
        with open(json_file, "w") as f:
            json.dump([
                {"dataset_name": "M_A_L", "image_channel": "PV", "crop_centers": [[10, 5, 8]], "roi_halo": [4, 3, 2]},
                {"dataset_name": "M_B_R", "image_channel": "PV", "crop_centers": [[10, 5, 8]]},
            ], f)
        extract_block_json_wrapper(self.output_dir, json_file=json_file, mobie_dir=mobie_dir, input_key="s0",
                                   voxel_size=self.voxel_size, roi_halo=[2, 2, 2])
        first = imageio.imread(os.path.join(self.output_dir, "M-A-L_crop_0010-0005-0008_PV.tif"))
        second = imageio.imread(os.path.join(self.output_dir, "M-B-R_crop_0010-0005-0008_PV.tif"))
        self.assertEqual(first.shape, (4, 6, 8))
        # The second entry has no 'roi_halo', so it uses the halo of the call, not the halo of the first entry.
        self.assertEqual(second.shape, (4, 4, 4))

    def test_image_channel_suffix_without_json(self):
        from flamingo_tools.extract_block_util import extract_block_json_wrapper

        extract_block_json_wrapper(
            self.output_dir, input_path=self.input_path, coords=[10, 5, 8], image_channel="PV", input_key=None,
            voxel_size=self.voxel_size, roi_halo=[4, 3, 3],
        )
        self.assertEqual(os.listdir(self.output_dir), ["crop_0010-0005-0008_PV.tif"])

    def test_point_tables_need_one_cochlea(self):
        import json
        import pandas as pd
        from flamingo_tools.extract_block_util import extract_block_json_wrapper

        json_file = os.path.join(self.tmpdir.name, "crops.json")
        params = {"image_channel": "PV", "crop_centers": [[10, 5, 8]], "roi_halo": [4, 3, 3]}
        with open(json_file, "w") as f:
            json.dump([{"dataset_name": "M_A_L", **params}, {"dataset_name": "M_B_R", **params}], f)
        points = pd.DataFrame({"x": [10.0], "y": [5.0], "z": [8.0]})
        with self.assertRaises(ValueError):
            extract_block_json_wrapper(self.output_dir, json_file=json_file, point_tables={"spots": points})

    def test_json_entry_without_crop_centers(self):
        """A file of flamingo_tools.json_extract_blocks has no crop centers until '--find_centers' finds them."""
        import json
        from flamingo_tools.extract_block_util import extract_block_json_wrapper

        json_file = os.path.join(self.tmpdir.name, "crops.json")
        with open(json_file, "w") as f:
            json.dump([
                {"dataset_name": "M_A_L", "image_channel": "PV", "crop_centers": [[10, 5, 8]], "roi_halo": [4, 3, 3]},
                {"dataset_name": "M_A_L", "image_channel": "PV", "segmentation_channel": "SGN_v2", "n_blocks": 6},
            ], f)
        output_dir = os.path.join(self.tmpdir.name, "new_crops")
        with self.assertRaisesRegex(ValueError, "Entry 1 \\(M_A_L\\).*--find_centers"):
            extract_block_json_wrapper(output_dir, json_file=json_file)
        # The check runs before the first block is extracted.
        self.assertFalse(os.path.exists(output_dir))


class TestExtractPoints(unittest.TestCase):

    shape = (20, 30, 40)  # Z, Y, X
    voxel_size = (0.5, 0.5, 1.0)  # x, y, z

    def test_points_in_crop(self):
        import pandas as pd
        from flamingo_tools.extract_block_util import extract_points_single

        points = pd.DataFrame({
            "spot_id": [1, 2, 3],
            "x": [10.0, 11.0, 19.0], "y": [5.0, 5.25, 5.0], "z": [8.0, 8.0, 8.0],
            "Homer1_snr": [4.0, 5.0, 6.0],
        })
        with tempfile.TemporaryDirectory() as tmp:
            path = extract_points_single(
                points, [10, 5, 8], tmp, self.shape, name="Homer1_spots", dataset_name="M_AMD_000117_L",
                voxel_size=self.voxel_size, roi_halo=[4, 3, 2],
            )
            self.assertEqual(os.path.basename(path), "M-AMD-000117-L_crop_0010-0005-0008_Homer1-spots.csv")
            table = pd.read_csv(path)
        # The crop starts at the voxel (6, 7, 16). Point 3 at x voxel 38 lies outside.
        self.assertEqual(table.columns.tolist()[:4], ["index", "axis-0", "axis-1", "axis-2"])
        self.assertEqual(table.spot_id.tolist(), [1, 2])
        np.testing.assert_allclose(table[["axis-0", "axis-1", "axis-2"]].values, [[2, 3, 4], [2, 3.5, 6]])
        self.assertEqual(table.Homer1_snr.tolist(), [4.0, 5.0])


if __name__ == "__main__":
    unittest.main()
