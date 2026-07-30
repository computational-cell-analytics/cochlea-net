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


if __name__ == "__main__":
    unittest.main()
