import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile


class TestEvalAnnotations(unittest.TestCase):
    voxel_size = (0.38, 0.38, 0.38)

    def _measurement_table(self, values):
        return pd.DataFrame({"label_id": list(range(1, len(values) + 1)), "median": values})

    def _write_annotation(self, directory, name, labels, shape=(4, 8, 8)):
        arr = np.zeros(shape, dtype="uint32")
        for num, label in enumerate(labels):
            arr[num, 0, 0] = label
        path = os.path.join(directory, name)
        tifffile.imwrite(path, arr, photometric="minisblack")
        return path

    # The ROI must match the crop that `compute_crop_bb` wrote, also when the crop was clipped.
    def test_get_roi_matches_compute_crop_bb(self):
        from flamingo_tools.export_data_utils import compute_crop_bb
        from flamingo_tools.intensity_annotation.eval_annotations import get_roi

        data_shape = (300, 3886, 4000)
        roi_halo = [256, 256, 128]
        # A centered crop, a crop clipped at the upper y border, and one clipped at the lower y border.
        centers = [[1000.0, 1000.0, 100.0], [1085.0, 1389.0, 594.0], [1000.0, 50.0, 100.0]]

        for center in centers:
            with self.subTest(center=center):
                start, stop = compute_crop_bb(center, roi_halo, self.voxel_size, scale=0, shape=data_shape)
                crop_shape = tuple((stop - start).tolist())

                roi = get_roi(center, crop_shape, data_shape, voxel_size=self.voxel_size)

                self.assertEqual(tuple(sl.start for sl in roi), tuple(start.tolist()))
                self.assertEqual(tuple(sl.stop for sl in roi), tuple(stop.tolist()))

    # The failing case: the y axis of the crop is odd, because it was clipped at the volume border.
    def test_get_roi_clipped_odd_extent(self):
        from flamingo_tools.intensity_annotation.eval_annotations import get_roi

        data_shape = (300, 3886, 4000)
        roi = get_roi((1085, 1389, 594), (256, 487, 512), data_shape, voxel_size=self.voxel_size)

        self.assertEqual(roi[1], slice(3399, 3886))
        self.assertEqual(tuple(sl.stop - sl.start for sl in roi), (256, 487, 512))

    def test_get_roi_anisotropic_voxel_size(self):
        from flamingo_tools.intensity_annotation.eval_annotations import get_roi

        # voxel_size is (x, y, z) while the arrays are (z, y, x), so z is divided by the last value.
        roi = get_roi((200, 200, 300), (10, 20, 20), (400, 400, 400), voxel_size=(2.0, 2.0, 3.0))

        self.assertEqual(roi[0], slice(95, 105))
        self.assertEqual(roi[2], slice(90, 110))

    def test_get_roi_crop_larger_than_volume(self):
        from flamingo_tools.intensity_annotation.eval_annotations import get_roi

        with self.assertRaises(ValueError):
            get_roi((100, 100, 100), (10, 500, 10), (400, 400, 400), voxel_size=self.voxel_size)

    def test_threshold_from_filename(self):
        from flamingo_tools.intensity_annotation.eval_annotations import threshold_from_filename

        expected = {
            "positive-negative_M-AMD-OTOF27-L_crop_1085-1389-0594_OTOF_allnegativeexcluded_46.tif": 46.0,
            "positive-negative_M-LR-000143-L_crop_0802-1067-0776_allNegativeExcluded_thr39.tif": 39.0,
            "positive-negative_M-AMD-OTOF28-L_crop_0181-1327-0785_Alphatag_allnegativeexcluded_66tif.tif": 66.0,
            "positive-negative_M-AMD-OTOF27-L_crop_1085-1389-0594_OTOF_allnegativeexcluded_46.5.tif": 46.5,
            "positive-negative_M-AMD-OTOF27-L_crop_1085-1389-0594_OTOF_allnegativeexcluded.tif": None,
        }
        for name, threshold in expected.items():
            with self.subTest(name=name):
                self.assertEqual(threshold_from_filename(name), threshold)

    def test_threshold_from_filenames(self):
        from flamingo_tools.intensity_annotation.eval_annotations import threshold_from_filenames

        file_neg = "crop_1259-0662-0447_OTOF_allnegativeexcluded_56.tif"
        file_pos = "crop_1259-0662-0447_OTOF_allpositiveincluded_40.tif"

        self.assertEqual(threshold_from_filenames([file_neg, file_pos]), 48.0)
        self.assertEqual(threshold_from_filenames([file_neg, None]), 56.0)
        self.assertIsNone(threshold_from_filenames([None, None]))

    def test_get_single_annotation_parameters(self):
        from flamingo_tools.intensity_annotation.eval_annotations import get_single_annotation_parameters

        table = self._measurement_table([1.0, 5.0, 20.0])
        with tempfile.TemporaryDirectory() as tmp_dir:
            all_negative = self._write_annotation(tmp_dir, "all_negative_42.tif", [1, 1, 1])
            all_positive = self._write_annotation(tmp_dir, "all_positive_42.tif", [2, 2, 2])
            mixed = self._write_annotation(tmp_dir, "mixed_42.tif", [1, 2, 1])
            empty = self._write_annotation(tmp_dir, "empty_42.tif", [])

            param_dic = get_single_annotation_parameters(all_negative, table)
            self.assertEqual(param_dic["median_intensity"], 30.0)
            self.assertEqual(param_dic["threshold_source"], "single-annotation-negative")

            param_dic = get_single_annotation_parameters(all_positive, table)
            self.assertEqual(param_dic["median_intensity"], 0.0)
            self.assertEqual(param_dic["threshold_source"], "single-annotation-positive")

            for path in (mixed, empty):
                param_dic = get_single_annotation_parameters(path, table)
                self.assertIsNone(param_dic["median_intensity"])
                self.assertIsNone(param_dic["threshold_source"])

    # A truncated TIF is read back as a single 2D plane instead of raising.
    def test_read_annotation_rejects_2d(self):
        from flamingo_tools.intensity_annotation.eval_annotations import read_annotation

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "plane.tif")
            tifffile.imwrite(path, np.zeros((8, 8), dtype="uint32"))

            with self.assertRaises(ValueError):
                read_annotation(path)

    def test_find_annotations_keeps_single_annotation(self):
        from flamingo_tools.intensity_annotation.eval_annotations import find_annotations

        cochlea = "M-AMD-OTOF27-R"
        with tempfile.TemporaryDirectory() as tmp_dir:
            paired = f"positive-negative_{cochlea}_crop_0625-0676-0122_OTOF"
            self._write_annotation(tmp_dir, f"{paired}_allnegativeexcluded_42.tif", [1])
            self._write_annotation(tmp_dir, f"{paired}_allpositiveincluded_40.tif", [2])
            single = f"positive-negative_{cochlea}_crop_0659-1149-0749_OTOF"
            self._write_annotation(tmp_dir, f"{single}_allnegativeexcluded_64.tif", [1])
            # A file that does not belong to a crop must not break the file name parsing.
            with open(os.path.join(tmp_dir, "Thumbs.db"), "w") as f:
                f.write("")

            dic = find_annotations(tmp_dir, cochlea, pattern="OTOF_")

            self.assertEqual(dic["center_strings"], ["0625-0676-0122", "0659-1149-0749"])
            self.assertIsNotNone(dic["0625-0676-0122"]["file_pos"])
            self.assertIsNone(dic["0659-1149-0749"]["file_pos"])
            self.assertTrue(dic["0659-1149-0749"]["file_neg"].endswith("_allnegativeexcluded_64.tif"))

    def test_find_annotations_skips_ambiguous_crop(self):
        from flamingo_tools.intensity_annotation.eval_annotations import find_annotations

        cochlea = "M-AMD-OTOF27-L"
        with tempfile.TemporaryDirectory() as tmp_dir:
            prefix = f"positive-negative_{cochlea}_crop_0568-0112-0692_OTOF"
            self._write_annotation(tmp_dir, f"{prefix}_allnegativeexcluded_36.tif", [1])
            self._write_annotation(tmp_dir, f"{prefix}_allnegativeexcluded_38.tif", [1])

            dic = find_annotations(tmp_dir, cochlea, pattern="OTOF_")

            self.assertEqual(dic["center_strings"], [])

    def _segmentation_table(self, length_fractions):
        n = len(length_fractions)
        return pd.DataFrame({
            "label_id": list(range(1, n + 1)),
            "anchor_x": [100.0] * n,
            "anchor_y": [100.0] * n,
            "anchor_z": [100.0] * n,
            "length_fraction": length_fractions,
        })

    def test_map_crops_to_length_fraction_drops_crop_without_threshold(self):
        from flamingo_tools.intensity_annotation.eval_annotations import map_crops_to_length_fraction

        table_seg = self._segmentation_table([0.2, 0.5])
        table_seg.loc[:, "anchor_x"] = [100.0, 200.0]
        intensity_dic = {
            "0100-0100-0100": {"median_intensity": 5.0},
            "0200-0100-0100": {"median_intensity": None},
        }

        lf_intensity = map_crops_to_length_fraction(intensity_dic, table_seg)

        self.assertEqual([entry["center"] for entry in lf_intensity.values()], ["0100-0100-0100"])

    def test_apply_nearest_threshold_raises_without_any_threshold(self):
        from flamingo_tools.intensity_annotation.eval_annotations import apply_nearest_threshold

        table_seg = self._segmentation_table([0.2])
        intensity_dic = {"0100-0100-0100": {"median_intensity": None}}

        with self.assertRaises(ValueError):
            apply_nearest_threshold(intensity_dic, table_seg, self._measurement_table([5.0]))


if __name__ == "__main__":
    unittest.main()
