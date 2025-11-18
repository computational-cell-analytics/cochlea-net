import unittest
from shutil import rmtree

import imageio.v3 as imageio
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential


class TestValidation(unittest.TestCase):
    folder = "./tmp"

    def setUp(self):
        from flamingo_tools.test_data import get_test_volume_and_segmentation

        _, self.seg_path, _ = get_test_volume_and_segmentation(self.folder)

    def tearDown(self):
        try:
            rmtree(self.folder)
        except Exception:
            pass

    def test_compute_scores_for_annotated_slice_2d(self):
        from flamingo_tools.validation import compute_scores_for_annotated_slice

        segmentation = imageio.imread(self.seg_path)
        segmentation = segmentation[segmentation.shape[0] // 2]
        segmentation, _, _ = relabel_sequential(segmentation)

        properties = ("label", "centroid")
        annotations = regionprops_table(segmentation, properties=properties)
        annotations = pd.DataFrame(annotations).rename(columns={"centroid-0": "axis-1", "centroid-1": "axis-2"})
        annotations = annotations.drop(columns="label")

        result = compute_scores_for_annotated_slice(segmentation, annotations)

        # Check the results. Note: we actually get 1 FP and 1 FN because 1 of the centroids is outside the object.
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tp"], segmentation.max() - 1)

    def test_compute_scores_for_annotated_slice_3d(self):
        from flamingo_tools.validation import compute_scores_for_annotated_slice

        segmentation = imageio.imread(self.seg_path)
        z0, z1 = segmentation.shape[0] // 2 - 2, segmentation.shape[0] // 2 + 2
        segmentation = segmentation[z0:z1]
        segmentation, _, _ = relabel_sequential(segmentation)

        properties = ("label", "centroid")
        annotations = regionprops_table(segmentation, properties=properties)
        annotations = pd.DataFrame(annotations).rename(
            columns={"centroid-0": "axis-0", "centroid-1": "axis-1", "centroid-2": "axis-2"}
        )
        annotations = annotations.drop(columns="label")

        result = compute_scores_for_annotated_slice(segmentation, annotations)

        # Check the results. Note: we actually get 1 FP and 1 FN because 1 of the centroids is outside the object.
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["tp"], segmentation.max() - 1)


if __name__ == "__main__":
    unittest.main()
