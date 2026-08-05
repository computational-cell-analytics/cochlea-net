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


class TestConsensusScores(unittest.TestCase):
    def _table(self, rows):
        return pd.DataFrame(rows, columns=["annotator", "file_name", "tps", "fps", "fns"])

    def test_compute_consensus_scores(self):
        from flamingo_tools.validation import compute_consensus_scores

        table = self._table([
            ["AMD", "crop1", 8, 2, 2],
            ["AMD", "crop2", 12, 3, 3],
            ["EK", "crop1", 9, 1, 1],
            ["EK", "crop2", 11, 4, 4],
        ])
        scores = compute_consensus_scores(table)

        self.assertEqual(sorted(scores.keys()), ["AMD", "EK", "all"])
        self.assertEqual(scores["AMD"]["crops"], ["crop1", "crop2"])
        self.assertEqual(scores["AMD"]["tp"], [8, 12])
        self.assertEqual(scores["AMD"]["fp"], [2, 3])
        self.assertEqual(scores["AMD"]["fn"], [2, 3])

        # AMD: tp=20, fp=5, fn=5.
        self.assertAlmostEqual(scores["AMD"]["precision"], 0.8)
        self.assertAlmostEqual(scores["AMD"]["recall"], 0.8)
        self.assertAlmostEqual(scores["AMD"]["f1-score"], 0.8)

        # Pooled over both annotators: tp=40, fp=10, fn=10.
        self.assertNotIn("crops", scores["all"])
        self.assertAlmostEqual(scores["all"]["precision"], 0.8)
        self.assertAlmostEqual(scores["all"]["recall"], 0.8)
        self.assertAlmostEqual(scores["all"]["f1-score"], 0.8)

    def test_compute_consensus_scores_undefined(self):
        from flamingo_tools.validation import compute_consensus_scores

        scores = compute_consensus_scores(self._table([["AMD", "crop1", 0, 0, 0]]))
        for key in ("precision", "recall", "f1-score"):
            self.assertIsNone(scores["AMD"][key])
            self.assertIsNone(scores["all"][key])

    def test_compute_consensus_scores_missing_column(self):
        from flamingo_tools.validation import compute_consensus_scores

        table = self._table([["AMD", "crop1", 8, 2, 2]]).drop(columns="file_name")
        with self.assertRaises(ValueError):
            compute_consensus_scores(table)


if __name__ == "__main__":
    unittest.main()
