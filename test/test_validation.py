import os
import unittest
from shutil import rmtree
from tempfile import TemporaryDirectory

import imageio.v3 as imageio
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential

COORDS = ["axis-0", "axis-1", "axis-2"]


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

    def test_compute_scores_from_counts(self):
        from flamingo_tools.validation import compute_scores_from_counts

        scores = compute_scores_from_counts(8, 2, 0)
        self.assertAlmostEqual(scores["precision"], 0.8)
        self.assertAlmostEqual(scores["recall"], 1.0)
        self.assertAlmostEqual(scores["f1-score"], 0.889)

        # Without any prediction the precision is undefined, and so is the F1-score.
        scores = compute_scores_from_counts(0, 0, 5)
        self.assertIsNone(scores["precision"])
        self.assertAlmostEqual(scores["recall"], 0.0)
        self.assertIsNone(scores["f1-score"])

    def test_average_scores_per_row(self):
        from flamingo_tools.validation import average_scores_per_row

        # Row 1: P=0.8, R=1.0, F1=0.889.  Row 2: P=0.5, R=0.5, F1=0.5.
        table = self._table([
            ["AMD", "crop1", 8, 2, 0],
            ["EK", "crop2", 5, 5, 5],
        ])
        averages = average_scores_per_row(table)
        self.assertAlmostEqual(averages["precision"], 0.65)
        self.assertAlmostEqual(averages["recall"], 0.75)
        self.assertAlmostEqual(averages["f1-score"], 0.695)

    def test_average_scores_per_row_skips_undefined(self):
        from flamingo_tools.validation import average_scores_per_row

        # The second row has no annotation at all, so none of its scores are defined.
        table = self._table([
            ["AMD", "crop1", 8, 2, 0],
            ["EK", "crop2", 0, 0, 0],
        ])
        averages = average_scores_per_row(table)
        self.assertAlmostEqual(averages["precision"], 0.8)
        self.assertAlmostEqual(averages["recall"], 1.0)
        self.assertAlmostEqual(averages["f1-score"], 0.889)

        averages = average_scores_per_row(self._table([["EK", "crop2", 0, 0, 0]]))
        for key in ("precision", "recall", "f1-score"):
            self.assertIsNone(averages[key])


class TestPairwiseAgreement(unittest.TestCase):
    """Pairwise agreement between annotators, without a consensus annotation."""

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.root = self.tmp_dir.name

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _write(self, annotator, crop, coordinates):
        folder = os.path.join(self.root, annotator)
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{crop}.csv")
        pd.DataFrame(np.asarray(coordinates, dtype=float), columns=COORDS).to_csv(path, index=False)
        return path

    def _grid(self, n, offset=0.0):
        """Points spaced far enough apart that only identical points match."""
        return [[offset + 100.0 * i, 0.0, 0.0] for i in range(n)]

    def test_symmetric_case(self):
        from flamingo_tools.validation import average_pairwise_scores, compute_pairwise_agreement

        # Every annotator shares 8 points and adds 2 of its own, so each pair matches 8 of 10.
        shared = self._grid(8)
        annotations_per_crop = {}
        for crop in ("crop1", "crop2"):
            annotations_per_crop[crop] = {
                annotator: self._write(annotator, crop, shared + self._grid(2, offset=10_000 * (idx + 1)))
                for idx, annotator in enumerate(["AMD", "EK", "LR"])
            }

        per_crop, summary = compute_pairwise_agreement(annotations_per_crop, matching_distance=1.0)

        self.assertEqual(len(per_crop), 3 * 2)
        self.assertEqual(len(summary), 3)
        for column in ("matching_distance", "voxel_size", "mean_match_distance", "max_match_distance"):
            self.assertIn(column, per_crop.columns)
            self.assertIn(column, summary.columns)
        for column in ("macro_precision", "macro_recall", "macro_f1-score"):
            self.assertIn(column, summary.columns)
        self.assertTrue((per_crop.n_matches == 8).all())

        scores = average_pairwise_scores(per_crop)
        self.assertEqual(scores, {"precision": 0.8, "recall": 0.8, "f1-score": 0.8})

    def test_asymmetric_case(self):
        from flamingo_tools.validation import average_pairwise_scores, compute_pairwise_agreement

        # AA holds 10 points, BB only the 6 shared ones.
        shared = self._grid(6)
        annotations_per_crop = {"crop1": {
            "AA": self._write("AA", "crop1", shared + self._grid(4, offset=10_000)),
            "BB": self._write("BB", "crop1", shared),
        }}
        per_crop, _ = compute_pairwise_agreement(annotations_per_crop, matching_distance=1.0)
        row = per_crop.iloc[0]
        self.assertEqual((row.n_annotations_a, row.n_annotations_b, row.n_matches), (10, 6, 6))

        # One direction gives P=0.6 and R=1.0; the reverse swaps them. F1 is 12/16 either way.
        scores = average_pairwise_scores(per_crop)
        self.assertEqual(scores["precision"], scores["recall"])
        self.assertAlmostEqual(scores["precision"], 0.8)
        self.assertAlmostEqual(scores["f1-score"], 0.75)

    def test_voxel_size_scales_coordinates(self):
        from flamingo_tools.validation import compute_pairwise_agreement

        # The two points are 1 voxel apart, so they only match once scaled below the distance.
        annotations_per_crop = {"crop1": {
            "AA": self._write("AA", "crop1", [[0.0, 0.0, 0.0]]),
            "BB": self._write("BB", "crop1", [[1.0, 0.0, 0.0]]),
        }}
        per_crop, _ = compute_pairwise_agreement(annotations_per_crop, matching_distance=0.5)
        self.assertEqual(per_crop.iloc[0].n_matches, 0)

        per_crop, _ = compute_pairwise_agreement(annotations_per_crop, matching_distance=0.5, voxel_size=0.38)
        self.assertEqual(per_crop.iloc[0].n_matches, 1)

    def test_missing_annotator_is_skipped(self):
        from flamingo_tools.validation import compute_pairwise_agreement

        # LR annotated only the second crop, so it contributes one row per pair instead of two.
        shared = self._grid(8)
        annotations_per_crop = {
            "crop1": {a: self._write(a, "crop1", shared) for a in ("AMD", "EK")},
            "crop2": {a: self._write(a, "crop2", shared) for a in ("AMD", "EK", "LR")},
        }
        per_crop, summary = compute_pairwise_agreement(annotations_per_crop, matching_distance=1.0)

        counts = per_crop.groupby(["annotator_a", "annotator_b"]).size().to_dict()
        self.assertEqual(counts[("AMD", "EK")], 2)
        self.assertEqual(counts[("AMD", "LR")], 1)
        self.assertEqual(counts[("EK", "LR")], 1)
        self.assertEqual(len(summary), 3)

    def test_too_few_annotators(self):
        from flamingo_tools.validation import compute_pairwise_agreement

        annotations_per_crop = {"crop1": {"AMD": self._write("AMD", "crop1", self._grid(4))}}
        with self.assertRaises(ValueError):
            compute_pairwise_agreement(annotations_per_crop, matching_distance=1.0)

    def test_evaluate_pairwise_agreement_saves_tables(self):
        from flamingo_tools.validation import evaluate_pairwise_agreement

        shared = self._grid(8)
        annotations_per_crop = {"crop1": {
            "AMD": self._write("AMD", "crop1", shared + self._grid(2, offset=10_000)),
            "EK": self._write("EK", "crop1", shared + self._grid(2, offset=20_000)),
        }}
        table_dir = os.path.join(self.root, "tables")
        os.makedirs(table_dir)
        scores = evaluate_pairwise_agreement(annotations_per_crop, table_dir, matching_distance=1.0)

        self.assertEqual(scores, {"precision": 0.8, "recall": 0.8, "f1-score": 0.8})
        self.assertEqual(
            sorted(os.listdir(table_dir)),
            ["pairwise_agreement_per_crop.csv", "pairwise_agreement_summary.csv"],
        )


if __name__ == "__main__":
    unittest.main()
