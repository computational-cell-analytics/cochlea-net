import unittest

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree


def _dense_match(detections, annotations, max_dist):
    """The former implementation of `match_detections`, with one N x M cost matrix."""
    N, M = len(detections), len(annotations)
    coo = cKDTree(detections).sparse_distance_matrix(cKDTree(annotations), max_dist, output_type="coo_matrix")
    if coo.nnz == 0:
        return np.empty(0, int), np.empty(0, int), np.arange(N), np.arange(M)
    cost = np.full((N, M), 5 * max_dist, dtype=float)
    cost[coo.row, coo.col] = coo.data
    row_ind, col_ind = linear_sum_assignment(cost)
    valid = cost[row_ind, col_ind] <= max_dist
    tp_det, tp_ann = row_ind[valid], col_ind[valid]
    return (tp_det, tp_ann, np.setdiff1d(np.arange(N), tp_det), np.setdiff1d(np.arange(M), tp_ann))


def _clustered_points(rng, n_clusters, n_points, spread):
    centers = rng.uniform(0, 100, size=(n_clusters, 3))
    return centers[rng.integers(0, n_clusters, n_points)] + rng.normal(0, spread, size=(n_points, 3))


class TestMatchDetections(unittest.TestCase):

    def test_equals_dense_assignment(self):
        from flamingo_tools.validation import match_detections

        for seed in range(8):
            rng = np.random.default_rng(seed)
            # Dense clusters give large connected components, the sparse background gives small ones.
            detections = np.concatenate([_clustered_points(rng, 6, 250, 1.5), rng.uniform(0, 100, (150, 3))])
            annotations = np.concatenate([_clustered_points(rng, 6, 180, 1.5), rng.uniform(0, 100, (120, 3))])
            expected = _dense_match(detections, annotations, 2.0)
            result = match_detections(detections, annotations, 2.0)
            for exp, res in zip(expected, result):
                np.testing.assert_array_equal(res, exp)
            distance = np.linalg.norm(detections[result[0]] - annotations[result[1]], axis=1)
            self.assertTrue(np.all(distance <= 2.0))

    def test_corner_cases(self):
        from flamingo_tools.validation import match_detections

        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
        empty = np.empty((0, 3))
        tp_det, tp_ann, fp, fn = match_detections(empty, points, 2.0)
        self.assertEqual((len(tp_det), len(tp_ann), len(fp)), (0, 0, 0))
        np.testing.assert_array_equal(fn, [0, 1])
        tp_det, tp_ann, fp, fn = match_detections(points, empty, 2.0)
        np.testing.assert_array_equal(fp, [0, 1])
        self.assertEqual(len(fn), 0)
        # No pair is within the distance.
        tp_det, _, fp, fn = match_detections(points, points + [0.0, 5.0, 0.0], 2.0)
        self.assertEqual(len(tp_det), 0)
        np.testing.assert_array_equal(fp, [0, 1])
        np.testing.assert_array_equal(fn, [0, 1])
        # N != M. Detection 0 must take annotation 1, so that detection 1 can take annotation 0.
        detections = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [50.0, 0.0, 0.0]])
        annotations = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
        tp_det, tp_ann, fp, fn = match_detections(detections, annotations, 2.0)
        np.testing.assert_array_equal(tp_det, [0, 1])
        np.testing.assert_array_equal(tp_ann, [1, 0])
        np.testing.assert_array_equal(fp, [2])
        self.assertEqual(len(fn), 0)


if __name__ == "__main__":
    unittest.main()
