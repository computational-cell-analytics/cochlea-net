import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd


def _make_tube(n_points=6000, radius=700.0, turns=1.1, tube_radius=50.0, pitch=400.0,
               origin=900.0, seed=0):
    """Fill a helical tube of known axis with points, as a stand-in for Rosenthal's canal.

    The origin keeps every coordinate positive, which the volumetric path methods require.

    Returns:
        Points of shape (n_points, 3).
    """
    rng = np.random.default_rng(seed)
    angle = rng.random(n_points) * (2 * np.pi * turns)
    axis = _tube_axis(angle, radius, pitch, origin)
    tangent = np.stack([-radius * np.sin(angle), radius * np.cos(angle),
                        np.full_like(angle, pitch / (2 * np.pi))], axis=1)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    first = np.cross(tangent, np.tile([0.0, 0.0, 1.0], (n_points, 1)))
    first /= np.linalg.norm(first, axis=1, keepdims=True)
    second = np.cross(tangent, first)

    around = rng.random(n_points) * 2 * np.pi
    # The square root spreads the points evenly over the area of the cross-section.
    offset = tube_radius * np.sqrt(rng.random(n_points))
    return axis + (offset * np.cos(around))[:, None] * first + (offset * np.sin(around))[:, None] * second


def _tube_axis(angle, radius=700.0, pitch=400.0, origin=900.0):
    """Axis of the helical tube at the given angles."""
    return np.stack([origin + radius * np.cos(angle), origin + radius * np.sin(angle),
                     origin + pitch * angle / (2 * np.pi)], axis=1)


def _distance_to_tube_axis(path, turns=1.1, **kwargs):
    """Distance from every node of a path to the axis of the helical tube."""
    from scipy.spatial import cKDTree
    axis = _tube_axis(np.linspace(0, 2 * np.pi * turns, 20000), **kwargs)
    distances, _ = cKDTree(axis).query(np.asarray(path, dtype=float))
    return distances


def _make_arc_table(radius=600.0, spacing=10.0, comp_sizes=(60, 40, 20), gap=400.0):
    """Build a synthetic IHC table along a circular arc that is split into separate components.

    The components are separated by a gap along the arc. Component 1 is the largest one,
    which matches the label order that the graph path method expects.

    Returns:
        The segmentation table.
        Run length of each instance along the arc, with the gaps excluded.
        Total run length of all components.
    """
    arc_position, run_length, component_labels = [], [], []
    arc, run = 0.0, 0.0

    for comp_index, size in enumerate(comp_sizes):
        for point in range(size):
            if point > 0:
                arc += spacing
                run += spacing
            arc_position.append(arc)
            run_length.append(run)
            component_labels.append(comp_index + 1)
        arc += gap

    angle = np.array(arc_position) / radius
    anchor_x = radius * np.cos(angle)
    anchor_y = radius * np.sin(angle)
    anchor_z = np.zeros_like(anchor_x)
    n_points = len(anchor_x)
    half = 3.0

    table = pd.DataFrame({
        "label_id": np.arange(1, n_points + 1),
        "anchor_x": anchor_x,
        "anchor_y": anchor_y,
        "anchor_z": anchor_z,
        "bb_min_x": anchor_x - half,
        "bb_max_x": anchor_x + half,
        "bb_min_y": anchor_y - half,
        "bb_max_y": anchor_y + half,
        "bb_min_z": anchor_z - half,
        "bb_max_z": anchor_z + half,
        "n_pixels": np.full(n_points, 500),
        "component_labels": np.array(component_labels, dtype=int),
    })
    return table, np.array(run_length), run_length[-1]


class TestEquidistantCenters(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.postprocessing.cochlea_mapping import equidistant_centers
        self.fn = equidistant_centers
        self.n_blocks = 6

    def _run_lengths_of_centers(self, centers, table, run_length):
        """Map each center back to the run length of the matching instance."""
        positions = np.stack([table["anchor_x"], table["anchor_y"], table["anchor_z"]], axis=1)
        values = []
        for center in centers:
            distances = np.linalg.norm(positions - np.array(center), axis=1)
            index = int(np.argmin(distances))
            self.assertLess(distances[index], 1e-6, "Center is not one of the instance positions.")
            values.append(run_length[index])
        return np.array(values)

    def _target_fractions(self):
        target = np.linspace(0, 1, self.n_blocks * 2 + 1)
        return target[1::2]

    def test_multi_component_centers_are_distinct(self):
        table, _, _ = _make_arc_table()
        centers = self.fn(table, component_label=[1, 2, 3], cell_type="ihc", n_blocks=self.n_blocks)

        self.assertEqual(len(centers), self.n_blocks)
        self.assertEqual(len({tuple(c) for c in centers}), self.n_blocks)

    def test_multi_component_centers_track_target_fractions(self):
        table, run_length, total = _make_arc_table()
        centers = self.fn(table, component_label=[1, 2, 3], cell_type="ihc", n_blocks=self.n_blocks)

        values = self._run_lengths_of_centers(centers, table, run_length)
        targets = self._target_fractions() * total
        # The path direction depends on which end of the arc is treated as the apex.
        forward = np.abs(values - targets).max()
        reverse = np.abs(values - targets[::-1]).max()
        self.assertLess(min(forward, reverse), 0.03 * total)

    def test_single_component_centers_are_distinct(self):
        table, run_length, total = _make_arc_table(comp_sizes=(120,))
        centers = self.fn(table, component_label=[1], cell_type="ihc", n_blocks=self.n_blocks)

        self.assertEqual(len({tuple(c) for c in centers}), self.n_blocks)
        values = self._run_lengths_of_centers(centers, table, run_length)
        targets = self._target_fractions() * total
        forward = np.abs(values - targets).max()
        reverse = np.abs(values - targets[::-1]).max()
        self.assertLess(min(forward, reverse), 0.03 * total)


class TestPathMethodRegistry(unittest.TestCase):

    def test_registry_holds_the_expected_methods(self):
        from flamingo_tools.postprocessing.cochlea_mapping import CENTRAL_PATH_METHODS
        self.assertEqual(set(CENTRAL_PATH_METHODS), {"edt", "edt_refined", "graph"})

    def test_defaults_per_cell_type(self):
        from flamingo_tools.postprocessing.cochlea_mapping import _resolve_path_method
        self.assertEqual(_resolve_path_method("sgn"), "edt_refined")
        self.assertEqual(_resolve_path_method("SGN"), "edt_refined")
        self.assertEqual(_resolve_path_method("ihc"), "graph")
        self.assertEqual(_resolve_path_method("IHC"), "graph")

    def test_explicit_method_wins_over_the_default(self):
        from flamingo_tools.postprocessing.cochlea_mapping import _resolve_path_method
        self.assertEqual(_resolve_path_method("sgn", "edt"), "edt")
        self.assertEqual(_resolve_path_method("ihc", "edt_refined"), "edt_refined")

    def test_unknown_cell_type_raises(self):
        from flamingo_tools.postprocessing.cochlea_mapping import _resolve_path_method
        with self.assertRaises(ValueError):
            _resolve_path_method("ohc")

    def test_unknown_method_raises_and_lists_the_choices(self):
        from flamingo_tools.postprocessing.cochlea_mapping import _resolve_path_method, measure_run_length
        with self.assertRaises(ValueError) as raised:
            _resolve_path_method("sgn", "skeleton")
        self.assertIn("edt_refined", str(raised.exception))
        with self.assertRaises(ValueError):
            measure_run_length([np.zeros((4, 3))], path_method="skeleton")

    def test_notice_is_printed_only_for_the_new_default(self):
        from flamingo_tools.postprocessing.cochlea_mapping import _resolve_path_method
        stream = io.StringIO()
        with redirect_stdout(stream):
            _resolve_path_method("sgn")
        self.assertIn("edt_refined", stream.getvalue())

        stream = io.StringIO()
        with redirect_stdout(stream):
            _resolve_path_method("sgn", "edt_refined")
            _resolve_path_method("sgn", "edt")
            _resolve_path_method("ihc")
        self.assertEqual(stream.getvalue(), "")


# Measured on Linux before the two run-length functions were unified. Do not update these to make
# a change pass: 'edt' exists to reproduce published results.
REFERENCE_TOTAL = 2254.2010466280785
REFERENCE_MID_FRACTION = 0.4965957564441733
REFERENCE_N_NODES = 299

# The EDT shortest path is tie-degenerate: the edge weights are 1 / (1e-3 + min(dt)) and the dt
# values are square roots of small integers, so many weights are exactly equal and many routes tie
# to within a few ulp. Which route Dijkstra returns is therefore decided by the last bits, which
# differ on arm64. Perturbing every weight by one ulp moves the total by up to 0.03 % on this
# fixture and 0.1 % on a real cochlea, and macOS CI lands 0.027 % below the value above. 0.5 % is
# wide enough for that and far below any change worth catching.
TOTAL_TOLERANCE = 0.005


class TestEdtReproducesPublishedPaths(unittest.TestCase):
    """The 'edt' method must keep reproducing the paths used for the CochleaNet paper.

    The exact run length is **not** a cross-platform invariant, for the reason given above
    `TOTAL_TOLERANCE`, so it is asserted within a tolerance. The assertions that do hold bit-exactly
    carry the real guarantee:

    - the int64 dtype is what guards the truncation quirk of `component_paths_edt`; removing the
      truncation changes the dtype immediately, and shifts the run length by only 0.3 %;
    - the node count, the consecutive keys and the bounds of the length fractions are unaffected by
      the tie, because a different route has the same number of hops.

    Byte-identity against the published tables is checked by hand against the real segmentation
    tables, which cannot live in CI. See the branch notes.
    """

    def setUp(self):
        from flamingo_tools.postprocessing.cochlea_mapping import measure_run_length
        self.points = [list(map(tuple, _make_tube()))]
        self.total, self.path_dict = measure_run_length(self.points, path_method="edt")

    def test_total_distance_is_unchanged(self):
        self.assertAlmostEqual(self.total, REFERENCE_TOTAL, delta=TOTAL_TOLERANCE * REFERENCE_TOTAL)

    def test_path_keeps_the_integer_quantization(self):
        # The smoothed path is truncated to whole µm. See component_paths_edt.
        from flamingo_tools.postprocessing.cochlea_mapping import CENTRAL_PATH_METHODS
        path = CENTRAL_PATH_METHODS["edt"](self.points)[0]
        self.assertEqual(path.dtype, np.int64)
        self.assertEqual(len(path), REFERENCE_N_NODES)

    def test_length_fractions_are_unchanged(self):
        fractions = np.array([self.path_dict[key]["length_fraction"] for key in sorted(self.path_dict)])
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)
        self.assertAlmostEqual(fractions[len(fractions) // 2], REFERENCE_MID_FRACTION, delta=TOTAL_TOLERANCE)
        self.assertTrue(np.all(np.diff(fractions) >= 0))

    def test_path_dict_keys_are_consecutive(self):
        self.assertEqual(sorted(self.path_dict), list(range(len(self.path_dict))))

    def test_tolerance_admits_the_measured_platform_spread(self):
        """The bound has to admit the macOS value and still reject a real algorithmic change."""
        bound = TOTAL_TOLERANCE * REFERENCE_TOTAL
        self.assertLess(abs(2253.590453058356 - REFERENCE_TOTAL), bound)   # macOS CI
        self.assertGreater(abs(1.03 * REFERENCE_TOTAL - REFERENCE_TOTAL), bound)


class TestRefinedCentralPath(unittest.TestCase):

    def test_refined_path_follows_the_known_axis(self):
        from flamingo_tools.postprocessing.cochlea_mapping import CENTRAL_PATH_METHODS
        points = [list(map(tuple, _make_tube()))]
        # The terminal nodes legitimately stop short of the ends of the axis.
        edt = _distance_to_tube_axis(CENTRAL_PATH_METHODS["edt"](points)[0])[5:-5]
        refined = _distance_to_tube_axis(CENTRAL_PATH_METHODS["edt_refined"](points)[0])[5:-5]

        self.assertLess(refined.max(), 5.0)
        self.assertLess(np.median(refined), 0.5 * np.median(edt))
        self.assertGreater(edt.max(), 5.0)

    def test_hull_centroid_is_not_biased_by_the_cell_density(self):
        """A plain mean of the points follows the density, the hull area centroid follows the shape."""
        from flamingo_tools.postprocessing.cochlea_mapping import hull_area_centroid
        rng = np.random.default_rng(1)
        hull_bias, mean_bias = [], []
        for _ in range(50):
            around = rng.random(4000) * 2 * np.pi
            radius = 60.0 * np.sqrt(rng.random(4000))
            points = np.stack([radius * np.cos(around), radius * np.sin(around)], axis=1)
            # Linear density gradient across the cross-section.
            keep = rng.random(len(points)) < (1 + 0.8 * points[:, 0] / 60.0) / 1.8
            points = points[keep][:200]
            hull_bias.append(hull_area_centroid(points)[0])
            mean_bias.append(points.mean(axis=0)[0])

        self.assertGreater(np.mean(mean_bias), 9.0)
        self.assertLess(abs(np.mean(hull_bias)), 5.0)

    def test_neighboring_turn_of_the_spiral_is_kept_out(self):
        """The refined path must not be pulled toward the turn of the spiral next to it.

        This exercises the production selection end to end: the same path is refined once with
        only its own cells and once with the cells of a neighboring turn added. If a cross-section
        picked up cells from that turn, the two results would differ.
        """
        from flamingo_tools.postprocessing.cochlea_mapping import refine_central_path, resample_path
        own = _make_tube(n_points=3000, turns=0.9, seed=2)
        # 200 µm between the two axes is the tightest separation the method tolerates; at 150 µm
        # the turns merge and the path is pulled about 70 µm off course.
        neighbor = _make_tube(n_points=3000, turns=0.9, seed=3) + np.array([0.0, 0.0, 200.0])
        axis = resample_path(_tube_axis(np.linspace(0, 2 * np.pi * 0.9, 400)), 20.0)

        alone = refine_central_path(axis, own)
        with_neighbor = refine_central_path(axis, np.concatenate([own, neighbor]))

        self.assertEqual(len(alone), len(with_neighbor))
        np.testing.assert_allclose(alone, with_neighbor, atol=1e-9)

    def test_refinement_does_not_erode_the_endpoints(self):
        """A pass must not shorten the path, or the run length would depend on the pass count.

        The moving average pads with the edge value, which pulls both terminal nodes inward.
        Resampling then pins the shortened ends, so the loss would accumulate over the passes.
        """
        from flamingo_tools.postprocessing.cochlea_mapping import arc_length, refine_central_path
        # A straight path through a uniform cylinder: the refinement is a geometric no-op, so any
        # change in length is an artefact of the smoothing.
        straight = np.stack([np.linspace(0, 3000, 151), np.zeros(151), np.zeros(151)], axis=1)
        rng = np.random.default_rng(0)
        along = rng.random(40000) * 3000
        around = rng.random(40000) * 2 * np.pi
        radius = 60.0 * np.sqrt(rng.random(40000))
        cells = np.stack([along, radius * np.cos(around), radius * np.sin(around)], axis=1)

        for n_iterations in (1, 3, 5):
            refined = refine_central_path(straight, cells, n_iterations=n_iterations, convergence_tol=0.0)
            self.assertAlmostEqual(arc_length(refined)[-1], 3000.0, delta=1.0)

    def test_degenerate_tangent_does_not_produce_nan(self):
        """A path that doubles back must not turn into NaN tangents or NaN positions."""
        from flamingo_tools.postprocessing.cochlea_mapping import (
            plane_basis, path_tangents, refine_central_path,
        )
        doubling_back = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
                                  [10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        tangents = path_tangents(doubling_back, window=40.0, spacing=10.0)
        first, second = plane_basis(tangents)
        self.assertTrue(np.isfinite(tangents).all())
        self.assertTrue(np.isfinite(first).all() and np.isfinite(second).all())

        # A genuinely zero tangent must still not divide into NaN.
        first, second = plane_basis(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        self.assertTrue(np.isfinite(first).all() and np.isfinite(second).all())

        cells = np.random.default_rng(0).normal(size=(500, 3)) * 5 + np.array([10.0, 0.0, 0.0])
        self.assertTrue(np.isfinite(refine_central_path(doubling_back, cells)).all())


class TestSharedPathSteps(unittest.TestCase):

    def test_run_length_skips_the_gaps_between_components(self):
        from flamingo_tools.postprocessing.cochlea_mapping import _centroids_per_component, measure_run_length
        table, _, _ = _make_arc_table()
        components = _centroids_per_component(table, [1, 2, 3])

        without_gap, path_dict = measure_run_length(components, path_method="graph")
        with_gap, _ = measure_run_length(components, path_method="graph", include_gap=True)

        # 60, 40 and 20 points per component at a spacing of 10 µm along the arc.
        self.assertAlmostEqual(without_gap, (59 + 39 + 19) * 10.0, delta=5.0)
        self.assertGreater(with_gap, without_gap + 700.0)

        fractions = np.array([path_dict[key]["length_fraction"] for key in sorted(path_dict)])
        self.assertTrue(np.all(np.diff(fractions) >= 0))
        self.assertEqual(fractions[0], 0.0)
        self.assertEqual(fractions[-1], 1.0)

    def test_moving_average_keeps_float_precision(self):
        from flamingo_tools.postprocessing.cochlea_mapping import moving_average_3d
        path = np.arange(30, dtype=float).reshape(10, 3) / 3.0
        smoothed = moving_average_3d(path, window=1)
        self.assertEqual(smoothed.dtype, np.float64)
        # A straight line through evenly spaced nodes is its own moving average.
        np.testing.assert_allclose(smoothed[1:-1], path[1:-1])

    def test_edt_graph_handles_a_mask_that_touches_the_border(self):
        from flamingo_tools.postprocessing.cochlea_mapping import central_path_edt_graph
        mask = np.zeros((4, 4, 4), dtype=bool)
        mask[:, 1, 1] = True
        path = central_path_edt_graph(mask, (0, 1, 1), (3, 1, 1))
        self.assertIsNotNone(path)
        self.assertEqual(len(path), 4)

    def test_volumetric_method_rejects_negative_coordinates(self):
        from flamingo_tools.postprocessing.cochlea_mapping import measure_run_length
        points = _make_tube(n_points=500) - 5000.0
        with self.assertRaises(ValueError) as raised:
            measure_run_length([list(map(tuple, points))], path_method="edt")
        self.assertIn("non-negative", str(raised.exception))


class TestEquidistantCentersCellTypes(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.postprocessing.cochlea_mapping import equidistant_centers
        self.fn = equidistant_centers
        self.table, _, _ = _make_arc_table(comp_sizes=(120,))

    def test_cell_type_is_case_insensitive(self):
        lower = self.fn(self.table, component_label=[1], cell_type="ihc", n_blocks=4)
        upper = self.fn(self.table, component_label=[1], cell_type="IHC", n_blocks=4)
        np.testing.assert_array_equal(np.array(lower), np.array(upper))

    def test_unknown_cell_type_raises(self):
        with self.assertRaises(ValueError):
            self.fn(self.table, component_label=[1], cell_type="ohc", n_blocks=4)


if __name__ == "__main__":
    unittest.main()
