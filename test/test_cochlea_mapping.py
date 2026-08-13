import unittest

import numpy as np
import pandas as pd


def _make_arc_table(radius=600.0, spacing=10.0, comp_sizes=(60, 40, 20), gap=400.0):
    """Build a synthetic IHC table along a circular arc that is split into separate components.

    The components are separated by a gap along the arc. Component 1 is the largest one,
    which matches the label order that `measure_run_length_ihcs` expects.

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


if __name__ == "__main__":
    unittest.main()
