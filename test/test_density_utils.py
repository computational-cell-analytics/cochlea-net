import unittest

import numpy as np
import pandas as pd


def _make_table(n=20, z_center=50.0, frac_center=0.5, frac_spread=0.05, component_label=1):
    """Build a synthetic SGN table with n instances in a grid around z_center."""
    rng = np.random.default_rng(42)
    nx, ny = 4, 5
    assert nx * ny == n

    ax = np.tile(np.linspace(0.0, 60.0, nx), ny)
    ay = np.repeat(np.linspace(0.0, 80.0, ny), nx)
    az = np.full(n, z_center) + rng.uniform(-3.0, 3.0, n)
    half = 5.0

    return pd.DataFrame({
        "label_id": np.arange(1, n + 1),
        "anchor_x": ax,
        "anchor_y": ay,
        "anchor_z": az,
        "bb_min_x": ax - half,
        "bb_max_x": ax + half,
        "bb_min_y": ay - half,
        "bb_max_y": ay + half,
        "bb_min_z": az - half,
        "bb_max_z": az + half,
        "n_pixels": np.full(n, 500),
        "component_labels": np.full(n, component_label, dtype=int),
        "length_fraction": np.linspace(frac_center - frac_spread, frac_center + frac_spread, n),
    })


class TestSgnDensityAtPosition(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.analysis.density_utils import sgn_density_at_position
        self.fn = sgn_density_at_position
        self.table = _make_table()

    def test_basic_mid(self):
        result = self.fn(self.table, reference_position="mid", slice_thickness=40.0)
        self.assertEqual(result["n_sgns"], 20)
        self.assertGreater(result["area_µm²"], 0.0)
        self.assertAlmostEqual(result["density_µm⁻²"], result["n_sgns"] / result["area_µm²"])
        self.assertEqual(result["axis"], "z")

    def test_preset_fractions(self):
        from flamingo_tools.analysis.density_utils import REFERENCE_PRESETS
        for name, val in REFERENCE_PRESETS.items():
            t = _make_table(frac_center=val)
            result = self.fn(t, reference_position=name, run_length_tolerance=0.1)
            self.assertEqual(result["reference_fraction"], val)

    def test_custom_float(self):
        result = self.fn(self.table, reference_position=0.5)
        self.assertEqual(result["reference_fraction"], 0.5)

    def test_run_length_filtering(self):
        # Add 5 extra instances far from frac=0.5 that still overlap the z-slice.
        extra = _make_table(n=20, z_center=50.0, frac_center=0.8, frac_spread=0.01)
        extra["label_id"] += 100
        combined = pd.concat([self.table, extra], ignore_index=True)

        # With tight tolerance, extras should be excluded.
        result = self.fn(combined, reference_position="mid", slice_thickness=40.0, run_length_tolerance=0.1)
        self.assertEqual(result["n_sgns"], 20)

    def test_slice_thickness_limits(self):
        # With zero thickness, only instances whose bb straddles the center survive.
        result_thin = self.fn(self.table, reference_position="mid", slice_thickness=1.0)
        result_thick = self.fn(self.table, reference_position="mid", slice_thickness=40.0)
        self.assertLessEqual(result_thin["n_sgns"], result_thick["n_sgns"])

    def test_configurable_axis(self):
        for ax in ("x", "y", "z"):
            result = self.fn(self.table, reference_position="mid", axis=ax)
            self.assertEqual(result["axis"], ax)

    def test_component_filter(self):
        # Component label 2 instances are outside the default filter.
        extra = _make_table(n=20, frac_center=0.5, component_label=2)
        extra["label_id"] += 100
        combined = pd.concat([self.table, extra], ignore_index=True)
        result = self.fn(combined, reference_position="mid", component_label=1)
        self.assertEqual(result["n_sgns"], 20)

    def test_invalid_axis(self):
        with self.assertRaises(ValueError):
            self.fn(self.table, axis="w")

    def test_invalid_preset(self):
        with self.assertRaises(ValueError):
            self.fn(self.table, reference_position="tip")

    def test_out_of_range_float(self):
        with self.assertRaises(ValueError):
            self.fn(self.table, reference_position=1.5)

    def test_missing_column(self):
        bad = self.table.drop(columns=["length_fraction"])
        with self.assertRaises(ValueError):
            self.fn(bad)

    def test_result_keys(self):
        result = self.fn(self.table)
        expected_keys = {
            "reference_fraction", "reference_label_id", "slice_center",
            "slice_min", "slice_max", "n_sgns", "area_µm²", "density_µm⁻²", "axis",
        }
        self.assertEqual(set(result.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
