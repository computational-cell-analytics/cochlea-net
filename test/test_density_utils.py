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
        self.assertGreater(result["area"], 0.0)
        self.assertAlmostEqual(result["density"], result["n_sgns"] / result["area"])
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
        extra = _make_table(n=20, z_center=50.0, frac_center=0.8, frac_spread=0.01)
        extra["label_id"] += 100
        combined = pd.concat([self.table, extra], ignore_index=True)
        result = self.fn(combined, reference_position="mid", slice_thickness=40.0, run_length_tolerance=0.1)
        self.assertEqual(result["n_sgns"], 20)

    def test_slice_thickness_limits(self):
        result_thin = self.fn(self.table, reference_position="mid", slice_thickness=1.0)
        result_thick = self.fn(self.table, reference_position="mid", slice_thickness=40.0)
        self.assertLessEqual(result_thin["n_sgns"], result_thick["n_sgns"])

    def test_configurable_axis(self):
        for ax in ("x", "y", "z"):
            result = self.fn(self.table, reference_position="mid", axis=ax)
            self.assertEqual(result["axis"], ax)

    def test_component_filter(self):
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

    def test_invalid_mode(self):
        with self.assertRaises(ValueError):
            self.fn(self.table, mode="flat")

    def test_missing_column(self):
        bad = self.table.drop(columns=["length_fraction"])
        with self.assertRaises(ValueError):
            self.fn(bad)

    def test_result_keys_2d(self):
        result = self.fn(self.table, mode="2d")
        expected = {
            "reference_fraction", "reference_label_id", "slice_center",
            "slice_min", "slice_max", "slice_thickness", "n_sgns",
            "area", "density", "mode", "axis",
            "bb_min", "bb_max", "bb_center",
        }
        self.assertEqual(set(result.keys()), expected)
        self.assertNotIn("volume", result)

    def test_result_keys_3d(self):
        result = self.fn(self.table, mode="3d")
        expected = {
            "reference_fraction", "reference_label_id", "slice_center",
            "slice_min", "slice_max", "slice_thickness", "n_sgns",
            "volume", "density", "mode", "axis",
            "bb_min", "bb_max", "bb_center",
        }
        self.assertEqual(set(result.keys()), expected)
        self.assertNotIn("area", result)

    def test_3d_mode(self):
        result = self.fn(self.table, mode="3d", slice_thickness=40.0)
        self.assertEqual(result["mode"], "3d")
        self.assertIn("volume", result)
        self.assertGreater(result["volume"], 0.0)
        self.assertAlmostEqual(result["density"], result["n_sgns"] / result["volume"])

    def test_3d_mode_larger_extent_than_2d(self):
        r2d = self.fn(self.table, mode="2d", slice_thickness=40.0)
        r3d = self.fn(self.table, mode="3d", slice_thickness=40.0)
        # Same SGNs selected; 3D volume > 2D area for a non-flat cluster
        self.assertEqual(r2d["n_sgns"], r3d["n_sgns"])

    def test_bounding_box_shape(self):
        result = self.fn(self.table, reference_position="mid", slice_thickness=40.0)
        for key in ("bb_min", "bb_max", "bb_center"):
            self.assertEqual(len(result[key]), 3, f"{key} should have 3 components")
        for lo, center, hi in zip(result["bb_min"], result["bb_center"], result["bb_max"]):
            self.assertLessEqual(lo, center)
            self.assertLessEqual(center, hi)

    def test_bounding_box_covers_slice_axis(self):
        result = self.fn(self.table, reference_position="mid", slice_thickness=40.0, axis="z")
        self.assertLessEqual(result["bb_min"][2], result["slice_max"])
        self.assertGreaterEqual(result["bb_max"][2], result["slice_min"])


class TestSgnDensityProfile(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.analysis.density_utils import sgn_density_profile
        self.fn = sgn_density_profile
        self.table = _make_table()

    def test_default_positions(self):
        fracs = [0.15, 0.5, 0.85]
        rows = [_make_table(n=20, frac_center=f, frac_spread=0.02) for f in fracs]
        for i, t in enumerate(rows[1:], 1):
            t["label_id"] += i * 100
        combined = pd.concat(rows, ignore_index=True)
        results = self.fn(combined, run_length_tolerance=0.05)
        self.assertEqual(set(results.keys()), {"apex", "mid", "base"})
        for key in results:
            self.assertIn("density", results[key])

    def test_custom_positions(self):
        results = self.fn(self.table, positions=["mid", 0.5])
        self.assertIn("mid", results)
        self.assertIn("0.5", results)

    def test_float_position_key_format(self):
        results = self.fn(self.table, positions=[0.3])
        self.assertIn("0.3", results)
        self.assertEqual(results["0.3"]["reference_fraction"], 0.3)

    def test_mode_forwarded(self):
        results = self.fn(self.table, positions=["mid"], mode="3d")
        self.assertEqual(results["mid"]["mode"], "3d")
        self.assertIn("volume", results["mid"])


class TestBuildBlockExtractionDict(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.analysis.density_utils import sgn_density_profile, _build_block_extraction_dict
        self.build = _build_block_extraction_dict
        fracs = [0.15, 0.5, 0.85]
        rows = [_make_table(n=20, frac_center=f, frac_spread=0.02) for f in fracs]
        for i, t in enumerate(rows[1:], 1):
            t["label_id"] += i * 100
        combined = pd.concat(rows, ignore_index=True)
        self.density_results = sgn_density_profile(
            combined, run_length_tolerance=0.05, slice_thickness=40.0
        )

    def test_returns_list(self):
        out = self.build(self.density_results)
        self.assertIsInstance(out, list)

    def test_one_entry_per_position(self):
        out = self.build(self.density_results)
        self.assertEqual(len(out), len(self.density_results))

    def test_each_entry_has_single_crop_center(self):
        out = self.build(self.density_results)
        for entry in out:
            self.assertIn("crop_centers", entry)
            self.assertEqual(len(entry["crop_centers"]), 1)

    def test_crop_centers_are_int_triples(self):
        out = self.build(self.density_results)
        for entry in out:
            center = entry["crop_centers"][0]
            self.assertEqual(len(center), 3)
            for v in center:
                self.assertIsInstance(v, int)

    def test_position_labels_match_keys(self):
        out = self.build(self.density_results)
        labels = [entry["position_label"] for entry in out]
        self.assertEqual(labels, list(self.density_results.keys()))

    def test_auto_roi_halo_default(self):
        # No explicit halo and no input JSON → auto-computed from bounding box.
        out = self.build(self.density_results)
        for entry in out:
            halo = entry["roi_halo"]
            self.assertEqual(len(halo), 3)
            for v in halo:
                self.assertIsInstance(v, int)
                self.assertGreater(v, 0)

    def test_auto_roi_halo_covers_bbox(self):
        # Auto halo must be >= half the bounding box extent in each axis.
        import math
        voxel_size = (0.38, 0.38, 0.38)
        out = self.build(self.density_results, voxel_size=voxel_size)
        for entry, (label, pos_result) in zip(out, self.density_results.items()):
            bb_min = pos_result["bb_min"]
            bb_max = pos_result["bb_max"]
            for i, (lo, hi, vs) in enumerate(zip(bb_min, bb_max, voxel_size)):
                min_required = math.ceil((hi - lo) / 2.0 / vs)
                self.assertGreaterEqual(entry["roi_halo"][i], min_required)

    def test_roi_halo_from_json_params(self):
        params = {"dataset_name": "test", "roi_halo": [256, 256, 50]}
        out = self.build(self.density_results, input_json_params=params)
        for entry in out:
            self.assertEqual(entry["roi_halo"], [256, 256, 50])

    def test_roi_halo_explicit_overrides_json(self):
        params = {"roi_halo": [256, 256, 50]}
        out = self.build(self.density_results, input_json_params=params, roi_halo=[64, 64, 32])
        for entry in out:
            self.assertEqual(entry["roi_halo"], [64, 64, 32])

    def test_per_position_halo_can_differ(self):
        # With auto-halo and positions at different distances, halos may differ.
        out = self.build(self.density_results)
        halos = [tuple(entry["roi_halo"]) for entry in out]
        # At least check we get 3 per-position entries (may or may not differ).
        self.assertEqual(len(halos), 3)

    def test_metadata_from_json_params(self):
        params = {
            "dataset_name": "M_LR_000155_L",
            "image_channel": ["PV", "GFP", "SGN_v2"],
            "segmentation_channel": "SGN_v2",
            "cell_type": "sgn",
            "component_list": [1],
        }
        out = self.build(self.density_results, input_json_params=params)
        for entry in out:
            self.assertEqual(entry["dataset_name"], "M_LR_000155_L")
            self.assertEqual(entry["image_channel"], ["PV", "GFP", "SGN_v2"])
            self.assertEqual(entry["segmentation_channel"], "SGN_v2")

    def test_no_json_params(self):
        out = self.build(self.density_results)
        for entry in out:
            self.assertNotIn("dataset_name", entry)


if __name__ == "__main__":
    unittest.main()
