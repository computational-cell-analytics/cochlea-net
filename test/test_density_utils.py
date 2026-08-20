import unittest
import warnings

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
        result = self.fn(combined, reference_position="mid", component_list=[1])
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
            "bb_min", "bb_max", "bb_center", "label_ids", "label_removed",
            "min_overlap_fraction", "min_overlap_volume", "hull_vertices",
            "component_list", "n_clusters", "cluster_removed",
        }
        self.assertEqual(set(result.keys()), expected)
        self.assertNotIn("volume", result)

    def test_result_keys_3d(self):
        result = self.fn(self.table, mode="3d")
        expected = {
            "reference_fraction", "reference_label_id", "slice_center",
            "slice_min", "slice_max", "slice_thickness", "n_sgns",
            "volume", "density", "mode", "axis",
            "bb_min", "bb_max", "bb_center", "label_ids", "label_removed",
            "min_overlap_fraction", "min_overlap_volume", "hull_vertices",
            "component_list", "n_clusters", "cluster_removed",
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

    def test_auto_roi_halo_slice_axis_matches_slice_thickness(self):
        # Along the slice axis the halo must equal ceil(slice_thickness / 2 / voxel_size).
        import math
        voxel_size = (0.38, 0.38, 0.38)
        axis_index = {"x": 0, "y": 1, "z": 2}
        out = self.build(self.density_results, voxel_size=voxel_size)
        for entry, (label, pos_result) in zip(out, self.density_results.items()):
            ax = pos_result["axis"]
            st = pos_result["slice_thickness"]
            vs = voxel_size[axis_index[ax]]
            expected = max(10, math.ceil(st / 2.0 / vs))
            self.assertEqual(entry["roi_halo"][axis_index[ax]], expected)

    def test_auto_roi_halo_projection_axes_cover_bbox(self):
        # Along the two projection axes the halo must cover the bounding box half-extents.
        import math
        voxel_size = (0.38, 0.38, 0.38)
        axis_index = {"x": 0, "y": 1, "z": 2}
        out = self.build(self.density_results, voxel_size=voxel_size)
        for entry, (label, pos_result) in zip(out, self.density_results.items()):
            ax_i = axis_index[pos_result["axis"]]
            bb_min = pos_result["bb_min"]
            bb_max = pos_result["bb_max"]
            for i, (lo, hi, vs) in enumerate(zip(bb_min, bb_max, voxel_size)):
                if i == ax_i:
                    continue  # slice axis tested separately
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

    def test_skips_position_with_nan_bb_center(self):
        # A position with zero SGNs (e.g. wrong component_list) has bb_center = [nan, nan, nan].
        # round(nan) must not crash; the position is skipped with a warning instead.
        density_results = dict(self.density_results)
        first_key = next(iter(density_results))
        no_sgn_result = dict(density_results[first_key])
        no_sgn_result["bb_center"] = [float("nan")] * 3
        density_results[first_key] = no_sgn_result

        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = self.build(density_results)
            self.assertTrue(any(first_key in str(w.message) for w in caught))

        labels = [entry["position_label"] for entry in out]
        self.assertNotIn(first_key, labels)
        self.assertEqual(len(out), len(density_results) - 1)


def _make_seg_array(table, shape_zyx, voxel_size=(1.0, 1.0, 1.0), origin_xyz=(0.0, 0.0, 0.0)):
    """Paint one voxel per label_id at its anchor position into a (Z, Y, X) array.

    `origin_xyz` is the physical coordinate of pixel (0,0,0) of the array (µm, x/y/z order).
    For a full volume use (0,0,0); for a crop set to the crop's lower-left corner.
    """
    seg = np.zeros(shape_zyx, dtype=np.int32)
    for _, row in table.iterrows():
        z = round((row["anchor_z"] - origin_xyz[2]) / voxel_size[2])
        y = round((row["anchor_y"] - origin_xyz[1]) / voxel_size[1])
        x = round((row["anchor_x"] - origin_xyz[0]) / voxel_size[0])
        if 0 <= z < shape_zyx[0] and 0 <= y < shape_zyx[1] and 0 <= x < shape_zyx[2]:
            seg[z, y, x] = int(row["label_id"])
    return seg


class TestFilterBySegmentationOverlap(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.analysis.density_utils import sgn_density_at_position
        self.fn = sgn_density_at_position
        # Use integer voxel size so physical coords = pixel coords (easy to reason about).
        self.voxel_size = (1.0, 1.0, 1.0)
        self.table = _make_table(n=20, z_center=50.0, frac_center=0.5, frac_spread=0.05)
        # Shape covers the full coordinate range of _make_table (x≤65, y≤85, z~50±8).
        self.shape_zyx = (120, 100, 80)

    def _seg_in_slice(self):
        """Segmentation array with all anchors visible in a z-slice around 50."""
        return _make_seg_array(self.table, self.shape_zyx, self.voxel_size)

    def _seg_out_of_slice(self):
        """Segmentation array with all anchors shifted 60 µm above the slice."""
        shifted = self.table.copy()
        shifted["anchor_z"] += 60.0
        shifted["bb_min_z"] += 60.0
        shifted["bb_max_z"] += 60.0
        return _make_seg_array(shifted, self.shape_zyx, self.voxel_size)

    def test_all_kept_when_fully_in_slice(self):
        seg = self._seg_in_slice()
        # Each label_id has exactly 1 voxel in the seg and n_pixels=500, so
        # overlap = 1/500 = 0.002.  Use a very small threshold so all pass.
        result_low = self.fn(
            self.table, reference_position="mid", slice_thickness=40.0,
            segmentation=seg, voxel_size=self.voxel_size, min_overlap_fraction=1 / 500,
        )
        self.assertEqual(result_low["n_sgns"], 20)

    def test_all_excluded_when_outside_slice(self):
        seg = self._seg_out_of_slice()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.fn(
                self.table, reference_position="mid", slice_thickness=40.0,
                segmentation=seg, voxel_size=self.voxel_size, min_overlap_fraction=1 / 500,
            )
        self.assertEqual(result["n_sgns"], 0)

    def test_partial_overlap_threshold(self):
        # Build a seg where only label_ids 1–10 are painted (first half of table).
        partial_table = self.table.iloc[:10]
        seg = _make_seg_array(partial_table, self.shape_zyx, self.voxel_size)
        result = self.fn(
            self.table, reference_position="mid", slice_thickness=40.0,
            segmentation=seg, voxel_size=self.voxel_size, min_overlap_fraction=1 / 500,
        )
        self.assertEqual(result["n_sgns"], 10)

    def test_no_filtering_when_none(self):
        # Without segmentation, result is the same as before (20 SGNs).
        result = self.fn(
            self.table, reference_position="mid", slice_thickness=40.0,
            min_overlap_fraction=None,
        )
        self.assertEqual(result["n_sgns"], 20)

    def test_result_includes_min_overlap_key(self):
        seg = self._seg_in_slice()
        result = self.fn(
            self.table, reference_position="mid", slice_thickness=40.0,
            segmentation=seg, voxel_size=self.voxel_size, min_overlap_fraction=0.002,
        )
        self.assertIn("min_overlap_fraction", result)
        self.assertAlmostEqual(result["min_overlap_fraction"], 0.002)

    def test_min_overlap_fraction_none_in_result_when_not_used(self):
        result = self.fn(self.table, reference_position="mid")
        self.assertIn("min_overlap_fraction", result)
        self.assertIsNone(result["min_overlap_fraction"])

    def test_min_overlap_volume_filters(self):
        # n_pixels=500, voxel_size=1.0 → one voxel = 1 µm³, threshold=0.5 µm³ should keep all.
        seg = self._seg_in_slice()
        result = self.fn(
            self.table, reference_position="mid", slice_thickness=40.0,
            segmentation=seg, voxel_size=self.voxel_size, min_overlap_volume=0.5,
        )
        self.assertEqual(result["n_sgns"], 20)
        # Threshold above 1 µm³ (each label has exactly 1 voxel = 1 µm³) → all excluded.
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_excl = self.fn(
                self.table, reference_position="mid", slice_thickness=40.0,
                segmentation=seg, voxel_size=self.voxel_size, min_overlap_volume=2.0,
            )
        self.assertEqual(result_excl["n_sgns"], 0)

    def test_crop_auto_detection(self):
        # Build a small crop array whose z-extent (40 µm) is less than the max anchor
        # coordinate (~53 µm), triggering the crop-detection path.  Labels are painted
        # relative to the crop origin at z=30 µm, so all anchors (z≈47-53) map to
        # pixels z≈17-23 inside the 40-px array.
        crop_origin_z = 30.0
        crop_shape_zyx = (40, 100, 80)
        seg = _make_seg_array(
            self.table, crop_shape_zyx, self.voxel_size, origin_xyz=(0.0, 0.0, crop_origin_z)
        )
        result = self.fn(
            self.table, reference_position="mid", slice_thickness=40.0,
            segmentation=seg, voxel_size=self.voxel_size, min_overlap_fraction=1 / 500,
        )
        self.assertEqual(result["n_sgns"], 20)


class TestClusterFilter(unittest.TestCase):
    """The slice must measure one turn of the Rosenthal's canal, not the gap between two."""

    def setUp(self):
        from flamingo_tools.analysis.density_utils import sgn_density_at_position
        self.fn = sgn_density_at_position

    @staticmethod
    def _two_turns(x_offset=500.0, n_second=20, frac_center=0.5):
        """One population at the reference fraction plus a second turn inside the same slice.

        The second population sits `x_offset` away in x, so it forms its own spatial cluster, and
        slightly further along the canal, so the run length identifies which one was asked for.
        """
        near = _make_table(n=20, z_center=50.0, frac_center=frac_center, frac_spread=0.005)
        far = _make_table(n=n_second, z_center=50.0, frac_center=frac_center + 0.05, frac_spread=0.005)
        far["label_id"] += 100
        for column in ("anchor_x", "bb_min_x", "bb_max_x"):
            far[column] += x_offset
        return pd.concat([near, far], ignore_index=True)

    def test_keeps_only_the_requested_turn(self):
        table = self._two_turns()
        result = self.fn(table, reference_position=0.5, slice_thickness=40.0)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["n_sgns"], 20)
        self.assertTrue(all(i <= 20 for i in result["label_ids"]))
        self.assertEqual(sorted(result["cluster_removed"]), sorted(range(101, 121)))

    def test_disabled_keeps_both_turns(self):
        table = self._two_turns()
        filtered = self.fn(table, reference_position=0.5, slice_thickness=40.0)
        unfiltered = self.fn(table, reference_position=0.5, slice_thickness=40.0, cluster_filter=False)
        self.assertEqual(unfiltered["n_sgns"], 40)
        self.assertEqual(unfiltered["cluster_removed"], [])
        # No clustering ran, so the result must not claim a single turn was verified.
        self.assertIsNone(unfiltered["n_clusters"])
        # The hull of both turns spans the blank space between them, so the area is far larger.
        self.assertGreater(unfiltered["area"], 5 * filtered["area"])
        self.assertLess(unfiltered["density"], filtered["density"])

    def test_single_cluster_is_untouched(self):
        table = _make_table(n=20, z_center=50.0, frac_center=0.5, frac_spread=0.005)
        filtered = self.fn(table, reference_position=0.5, slice_thickness=40.0)
        unfiltered = self.fn(table, reference_position=0.5, slice_thickness=40.0, cluster_filter=False)
        self.assertEqual(filtered["n_clusters"], 1)
        self.assertEqual(filtered["n_sgns"], unfiltered["n_sgns"])
        self.assertEqual(filtered["area"], unfiltered["area"])

    def test_single_instance_keeps_bbox_fallback(self):
        table = _make_table(n=20, z_center=50.0, frac_center=0.5, frac_spread=0.005).iloc[[0]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.fn(table, reference_position=0.5, slice_thickness=1.0)
        self.assertEqual(result["n_sgns"], 1)
        self.assertIsNone(result["hull_vertices"])

    def test_falls_back_to_largest_cluster(self):
        # Both clusters are below min_cluster_size, so the position must still report a density.
        table = self._two_turns(n_second=20)
        result = self.fn(table, reference_position=0.5, slice_thickness=40.0, min_cluster_size=100)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["n_sgns"], 20)
        self.assertGreater(result["density"], 0)

    def test_selects_by_run_length_not_by_size(self):
        # The far turn is the larger cluster, but the near one matches the requested fraction.
        near = _make_table(n=20, z_center=50.0, frac_center=0.5, frac_spread=0.005)
        far_a = _make_table(n=20, z_center=50.0, frac_center=0.58, frac_spread=0.005)
        far_b = _make_table(n=20, z_center=50.0, frac_center=0.58, frac_spread=0.005)
        far_a["label_id"] += 100
        far_b["label_id"] += 200
        for column in ("anchor_x", "bb_min_x", "bb_max_x"):
            far_a[column] += 500.0
            far_b[column] += 510.0
        table = pd.concat([near, far_a, far_b], ignore_index=True)
        result = self.fn(table, reference_position=0.5, slice_thickness=40.0, run_length_tolerance=0.2)
        self.assertEqual(result["n_sgns"], 20)
        self.assertTrue(all(i <= 20 for i in result["label_ids"]))

    def test_max_edge_distance_merges_the_turns(self):
        table = self._two_turns(x_offset=30.0)
        result = self.fn(table, reference_position=0.5, slice_thickness=40.0, max_edge_distance=40.0)
        self.assertEqual(result["n_clusters"], 1)
        self.assertEqual(result["n_sgns"], 40)

    def test_component_list_is_recorded(self):
        table = _make_table(n=20, frac_center=0.5, frac_spread=0.005, component_label=3)
        result = self.fn(table, reference_position=0.5, component_list=[3])
        self.assertEqual(result["component_list"], [3])


class TestHullToMask(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.analysis.density_utils import sgn_density_at_position, hull_to_mask
        self.density_fn = sgn_density_at_position
        self.mask_fn = hull_to_mask
        self.table = _make_table(n=20, z_center=50.0, frac_center=0.5, frac_spread=0.05)
        self.voxel_size = (1.0, 1.0, 1.0)
        self.roi_halo = [40, 50, 30]  # [hx, hy, hz]

    def _density_result(self, mode="2d"):
        return self.density_fn(
            self.table, reference_position="mid", slice_thickness=40.0, mode=mode,
        )

    def test_2d_mask_shape(self):
        result = self._density_result(mode="2d")
        mask = self.mask_fn(
            result["hull_vertices"], result["bb_center"],
            self.roi_halo, self.voxel_size, axis="z", mode="2d",
        )
        hz, hy, hx = self.roi_halo[2], self.roi_halo[1], self.roi_halo[0]
        self.assertEqual(mask.shape, (2 * hz, 2 * hy, 2 * hx))

    def test_2d_mask_nonzero(self):
        result = self._density_result(mode="2d")
        mask = self.mask_fn(
            result["hull_vertices"], result["bb_center"],
            self.roi_halo, self.voxel_size, axis="z", mode="2d",
        )
        self.assertTrue(mask.any())

    def test_2d_mask_extruded_uniformly(self):
        # For axis="z" every z-slice must be identical (the 2D polygon extruded along z).
        result = self._density_result(mode="2d")
        mask = self.mask_fn(
            result["hull_vertices"], result["bb_center"],
            self.roi_halo, self.voxel_size, axis="z", mode="2d",
        )
        for zi in range(mask.shape[0]):
            np.testing.assert_array_equal(mask[zi], mask[0])

    def test_3d_mask_shape(self):
        result = self._density_result(mode="3d")
        mask = self.mask_fn(
            result["hull_vertices"], result["bb_center"],
            self.roi_halo, self.voxel_size, axis="z", mode="3d",
        )
        hz, hy, hx = self.roi_halo[2], self.roi_halo[1], self.roi_halo[0]
        self.assertEqual(mask.shape, (2 * hz, 2 * hy, 2 * hx))

    def test_3d_mask_nonzero(self):
        result = self._density_result(mode="3d")
        mask = self.mask_fn(
            result["hull_vertices"], result["bb_center"],
            self.roi_halo, self.voxel_size, axis="z", mode="3d",
        )
        self.assertTrue(mask.any())

    def test_hull_vertices_none_when_too_few_points(self):
        # A table with only 1 SGN cannot form a convex hull → hull_vertices must be None.
        single = self.table.iloc[:1].copy()
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = self.density_fn(single, reference_position="mid", slice_thickness=40.0)
        self.assertIsNone(result["hull_vertices"])


class TestComponentListAutoDefault(unittest.TestCase):
    """calc_sgn_density's component_list should fall back to json_input's component_list,
    then to [1], when not passed explicitly."""

    def setUp(self):
        import json
        import os
        import tempfile
        from flamingo_tools.analysis.density_utils import calc_sgn_density
        self.calc = calc_sgn_density

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        # Component 1 only covers the base region; component 2 only covers the apex region.
        base_rows = _make_table(n=20, frac_center=0.85, frac_spread=0.02, component_label=1)
        apex_rows = _make_table(n=20, frac_center=0.15, frac_spread=0.02, component_label=2)
        apex_rows["label_id"] += 100
        table = pd.concat([base_rows, apex_rows], ignore_index=True)

        self.table_path = os.path.join(self.tmpdir.name, "table.tsv")
        table.to_csv(self.table_path, sep="\t", index=False)

        self.json_path = os.path.join(self.tmpdir.name, "info.json")
        with open(self.json_path, "w") as f:
            json.dump({"dataset_name": "TEST", "component_list": [1, 2]}, f)

    def _run(self, name, json_input=None, component_list=None):
        import json
        import os
        output_path = os.path.join(self.tmpdir.name, f"{name}.json")
        self.calc(
            output=output_path,
            seg_table_path=self.table_path,
            json_input=json_input,
            positions=["apex"],
            run_length_tolerance=0.05,
            slice_thickness=40.0,
            component_list=component_list,
            force_overwrite=True,
        )
        with open(output_path) as f:
            return json.load(f)

    def test_auto_default_uses_json_component_list(self):
        # Apex SGNs only exist in component 2; the json's component_list=[1, 2] must be picked up.
        result = self._run("with_json", json_input=self.json_path, component_list=None)
        self.assertEqual(result["apex"]["n_sgns"], 20)

    def test_no_json_input_falls_back_to_component_1(self):
        # Without json_input, component_list falls back to [1], which has no apex SGNs.
        result = self._run("no_json", json_input=None, component_list=None)
        self.assertEqual(result["apex"]["n_sgns"], 0)

    def test_explicit_component_list_overrides_json(self):
        # An explicit component_list argument takes precedence over json_input's value.
        result = self._run("explicit_override", json_input=self.json_path, component_list=[1])
        self.assertEqual(result["apex"]["n_sgns"], 0)


class TestImgPathsWithoutCropOutput(unittest.TestCase):

    def setUp(self):
        import os
        import tempfile
        from flamingo_tools.analysis.density_utils import calc_sgn_density
        self.calc = calc_sgn_density

        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)

        self.table_path = os.path.join(self.tmpdir.name, "table.tsv")
        _make_table().to_csv(self.table_path, sep="\t", index=False)
        self.output_path = os.path.join(self.tmpdir.name, "out.json")

    def test_warns_and_ignores_img_paths(self):
        import warnings
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.calc(
                output=self.output_path,
                seg_table_path=self.table_path,
                positions=["mid"],
                img_paths=["/does/not/exist.ome.zarr"],
                crop_output=None,
                force_overwrite=True,
            )
            self.assertTrue(any("img_paths" in str(w.message) for w in caught))


def _overlap_entry(fraction, label_ids=None, hull_vertices=None, slice_min=0.0, slice_max=10.0, mode="2d"):
    """Build a minimal density result entry for the overlap tests."""
    entry = {
        "reference_fraction": fraction,
        "n_sgns": 0 if label_ids is None else len(label_ids),
        "slice_min": slice_min,
        "slice_max": slice_max,
        "mode": mode,
        "axis": "z",
        "hull_vertices": hull_vertices,
    }
    if label_ids is not None:
        entry["label_ids"] = list(label_ids)
    return entry


UNIT_SQUARE = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
OFFSET_SQUARE = [[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5]]
DISTANT_SQUARE = [[5.0, 5.0], [6.0, 5.0], [6.0, 6.0], [5.0, 6.0]]


class TestDensityPositionOverlap(unittest.TestCase):
    def test_shared_ids_and_hull_iou(self):
        from flamingo_tools.analysis.density_utils import density_position_overlap

        results = {
            "0.1": _overlap_entry(0.1, [1, 2, 3, 4], UNIT_SQUARE, slice_min=0.0, slice_max=10.0),
            "0.15": _overlap_entry(0.15, [3, 4, 5, 6], OFFSET_SQUARE, slice_min=5.0, slice_max=15.0),
        }
        records = density_position_overlap(results)

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["n_shared"], 2)
        self.assertAlmostEqual(record["jaccard"], 1 / 3)
        self.assertAlmostEqual(record["shared_fraction"], 0.5)
        # Half-offset unit squares: intersection 0.25, union 1.75.
        self.assertAlmostEqual(record["hull_intersection"], 0.25)
        self.assertAlmostEqual(record["hull_iou"], 1 / 7)
        self.assertAlmostEqual(record["slice_overlap"], 5.0)

    def test_disjoint_hulls(self):
        from flamingo_tools.analysis.density_utils import density_position_overlap

        results = {
            "0.1": _overlap_entry(0.1, [1, 2, 3], UNIT_SQUARE, slice_min=0.0, slice_max=10.0),
            "0.9": _overlap_entry(0.9, [4, 5, 6], DISTANT_SQUARE, slice_min=20.0, slice_max=30.0),
        }
        record = density_position_overlap(results)[0]

        self.assertEqual(record["n_shared"], 0)
        self.assertAlmostEqual(record["jaccard"], 0.0)
        self.assertAlmostEqual(record["hull_iou"], 0.0)
        self.assertAlmostEqual(record["slice_overlap"], 0.0)

    def test_missing_ids_and_hull(self):
        from flamingo_tools.analysis.density_utils import density_position_overlap

        results = {
            "apex": _overlap_entry(0.15, slice_min=0.0, slice_max=10.0),
            "base": _overlap_entry(0.85, slice_min=2.0, slice_max=10.0),
        }
        record = density_position_overlap(results)[0]

        self.assertIsNone(record["n_shared"])
        self.assertIsNone(record["jaccard"])
        self.assertIsNone(record["shared_fraction"])
        self.assertIsNone(record["hull_iou"])
        self.assertAlmostEqual(record["slice_overlap"], 8.0)

    def test_3d_mode_has_no_hull_iou(self):
        from flamingo_tools.analysis.density_utils import density_position_overlap

        hull_3d = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        results = {
            "0.1": _overlap_entry(0.1, [1, 2], hull_3d, mode="3d"),
            "0.15": _overlap_entry(0.15, [2, 3], hull_3d, mode="3d"),
        }
        record = density_position_overlap(results)[0]

        self.assertIsNone(record["hull_iou"])
        self.assertEqual(record["n_shared"], 1)

    def test_mismatched_mode_not_compared(self):
        from flamingo_tools.analysis.density_utils import density_position_overlap

        results = {
            "0.1": _overlap_entry(0.1, [1, 2], UNIT_SQUARE, mode="2d"),
            "0.15": _overlap_entry(0.15, [2, 3], UNIT_SQUARE, mode="3d"),
        }
        record = density_position_overlap(results)[0]

        self.assertIsNone(record["hull_iou"])
        self.assertIsNone(record["slice_overlap"])
        self.assertEqual(record["n_shared"], 1)

    def test_all_pairs_sorted_by_fraction(self):
        from flamingo_tools.analysis.density_utils import density_position_overlap

        results = {
            "0.9": _overlap_entry(0.9, [1]),
            "0.1": _overlap_entry(0.1, [2]),
            "0.5": _overlap_entry(0.5, [3]),
        }
        records = density_position_overlap(results)

        self.assertEqual(len(records), 3)
        self.assertEqual(
            [(r["fraction_a"], r["fraction_b"]) for r in records],
            [(0.1, 0.5), (0.1, 0.9), (0.5, 0.9)],
        )

    def test_json_file_input(self):
        import json
        import os
        import tempfile
        from flamingo_tools.analysis.density_utils import density_position_overlap

        results = {
            "0.1": _overlap_entry(0.1, [1, 2, 3], UNIT_SQUARE),
            "0.15": _overlap_entry(0.15, [2, 3, 4], OFFSET_SQUARE),
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = os.path.join(tmp_dir, "SGN_density_2d_extended.json")
            with open(json_path, "w") as f:
                json.dump(results, f)
            records = density_position_overlap(json_path)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["n_shared"], 2)


class TestReportDensityOverlap(unittest.TestCase):
    def test_warns_above_threshold(self):
        import warnings
        from flamingo_tools.analysis.density_utils import report_density_overlap

        results = {
            "0.1": _overlap_entry(0.1, [1, 2, 3, 4], UNIT_SQUARE),
            "0.15": _overlap_entry(0.15, [3, 4, 5, 6], OFFSET_SQUARE),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            records = report_density_overlap(results, name="test", warn_threshold=0.1)

        self.assertEqual(len(records), 1)
        self.assertTrue(any("not independent measurements" in str(w.message) for w in caught))

    def test_no_warning_without_shared_ids(self):
        import warnings
        from flamingo_tools.analysis.density_utils import report_density_overlap

        results = {
            "0.1": _overlap_entry(0.1, [1, 2], UNIT_SQUARE),
            "0.15": _overlap_entry(0.15, [3, 4], OFFSET_SQUARE),
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            report_density_overlap(results, warn_threshold=0.1)

        self.assertFalse(any("not independent measurements" in str(w.message) for w in caught))


if __name__ == "__main__":
    unittest.main()
