import os
import sys
import unittest
from unittest import mock

import pandas as pd

SCRIPTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "export_data")
sys.path.insert(0, SCRIPTS_DIR)


def _make_table():
    return pd.DataFrame({
        "label_id": [1, 2, 3],
        "anchor_x": [10.0, 20.0, 30.0],
        "anchor_y": [1.0, 2.0, 3.0],
        "anchor_z": [100.0, 200.0, 300.0],
        "component_labels": [1, 1, 1],
        "length_fraction": [0.1, 0.5, 0.9],
    })


class TestResolvePositions(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position
        self.table = _make_table()

    def test_default_positions(self):
        resolved = self.mod.resolve_positions(self.table, self.mod.DEFAULT_POSITIONS)
        labels = [r["label"] for r in resolved]
        self.assertEqual(labels, ["apex", "mid", "base"])
        self.assertEqual([r["label_id"] for r in resolved], [1, 2, 3])
        self.assertEqual(resolved[0]["crop_center"], [10.0, 1.0, 100.0])

    def test_multiple_values_per_label(self):
        positions = {"length_fraction": {"apex": [0.1, 0.9]}}
        resolved = self.mod.resolve_positions(self.table, positions)
        self.assertEqual(len(resolved), 2)
        self.assertEqual([r["label_id"] for r in resolved], [1, 3])
        self.assertTrue(all(r["label"] == "apex" for r in resolved))


class TestRunExports(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position

    def test_dispatch_calls_correct_functions_with_expected_kwargs(self):
        calls = {}

        def fake_grid(**kwargs):
            calls["lower_resolution"] = kwargs

        def fake_synapse(**kwargs):
            calls["synapse"] = kwargs

        def fake_frequency(**kwargs):
            calls.setdefault("frequency", []).append(kwargs)

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, fake_grid),
            "synapse": (self.mod._run_synapse_export, fake_synapse),
            "frequency": (self.mod._run_frequency_export, fake_frequency),
        }):
            resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                        "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]
            self.mod.run_exports(
                "cochlea_x", resolved, ["lower_resolution", "synapse", "frequency"],
                scale=[0, 1], output_folder="/tmp/out", roi_halo=None, axis=2,
                json_info={"synapse": {"synapse_name": "syn_v1"}, "frequency": {"source_name": "SGN_v2"}},
            )

        self.assertEqual(calls["lower_resolution"]["cochlea"], "cochlea_x")
        self.assertEqual(calls["lower_resolution"]["scale"], [0, 1])
        self.assertEqual(calls["lower_resolution"]["suffix"], "apex")
        self.assertEqual(calls["lower_resolution"]["crop_center"], [1.0, 2.0, 3.0])
        self.assertEqual(calls["lower_resolution"]["axis"], 2)
        self.assertIsNone(calls["lower_resolution"]["roi_halo"])

        self.assertEqual(calls["synapse"]["scales"], [0, 1])
        self.assertEqual(calls["synapse"]["synapse_name"], "syn_v1")
        self.assertEqual(calls["synapse"]["suffix"], "apex")

        # export_frequency_mapping only accepts a single scale, so it is invoked once per value.
        self.assertEqual(len(calls["frequency"]), 2)
        self.assertEqual([c["scale"] for c in calls["frequency"]], [0, 1])
        self.assertTrue(all(c["source_name"] == "SGN_v2" for c in calls["frequency"]))

    def test_no_json_info_uses_shared_kwargs_only(self):
        recorded = []

        def fake_grid(**kwargs):
            recorded.append(kwargs)

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "marker": (self.mod._run_grid_export, fake_grid),
        }):
            resolved = [{"column": "length_fraction", "label": "mid", "value": 0.5,
                        "label_id": 2, "crop_center": [4.0, 5.0, 6.0]}]
            self.mod.run_exports(
                "cochlea_x", resolved, ["marker"], scale=[0], output_folder="/tmp/out",
                roi_halo=[8, 8, 8], axis=None,
            )

        self.assertEqual(recorded[0]["roi_halo"], [8, 8, 8])
        self.assertIsNone(recorded[0]["axis"])
        self.assertEqual(recorded[0]["suffix"], "mid")


if __name__ == "__main__":
    unittest.main()
