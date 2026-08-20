import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import pandas as pd
import tifffile

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
                voxel_size=[1.887779, 1.887779, 3.0],
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

        # voxel_size reaches every adapter shape.
        voxel_size = [1.887779, 1.887779, 3.0]
        self.assertEqual(calls["lower_resolution"]["voxel_size"], voxel_size)
        self.assertEqual(calls["synapse"]["voxel_size"], voxel_size)
        self.assertTrue(all(c["voxel_size"] == voxel_size for c in calls["frequency"]))

    def test_json_info_voxel_size_overrides_shared_value(self):
        calls = []

        def fake_grid(**kwargs):
            calls.append(kwargs)

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, fake_grid),
            "marker": (self.mod._run_grid_export, fake_grid),
        }):
            resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                        "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]
            self.mod.run_exports(
                "cochlea_x", resolved, ["lower_resolution", "marker"], scale=[0],
                output_folder="/tmp/out", roi_halo=[8, 8, 8], axis=None,
                json_info={"marker": {"voxel_size": [0.76, 0.76, 3.0]}},
                voxel_size=[1.887779, 1.887779, 3.0],
            )

        self.assertEqual(calls[0]["voxel_size"], [1.887779, 1.887779, 3.0])
        self.assertEqual(calls[1]["voxel_size"], [0.76, 0.76, 3.0])

    def test_voxel_size_defaults_to_isotropic_038(self):
        calls = []

        def fake_grid(**kwargs):
            calls.append(kwargs)

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, fake_grid),
        }):
            resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                        "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]
            self.mod.run_exports(
                "cochlea_x", resolved, ["lower_resolution"], scale=[0],
                output_folder="/tmp/out", roi_halo=[8, 8, 8],
            )

        self.assertEqual(tuple(calls[0]["voxel_size"]), (0.38, 0.38, 0.38))

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

    def test_json_info_list_runs_each_entry_as_independent_pass(self):
        calls = []

        def fake_grid(**kwargs):
            calls.append(kwargs)

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, fake_grid),
        }):
            resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                        "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]
            json_info = [
                {"lower_resolution": {"channels": ["PV", "VGlut3", "CTBP2"], "roi_halo": [256, 256, 0]}},
                {"lower_resolution": {"channels": ["IHC_v11"], "filter_by_components": [1]}},
                {"lower_resolution": {"channels": ["SGN_v2"], "filter_by_components": [1]}},
            ]
            # export_functions is deliberately wrong ("synapse" isn't in any entry) to prove
            # it has no effect in list mode -- each entry's own key(s) determine what runs.
            self.mod.run_exports(
                "cochlea_x", resolved, ["synapse"], scale=[0], output_folder="/tmp/out",
                roi_halo=None, axis=2, json_info=json_info,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0]["channels"], ["PV", "VGlut3", "CTBP2"])
        self.assertEqual(calls[0]["roi_halo"], [256, 256, 0])  # entry override wins over the shared None
        self.assertEqual(calls[1]["channels"], ["IHC_v11"])
        self.assertEqual(calls[1]["filter_by_components"], [1])
        self.assertIsNone(calls[1]["roi_halo"])  # falls back to the shared roi_halo (None)
        self.assertEqual(calls[2]["channels"], ["SGN_v2"])
        # shared arguments still apply to every pass unless a pass overrides them.
        self.assertTrue(all(c["cochlea"] == "cochlea_x" for c in calls))
        self.assertTrue(all(c["axis"] == 2 for c in calls))
        self.assertTrue(all(c["suffix"] == "apex" for c in calls))

    def test_json_info_list_unknown_key_raises(self):
        resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                    "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]
        with self.assertRaises(ValueError):
            self.mod.run_exports(
                "cochlea_x", resolved, ["lower_resolution"], scale=[0], output_folder="/tmp/out",
                roi_halo=[8, 8, 8], json_info=[{"not_a_real_key": {}}],
            )

    def test_dict_mode_unknown_key_raises(self):
        resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                    "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]
        with self.assertRaises(ValueError):
            self.mod.run_exports(
                "cochlea_x", resolved, ["not_a_real_key"], scale=[0], output_folder="/tmp/out",
                roi_halo=[8, 8, 8],
            )


SOURCES = {
    "PV": "image",
    "VGlut3": "image",
    "CTBP2": "image",
    "SGN_v2": "segmentation",
    "IHC_v4c": "segmentation",
    "synapse_v3_ihc_v4b": "spots",
    "synapse_v3_ihc_v4c": "spots",
}

# A cochlea that also carries the pinned default versions.
SOURCES_V11 = {
    **SOURCES,
    "IHC_v11": "segmentation",
    "synapse_v3_ihc_v11": "spots",
    "synapse_v5_ihc_v11": "spots",
}


class TestBuildGroupPasses(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position

    def test_sgn_group_splits_channels_and_segmentations(self):
        passes = self.mod.build_group_passes(SOURCES, self.mod.EXPORT_GROUPS["sgn"])

        self.assertEqual([key for key, _ in passes], ["lower_resolution", "lower_resolution"])
        self.assertEqual(passes[0][1], {"channels": ["PV"]})
        self.assertEqual(passes[1][1], {"channels": ["SGN_v2"], "filter_by_components": [1]})

    def test_ihc_group_adds_synapse_pass(self):
        passes = self.mod.build_group_passes(SOURCES, self.mod.EXPORT_GROUPS["ihc"])

        self.assertEqual([key for key, _ in passes], ["lower_resolution", "lower_resolution", "synapse"])
        self.assertEqual(passes[0][1]["channels"], ["VGlut3", "CTBP2"])
        self.assertEqual(passes[1][1], {"channels": ["IHC_v4c"], "filter_by_components": [1]})
        # "auto" pairs the synapse source with the group's IHC segmentation.
        self.assertEqual(passes[2][1], {
            "synapse_name": "synapse_v3_ihc_v4c", "reference_ihcs": "IHC_v4c", "filter_ihc_components": [1],
        })

    def test_pinned_versions_are_used_when_available(self):
        passes = self.mod.build_group_passes(SOURCES_V11, self.mod.EXPORT_GROUPS["ihc"])

        self.assertEqual(passes[1][1]["channels"], ["IHC_v11"])
        self.assertEqual(passes[2][1]["synapse_name"], "synapse_v3_ihc_v11")
        self.assertEqual(passes[2][1]["reference_ihcs"], "IHC_v11")

    def test_explicit_synapse_source(self):
        group = {**self.mod.EXPORT_GROUPS["ihc"], "synapses": "synapse_v3_ihc_v4b"}
        passes = self.mod.build_group_passes(SOURCES, group)

        self.assertEqual(passes[2][1]["synapse_name"], "synapse_v3_ihc_v4b")

    def test_group_overrides_reach_every_pass(self):
        group = {**self.mod.EXPORT_GROUPS["ihc"], "components": [1, 2], "axis": 1, "roi_halo": [128, 128, 0]}
        passes = self.mod.build_group_passes(SOURCES, group)

        self.assertTrue(all(kwargs["axis"] == 1 for _, kwargs in passes))
        self.assertTrue(all(kwargs["roi_halo"] == [128, 128, 0] for _, kwargs in passes))
        self.assertEqual(passes[1][1]["filter_by_components"], [1, 2])
        self.assertEqual(passes[2][1]["filter_ihc_components"], [1, 2])

    def test_filter_cochlea_masks_the_channel_pass(self):
        group = {**self.mod.EXPORT_GROUPS["sgn"], "filter_cochlea": True}
        passes = self.mod.build_group_passes(SOURCES, group)

        self.assertEqual(passes[0][1]["filter_cochlea_channels"], ["SGN_v2"])
        self.assertEqual(passes[0][1]["filter_sgn_components"], [1])
        self.assertNotIn("filter_cochlea_channels", passes[1][1])

    def test_channels_only_group(self):
        group = {"reference": "SGN", "channels": ["PV"]}
        self.assertEqual(self.mod.build_group_passes(SOURCES, group), [("lower_resolution", {"channels": ["PV"]})])

    def test_synapses_without_ihc_segmentation_raises(self):
        group = {"reference": "SGN", "segmentations": ["SGN"], "synapses": "synapses"}
        with self.assertRaises(ValueError):
            self.mod.build_group_passes(SOURCES, group)

    def test_empty_group_raises(self):
        with self.assertRaises(ValueError):
            self.mod.build_group_passes(SOURCES, {"reference": "SGN"})


class TestRunGroups(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position
        self.centers = {
            "SGN_v2": [10.0, 20.0, 30.0],
            "IHC_v4c": [40.0, 50.0, 60.0],
        }

    def _fake_positions(self, cochlea, reference_seg, positions, component_list=None):
        self.resolve_calls.append((reference_seg, tuple(component_list or ())))
        return [{"column": "length_fraction", "label": "apex", "value": 0.15,
                 "label_id": 1, "crop_center": self.centers[reference_seg]}]

    def _run(self, groups, **kwargs):
        self.resolve_calls = []
        calls = []

        def fake_export(**call_kwargs):
            calls.append(call_kwargs)

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, fake_export),
            "synapse": (self.mod._run_synapse_export, fake_export),
        }), \
                mock.patch.object(self.mod, "source_types", return_value=SOURCES), \
                mock.patch.object(self.mod, "resolve_reference_positions", self._fake_positions):
            self.mod.run_groups("cochlea_x", groups, scale=[2], output_folder="/tmp/out", **kwargs)
        return calls

    def test_each_group_uses_its_own_reference_and_output_folder(self):
        calls = self._run({name: self.mod.EXPORT_GROUPS[name] for name in ("sgn", "ihc")}, axis=0)

        self.assertEqual(self.resolve_calls, [("SGN_v2", (1,)), ("IHC_v4c", (1,))])

        sgn_calls = [call for call in calls if call["output_folder"] == "/tmp/out/sgn"]
        ihc_calls = [call for call in calls if call["output_folder"] == "/tmp/out/ihc"]
        self.assertEqual(len(sgn_calls), 2)
        self.assertEqual(len(ihc_calls), 3)

        self.assertTrue(all(call["crop_center"] == self.centers["SGN_v2"] for call in sgn_calls))
        self.assertTrue(all(call["crop_center"] == self.centers["IHC_v4c"] for call in ihc_calls))
        self.assertTrue(all(call["suffix"] == "apex" for call in calls))
        self.assertTrue(all(call["axis"] == 0 for call in calls))

        self.assertEqual([call.get("channels") for call in sgn_calls], [["PV"], ["SGN_v2"]])
        self.assertEqual(ihc_calls[-1]["synapse_name"], "synapse_v3_ihc_v4c")
        # the synapse export takes `scales` instead of `scale`.
        self.assertEqual(ihc_calls[-1]["scales"], [2])

    def test_components_argument_overrides_the_group(self):
        groups = {"sgn": {**self.mod.EXPORT_GROUPS["sgn"], "components": [3]}}
        calls = self._run(groups, axis=0, components=[1, 2])

        self.assertEqual(self.resolve_calls, [("SGN_v2", (1, 2))])
        self.assertEqual(calls[1]["filter_by_components"], [1, 2])

    def test_group_components_apply_without_the_argument(self):
        groups = {"sgn": {**self.mod.EXPORT_GROUPS["sgn"], "components": [3]}}
        calls = self._run(groups, axis=0)

        self.assertEqual(self.resolve_calls, [("SGN_v2", (3,))])
        self.assertEqual(calls[1]["filter_by_components"], [3])

    def test_shared_reference_reads_the_table_once(self):
        groups = {
            "sgn": self.mod.EXPORT_GROUPS["sgn"],
            "sgn_only_seg": {"reference": "SGN", "segmentations": ["SGN"]},
        }
        self._run(groups, axis=0)

        self.assertEqual(self.resolve_calls, [("SGN_v2", (1,))])

    def test_missing_geometry_raises(self):
        with self.assertRaises(ValueError):
            self._run({"sgn": self.mod.EXPORT_GROUPS["sgn"]})

    def test_group_geometry_replaces_the_shared_one(self):
        groups = {"sgn": {**self.mod.EXPORT_GROUPS["sgn"], "roi_halo": [64, 64, 8], "voxel_size": [0.76, 0.76, 3.0]}}
        calls = self._run(groups)

        self.assertTrue(all(call["roi_halo"] == [64, 64, 8] for call in calls))
        self.assertTrue(all(call["voxel_size"] == [0.76, 0.76, 3.0] for call in calls))

    def test_view_opens_each_group_after_its_export(self):
        self.resolve_calls = []
        order, view_calls = [], []

        def fake_export(**kwargs):
            order.append(("export", kwargs["output_folder"]))

        def fake_view(cochlea, folder, resolved, scale, sources, **kwargs):
            order.append(("view", folder))
            view_calls.append((folder, resolved, kwargs))

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, fake_export),
            "synapse": (self.mod._run_synapse_export, fake_export),
        }), \
                mock.patch.object(self.mod, "source_types", return_value=SOURCES), \
                mock.patch.object(self.mod, "resolve_reference_positions", self._fake_positions), \
                mock.patch.object(self.mod, "view_crops", fake_view):
            self.mod.run_groups(
                "cochlea_x", {name: self.mod.EXPORT_GROUPS[name] for name in ("sgn", "ihc")},
                scale=[2], output_folder="/tmp/out", axis=0, view=True,
            )

        # every export of a group runs before that group is viewed.
        self.assertEqual(
            order,
            [("export", "/tmp/out/sgn")] * 2 + [("view", "/tmp/out/sgn")]
            + [("export", "/tmp/out/ihc")] * 3 + [("view", "/tmp/out/ihc")],
        )
        self.assertEqual([call[0] for call in view_calls], ["/tmp/out/sgn", "/tmp/out/ihc"])
        self.assertEqual(view_calls[0][1], self._fake_positions("cochlea_x", "SGN_v2", None))
        self.assertEqual(view_calls[0][2]["axis"], 0)
        self.assertEqual(view_calls[0][2]["label"], "sgn")

    def test_view_only_does_not_export(self):
        self.resolve_calls = []
        calls, view_calls = [], []

        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, lambda **kwargs: calls.append(kwargs)),
        }), \
                mock.patch.object(self.mod, "source_types", return_value=SOURCES), \
                mock.patch.object(self.mod, "resolve_reference_positions", self._fake_positions), \
                mock.patch.object(self.mod, "view_crops", lambda *args, **kwargs: view_calls.append(args[1])):
            self.mod.run_groups(
                "cochlea_x", {"sgn": self.mod.EXPORT_GROUPS["sgn"]}, scale=[2], output_folder="/tmp/out",
                axis=0, view_only=True,
            )

        self.assertEqual(calls, [])
        self.assertEqual(view_calls, ["/tmp/out/sgn"])

    def test_group_axis_is_used_for_the_view(self):
        self.resolve_calls = []
        view_calls = []

        groups = {"sgn": {**self.mod.EXPORT_GROUPS["sgn"], "axis": 2, "voxel_size": [0.76, 0.76, 3.0]}}
        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, lambda **kwargs: None),
        }), \
                mock.patch.object(self.mod, "source_types", return_value=SOURCES), \
                mock.patch.object(self.mod, "resolve_reference_positions", self._fake_positions), \
                mock.patch.object(self.mod, "view_crops", lambda *args, **kwargs: view_calls.append(kwargs)):
            self.mod.run_groups("cochlea_x", groups, scale=[2], output_folder="/tmp/out", view=True)

        self.assertEqual(view_calls[0]["axis"], 2)
        self.assertEqual(view_calls[0]["voxel_size"], [0.76, 0.76, 3.0])

    def test_dry_run_does_not_export(self):
        calls = self._run({"ihc": self.mod.EXPORT_GROUPS["ihc"]}, axis=0, dry_run=True)

        self.assertEqual(calls, [])
        self.assertEqual(self.resolve_calls, [("IHC_v4c", (1,))])


class TestViewCrops(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position
        self.tmp_dir = tempfile.TemporaryDirectory()

        # A 2D slice at axis 0: the x axis holds a single pixel.
        self.folder = os.path.join(self.tmp_dir.name, "cochlea_x", "scale4")
        os.makedirs(self.folder)
        self.crop_center = [10.0, 20.0, 30.0]
        tifffile.imwrite(
            os.path.join(self.folder, "PV_crop_0010-0020-0030_axis-0_apex.tif"),
            np.random.rand(6, 5, 1).astype("float32"),
        )
        tifffile.imwrite(
            os.path.join(self.folder, "SGN_v2_crop_0010-0020-0030_axis-0_apex.tif"),
            np.zeros((6, 5, 1), dtype="float32"),
        )
        self.resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                          "label_id": 1, "crop_center": self.crop_center}]

    def tearDown(self):
        self.tmp_dir.cleanup()

    def _view(self, **kwargs):
        self.viewers = []
        fake_napari = mock.MagicMock()
        fake_napari.Viewer.side_effect = lambda: self.viewers.append(mock.MagicMock()) or self.viewers[-1]

        with mock.patch.dict(sys.modules, {"napari": fake_napari}):
            self.mod.view_crops(
                "cochlea_x", self.tmp_dir.name, self.resolved, [4], SOURCES, axis=0, label="sgn", **kwargs
            )
        return fake_napari

    def test_one_viewer_per_crop(self):
        fake_napari = self._view()

        self.assertEqual(len(self.viewers), 1)
        self.assertEqual(fake_napari.run.call_count, 1)
        self.assertIn("apex", self.viewers[0].title)
        self.assertIn("sgn", self.viewers[0].title)
        self.assertIn("scale4", self.viewers[0].title)

    def test_images_and_labels_are_split(self):
        self._view()
        viewer = self.viewers[0]

        self.assertEqual([call.kwargs["name"] for call in viewer.add_image.call_args_list], ["PV"])
        self.assertEqual([call.kwargs["name"] for call in viewer.add_labels.call_args_list], ["SGN_v2"])
        self.assertEqual(viewer.add_labels.call_args.args[0].dtype, np.dtype("uint32"))
        self.assertEqual(viewer.add_image.call_args.kwargs["blending"], "additive")
        self.assertIn(viewer.add_image.call_args.kwargs["colormap"], self.mod.CHANNEL_COLORMAPS)

    def test_labels_are_added_after_the_images(self):
        # napari draws later layers on top, so the segmentations must come last.
        tifffile.imwrite(
            os.path.join(self.folder, "CTBP2_crop_0010-0020-0030_axis-0_apex.tif"),
            np.zeros((6, 5, 1), dtype="float32"),
        )
        self._view()
        viewer = self.viewers[0]

        names = [call.kwargs["name"] for call in viewer.method_calls if call[0] in ("add_image", "add_labels")]
        self.assertEqual(names, ["CTBP2", "PV", "SGN_v2"])

    def test_single_pixel_axis_is_dropped(self):
        self._view(voxel_size=[0.5, 0.5, 2.0])
        viewer = self.viewers[0]

        self.assertEqual(viewer.add_image.call_args.args[0].shape, (6, 5))
        self.assertEqual(viewer.add_labels.call_args.args[0].shape, (6, 5))
        # scale4 with voxel size (x, y, z) = (0.5, 0.5, 2.0): the ZYX scale is (32, 8), x is dropped.
        self.assertEqual(viewer.add_image.call_args.kwargs["scale"], (32.0, 8.0))

    def test_missing_scale_is_reported_without_a_viewer(self):
        self.viewers = []
        fake_napari = mock.MagicMock()
        fake_napari.Viewer.side_effect = lambda: self.viewers.append(mock.MagicMock()) or self.viewers[-1]

        with mock.patch.dict(sys.modules, {"napari": fake_napari}):
            self.mod.view_crops("cochlea_x", self.tmp_dir.name, self.resolved, [5], SOURCES, axis=0)

        self.assertEqual(self.viewers, [])
        self.assertEqual(fake_napari.run.call_count, 0)

    def test_dilation_folder_is_found(self):
        os.makedirs(os.path.join(self.tmp_dir.name, "cochlea_x", "scale4_dilation8"))
        tifffile.imwrite(
            os.path.join(self.tmp_dir.name, "cochlea_x", "scale4_dilation8",
                         "PV_crop_0010-0020-0030_axis-0_apex.tif"),
            np.zeros((6, 5, 1), dtype="float32"),
        )
        self._view()

        self.assertEqual(len(self.viewers), 2)


class TestDescribePass(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position

    def test_channels(self):
        self.assertEqual(
            self.mod.describe_pass("lower_resolution", {"channels": ["Vglut3", "CTBP2"]}),
            "'lower_resolution' for ['Vglut3', 'CTBP2']",
        )

    def test_segmentations_with_the_component_filter(self):
        self.assertEqual(
            self.mod.describe_pass("lower_resolution", {"channels": ["IHC_v11"], "filter_by_components": [1]}),
            "'lower_resolution' for ['IHC_v11'], filtered by components [1]",
        )

    def test_synapses_name_the_reference(self):
        self.assertEqual(
            self.mod.describe_pass("synapse", {"synapse_name": "synapse_v3_ihc_v11", "reference_ihcs": "IHC_v11"}),
            "'synapse' for ['synapse_v3_ihc_v11'], matched to IHC_v11",
        )

    def test_without_names(self):
        self.assertEqual(self.mod.describe_pass("lower_resolution", {}), "'lower_resolution' with its default sources")


class TestExportPrinting(unittest.TestCase):

    def setUp(self):
        import export_by_position
        self.mod = export_by_position
        self.resolved = [{"column": "length_fraction", "label": "apex", "value": 0.15,
                          "label_id": 1, "crop_center": [1.0, 2.0, 3.0]}]

    def test_run_exports_names_every_source(self):
        stdout = io.StringIO()
        with mock.patch.dict(self.mod.EXPORT_DISPATCH, {
            "lower_resolution": (self.mod._run_grid_export, lambda **kwargs: None),
            "synapse": (self.mod._run_synapse_export, lambda **kwargs: None),
        }), redirect_stdout(stdout):
            self.mod.run_exports(
                "cochlea_x", self.resolved, [], scale=[4], output_folder="/tmp/out", axis=0,
                json_info=[
                    {"lower_resolution": {"channels": ["Vglut3", "CTBP2"]}},
                    {"lower_resolution": {"channels": ["IHC_v11"], "filter_by_components": [1]}},
                    {"synapse": {"synapse_name": "synapse_v3_ihc_v11", "reference_ihcs": "IHC_v11"}},
                ],
            )

        output = stdout.getvalue()
        self.assertIn("Exporting 'lower_resolution' for ['Vglut3', 'CTBP2']", output)
        self.assertIn("Exporting 'lower_resolution' for ['IHC_v11'], filtered by components [1]", output)
        self.assertIn("Exporting 'synapse' for ['synapse_v3_ihc_v11'], matched to IHC_v11", output)


if __name__ == "__main__":
    unittest.main()
