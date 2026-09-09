import json
import os
import unittest
from tempfile import TemporaryDirectory


class TestUpdateJson(unittest.TestCase):
    def test_update_json_creates_file(self):
        from flamingo_tools.json_util import update_json

        with TemporaryDirectory() as tmp_dir:
            # The nested directory does not exist yet.
            output_path = os.path.join(tmp_dir, "nested", "consensus_SGN.json")
            update_json({"AMD": {"precision": 0.9}}, output_path)

            with open(output_path) as f:
                data = json.load(f)
            self.assertEqual(data, {"AMD": {"precision": 0.9}})

    def test_update_json_merges_keys(self):
        from flamingo_tools.json_util import update_json

        with TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "consensus_SGN.json")
            update_json({"AMD": {"precision": 0.9}, "EK": {"precision": 0.8}}, output_path)
            update_json({"EK": {"precision": 0.85}}, output_path)

            with open(output_path) as f:
                data = json.load(f)

            # The untouched key survives and the given key is replaced.
            self.assertEqual(data["AMD"], {"precision": 0.9})
            self.assertEqual(data["EK"], {"precision": 0.85})


class TestLoadProcessingParams(unittest.TestCase):
    """Cover the reader of the shared processing parameter files."""

    ENTRY = {
        "dataset_name": "M_AMD_000126_L",
        "segmentation_channel": "IHC_v9",
        "cell_type": "ihc",
        "image_channel": ["Vglut3"],
        "component_list": [1],
        "label_components": {"component_list_path": [1, 2, 6], "min_component_length": 10},
        "tonotopic_mapping": {},
        "object_measures": {"use_bg_mask": "yes"},
    }

    def _write(self, tmp_dir, entry, name="M_AMD_000126_L_IHC.json"):
        path = os.path.join(tmp_dir, name)
        with open(path, "w") as f:
            json.dump(entry, f)
        return path

    def test_common_keys_reach_every_step(self):
        from flamingo_tools.json_util import load_processing_params

        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, self.ENTRY)
            for step in ("label_components", "tonotopic_mapping", "object_measures"):
                params = load_processing_params(path, step)
                self.assertEqual(len(params), 1)
                self.assertEqual(params[0]["dataset_name"], "M_AMD_000126_L")
                self.assertEqual(params[0]["segmentation_channel"], "IHC_v9")

    def test_section_keys_stay_in_their_step(self):
        from flamingo_tools.json_util import load_processing_params

        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, self.ENTRY)
            components = load_processing_params(path, "label_components")[0]
            tonotopy = load_processing_params(path, "tonotopic_mapping")[0]
            measures = load_processing_params(path, "object_measures")[0]

            self.assertEqual(components["min_component_length"], 10)
            self.assertNotIn("min_component_length", tonotopy)
            self.assertNotIn("min_component_length", measures)
            self.assertEqual(measures["use_bg_mask"], "yes")
            self.assertNotIn("use_bg_mask", components)

    def test_component_list_path_overrides_the_common_list(self):
        from flamingo_tools.json_util import load_processing_params

        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, self.ENTRY)
            # label_components uses the components of the central path.
            self.assertEqual(
                load_processing_params(path, "label_components")[0]["component_list"], [1, 2, 6])
            # The later steps select the labeled component.
            for step in ("tonotopic_mapping", "object_measures"):
                self.assertEqual(load_processing_params(path, step)[0]["component_list"], [1])

    def test_empty_section_yields_the_common_keys(self):
        from flamingo_tools.json_util import load_processing_params

        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, self.ENTRY)
            params = load_processing_params(path, "tonotopic_mapping")[0]
            self.assertEqual(set(params), {"dataset_name", "segmentation_channel", "cell_type",
                                           "image_channel", "component_list"})

    def test_flat_file_is_passed_through(self):
        """A block extraction file has no sections. doc/analysis.md feeds those to
        flamingo_tools.object_measures, so they must keep working unvalidated."""
        from flamingo_tools.json_util import load_processing_params

        flat = {
            "dataset_name": "M_LR_000153_L",
            "image_channel": ["PV", "CR", "GFP", "SGN_v2"],
            "segmentation_channel": "SGN_v2",
            "cell_type": "sgn",
            "n_blocks": 6,
            "roi_halo": [256, 256, 64],
            "component_list": [1, 2, 3],
            "crop_centers": [[1, 2, 3]],
        }
        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, flat, "M_LR_000153_L.json")
            for step in ("label_components", "object_measures", "tonotopic_mapping"):
                params = load_processing_params(path, step)
                self.assertEqual(params, [flat])

    def test_voxel_size_is_common(self):
        """scripts/analysis/create_main_table.py reads voxel_size with a flat update."""
        from flamingo_tools.json_util import COMMON_KEYS, STEP_KEYS

        self.assertIn("voxel_size", COMMON_KEYS)
        for keys in STEP_KEYS.values():
            self.assertNotIn("voxel_size", keys)

    def test_absent_section_raises(self):
        from flamingo_tools.json_util import load_processing_params

        entry = {key: value for key, value in self.ENTRY.items() if key != "object_measures"}
        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, entry)
            with self.assertRaises(ValueError):
                load_processing_params(path, "object_measures")

    def test_absent_section_of_one_entry_is_skipped(self):
        from flamingo_tools.json_util import load_processing_params

        second = {key: value for key, value in self.ENTRY.items() if key != "object_measures"}
        second["dataset_name"] = "M_AMD_000127_L"
        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, [self.ENTRY, second])
            params = load_processing_params(path, "object_measures")
            self.assertEqual([p["dataset_name"] for p in params], ["M_AMD_000126_L"])

    def test_unknown_keys_raise(self):
        from flamingo_tools.json_util import load_processing_params

        cases = [
            dict(self.ENTRY, min_component_lenght=10),          # misspelled, at the top level
            dict(self.ENTRY, table_path="/somewhere/default.tsv"),   # supplied by the wrapper
            dict(self.ENTRY, sgn_density={}),                    # not a processing step
        ]
        section = dict(self.ENTRY)
        section["label_components"] = {"use_bg_mask": "yes"}     # key of another step
        cases.append(section)
        with TemporaryDirectory() as tmp_dir:
            for index, entry in enumerate(cases):
                path = self._write(tmp_dir, entry, f"case{index}.json")
                with self.assertRaises(ValueError):
                    load_processing_params(path, "label_components")

    def test_required_keys_are_enforced(self):
        from flamingo_tools.json_util import load_processing_params

        entry = {key: value for key, value in self.ENTRY.items() if key != "segmentation_channel"}
        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, entry)
            with self.assertRaises(ValueError):
                load_processing_params(path, "label_components")

    def test_unknown_step_raises(self):
        from flamingo_tools.json_util import load_processing_params

        with TemporaryDirectory() as tmp_dir:
            path = self._write(tmp_dir, self.ENTRY)
            with self.assertRaises(ValueError):
                load_processing_params(path, "sgn_density")

    def test_step_keys_match_the_function_signatures(self):
        """Guard STEP_KEYS against drifting away from the functions it feeds."""
        import inspect

        from flamingo_tools.json_util import COMMON_KEYS, STEP_KEYS, STEP_RENAMES
        from flamingo_tools.measurements import object_measures_single
        from flamingo_tools.postprocessing.cochlea_mapping import tonotopic_mapping_single
        from flamingo_tools.postprocessing.label_components import label_components_single

        functions = {
            "label_components": label_components_single,
            "tonotopic_mapping": tonotopic_mapping_single,
            "object_measures": object_measures_single,
        }
        self.assertEqual(set(functions), set(STEP_KEYS))
        for step, function in functions.items():
            accepted = set(inspect.signature(function).parameters)
            renames = STEP_RENAMES.get(step, {})
            for key in STEP_KEYS[step]:
                self.assertIn(renames.get(key, key), accepted,
                              f"{step} section key {key!r} is not a parameter of {function.__name__}")
        # No section may claim a common key, which would make the source of a value ambiguous.
        for step, keys in STEP_KEYS.items():
            overlap = keys & COMMON_KEYS
            self.assertFalse(overlap, f"{step} section also declares the common key(s) {overlap}")


if __name__ == "__main__":
    unittest.main()
