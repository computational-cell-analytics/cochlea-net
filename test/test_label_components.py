import json
import os
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd


class TestRefineIhcRow(unittest.TestCase):

    def setUp(self):
        # 100 IHC positions, 9 µm apart on an arc with a radius of 300 µm. Cells 0-39 are component 1 and
        # cells 50-99 component 2. In the gap, cells 40-46 and 49 are missing, cell 47 is in no component
        # and cell 48 is small. Label 101 is a large object 40 µm outside the arc in component 1, and
        # labels 102-106 a component 200 µm away from the row.
        angles = np.arange(100) * 9.0 / 300.0
        present = [i for i in range(100) if not 40 <= i <= 46 and i != 49]
        rows = []
        for i in present:
            component = 1 if i < 40 else (2 if i >= 50 else 0)
            rows.append((i + 1, 300.0, angles[i], 500 if i == 48 else 20000, component))
        rows.append((101, 340.0, angles[20], 20000, 1))
        rows += [(102 + j, 100.0, angles[60] + 0.03 * j, 20000, 3) for j in range(5)]
        label_ids, radii, phis, sizes, components = map(np.array, zip(*rows))
        self.table = pd.DataFrame({
            "label_id": label_ids,
            "anchor_x": 500 + radii * np.cos(phis),
            "anchor_y": 500 + radii * np.sin(phis),
            "anchor_z": 100.0,
            "n_pixels": sizes,
            "component_labels": components,
        })
        self.expected_row = {i + 1 for i in present if i != 48}

    def test_refine_ihc_row(self):
        from flamingo_tools.postprocessing.label_components import refine_ihc_row

        table, path = refine_ihc_row(self.table.copy(), [1, 2], min_size=10000, max_path_deviation=15.0)
        self.assertEqual(set(table.loc[table["component_labels"] == 1, "label_id"]), self.expected_row)
        self.assertEqual(set(table["component_labels"]), {0, 1})
        self.assertEqual(path.shape[1], 3)

    def test_refine_ihc_row_missing_component(self):
        from flamingo_tools.postprocessing.label_components import refine_ihc_row

        with self.assertRaises(ValueError):
            refine_ihc_row(self.table.copy(), [1, 9], min_size=10000, max_path_deviation=15.0)

    def test_refine_ihc_row_branch(self):
        from flamingo_tools.postprocessing.label_components import refine_ihc_row

        # Component 1 runs along x. The row continues past a 300 µm gap with component 2, which comes back to
        # within 250 µm of the middle of component 1, as on the next turn of the spiral. A path through the
        # nearest gap would skip the part of component 1 past x = 300 µm.
        line = np.stack([np.arange(-500, 1001, 10.0), np.zeros(151), np.zeros(151)], axis=1)
        turn = np.array([1000.0, 300.0, 0.0]) + np.linspace(0, 1, 72)[:, None] * np.array([-700.0, -50.0, 0.0])
        coords = np.concatenate([line, turn])
        table = pd.DataFrame({
            "label_id": np.arange(1, len(coords) + 1),
            "anchor_x": coords[:, 0], "anchor_y": coords[:, 1], "anchor_z": coords[:, 2],
            "n_pixels": 20000,
            "component_labels": [1] * len(line) + [2] * len(turn),
        })
        table, _ = refine_ihc_row(table, [1, 2], min_size=10000, max_path_deviation=15.0)
        self.assertTrue((table["component_labels"] == 1).all())

    def test_refine_row_requires_ihc(self):
        from flamingo_tools.postprocessing.label_components import label_components_single

        with self.assertRaises(ValueError):
            label_components_single("table.tsv", cell_type="sgn", refine_row=True)

    def _label_with_prompt(self, cell_type, answers):
        from flamingo_tools.json_util import load_processing_params
        from flamingo_tools.postprocessing.label_components import label_components_single

        with TemporaryDirectory() as tmp_dir:
            table_path = os.path.join(tmp_dir, "default.tsv")
            path_file = os.path.join(tmp_dir, "path.tsv")
            self.table.to_csv(table_path, sep="\t", index=False)
            with patch("flamingo_tools.postprocessing.label_components._show_components"), \
                    patch("builtins.input", side_effect=answers):
                label_components_single(
                    table_path, cell_type=cell_type, min_size=10000, min_component_length=3, use_napari=True,
                    dataset_name="M_AMD_000126_L", segmentation_channel=f"{cell_type.upper()}_v11", archive=tmp_dir,
                    path_file=path_file,
                )
            path = os.path.join(tmp_dir, f"M_AMD_000126_L_{cell_type.upper()}.json")
            with open(path) as f:
                entry = json.load(f)
            self.assertEqual(os.path.isfile(path_file), cell_type == "ihc")
            return entry, load_processing_params(path, "label_components")[0]

    def test_archive_ihc_row_from_prompt(self):
        entry, params = self._label_with_prompt("ihc", ["1 2", ""])
        self.assertEqual(entry["component_list"], [1])
        self.assertEqual(params["component_list_path"], [1, 2])
        self.assertTrue(params["refine_row"])
        self.assertEqual(params["min_size"], 10000)

    def test_archive_sgn_order_from_prompt(self):
        entry, _ = self._label_with_prompt("sgn", ["x", "3 1"])
        self.assertEqual(entry["component_list"], [3, 1])
        self.assertNotIn("component_list_path", entry["label_components"])

    def test_replay_archive_keeps_both_component_lists(self):
        from flamingo_tools.json_util import load_processing_params
        from flamingo_tools.postprocessing.label_components import label_components_json_wrapper

        entry = {
            "dataset_name": "M_AMD_000126_L", "segmentation_channel": "IHC_v9", "cell_type": "ihc",
            "component_list": [1],
            "label_components": {"component_list_path": [1, 2], "min_component_length": 3, "min_size": 10000},
            "tonotopic_mapping": {},
        }
        with TemporaryDirectory() as tmp_dir:
            table_dir = os.path.join(tmp_dir, "M_AMD_000126_L", "tables", "IHC_v9")
            os.makedirs(table_dir)
            self.table.to_csv(os.path.join(table_dir, "default.tsv"), sep="\t", index=False)
            path = os.path.join(tmp_dir, "M_AMD_000126_L_IHC.json")
            with open(path, "w") as f:
                json.dump(entry, f)
            before = load_processing_params(path, "label_components")[0]
            label_components_json_wrapper(path, out_path=os.path.join(tmp_dir, "out"), mobie_dir=tmp_dir, archive=path)
            after = load_processing_params(path, "label_components")[0]
            self.assertEqual(after, {**before, "max_edge_distance": 30})
            self.assertEqual(load_processing_params(path, "tonotopic_mapping")[0]["component_list"], [1])


if __name__ == "__main__":
    unittest.main()
