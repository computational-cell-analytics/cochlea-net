import os
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_annotation_fill import synthetic_row  # noqa: E402


@mock.patch("flamingo_tools.annotation.issues.MIN_CELL_SIZE", 500)
class TestIssues(unittest.TestCase):

    def setUp(self):
        self.gt, _ = synthetic_row(n_cells=9)

    def reasons(self, items):
        return {item.label_id: item.reasons for item in items if item.kind == "cell"}

    def test_generic(self):
        from flamingo_tools.annotation.issues import (
            cell_voxels, indent_findings, piece_findings, reference_findings, size_findings,
        )

        labels = self.gt.copy()
        labels[0:4, 0:4, 0:4] = 3  # a second piece of cell 3
        labels[24:37, 44:49, 18:23] = 0  # a cavity in cell 5, as at a nucleus with low signal
        labels[labels == 7] = 0
        coords = cell_voxels(labels)
        self.assertEqual([f.label_id for f in piece_findings(labels, coords)], [3])
        indents = indent_findings(labels, coords, voxel_size=1.0, radius=2.0)
        self.assertIn(5, [f.label_id for f in indents])
        self.assertGreater(len(indents[[f.label_id for f in indents].index(5)].proposal), 0)

        small = self.gt.copy()
        small[small == 3] = 2  # a merge of two cells
        self.assertEqual([f.label_id for f in size_findings(small, cell_voxels(small))], [2])

        # The reference has cell 7, and cell 8 reaches 6 px further along z than in the annotation.
        reference = self.gt.copy()
        labels[(labels == 8) & (np.indices(labels.shape)[0] > 38)] = 0
        findings = reference_findings(labels, cell_voxels(labels), reference)
        kinds = {(f.kind, f.reason, f.label_id) for f in findings}
        self.assertIn(("missing", "missing_s3", 0), kinds)
        self.assertIn(("cell", "extent_s3", 8), kinds)
        missing = [f for f in findings if f.kind == "missing"][0]
        self.assertEqual(len(missing.proposal), (self.gt == 7).sum())

    def test_ihc_row(self):
        from flamingo_tools.annotation.issues import detect

        labels = self.gt.copy()
        labels[labels == 5] = 0  # a missing cell in the row
        labels[(labels == 7) & (np.indices(labels.shape)[0] > 36)] = 0  # a short cell
        items, row = detect(labels, 1.0, "ihc")
        self.assertIsNotNone(row)
        missing = [item for item in items if item.kind == "missing"]
        self.assertEqual(len(missing), 1)
        self.assertEqual(missing[0].reasons, ["gap"])
        self.assertLess(abs(missing[0].anchor[1] - 46), 4)
        self.assertGreater(len(missing[0].proposal), 0)
        self.assertIn("short", self.reasons(items)[7])
        # The items are ordered along the row.
        orders = [item.order for item in items]
        self.assertEqual(orders, sorted(orders))

    def test_table_and_large_ids(self):
        from flamingo_tools.annotation.ids import compact
        from flamingo_tools.annotation.issues import detect

        labels = self.gt.astype("uint64")
        labels[labels > 0] += 2_997_629_970
        labels = labels.astype("uint32")
        compact_labels, original = compact(labels)
        table = pd.DataFrame([
            {"crop": "c", "label_id": "2997629973", "z": 30, "y": 28, "x": 20, "reason": "cut_tip", "comment": "a"},
            {"crop": "c", "label_id": None, "z": 30, "y": 5, "x": 20, "reason": "missing", "comment": "b"},
        ])
        items, _ = detect(compact_labels, 1.0, "sgn", issue_rows=table, original=original)
        cells = self.reasons(items)
        self.assertEqual(cells, {3: ["table_cut_tip"]})
        missing = [item for item in items if item.kind == "missing"]
        self.assertEqual(missing[0].reasons, ["table_missing"])
        self.assertEqual(missing[0].comment, "b")
        # Without an IHC row, the proposal of a missing cell is a sphere of the median cell size.
        self.assertGreater(len(missing[0].proposal), 0)

        table.loc[0, "label_id"] = "5"
        with self.assertRaises(ValueError):
            detect(compact_labels, 1.0, "sgn", issue_rows=table, original=original)

    def test_comparison_annotation(self):
        from flamingo_tools.annotation.issues import cell_voxels, reference_findings, table_findings

        # The base misses cell 5; the annotation (comparison) has it with the id 5.
        base = np.where(self.gt == 5, 0, self.gt).astype("uint32")
        findings = reference_findings(base, cell_voxels(base), self.gt, tag="annotation")
        self.assertIn("missing_annotation", [f.reason for f in findings])
        table = pd.DataFrame([{"crop": "c", "label_id": "5", "z": None, "y": None, "x": None,
                               "reason": "hole", "comment": ""}])
        found = table_findings(table, base, cell_voxels(base), source_labels=self.gt)
        self.assertEqual([(f.kind, f.reason) for f in found], [("missing", "table_hole")])
        self.assertEqual(len(found[0].proposal), (self.gt == 5).sum())
        # Without the annotation, a label row needs a position.
        table.loc[0, ["z", "y", "x"]] = [30, 37, 20]
        found = table_findings(table, base, cell_voxels(base))
        self.assertEqual([(f.kind, f.label_id) for f in found], [("cell", 4)])

    def test_redetect_subset(self):
        from flamingo_tools.annotation.issues import detect

        labels = self.gt.copy()
        labels[0:4, 0:4, 0:4] = 3
        labels[0:4, 0:4, 30:34] = 6
        items, _ = detect(labels, 1.0, "sgn", label_ids=[6])
        self.assertEqual(self.reasons(items), {6: ["pieces"]})


if __name__ == "__main__":
    unittest.main()
