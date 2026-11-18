import os
import sys
import unittest
from unittest import mock

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_annotation_fill import dice, synthetic_row  # noqa: E402


class TestRegion(unittest.TestCase):
    """A row of 9 cells. The annotation lacks the cells 4-6, so the stretch between 3 and 7 has 3 positions."""

    def setUp(self):
        self.patch = mock.patch("flamingo_tools.annotation.issues.MIN_CELL_SIZE", 500)
        self.patch.start()
        self.gt, _ = synthetic_row(n_cells=9)
        self.annotation = np.where(np.isin(self.gt, [4, 5, 6]), 0, self.gt).astype("uint32")
        # IHC_v11 misses cell 5; IHC_v9 has all cells.
        self.v11 = np.where(self.gt == 5, 0, self.gt).astype("uint32")
        self.v9 = self.gt.copy()

    def tearDown(self):
        self.patch.stop()

    def propose(self, sources, **kwargs):
        from flamingo_tools.annotation.region import propose_region

        kwargs.setdefault("correct", None)
        return propose_region(self.annotation, self.annotation, (3, 7), sources, 1.0, **kwargs)

    def best_match(self, voxels):
        values, counts = np.unique(self.gt[tuple(voxels.T)], return_counts=True)
        label_id = values[np.argmax(counts)]
        mask = np.zeros(self.gt.shape, dtype=bool)
        mask[tuple(voxels.T)] = True
        return label_id, dice(mask, self.gt == label_id)

    def test_newest_eligible(self):
        proposal = self.propose({"IHC_v11": self.v11, "IHC_v9": self.v9})
        self.assertEqual(len(proposal.cells), 3)
        sources = [cell.comment.split(":")[0] for cell in proposal.cells]
        self.assertEqual(sources, ["IHC_v11", "IHC_v9", "IHC_v11"])
        for cell, expected in zip(proposal.cells, (4, 5, 6)):
            label_id, score = self.best_match(cell.voxels)
            self.assertEqual(label_id, expected)
            self.assertGreater(score, 0.95)
        self.assertEqual(proposal.end_labels, [3, 7])
        self.assertEqual(proposal.removed, [])

    def test_rule_and_row_correction(self):
        # The IHC_v11 cell 4 lacks a tip, but stays typical; the rule 'score' takes the better IHC_v9 cell, and
        # 'newest' keeps IHC_v11.
        self.v11[(self.gt == 4) & (np.indices(self.gt.shape)[0] >= 41)] = 0
        sources = {"IHC_v11": self.v11, "IHC_v9": self.v9}
        self.assertTrue(self.propose(sources).cells[0].comment.startswith("IHC_v11:"))
        self.assertTrue(self.propose(sources, rule="score").cells[0].comment.startswith("IHC_v9:"))
        # The row correction of a model ranks above the model.
        proposal = self.propose(sources, correct=lambda labels, path, reference, voxel_size: labels)
        self.assertTrue(proposal.cells[0].comment.startswith("IHC_v11 row-corrected:"))

    def test_template_and_apply(self):
        from flamingo_tools.annotation.region import apply_region, region_items

        # The base has a fragment in the stretch; without sources, the cells are typical IHC shapes.
        base = self.annotation.copy()
        base[(self.gt == 5) & (np.indices(self.gt.shape)[0] < 30)] = 20
        proposal = self.propose({}, base_source="annotation")
        self.assertTrue(all(cell.comment.startswith("typical IHC shape") for cell in proposal.cells))
        for cell, expected in zip(proposal.cells, (4, 5, 6)):
            center = self.gt == expected
            self.assertLess(np.linalg.norm(cell.voxels.mean(axis=0) - np.argwhere(center).mean(axis=0)), 3)

        from flamingo_tools.annotation.region import propose_region

        proposal = propose_region(base, self.annotation, (3, 7), {"IHC_v9": self.v9}, 1.0, correct=None)
        self.assertEqual(proposal.removed, [20])
        new_ids = iter(range(21, 30))
        box, after, cells = apply_region(base, proposal, new_ids.__next__, 1.0)
        labels = base.copy()
        labels[box] = after
        self.assertFalse((labels == 20).any())
        self.assertEqual([label_id for label_id, _ in cells], [21, 22, 23])
        for label_id, expected in zip((21, 22, 23), (4, 5, 6)):
            self.assertGreater(dice(labels == label_id, self.gt == expected), 0.95)
        np.testing.assert_array_equal(labels == 3, self.gt == 3)
        items = region_items(labels, cells, proposal.end_labels)
        self.assertEqual([item.reasons[0] for item in items], ["region_border"] + ["region"] * 3 + ["region_border"])
        self.assertEqual([item.item_id for item in items], [0, 1, 2, 3, 4])

    def test_end_cell_of_base(self):
        # IHC_v11 has a cell-sized object at the position of cell 6 that lies 5 px toward the end cell 7. The base
        # end cell 7 reaches 3 px into cell 6, so the object is mostly the end cell of the base.
        y = np.indices(self.gt.shape)[1]
        shifted = np.zeros_like(self.gt)
        shifted[:, :-5][self.gt[:, 5:] == 7] = 6
        v11 = np.where(np.isin(self.gt, [4, 5]), self.gt, shifted).astype("uint32")
        sources = {"IHC_v11": v11, "IHC_v9": self.v9}
        # With an end cell as in the annotation, 60% of the object is free, so it wins.
        proposal = self.propose(sources)
        self.assertTrue(proposal.cells[2].comment.startswith("IHC_v11:"))
        base = self.gt.copy()
        base[(self.gt == 6) & (y >= 57)] = 7
        from flamingo_tools.annotation.region import propose_region

        proposal = propose_region(base, self.annotation, (3, 7), sources, 1.0, correct=None)
        self.assertTrue(proposal.cells[2].comment.startswith("IHC_v9:"))
        # The cell lacks the part of cell 6 that the end cell of the base holds.
        mask = np.zeros(self.gt.shape, dtype=bool)
        mask[tuple(proposal.cells[2].voxels.T)] = True
        self.assertGreater(dice(mask, (self.gt == 6) & (base != 7)), 0.95)
        self.assertEqual(proposal.end_labels, [3, 7])

    def test_neighbours(self):
        from flamingo_tools.annotation.region import propose_region

        with self.assertRaises(ValueError):
            propose_region(self.gt, self.gt, (3, 4), {}, 1.0, correct=None)


class TestModelRank(unittest.TestCase):

    def test_order(self):
        from flamingo_tools.annotation.crop_reference import ihc_sources, model_rank
        from flamingo_tools.annotation.region import source_rank

        sources = {name: "segmentation" for name in ("IHC_v4b", "IHC_v11", "IHC_v11b", "IHC_v9", "IHC_v10")}
        sources.update({"IHC_LOWRES-v3": "segmentation", "Vglut3": "image", "IHC_v12": "image"})
        self.assertEqual(ihc_sources(sources), ["IHC_v11b", "IHC_v11", "IHC_v10", "IHC_v9", "IHC_v4b"])
        self.assertGreater(model_rank("IHC_v11b"), model_rank("IHC_v11"))
        self.assertGreater(source_rank("IHC_v11b"), source_rank("IHC_v11 row-corrected"))
        self.assertGreater(source_rank("IHC_v11 row-corrected"), source_rank("IHC_v11"))
        self.assertGreater(source_rank("IHC_v4b"), source_rank("IHC_v11 + IHC_v4b"))
        self.assertGreater(source_rank("IHC_v11 + IHC_v4b"), source_rank("annotation"))
        with self.assertRaises(ValueError):
            model_rank("SGN_v2")


if __name__ == "__main__":
    unittest.main()
