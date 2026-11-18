import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from test_annotation_fill import dice, synthetic_row  # noqa: E402


class TestController(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.annotation.controller import Settings
        from flamingo_tools.annotation.issues import Item

        self.tmp = tempfile.TemporaryDirectory()
        self.gt, self.image = synthetic_row()
        self.labels = self.gt.copy()
        # Cell 4 has a cut tip, cell 6 is correct.
        self.labels[(self.gt == 4) & (np.indices(self.gt.shape)[0] >= 38)] = 0
        self.settings = Settings(voxel_size=1.0, spacing=4.0, cap=3.0, inset=1.0, margin=5.0)
        self.items = [Item(0, "cell", (30, 37, 20), 4, ["short"], order=0),
                      Item(1, "cell", (30, 55, 20), 6, ["size"], order=1)]

    def tearDown(self):
        self.tmp.cleanup()

    def controller(self, items=None, redetect=None, propose=None):
        from flamingo_tools.annotation.controller import Controller
        from flamingo_tools.annotation.session import Session

        session = Session.create(os.path.join(self.tmp.name, "session"), "sha1")
        return Controller.create(self.image, self.labels, items or self.items, session, self.settings,
                                 propose=propose, redetect=redetect)

    def gt_edit(self, controller):
        """The prefill with the annotation painted on the sparse slices."""
        visit = controller.view()
        edit = controller.prefill()
        local = visit.local_slices
        np.moveaxis(edit, visit.axis, 0)[local] = np.moveaxis(self.gt[visit.roi], visit.axis, 0)[local]
        return edit

    def test_loop_rule(self):
        redetect = mock.Mock(return_value=[])
        controller = self.controller(redetect=redetect)
        visit = controller.view()
        self.assertEqual(visit.label_id, 4)
        # The caps reach past the cut end, so the painted slices recover the tip.
        self.assertTrue(controller.next(self.gt_edit(controller)))
        self.assertEqual(controller.queue, [1, 0])
        self.assertGreater(dice(self.labels == 4, self.gt == 4), 0.95)

        # A visit without a change (also paint and undo) marks the item as done.
        self.assertEqual(controller.view().label_id, 6)
        self.assertFalse(controller.next(controller.prefill()))
        self.assertEqual(controller.items[1].status, "done")

        self.assertEqual(controller.view().label_id, 4)
        self.assertFalse(controller.next(controller.prefill()))
        self.assertIsNone(controller.view())
        self.assertTrue(controller.finished)
        # Cell 4 changed before it was done, so the final check has nothing to look at.
        redetect.assert_not_called()

    def test_accept_and_neighbour(self):
        from scipy import ndimage

        redetect = mock.Mock(return_value=[])
        controller = self.controller(redetect=redetect)
        visit = controller.view()
        # The cells 3 and 5 touch cell 4, so they can change; no other cell is in the region.
        self.assertEqual(visit.focus, [3, 4, 5])
        edit = self.gt_edit(controller)
        # Cell 4 takes the voxels of cell 5 on a slice, and cell 3 grows into the background on the slice.
        z = visit.local_slices[len(visit.local_slices) // 2]
        region = edit[z]
        region[region == 5] = 4
        region[ndimage.binary_dilation(region == 3, iterations=2) & (region == 0)] = 3
        before = self.labels.copy()
        self.assertTrue(controller.accept(edit))
        self.assertEqual(controller.items[0].status, "done")
        self.assertTrue(((before == 3) != (self.labels == 3)).any())
        # Cell 5 is cut in two, so the shape detectors flag it; cell 3 keeps a normal shape and has no item.
        neighbour = [item for item in controller.items.values() if item.label_id == 5]
        self.assertEqual(len(neighbour), 1)
        self.assertEqual(neighbour[0].reasons, ["neighbour_pieces"])
        self.assertEqual(controller.queue, [1, neighbour[0].item_id])
        self.assertFalse(any(item.label_id == 3 for item in controller.items.values()))

        controller.next(controller.prefill())
        controller.next(controller.prefill())
        self.assertIsNone(controller.view())
        # Cell 3 changed and has no item, so the final check looks at it.
        redetect.assert_called_once()
        self.assertEqual(redetect.call_args[0][1], [3])

    def test_locked_cells(self):
        from flamingo_tools.annotation.controller import fill_visit

        # Only cell 4 is editable: paint of cell 4 on cell 5 and paint with the locked label 5 change nothing.
        roi = (slice(10, 50), slice(24, 51), slice(5, 35))
        current = self.labels[roi].copy()
        slices = [16, 20, 24]
        edit = np.zeros_like(current)
        edit[slices] = np.where(current[slices] == 4, 4, 0)
        edit[20][current[20] == 5] = 4
        edit[20][(current[20] == 0) & (np.arange(current.shape[2]) < 5)] = 5
        result, splits = fill_visit(current, edit, 0, slices, [4], self.settings)
        np.testing.assert_array_equal(result, current)
        self.assertEqual(splits, {})

    def test_split_piece(self):
        # A second piece of the target that is a cell on its own gets a new label.
        controller = self.controller()
        visit = controller.view()
        edit = self.gt_edit(controller)
        local = visit.local_slices
        # A block in the background, off the cells and the region border (the cells end at x = 24).
        block = np.zeros(edit.shape[1:], dtype=bool)
        block[8:16, 26 - visit.roi[2].start:29 - visit.roi[2].start] = True
        self.assertFalse(self.labels[visit.roi][local[1]][block].any())
        edit[local[1]][block] = 4
        edit[local[2]][block] = 4
        self.settings.split_size = 20
        self.assertTrue(controller.next(edit))
        split = [item for item in controller.items.values() if item.reasons == ["split"]]
        self.assertEqual(len(split), 1)
        self.assertGreaterEqual(split[0].label_id, 8)
        self.assertTrue(any("new label" in message for message in controller.messages))

    def test_missing_cell(self):
        from flamingo_tools.annotation.issues import Item

        labels = self.gt.copy()
        labels[labels == 4] = 0
        self.labels = labels
        proposal = np.argwhere(self.gt == 4)
        controller = self.controller([Item(0, "missing", (30, 37, 20), 0, ["gap"], proposal=proposal)])
        visit = controller.view()
        self.assertEqual(visit.label_id, 8)
        self.assertTrue(controller.next(controller.prefill()))
        self.assertGreater(dice(self.labels == 8, self.gt == 4), 0.9)
        self.assertIsNone(controller.items[0].proposal)
        # The second visit shows the new cell without the proposal.
        self.assertEqual(controller.view().label_id, 8)
        self.assertFalse(controller.next(controller.prefill()))

    def test_reject_new_cell_undo(self):
        propose = mock.Mock(return_value=None)
        controller = self.controller(propose=propose)
        controller.next(self.gt_edit(controller))
        changed = self.labels.copy()

        self.assertEqual(controller.undo_last(), "next")
        np.testing.assert_array_equal(self.labels != changed, (self.labels == 0) & (changed == 4))
        self.assertEqual(controller.queue[0], 0)

        item = controller.new_cell((10, 20, 30))
        propose.assert_called_once_with((10, 20, 30))
        self.assertEqual(controller.queue[0], item.item_id)
        visit = controller.view()
        self.assertEqual(visit.item.item_id, item.item_id)
        before = self.labels.copy()
        controller.reject()
        self.assertEqual(controller.items[item.item_id].status, "rejected")
        np.testing.assert_array_equal(self.labels, before)

        controller.defer()
        self.assertEqual(controller.queue, [1, 0])

    def test_long_axis(self):
        from flamingo_tools.annotation.controller import Settings
        from flamingo_tools.annotation.issues import Item

        # The cells lie in the xy plane with the long axis along x; cell 4 has a tip cut along x.
        self.gt, self.image = synthetic_row("x")
        self.labels = self.gt.copy()
        self.labels[(self.gt == 4) & (np.indices(self.gt.shape)[2] > 28)] = 0
        self.settings = Settings(voxel_size=1.0, slice_axis="long", n_slices=10, cap=3.0, inset=1.0)
        controller = self.controller([Item(0, "cell", (30, 37, 20), 4, ["short"])])
        visit = controller.view()
        self.assertEqual(visit.axis, 2)
        self.assertIn(len(visit.slices), (11, 12))
        # The edit is the labels on the slices along x, and zero elsewhere.
        prefill = controller.prefill()
        on_slices = np.isin(np.arange(prefill.shape[2]), visit.local_slices)
        self.assertFalse(prefill[:, :, ~on_slices].any())
        self.assertTrue(controller.next(self.gt_edit(controller)))
        self.assertGreater(dice(self.labels == 4, self.gt == 4), 0.95)
        with self.assertRaises(ValueError):
            self.settings = Settings(slice_axis="oblique")
            self.controller()

    def test_delete_and_resume(self):
        from flamingo_tools.annotation.controller import Controller
        from flamingo_tools.annotation.session import Session

        input_labels = self.labels.copy()
        controller = self.controller()
        controller.next(self.gt_edit(controller))
        controller.view()
        controller.delete_cell()
        self.assertFalse((self.labels == 6).any())

        session = Session.open(os.path.join(self.tmp.name, "session"), "sha1")
        resumed = Controller.resume(self.image, input_labels, session, self.settings)
        np.testing.assert_array_equal(input_labels, self.labels)
        self.assertEqual(resumed.queue, controller.queue)
        self.assertEqual(resumed.items[1].status, "done")

    def test_undo_visits(self):
        from flamingo_tools.annotation.controller import Controller
        from flamingo_tools.annotation.session import Session

        input_labels = self.labels.copy()
        controller = self.controller()
        edit = self.gt_edit(controller)
        self.assertTrue(controller.accept(edit))
        changed = self.labels.copy()
        # Cell 6 is accepted without a change: no patch, but an action of the loop.
        self.assertEqual(controller.view().label_id, 6)
        self.assertFalse(controller.accept(controller.prefill()))
        self.assertIsNone(controller.view())

        # The first undo returns to cell 6, the last visit, and keeps the change of cell 4.
        self.assertEqual(controller.undo_last(), "accept")
        self.assertEqual(controller.visit.label_id, 6)
        self.assertEqual(controller.items[1].status, "pending")
        self.assertFalse(controller.finished)
        np.testing.assert_array_equal(self.labels, changed)
        # The second undo reverts cell 4 and restores the loop before its visit: cell 6 is pending again.
        self.assertEqual(controller.undo_last(), "accept")
        self.assertEqual(controller.visit.label_id, 4)
        self.assertEqual(controller.queue, [0, 1])
        self.assertEqual([item.status for item in controller.items.values()], ["pending", "pending"])
        np.testing.assert_array_equal(self.labels, input_labels)
        # The visit shows the edit of the user again, once.
        np.testing.assert_array_equal(controller.prefill(), edit)
        self.assertTrue((controller.prefill() != edit).any())
        self.assertIsNone(controller.undo_last())

        # The history is part of the session.
        controller.accept(edit)
        session = Session.open(os.path.join(self.tmp.name, "session"), "sha1")
        resumed = Controller.resume(self.image, input_labels.copy(), session, self.settings)
        self.assertEqual(len(resumed.history), 1)
        self.assertEqual(resumed.undo_last(), "accept")
        self.assertEqual(resumed.visit.label_id, 4)

    def test_undo_reject_and_proposal(self):
        from flamingo_tools.annotation.issues import Item

        labels = self.gt.copy()
        labels[labels == 4] = 0
        self.labels = labels
        proposal = np.argwhere(self.gt == 4)
        controller = self.controller([Item(0, "missing", (30, 37, 20), 0, ["gap"], proposal=proposal)])
        label_id = controller.view().label_id
        controller.reject()
        self.assertIsNone(controller.items[0].proposal)
        self.assertEqual(controller.undo_last(), "reject")
        # The proposal of the item is open again.
        np.testing.assert_array_equal(controller.items[0].proposal, proposal)
        self.assertEqual(controller.visit.label_id, label_id)
        self.assertEqual(len(controller.visit.proposal), len(proposal))

    def test_undo_region_patch(self):
        # A commit of the region proposal is not an action of the loop: undo leaves it.
        controller = self.controller()
        after = self.labels.copy()
        after[self.labels == 6] = 0
        controller.session.commit((0, 0, 0), self.labels.copy(), after, {"item": -1, "action": "region"})
        self.labels[...] = after
        self.assertIsNone(controller.undo_last())
        np.testing.assert_array_equal(self.labels, after)

    def test_focus_gap_and_unlock(self):
        from flamingo_tools.annotation.controller import focus_ids

        labels = np.zeros((10, 10, 30), dtype="uint32")
        labels[2:8, 2:8, 2:10] = 1
        labels[2:8, 2:8, 12:20] = 2  # 2 voxels from cell 1
        labels[2:8, 2:8, 24:28] = 3  # 14 voxels from cell 1
        self.assertEqual(focus_ids(labels, labels == 1, 1), [1])
        self.assertEqual(focus_ids(labels, labels == 1, 1, gap=2), [1, 2])

        # With a large minimal contact, only the target is in the focus; unlock adds a cell of the region.
        self.settings.min_contact = 10 ** 6
        controller = self.controller()
        visit = controller.view()
        self.assertEqual(visit.focus, [4])
        self.assertTrue(controller.unlock(5))
        self.assertFalse(controller.unlock(5))
        self.assertEqual(visit.editable, [4, 5])
        self.assertFalse((controller.locked() & (self.labels[visit.roi] == 5)).any())
        with self.assertRaises(ValueError):
            controller.unlock(0)
        # Cell 4 takes the voxels of cell 5 on a slice.
        edit = controller.prefill()
        z = visit.local_slices[len(visit.local_slices) // 2]
        on_five = np.moveaxis(self.labels[visit.roi], visit.axis, 0)[z] == 5
        self.assertTrue(on_five.any())
        np.moveaxis(edit, visit.axis, 0)[z][on_five] = 4
        before = (self.labels == 5).sum()
        controller.next(edit)
        self.assertLess((self.labels == 5).sum(), before)

    def test_fragment_slices(self):
        # A fragment on the upper end of cell 4 reaches 6 slices beyond it (z 10-19, cell 4 from z 16).
        self.labels[10:20, 35:40, 18:23] = 8
        self.settings.fragment_size = 0.0
        visit = self.controller().view()
        self.assertIn(8, visit.focus)
        self.assertGreater(min(visit.slices), 10)

        # The synthetic cells are below issues.MIN_CELL_SIZE, so the median size comes from the mock.
        self.settings.fragment_size = 0.5
        with mock.patch("flamingo_tools.annotation.controller.median_cell_size", return_value=1325.0):
            visit = self.controller().view()
        self.assertIn(8, visit.focus)
        self.assertLessEqual(min(visit.slices), 10)
        # The neighbours 3 and 5 reach z 44, past the cut end of cell 4 (z 37) and its cap: they are cells, not
        # fragments, so the slices stay at the cap.
        self.assertLessEqual(max(visit.slices), 37 + 3)
        self.assertIn(3, visit.focus)

        # A fragment out of the focus: unlock adds the slices that cover it.
        self.settings.min_contact = 10 ** 6
        with mock.patch("flamingo_tools.annotation.controller.median_cell_size", return_value=1325.0):
            controller = self.controller()
            visit = controller.view()
            self.assertEqual(visit.focus, [4])
            self.assertGreater(min(visit.slices), 10)
            high = max(visit.slices)
            self.assertTrue(controller.unlock(8))
            self.assertLessEqual(min(visit.slices), 10)
            self.assertEqual(max(visit.slices), high)
            # A cell is no fragment: the slices stay.
            slices = list(visit.slices)
            self.assertTrue(controller.unlock(3))
            self.assertEqual(visit.slices, slices)


class TestPrepareCorrection(unittest.TestCase):

    def setUp(self):
        import tifffile

        self.tmp = tempfile.TemporaryDirectory()
        gt, image = synthetic_row(n_cells=9)
        # The S3 segmentation misses cell 5 and has other ids than the annotation.
        self.s3 = np.where((gt > 0) & (gt != 5), gt + 100, 0).astype("uint32")
        self.annotation = gt
        self.image_path = os.path.join(self.tmp.name, "crop_Vglut3.tif")
        self.annotation_path = os.path.join(self.tmp.name, "crop_Vglut3_annotations.tif")
        tifffile.imwrite(self.image_path, image)
        tifffile.imwrite(self.annotation_path, gt)
        self.output = os.path.join(self.tmp.name, "out")
        self.patches = [mock.patch("flamingo_tools.annotation.issues.MIN_CELL_SIZE", 500)]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in self.patches:
            patch.stop()
        self.tmp.cleanup()

    def prepare(self, s3=True, **kwargs):
        from flamingo_tools.annotation.cli import prepare_correction

        result = (self.s3, "IHC_v11", None) if s3 else (None, None, None)
        with mock.patch("flamingo_tools.annotation.cli.load_reference", return_value=result):
            return prepare_correction(self.image_path, self.output, **kwargs)

    def test_s3_base(self):
        import pandas as pd

        table = os.path.join(self.tmp.name, "issues.tsv")
        pd.DataFrame([{"crop": "crop_Vglut3", "label_id": "3", "z": None, "y": None, "x": None,
                       "reason": "cut_tip", "comment": ""}]).to_csv(table, sep="\t", index=False)
        correction = self.prepare(annotation_path=self.annotation_path, issues=table)
        controller = correction.controller
        self.assertEqual(correction.comparison_name, "annotation (comparison)")
        self.assertEqual(correction.output_path, os.path.join(self.output, "crop_Vglut3_annotations.tif"))
        # The annotated cell 5 has no S3 object, and the table label 3 of the annotation is S3 label 103.
        reasons = {r: item for item in controller.items.values() for r in item.reasons}
        self.assertIn("missing_annotation", reasons)
        self.assertEqual(int(correction.original[reasons["table_cut_tip"].label_id]), 103)
        self.assertEqual(controller.session.state["base"], "s3:IHC_v11")

    def test_edit_annotation(self):
        correction = self.prepare(annotation_path=self.annotation_path, edit_annotation=True)
        self.assertEqual(correction.comparison_name, "IHC_v11 (comparison)")
        np.testing.assert_array_equal(correction.original[correction.controller.labels], self.annotation)
        with self.assertRaises(ValueError):
            self.prepare(s3=False, restart=True)
        with self.assertRaises(ValueError):
            self.prepare(edit_annotation=True, restart=True)

    def test_region(self):
        import tifffile
        from flamingo_tools.annotation.ids import IdMap

        # The annotation lacks cell 5; the region between 4 and 6 gets it from the S3 source.
        annotation = np.where(self.annotation == 5, 0, self.annotation).astype("uint32")
        path = os.path.join(self.tmp.name, "gap.tif")
        tifffile.imwrite(path, annotation)
        sources = {"IHC_v11": np.where(self.annotation > 0, self.annotation + 100, 0).astype("uint32")}
        with mock.patch("flamingo_tools.annotation.cli.load_region_sources", return_value=sources):
            correction = self.prepare(annotation_path=path, edit_annotation=True, regions=[(4, 6)])
        controller = correction.controller
        items = sorted(controller.items.values(), key=lambda item: item.item_id)
        self.assertEqual([item.reasons for item in items], [["region_border"], ["region"], ["region_border"]])
        self.assertIn("IHC_v11", items[1].comment)
        id_map = IdMap(correction.original)
        self.assertEqual([id_map.file_id(item.label_id) for item in items], [4, 10, 6])
        self.assertGreater(dice(controller.labels == items[1].label_id, self.annotation == 5), 0.9)
        self.assertEqual(controller.session.state["patches"][0]["action"], "region")
        self.assertEqual(controller.next_label, items[1].label_id + 1)
        # A resumed region session has the row context, so a new cell gets a typical IHC (28 px along z), not a
        # sphere of the median cell size.
        resumed = self.prepare(annotation_path=path, edit_annotation=True, regions=[(4, 6)]).controller
        self.assertEqual(len(resumed.items), 3)
        proposal = resumed.propose((30, 50, 35))
        self.assertGreater(np.ptp(proposal[:, 0]), 20)

    def test_resume_restart(self):
        controller = self.prepare(annotation_path=self.annotation_path).controller
        controller.view()
        controller.defer()
        queue = list(controller.queue)
        self.assertEqual(self.prepare(annotation_path=self.annotation_path).controller.queue, queue)
        restarted = self.prepare(annotation_path=self.annotation_path, restart=True).controller
        self.assertEqual(sorted(restarted.queue), sorted(queue))
        self.assertNotEqual(restarted.queue, queue)


class TestIdMap(unittest.TestCase):

    def test_round_trip(self):
        from flamingo_tools.annotation.ids import IdMap, compact, restore

        labels = np.zeros((3, 4, 5), dtype="uint32")
        labels[0] = 12
        labels[1] = 3_000_000_000
        compact_labels, original = compact(labels)
        id_map = IdMap(original)
        np.testing.assert_array_equal(id_map.to_file(compact_labels), labels)
        self.assertEqual(id_map.compact_id(12, 3), 1)
        # A new label has the file id of the export.
        compact_labels[2, 0, 0] = 3
        self.assertEqual(id_map.file_id(3), restore(compact_labels, original)[2, 0, 0])
        self.assertEqual(id_map.compact_id(3_000_000_001, 4), 3)
        # Ids that are not in the file and not allocated are unknown.
        self.assertIsNone(id_map.compact_id(3_000_000_001, 3))
        self.assertIsNone(id_map.compact_id(13, 4))


class TestPaintFilter(unittest.TestCase):

    def test_filter(self):
        from flamingo_tools.annotation.controller import filter_paint_atoms

        data = np.zeros((4, 3, 3), dtype="uint32")
        indices = (np.array([0, 1, 2, 2]), np.array([0, 0, 1, 2]), np.array([0, 1, 1, 2]))
        old = data[indices].copy()
        data[indices] = 5
        atoms = [(indices, old, 5)]
        self.assertTrue(filter_paint_atoms(atoms, [2], data))
        self.assertEqual(data[0, 0, 0], 0)
        self.assertEqual(data[1, 0, 1], 0)
        self.assertEqual(data[2, 1, 1], 5)
        self.assertEqual(len(atoms), 1)
        np.testing.assert_array_equal(atoms[0][0][0], [2, 2])
        self.assertFalse(filter_paint_atoms(atoms, [2], data))

        # Along x: only the voxels with x = 1 stay.
        data[...] = 0
        indices = (np.array([0, 1]), np.array([0, 0]), np.array([1, 2]))
        data[indices] = 7
        atoms = [(indices, np.zeros(2, dtype="uint32"), 7)]
        self.assertTrue(filter_paint_atoms(atoms, [1], data, axis=2))
        self.assertEqual(data[0, 0, 1], 7)
        self.assertEqual(data[1, 0, 2], 0)

    def test_locked(self):
        from flamingo_tools.annotation.controller import filter_paint_atoms

        data = np.zeros((2, 3, 3), dtype="uint32")
        locked = np.zeros(data.shape, dtype=bool)
        locked[1, 0, 0] = True
        indices = (np.array([1, 1]), np.array([0, 1]), np.array([0, 1]))
        data[indices] = 4
        atoms = [(indices, np.zeros(2, dtype="uint32"), 4)]
        self.assertEqual(filter_paint_atoms(atoms, [1], data, locked=locked), "locked")
        self.assertEqual(data[1, 0, 0], 0)
        self.assertEqual(data[1, 1, 1], 4)
        # A stroke with a locked label is undone completely.
        data[indices] = 9
        atoms = [(indices, np.array([0, 4], dtype="uint32"), 9)]
        reason = filter_paint_atoms(atoms, [1], data, locked=locked, locked_value=lambda value: value == 9)
        self.assertEqual(reason, "label")
        self.assertEqual(atoms, [])
        np.testing.assert_array_equal(data[indices], [0, 4])


if __name__ == "__main__":
    unittest.main()
