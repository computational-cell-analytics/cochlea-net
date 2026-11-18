import os
import tempfile
import unittest

import numpy as np
import pandas as pd
import tifffile


class TestSession(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = os.path.join(self.tmp.name, "crop_session")

    def tearDown(self):
        self.tmp.cleanup()

    def test_commit_replay_undo_export(self):
        from flamingo_tools.annotation.ids import compact
        from flamingo_tools.annotation.session import Session, array_sha1

        original_labels = np.zeros((10, 20, 20), dtype="uint32")
        original_labels[2:6, 2:8, 2:8] = 3_000_000_000
        original_labels[2:6, 10:16, 2:8] = 12
        labels, original = compact(original_labels)
        sha1 = array_sha1(original_labels)
        session = Session.create(self.folder, sha1, {"crop": "test"})

        before = labels[0:8, 0:10, 0:10].copy()
        after = before.copy()
        after[6, 2:8, 2:8] = 2
        edit = np.zeros_like(before)
        edit[6] = after[6]
        entry = session.commit((0, 0, 0), before, after, {"item": 0, "action": "next", "label_id": 2}, edit=edit)
        labels[0:8, 0:10, 0:10] = after
        # The raw edit is stored with the patch.
        with np.load(os.path.join(self.folder, "patches", "00000.npz")) as data:
            np.testing.assert_array_equal(data["edit"], edit)
        self.assertEqual(entry["n_changed"], 36)
        self.assertEqual(entry["delta"], {2: 36})
        self.assertEqual(entry["center"], [6, 4, 4])
        # Nothing changed: no patch.
        self.assertIsNone(session.commit((0, 0, 0), after, after))

        second = labels[0:10, 8:20, 0:10].copy()
        changed = second.copy()
        changed[changed == 1] = 0
        session.commit((0, 8, 0), second, changed, {"item": 1, "action": "delete", "label_id": 1})
        labels[0:10, 8:20, 0:10] = changed

        reopened = Session.open(self.folder, sha1)
        replayed = reopened.replay(compact(original_labels)[0])
        np.testing.assert_array_equal(replayed, labels)
        with self.assertRaises(ValueError):
            Session.open(self.folder, "other")

        entry, stored_edit = reopened.undo_last(replayed)
        self.assertEqual(entry["action"], "delete")
        self.assertIsNone(stored_edit)
        self.assertEqual(len(reopened.state["patches"]), 1)
        self.assertTrue((replayed == 1).any())
        # The undo of a visit returns its raw edit.
        undone = replayed.copy()
        entry, stored_edit = reopened.undo_last(undone)
        np.testing.assert_array_equal(stored_edit, edit)
        self.assertIsNone(reopened.undo_last(undone))
        np.testing.assert_array_equal(undone, compact(original_labels)[0])
        reopened.commit((0, 0, 0), before, after, {"item": 0, "action": "next", "label_id": 2}, edit=edit)

        output = os.path.join(self.tmp.name, "crop_annotations.tif")
        table_path = os.path.join(self.tmp.name, "crop_changes.tsv")
        reopened.export(replayed, original, output, table_path)
        # The session stays by default, as the record of the edits.
        self.assertTrue(Session.exists(self.folder))
        reopened.export(replayed, original, output, table_path, keep=False)
        exported = tifffile.imread(output)
        self.assertEqual(exported[6, 4, 4], 3_000_000_000)
        self.assertEqual(exported[3, 12, 4], 12)
        changes = pd.read_csv(table_path, sep="\t")
        self.assertEqual(changes["label_id"].tolist(), [3_000_000_000])
        self.assertEqual(changes["delta"].tolist(), ["3000000000:+36"])
        self.assertFalse(os.path.exists(self.folder))
        # No temporary files are left.
        self.assertEqual(sorted(os.listdir(self.tmp.name)), ["crop_annotations.tif", "crop_changes.tsv"])

    def test_arrays(self):
        from flamingo_tools.annotation.session import Session

        session = Session.create(self.folder, "abc")
        self.assertEqual(session.load_arrays(), {})
        session.save_arrays({"3": np.arange(6).reshape(2, 3)})
        np.testing.assert_array_equal(session.load_arrays()["3"], np.arange(6).reshape(2, 3))


if __name__ == "__main__":
    unittest.main()
