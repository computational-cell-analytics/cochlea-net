import importlib.util
import io
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout

import numpy as np
import z5py

SCRIPT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "check_n5_container.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("check_n5_container", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestCheckN5Container(unittest.TestCase):
    key = "setup0/timepoint0/s0"
    shape = (16, 32, 32)
    chunks = (8, 16, 16)

    def setUp(self):
        self.script = _load_script()
        self.tmp_dir = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp_dir, "data.n5")
        with z5py.File(self.path, "a") as f:
            f.create_dataset(
                self.key, data=np.random.randint(0, 255, size=self.shape).astype("uint16"),
                chunks=self.chunks,
            )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _check(self, **kwargs):
        """Run the check and return (problems, printed output)."""
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            problems = self.script.check_container(self.path, **kwargs)
        return problems, stdout.getvalue()

    def _chunk_files(self):
        dataset_dir = os.path.join(self.path, self.key)
        return sorted(
            os.path.join(root, name)
            for root, _, files in os.walk(dataset_dir) for name in files if name.isdigit()
        )

    def test_healthy_container(self):
        problems, output = self._check(key=self.key)
        self.assertEqual(problems, [])
        self.assertIn("n5 dataset", output)
        # The dimensions are reported in the (x, y, z) order of the n5 metadata.
        self.assertIn(str(tuple(reversed(self.shape))), output)
        self.assertIn("8/8 chunk files", output)

    def test_missing_attributes(self):
        """The reported failure: the dataset metadata is gone, so the key reads as a group."""
        os.remove(os.path.join(self.path, self.key, "attributes.json"))

        problems, output = self._check(key=self.key)
        self.assertTrue(any("attributes.json" in p for p in problems), msg=output)
        self.assertTrue(any("does not point to an array" in p for p in problems), msg=output)

    def test_truncated_chunk(self):
        with open(self._chunk_files()[0], "w"):
            pass

        problems, output = self._check(key=self.key)
        self.assertTrue(any("truncated chunk file" in p for p in problems), msg=output)

    def test_absent_chunk_is_not_a_problem(self):
        """n5 datasets may be sparse, so an absent chunk is reported but does not fail."""
        os.remove(self._chunk_files()[0])

        problems, output = self._check(key=self.key)
        self.assertEqual(problems, [])
        self.assertIn("7/8 chunk files", output)
        self.assertIn("1 chunk file(s) absent", output)

    def test_missing_key(self):
        problems, output = self._check(key="setup1/timepoint0/s0")
        self.assertTrue(any("does not exist" in p for p in problems), msg=output)

    def test_node_keys_use_forward_slashes(self):
        """Container keys are '/'-separated on every platform, unlike filesystem paths."""
        keys = {rel for rel, _, _, _ in self.script.walk_nodes(self.path)}
        self.assertEqual(keys, {"/", "setup0", "setup0/timepoint0", self.key})

    def test_backslash_key_is_accepted(self):
        problems, output = self._check(key=self.key.replace("/", "\\"))
        self.assertEqual(problems, [], msg=output)

    def test_node_dir_of_root_is_the_container(self):
        """A root key must not resolve to the filesystem root."""
        self.assertEqual(self.script._node_dir(self.path, "/"), self.path)

    def test_unreadable_chunk(self):
        with open(self._chunk_files()[0], "wb") as f:
            f.write(b"not a valid n5 chunk")

        problems, output = self._check(key=self.key)
        self.assertTrue(any("cannot be read" in p for p in problems), msg=output)


if __name__ == "__main__":
    unittest.main()
