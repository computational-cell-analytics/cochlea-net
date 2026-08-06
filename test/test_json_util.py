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


if __name__ == "__main__":
    unittest.main()
