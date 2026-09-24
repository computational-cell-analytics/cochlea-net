import importlib.util
import os
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATION_DIR = os.path.join(REPO_ROOT, "scripts", "validation", "synapses")


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestSynapseModelRegistry(unittest.TestCase):
    """Guard the two synapse model registries against drifting apart.

    prediction.py maps a version to a checkpoint, run_evaluation.py lists the versions it will
    score. A version present in only one of them means `-v <version>` predicts but cannot be
    evaluated, or the other way round, and the failure only appears on the cluster.
    """

    @classmethod
    def setUpClass(cls):
        cls.prediction = _load_module("synapse_prediction", os.path.join(VALIDATION_DIR, "prediction.py"))
        cls.evaluation = _load_module("synapse_run_evaluation", os.path.join(VALIDATION_DIR, "run_evaluation.py"))

    def test_registries_agree(self):
        self.assertEqual(
            set(self.prediction.PREDICTION_DICT), set(self.evaluation._PRODUCTION_VERSIONS)
        )

    def test_entries_are_complete(self):
        expected = {"image_root", "ref_root", "pred_root", "synapse_model", "ihc_model"}
        for version, entry in self.prediction.PREDICTION_DICT.items():
            self.assertEqual(set(entry), expected, msg=version)

    def test_prediction_output_matches_the_version(self):
        # _entry() takes the version twice over, as the dict key and as the argument that names
        # the output folder. A mismatch would write predictions where another version reads them.
        for version, entry in self.prediction.PREDICTION_DICT.items():
            self.assertEqual(os.path.basename(entry["pred_root"]), version)

    def test_checkpoints_are_distinct(self):
        models = [entry["synapse_model"] for entry in self.prediction.PREDICTION_DICT.values()]
        self.assertEqual(len(set(models)), len(models))

    def test_one_ihc_model_for_every_version(self):
        # Scores are only comparable across versions when they were filtered against the same IHC
        # segmentation, so the IHC model is deliberately not per entry.
        ihc_models = {entry["ihc_model"] for entry in self.prediction.PREDICTION_DICT.values()}
        self.assertEqual(len(ihc_models), 1)


if __name__ == "__main__":
    unittest.main()
