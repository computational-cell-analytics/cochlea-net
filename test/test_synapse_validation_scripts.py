import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

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
        # An '-ihc11' entry rescores its base version against IHC v11, so it shares that model.
        entries = self.prediction.PREDICTION_DICT
        models = [entry["synapse_model"] for version, entry in entries.items() if not version.endswith("-ihc11")]
        self.assertEqual(len(set(models)), len(models))
        for version, entry in entries.items():
            if version.endswith("-ihc11"):
                self.assertEqual(entry["synapse_model"], entries[version[:-len("-ihc11")]]["synapse_model"])

    def test_one_ihc_model_for_every_version(self):
        # Scores are only comparable across versions when they were filtered against the same IHC
        # segmentation, so the IHC model is deliberately not per entry.
        ihc_models = {entry["ihc_model"] for entry in self.prediction.PREDICTION_DICT.values()}
        self.assertEqual(len(ihc_models), 1)


class TestPredictionRoots(unittest.TestCase):
    """The version must supply defaults without swallowing a command line override.

    Passing --output_root together with --version used to be ignored, so a run meant for a
    private directory went to the shared production tree instead.
    """

    @classmethod
    def setUpClass(cls):
        cls.prediction = _load_module("synapse_prediction", os.path.join(VALIDATION_DIR, "prediction.py"))

    def _run(self, argv):
        recorded = {}
        with mock.patch.object(self.prediction, "process_everything", lambda **kw: recorded.update(kw)), \
             mock.patch.object(sys, "argv", ["prediction.py"] + argv):
            self.prediction.main()
        return recorded

    def test_version_defaults(self):
        entry = self.prediction.PREDICTION_DICT["v7"]
        recorded = self._run(["-v", "v7"])
        self.assertEqual(recorded["output_root"], entry["pred_root"])
        self.assertEqual(recorded["input_root"], entry["image_root"])
        self.assertEqual(recorded["synapse_model_path"], entry["synapse_model"])
        self.assertEqual(recorded["ihc_model_path"], entry["ihc_model"])

    def test_output_root_overrides_the_version(self):
        recorded = self._run(["-v", "v7", "-o", "/tmp/somewhere"])
        self.assertEqual(recorded["output_root"], os.path.join("/tmp/somewhere", "v7"))

    def test_every_entry_field_is_overridable(self):
        recorded = self._run([
            "-v", "v7", "-i", "/tmp/img", "-g", "/tmp/gt",
            "--model_synapse", "/tmp/syn.pt", "--model_ihc", "/tmp/ihc",
        ])
        self.assertEqual(recorded["input_root"], "/tmp/img")
        self.assertEqual(recorded["gt_root"], "/tmp/gt")
        self.assertEqual(recorded["synapse_model_path"], "/tmp/syn.pt")
        self.assertEqual(recorded["ihc_model_path"], "/tmp/ihc")

    def test_without_a_version_the_output_root_is_exact(self):
        recorded = self._run(["-i", "/tmp/img", "-o", "/tmp/out", "--model_synapse", "/tmp/syn.pt"])
        self.assertEqual(recorded["output_root"], "/tmp/out")


class TestPredictionNormalization(unittest.TestCase):
    """The crops are zero-padded for the production block shape; the padding must not enter the
    normalization statistics, which training takes over the unpadded crop."""

    @classmethod
    def setUpClass(cls):
        cls.prediction = _load_module("synapse_prediction", os.path.join(VALIDATION_DIR, "prediction.py"))

    def test_statistics_of_the_unpadded_crop(self):
        import numpy as np
        import zarr

        raw = np.random.default_rng(0).integers(100, 1000, (10, 20, 20)).astype("uint16")
        recorded = {}
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_path = os.path.join(tmp_dir, "crop.zarr")
            zarr.open(store=input_path, mode="w").create_array("raw", data=raw)

            module = self.prediction
            with mock.patch.object(module, "prediction_impl", lambda **kw: recorded.update(kw)), \
                 mock.patch.object(module, "_get_model_out_channels", lambda path: 1), \
                 mock.patch.object(module, "synapse_detection_from_prediction"), \
                 mock.patch.object(module, "_drop_padding_detections"):
                module.pred_synapse_impl(input_path, os.path.join(tmp_dir, "out"), "model.pt")

        self.assertNotEqual(recorded["input_path"], input_path)
        self.assertAlmostEqual(recorded["mean"], float(raw.mean()), places=3)
        self.assertAlmostEqual(recorded["std"], float(raw.std()), places=3)


class TestEvaluationRoots(unittest.TestCase):
    """-p must take the same directory that prediction.py was given as -o."""

    voxel_size = 0.38
    points = [(10.0, 20.0, 30.0), (40.0, 50.0, 60.0)]

    @classmethod
    def setUpClass(cls):
        cls.evaluation = _load_module("synapse_run_evaluation", os.path.join(VALIDATION_DIR, "run_evaluation.py"))

    def _create_data(self, tmp_dir, version):
        images = os.path.join(tmp_dir, "images")
        refs = os.path.join(tmp_dir, "refs")
        crop_dir = os.path.join(tmp_dir, "preds", version, "crop")
        for folder in (os.path.join(images, "crop.zarr"), refs, crop_dir):
            os.makedirs(folder)

        with open(os.path.join(refs, "crop.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["axis-0", "axis-1", "axis-2"])
            writer.writerows(self.points)

        # The detections are in physical coordinates, the annotations in voxels, so a perfect
        # prediction is the annotation scaled by the voxel size.
        with open(os.path.join(crop_dir, "synapse_detection.tsv"), "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["z", "y", "x"])
            writer.writerows([[c * self.voxel_size for c in point] for point in self.points])

        return images, refs, os.path.join(tmp_dir, "preds")

    def test_pred_root_overrides_the_version(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            images, refs, preds = self._create_data(tmp_dir, "v7")
            out_dir = os.path.join(tmp_dir, "accuracy")

            argv = ["run_evaluation.py", "-v", "v7", "-c", images, "-r", refs, "-p", preds, "-o", out_dir]
            with mock.patch.object(sys, "argv", argv):
                self.evaluation.main()

            with open(os.path.join(out_dir, "synapses.json")) as f:
                results = json.load(f)

            # Read from <preds>/v7, and stored under the version rather than the directory name.
            self.assertEqual(list(results), ["v7"])
            self.assertEqual(results["v7"]["tp"], [len(self.points)])
            self.assertEqual(results["v7"]["fp"], [0])
            self.assertEqual(results["v7"]["fn"], [0])

    def test_existing_entry_needs_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            images, refs, preds = self._create_data(tmp_dir, "v7")
            out_dir = os.path.join(tmp_dir, "accuracy")
            argv = ["run_evaluation.py", "-v", "v7", "-c", images, "-r", refs, "-p", preds, "-o", out_dir]

            with mock.patch.object(sys, "argv", argv):
                self.evaluation.main()
                with self.assertRaises(ValueError):
                    self.evaluation.main()
            with mock.patch.object(sys, "argv", argv + ["--overwrite"]):
                self.evaluation.main()


if __name__ == "__main__":
    unittest.main()
