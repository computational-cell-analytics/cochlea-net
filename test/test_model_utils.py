import os
import unittest
import urllib.request
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import mock

import torch
import yaml

from flamingo_tools.model_utils import MODELS, _create_model, get_model, get_model_path, get_model_url
from flamingo_tools.segmentation.cli import _parse_segmentation_kwargs
from flamingo_tools.segmentation.synapse_detection import _get_model_out_channels
from flamingo_tools.segmentation.unet_prediction import _load_model


class TestSegmentationDefaults(unittest.TestCase):
    def test_ihc_defaults(self):
        """run_segmentation must use the grid search result of IHC_v11, the model that get_model('IHC') loads."""
        kwargs = _parse_segmentation_kwargs([], "IHC")
        self.assertEqual(kwargs["center_distance_threshold"], 0.7)
        self.assertEqual(kwargs["boundary_distance_threshold"], 0.6)
        self.assertEqual(kwargs["distance_smoothing"], 1.0)


class TestModelSource(unittest.TestCase):
    """Check that the model specifications match the models on BioImage.IO."""

    def _read_rdf(self, model_type):
        with urllib.request.urlopen(get_model_url(model_type, "rdf.yaml"), timeout=60) as response:
            return yaml.safe_load(response.read())

    def test_rdf_matches_model_spec(self):
        for model_type, model in MODELS.items():
            with self.subTest(model_type=model_type):
                weights = self._read_rdf(model_type)["weights"]["pytorch_state_dict"]
                self.assertEqual(weights["sha256"], model["sha256"])

                architecture = weights["architecture"]
                self.assertEqual(architecture["callable"], model["model_class"].rsplit(".", 1)[1])
                # The model spec leaves out the arguments that have the default value of torch_em.
                kwargs = {
                    name: value for name, value in architecture["kwargs"].items()
                    if (name, value) not in (("return_side_outputs", False), ("postprocessing", None))
                }
                self.assertEqual(kwargs, model["model_kwargs"])

    def test_weights_are_available(self):
        for model_type in MODELS:
            with self.subTest(model_type=model_type):
                # Request only the first byte to check the URL without the download of the full weights.
                request = urllib.request.Request(get_model_url(model_type), headers={"Range": "bytes=0-0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    self.assertIn(response.status, (200, 206))


class TestGetModelPath(unittest.TestCase):
    model_type = "Synapses"

    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.cache_dir = os.path.join(self.tmp_dir.name, "cache")
        patcher = mock.patch("flamingo_tools.model_utils.get_cache_dir", return_value=self.cache_dir)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.tmp_dir.cleanup)

        # Save random weights with the architecture of the model in place of the BioImage.IO weights.
        self.state = _create_model(self.model_type).state_dict()
        self.weights_path = os.path.join(self.tmp_dir.name, "weights.pt")
        torch.save(self.state, self.weights_path)

    def _retrieve(self, url, known_hash, fname, path):
        self.assertEqual(url, get_model_url(self.model_type))
        self.assertEqual(known_hash, f"sha256:{MODELS[self.model_type]['sha256']}")
        out_path = os.path.join(path, fname)
        copyfile(self.weights_path, out_path)
        return out_path

    def _check_state(self, model):
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.cpu(), self.state[name]), name)

    def test_get_model_path(self):
        with mock.patch("flamingo_tools.model_utils.pooch.retrieve", side_effect=self._retrieve) as retrieve:
            model_path = get_model_path(self.model_type)
            self.assertEqual(get_model_path(self.model_type), model_path)
        retrieve.assert_called_once()

        # Only the checkpoint stays in the model directory.
        self.assertEqual(os.listdir(os.path.dirname(model_path)), [os.path.basename(model_path)])

        # The checkpoint works with the model loading of the segmentation and detection functions.
        model = _load_model(model_path)
        self.assertEqual(type(model).__name__, "AnisotropicUNet")
        self._check_state(model)
        self.assertEqual(_get_model_out_channels(model_path), 1)

        model = get_model(self.model_type, device="cpu")
        self.assertEqual(type(model).__name__, "AnisotropicUNet")
        self._check_state(model)

    def test_failed_download(self):
        with mock.patch("flamingo_tools.model_utils.pooch.retrieve", side_effect=ValueError("Hash mismatch")):
            with self.assertRaises(ValueError):
                get_model_path(self.model_type)
        self.assertEqual(os.listdir(os.path.join(self.cache_dir, "models")), [])

    def test_invalid_model_type(self):
        with self.assertRaises(ValueError):
            get_model_path("SGN-highres")


if __name__ == "__main__":
    unittest.main()
