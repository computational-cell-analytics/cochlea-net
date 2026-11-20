import os
import unittest
import urllib.error
import urllib.request
from shutil import copyfile
from tempfile import TemporaryDirectory
from unittest import mock

import torch
import yaml

from torch_em.model.unet import AnisotropicUNet

from flamingo_tools.model_utils import (
    MODELS, _create_model, export_model, get_default_segmentation_settings, get_model, get_model_path, get_model_url,
    load_model_from_path
)
from flamingo_tools.segmentation.synapse_detection import _get_model_out_channels


class TestSegmentationDefaults(unittest.TestCase):
    def test_ihc_defaults(self):
        """run_segmentation must use the grid search result of IHC_v11, the model that get_model('IHC') loads."""
        kwargs = get_default_segmentation_settings("IHC")
        self.assertEqual(kwargs["center_distance_threshold"], 0.7)
        self.assertEqual(kwargs["boundary_distance_threshold"], 0.6)
        self.assertEqual(kwargs["distance_smoothing"], 1.0)


class TestModelSource(unittest.TestCase):
    """Check that the model specifications match the models on BioImage.IO."""

    # After the first failed request, the other requests skip at once instead of waiting for the timeout.
    unavailable = None

    def _urlopen(self, request):
        """Skip the test when the server is not available. A missing file (404) still fails."""
        if type(self).unavailable:
            self.skipTest(type(self).unavailable)
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as error:
            if error.code != 429 and error.code < 500:
                raise
            reason = str(error)
        except (urllib.error.URLError, TimeoutError) as error:
            reason = str(error)
        type(self).unavailable = f"BioImage.IO is not available: {reason}"
        self.skipTest(type(self).unavailable)

    def _read_rdf(self, model_type):
        with self._urlopen(get_model_url(model_type, "rdf.yaml")) as response:
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
                with self._urlopen(request) as response:
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
        model = load_model_from_path(model_path)
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


def _create_small_unet():
    return AnisotropicUNet(in_channels=1, out_channels=2, scale_factors=[[1, 2, 2], [2, 2, 2]], initial_features=4)


class TestLoadModelFromPath(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.model_path = os.path.join(self.tmp_dir.name, "model.pt")

    def test_pickled_model(self):
        """A pickled model of an older torch_em version can miss an attribute that the constructor sets."""
        model = _create_small_unet().eval()
        del model.decoder.return_outputs
        torch.save(model, self.model_path)

        loaded = load_model_from_path(self.model_path)
        self.assertIsInstance(loaded, AnisotropicUNet)
        self.assertFalse(loaded.decoder.return_outputs)
        self.assertFalse(loaded.training)
        expected_state = model.state_dict()
        for name, value in loaded.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)
        with torch.no_grad():
            self.assertEqual(loaded(torch.zeros(1, 1, 8, 32, 32)).shape, (1, 2, 8, 32, 32))

    def test_pickled_model_without_init_kwargs(self):
        torch.save(torch.nn.Conv3d(1, 1, 3), self.model_path)
        self.assertIsInstance(load_model_from_path(self.model_path), torch.nn.Conv3d)


class TestExportModel(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)
        self.model = _create_small_unet()
        self.output_path = os.path.join(self.tmp_dir.name, "exported.pt")

    def _check_export(self, model_path):
        export_model(model_path, self.output_path)

        checkpoint = torch.load(self.output_path, map_location="cpu", weights_only=False)
        self.assertEqual(set(checkpoint), {"init", "model_state"})
        self.assertEqual(checkpoint["init"]["model_class"], "torch_em.model.unet.AnisotropicUNet")
        for name, value in checkpoint["model_state"].items():
            self.assertEqual(value.device.type, "cpu", name)

        model = load_model_from_path(self.output_path)
        self.assertIsInstance(model, AnisotropicUNet)
        expected_state = self.model.state_dict()
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, expected_state[name]), name)
        # The constructor sets the attributes again, also those that the input file does not have.
        self.assertFalse(model.decoder.return_outputs)
        with torch.no_grad():
            self.assertEqual(model(torch.zeros(1, 1, 8, 32, 32)).shape, (1, 2, 8, 32, 32))
        self.assertEqual(_get_model_out_channels(self.output_path), 2)

    def test_trainer_checkpoint(self):
        model_path = os.path.join(self.tmp_dir.name, "latest.pt")
        torch.save({
            "iteration": 10,
            "model_state": self.model.state_dict(),
            "optimizer_state": torch.optim.Adam(self.model.parameters()).state_dict(),
            "init": {"model_class": "torch_em.model.unet.AnisotropicUNet", "model_kwargs": self.model.init_kwargs},
        }, model_path)
        self._check_export(model_path)

    def test_pickled_model(self):
        """A pickled model of an older torch_em version can miss an attribute that the constructor sets."""
        model_path = os.path.join(self.tmp_dir.name, "model.pt")
        del self.model.decoder.return_outputs
        torch.save(self.model, model_path)
        self._check_export(model_path)

    def test_model_without_init_kwargs(self):
        model_path = os.path.join(self.tmp_dir.name, "model.pt")
        torch.save(torch.nn.Conv3d(1, 1, 3), model_path)
        with self.assertRaises(ValueError):
            export_model(model_path, self.output_path)


if __name__ == "__main__":
    unittest.main()
