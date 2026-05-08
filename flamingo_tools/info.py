import argparse
import os

from flamingo_tools.file_utils import get_cache_dir
from flamingo_tools.version import __version__ as version


def module_info():
    """Return information about the flamingo_tools module.

    Checks if segmentation models are downloaded
    Use this in environments where internet access is not available at inference time
    (e.g. restricted HPC compute nodes). Run on a login node before submitting jobs.
    """
    parser = argparse.ArgumentParser(
        description="Return information about the flamingo_tools module. "
        "Check whether pre-trained CochleaNet models are in the local cache. "
        "Run flamingo_tools.download_models to download models."
    )
    _ = parser.parse_args()

    # Return version
    print(f"Version: {version}")

    # Check downloaded models in cache directory
    cache_dir = get_cache_dir()
    model_dir = os.path.join(cache_dir, "models")
    if not os.path.isdir(model_dir):
        print(f"No models in cache directory {cache_dir}.")
        print(f"Create model directory {model_dir} and transfer pre-trained models.")
        print("If you have internet acces: Run flamingo_tools.download_models to download models.")
    else:
        downloaded_models = [entry.name for entry in os.scandir(model_dir)]
        print(f"Model directory: {model_dir}")
        print(f"Downloaded models: {downloaded_models}")
