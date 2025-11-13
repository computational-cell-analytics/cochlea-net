import runpy
from setuptools import setup, find_packages

version = runpy.run_path("flamingo_tools/version.py")["__version__"]
setup(
    name="cochlea_net",
    packages=find_packages(exclude=["test"]),
    version=version,
    author="Constantin Pape; Martin Schilling",
    license="MIT",
    entry_points={
        "console_scripts": [
            "convert_flamingo = flamingo_tools.data_conversion:convert_lightsheet_to_bdv_cli"
        ],
        "napari.manifest": [
            "cochlea_net = flamingo_tools:napari.yaml",
        ],
    }
)
