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
            "flamingo_tools.convert_data = flamingo_tools.data_conversion:convert_lightsheet_to_bdv_cli",
            "flamingo_tools.run_segmentation = flamingo_tools.segmentation.cli:run_segmentation",
            "flamingo_tools.run_detection = flamingo_tools.segmentation.cli:run_detection",
            # TODO: MoBIE conversion, tonotopic mapping
        ],
        "napari.manifest": [
            "cochlea_net = flamingo_tools:napari.yaml",
        ],
    }
)
