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
            "flamingo_tools.equidistant_centers = flamingo_tools.postprocessing.cli:equidistant_centers",
            "flamingo_tools.extract_block = flamingo_tools.postprocessing.cli:extract_block",
            "flamingo_tools.extract_central_blocks = flamingo_tools.postprocessing.cli:extract_central_blocks",
            "flamingo_tools.json_block_extraction = flamingo_tools.postprocessing.cli:json_block_extraction",
            "flamingo_tools.label_components = flamingo_tools.postprocessing.cli:label_components",
            "flamingo_tools.object_measures = flamingo_tools.postprocessing.cli:object_measures",
            "flamingo_tools.tonotopic_mapping = flamingo_tools.postprocessing.cli:tonotopic_mapping",
            # TODO: MoBIE conversion
        ],
        "napari.manifest": [
            "cochlea_net = flamingo_tools:napari.yaml",
        ],
    }
)
