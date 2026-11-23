import runpy
from setuptools import setup, find_packages

version = runpy.run_path("flamingo_tools/version.py")["__version__"]
setup(
    name="cochlea_net",
    packages=find_packages(exclude=["test"]),
    version=version,
    author="Constantin Pape; Martin Schilling",
    license="MIT",
    # spotiflow is only needed for the flow channels of the synapse detection training.
    extras_require={"flow": ["spotiflow"]},
    entry_points={
        "console_scripts": [
            "flamingo_tools.convert_data = flamingo_tools.data_conversion:convert_lightsheet_to_bdv_cli",
            "flamingo_tools.transfer_data = flamingo_tools.data_transfer:transfer_data_cli",
            "flamingo_tools.run_segmentation = flamingo_tools.segmentation.cli:run_segmentation",
            "flamingo_tools.run_detection = flamingo_tools.segmentation.cli:run_detection",
            "flamingo_tools.download_model = flamingo_tools.model_utils:download_model",
            "flamingo_tools.info = flamingo_tools.info:module_info",
            "flamingo_tools.extract_blocks = flamingo_tools.postprocessing.cli:extract_blocks",
            "flamingo_tools.json_extract_blocks = flamingo_tools.postprocessing.cli:json_extract_blocks",
            "flamingo_tools.label_components = flamingo_tools.postprocessing.cli:label_components",
            "flamingo_tools.object_measures = flamingo_tools.postprocessing.cli:object_measures",
            "flamingo_tools.tonotopic_mapping = flamingo_tools.postprocessing.cli:tonotopic_mapping",
            "flamingo_tools.sgn_density = flamingo_tools.postprocessing.cli:sgn_density",
            "flamingo_tools.central_path_tifs = flamingo_tools.postprocessing.cli:central_path_tifs",
            "flamingo_tools.table_info = flamingo_tools.analysis.cli:table_info",
            "flamingo_tools.correct_annotation = flamingo_tools.annotation.cli:correct_annotation",
            # TODO: MoBIE conversion
        ],
        "napari.manifest": [
            "cochlea_net = flamingo_tools:napari.yaml",
        ],
    }
)
