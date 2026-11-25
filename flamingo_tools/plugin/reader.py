from typing import Callable, List, Optional

from napari_builtins.io import csv_to_layer_data

# napari's default size of 10 voxels (3.8 um at 0.38 um/voxel) hides the signal of a synapse.
POINT_SIZE = 2


def get_csv_reader(path) -> Optional[Callable]:
    if isinstance(path, str) and path.endswith(".csv"):
        return read_csv
    return None


def read_csv(path: str) -> List[tuple]:
    """Read a napari CSV file like the napari builtin reader, with points of size `POINT_SIZE`.

    Other layer types, such as shapes, keep the result of the builtin reader.
    """
    data, meta, layer_type = csv_to_layer_data(path, require_type="any")
    if layer_type == "points":
        meta["size"] = POINT_SIZE
    return [(data, meta, layer_type)]
