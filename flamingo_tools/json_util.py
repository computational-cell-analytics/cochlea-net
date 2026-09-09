import json
import os
from typing import Dict, List


def export_dictionary_as_json(
    param_dict: dict,
    output_path: str,
    force_overwrite: bool = False,
):
    """Export a dictionary as a file in JSON format.

    Args:
        param_dict: Parameter dictionary.
        out_path: Output path for JSON file.
        force_overwrite: Flag for forcefully overwriting file.
    """

    if force_overwrite or not os.path.isfile(output_path):
        with open(output_path, "w") as f:
            json.dump(param_dict, f, indent='\t', separators=(',', ': '))
    else:
        print(f"Skipping creation of {output_path}. Table already exists.")


def update_json(
    param_dict: dict,
    output_path: str,
):
    """Merge a dictionary into a file in JSON format.

    Creates the file if it does not exist yet. Replaces the top-level keys of param_dict
    and keeps all other top-level keys of the existing file.

    Args:
        param_dict: Parameter dictionary.
        output_path: Output path for JSON file.
    """
    data = {}
    if os.path.isfile(output_path):
        with open(output_path, "r") as f:
            data = json.load(f)

    data.update(param_dict)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent='\t', separators=(',', ': '))
    print(f"Saved results to {output_path}")


# Keys of a shared processing parameter file that describe the cochlea itself. They are passed to
# every processing step.
COMMON_KEYS = frozenset({
    "dataset_name", "segmentation_channel", "cell_type", "image_channel", "component_list",
    "voxel_size",
})

# Keys that only one processing step consumes, per section of the file. The section names are the
# names of the console scripts. test/test_json_util.py asserts that every key here is a parameter
# of the matching *_single function, so the table cannot drift away from the signatures.
STEP_KEYS: Dict[str, frozenset] = {
    "label_components": frozenset({
        "component_list_path", "max_edge_distance", "min_component_length", "min_size",
        "custom_dic", "max_path_deviation",
    }),
    "tonotopic_mapping": frozenset({
        "apex_position", "component_mapping", "include_gap", "animal", "otof",
    }),
    "object_measures": frozenset({"use_bg_mask"}),
}

# Section keys that reach the processing function under a different name.
# In label_components the component list selects the components that form the central path and
# that the instance counts are reported for, which is not always the list the later steps select.
STEP_RENAMES: Dict[str, Dict[str, str]] = {
    "label_components": {"component_list_path": "component_list"},
}

REQUIRED_KEYS = ("dataset_name", "segmentation_channel")


def load_processing_params(json_file: str, step: str) -> List[dict]:
    """Read the parameters of one processing step from a shared processing parameter file.

    A file describes one cochlea and one segmentation. It holds the common keys at the top level
    and one section per processing step. An empty section means the step ran with its defaults.
    An absent section means the step was not run for this cochlea, and the entry is skipped. A
    file that has no entry for the step yields an empty list, so that a loop over a whole folder
    is not interrupted by a cochlea which skipped the step.

    An entry that holds no section at all is read as a flat parameter dictionary and passed
    through unvalidated. That keeps the block extraction files working, which are flat by design
    and which doc/analysis.md feeds to flamingo_tools.object_measures.

    Args:
        json_file: Shared parameter file, or a list of such parameter dictionaries.
        step: Name of the processing step, a key of STEP_KEYS.

    Returns:
        One flat parameter dictionary per entry that applies to the step, empty if none does.

    Raises:
        ValueError: If the step is unknown, if a sectioned entry holds an unknown key, or if a
            required key is missing.
    """
    if step not in STEP_KEYS:
        raise ValueError(f"Unknown processing step {step!r}. Expected one of {sorted(STEP_KEYS)}.")

    with open(json_file, "r") as f:
        data = json.load(f)
    entries = data if isinstance(data, list) else [data]

    allowed_top = COMMON_KEYS | set(STEP_KEYS)
    renames = STEP_RENAMES.get(step, {})
    params = []
    for index, entry in enumerate(entries):
        where = f"{json_file} entry {index}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} is not a parameter dictionary.")

        missing = [key for key in REQUIRED_KEYS if key not in entry]
        if missing:
            raise ValueError(f"{where} is missing the required key(s) {missing}.")

        if not set(entry) & set(STEP_KEYS):
            # A flat file from before the sectioned format. Pass it through as it is.
            params.append(dict(entry))
            continue

        unknown = sorted(set(entry) - allowed_top)
        if unknown:
            raise ValueError(
                f"{where} holds unknown top-level key(s) {unknown}. Expected the common keys "
                f"{sorted(COMMON_KEYS)} or a section named after a processing step "
                f"{sorted(STEP_KEYS)}."
            )
        if step not in entry:
            continue
        section = entry[step]
        if not isinstance(section, dict):
            raise ValueError(f"{where} has a {step!r} section that is not a dictionary.")
        unknown = sorted(set(section) - STEP_KEYS[step])
        if unknown:
            raise ValueError(
                f"{where} has unknown key(s) {unknown} in its {step!r} section. "
                f"Expected {sorted(STEP_KEYS[step])}."
            )

        flat = {key: value for key, value in entry.items() if key in COMMON_KEYS}
        flat.update({renames.get(key, key): value for key, value in section.items()})
        params.append(flat)

    return params
