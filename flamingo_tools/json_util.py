import json
import os


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
