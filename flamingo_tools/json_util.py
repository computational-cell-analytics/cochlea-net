import json


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
    if force_overwrite:
        with open(output_path, "w") as f:
            json.dump(param_dict, f, indent='\t', separators=(',', ': '))
    else:
        print(f"Skipping creation of {output_path}. Table already exists.")
