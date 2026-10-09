#!/usr/bin/env python3
"""
Write the file names of a raw-data folder on the SMB share to a JSON file. Nothing is downloaded.

The raw-data folder contains one subfolder per stain. The JSON file holds the raw-data folder name
and, for each stain folder, its sorted file names.

Usage:
    python smb_list_raw.py -u <username> -p <remote_parent_dir> -d <raw_data_folder> -o raw_files.json
"""

import argparse
import getpass
import json
import os

from flamingo_tools.data_transfer_utils import (
    SMB_SERVER,
    normalize_remote_dir,
    remote_size_map_with_retry,
    require_remote_dir,
)


def raw_file_list(raw_data: str, size_map: dict) -> dict:
    """Group the {"<stain_folder>/<file>": size} map of a raw-data folder by stain folder."""
    folders = {}
    for rel in sorted(size_map):
        folder, sep, name = rel.partition("/")
        if sep:  # A file directly in the raw-data folder is not a stain file.
            folders.setdefault(folder, []).append(name)
    return {"raw_data": raw_data, "stain_folders": folders}


def main():
    parser = argparse.ArgumentParser(
        description="Write the file names of a raw-data folder on the SMB share to a JSON file."
    )
    parser.add_argument("-u", "--username", required=True, help="GWDG username, e.g. schilling40")
    parser.add_argument("-p", "--remote_parent_dir", required=True,
                        help="Remote parent directory on the SMB share")
    parser.add_argument("-d", "--remote_data", required=True,
                        help="Raw-data folder with one subfolder per stain")
    parser.add_argument("-o", "--output", required=True, help="Path of the output JSON file")
    parser.add_argument("-s", "--smb_server", type=str, default=SMB_SERVER,
                        help=f"SMB server to list from. Default: {SMB_SERVER}")
    args = parser.parse_args()

    remote_dir = f"{normalize_remote_dir(args.remote_parent_dir)}/{args.remote_data}"

    password = getpass.getpass("Enter password: ")

    require_remote_dir(args.username, password, remote_dir, os.getcwd(), smb_server=args.smb_server)
    size_map = remote_size_map_with_retry(args.username, password, remote_dir, os.getcwd(),
                                          smb_server=args.smb_server)
    if size_map is None:
        raise SystemExit(f"Could not list the remote directory {remote_dir}.")

    file_list = raw_file_list(args.remote_data, size_map)
    with open(args.output, "w") as f:
        json.dump(file_list, f, indent=2)

    print(f"\nWrote {args.output}")
    for folder, names in file_list["stain_folders"].items():
        print(f"  {folder}: {len(names)} files")


if __name__ == "__main__":
    main()
