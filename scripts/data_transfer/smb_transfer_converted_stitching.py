#!/usr/bin/env python3
"""
Download the stitching input from a converted-data folder on the SMB share.

The folder (usually 2_converted_stitching) contains:
- the BigStitcher XML files and their backups (every name that contains "xml");
- interestpoints.n5, which is copied in full;
- the image N5, of which only the scale levels from --min_scale onwards are copied.

The data lands in <output_dir>/<remote_data>, with the same layout as on the share.

Usage:
    python smb_transfer_converted_stitching.py -u <username> -p <remote_parent_dir> -d 2_converted_stitching \
        [-o local_dir] [--min_scale 2]
"""

import argparse
import getpass
import glob
import os

from flamingo_tools.data_transfer_utils import (
    SMB_SERVER,
    list_remote_dirs,
    log_size,
    normalize_remote_dir,
    require_remote_dir,
    transfer_path,
)
from smb_transfer_resilient import iterative_n5_transfer, verify_and_repair_download_generic, verify_and_repair_n5

INTEREST_POINTS = "interestpoints.n5"


def main():
    parser = argparse.ArgumentParser(
        description="Download the XML files, the interest points and the downsampled image N5 "
                    "that BigStitcher needs from a converted-data folder on the SMB share."
    )
    parser.add_argument("-u", "--username", required=True, help="GWDG username, e.g. schilling40")
    parser.add_argument("-p", "--remote_parent_dir", required=True,
                        help="Remote parent directory on the SMB share")
    parser.add_argument("-d", "--remote_data", required=True,
                        help="Folder with the stitching data, e.g. 2_converted_stitching")
    parser.add_argument("-o", "--output-dir", default=os.getcwd(),
                        help="Local directory that receives <remote_data>. Default: cwd.")
    parser.add_argument("-s", "--smb_server", type=str, default=SMB_SERVER,
                        help=f"SMB server to transfer from. Default: {SMB_SERVER}")
    parser.add_argument("-l", "--log_file", type=str, default=None,
                        help="Log transfer errors. Default: transfer_log.txt in the output directory.")
    parser.add_argument("--min_scale", type=int, default=2,
                        help="First scale level of the image N5 to copy. Default: 2 (s2, s3, ...).")
    args = parser.parse_args()

    data_name = args.remote_data.rstrip("/\\")
    remote_dir = f"{normalize_remote_dir(args.remote_parent_dir)}/{data_name}"
    output_dir = os.path.realpath(args.output_dir)
    local_dir = os.path.join(output_dir, data_name)
    os.makedirs(output_dir, exist_ok=True)
    log_file = args.log_file if args.log_file is not None else os.path.join(output_dir, "transfer_log.txt")
    smb = dict(log_file=log_file, smb_server=args.smb_server)

    log_start = log_size(log_file)

    password = getpass.getpass("Enter password: ")

    # Create the local folder only after the check, so that a wrong -d leaves nothing behind.
    require_remote_dir(args.username, password, remote_dir, output_dir, smb_server=args.smb_server)
    os.makedirs(local_dir, exist_ok=True)
    dirs = list_remote_dirs(args.username, password, remote_dir, local_dir,
                            local_fallback=local_dir, smb_server=args.smb_server)
    image_n5s = [d for d in dirs if d.endswith(".n5") and d != INTEREST_POINTS]
    if not image_n5s:
        raise SystemExit(f"No image N5 found in {remote_dir} (found directories: {dirs})")

    print("\n=== XML files ===")
    transfer_path(args.username, password, remote_cd=remote_dir, mget_target="*xml*", local_cwd=local_dir, **smb)
    # mget exits 0 when no file matches, so check the result before the large image transfer.
    if not glob.glob(os.path.join(local_dir, "*.xml")):
        raise SystemExit(f"No XML file was downloaded from {remote_dir}. "
                         "Check that -d names the converted-data folder.")

    if INTEREST_POINTS in dirs:
        print(f"\n=== {INTEREST_POINTS} ===")
        transfer_path(args.username, password, remote_cd=remote_dir, mget_target=INTEREST_POINTS,
                      local_cwd=local_dir, **smb)
        verify_and_repair_download_generic(args.username, password, remote_dir, INTEREST_POINTS, local_dir, **smb)
    else:
        print(f"[warn] {INTEREST_POINTS} not found in {remote_dir}")

    for n5_name in image_n5s:
        print(f"\n=== {n5_name} (s{args.min_scale} onwards) ===")
        iterative_n5_transfer(args.username, password, remote_dir, n5_name, local_dir,
                              min_scale=args.min_scale, **smb)
        verify_and_repair_n5(args.username, password, remote_dir, n5_name, local_dir, **smb)

    if log_size(log_file) > log_start:
        raise SystemExit(f"\n[error] Some transfers failed. See {log_file}")


if __name__ == "__main__":
    main()
