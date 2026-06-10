#!/usr/bin/env python3
"""
Resilient SMB transfer for N5 data with automatic disconnect recovery.

Falls back from a single bulk transfer to per-subdirectory iterative mode
when an SMB connection drop is detected, retrying each piece independently
without re-prompting for the password.

Usage:
    python smb_transfer_resilient.py [-o output_dir] <username> <remote_parent_dir> <remote_data>
"""

import argparse
import getpass
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings
from typing import Optional

UKON_OLD = "//wfs-medizin.top.gwdg.de/ukon-all$/ukon100"
SMB_SERVER = "//wfs-medizin-spezial.top.gwdg.de/ukon-all$"
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds


def run_smbclient(
    username: str,
    password: str,
    commands: list[str],
    cwd: str,
    smb_server: str = SMB_SERVER,
) -> tuple[list[str], bool, int]:
    """Run smbclient with the given command list; stream output in real time.
    Terminates the process immediately on the first disconnect signal.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_cd: path to cd into inside smbclient before mget.
        cwd: Current working directory.smb_server

    Returns:
      lines, had_disconnect, returncode.

    """
    cmd = ["smbclient", smb_server, "-U", f"GWDG/{username}%{password}"]
    cmd_input = "\n".join(commands + ["exit"])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        cwd=cwd,
        text=True,
    )
    proc.stdin.write(cmd_input)
    proc.stdin.close()

    lines = []
    had_disconnect = False

    for line in proc.stdout:
        line = line.rstrip()
        print(line)
        lines.append(line)
        if "NT_STATUS_CONNECTION_DISCONNECTED" in line and not had_disconnect:
            had_disconnect = True
            proc.terminate()
            break

    proc.wait()
    return lines, had_disconnect, proc.returncode


def list_remote_dirs(
    username: str,
    password: str,
    remote_path: str,
    cwd: str,
    local_fallback: Optional[str] = None,
    smb_server: str = SMB_SERVER,
) -> list[str]:
    """Return subdirectory names at remote_path via smbclient ls.
    Falls back to listing local_fallback on disconnect (folder structure is
    usually present locally after the first drop).

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_path: Remote path on SMB client.
        local_fallback: Path to local data to check directory structure.

    Returns:
        list of remote directories.
    """
    # Normalise to forward slashes for smbclient
    remote_path = remote_path.replace("\\", "/")
    lines, had_disconnect, _ = run_smbclient(
        username, password, [f'cd "{remote_path}"', "ls"], cwd, smb_server=smb_server,
    )

    if had_disconnect and local_fallback and os.path.isdir(local_fallback):
        print(f"  [fallback] listing local directory: {local_fallback}")
        return sorted(
            d for d in os.listdir(local_fallback)
            if os.path.isdir(os.path.join(local_fallback, d))
        )

    dirs = []
    for line in lines:
        m = re.match(r"^\s+(.+?)\s+([DARHSE]+)\s+\d+", line)
        if m:
            name = m.group(1).strip()
            attrs = m.group(2)
            if name in (".", ".."):
                continue
            if "D" in attrs:
                dirs.append(name)
    return dirs


def transfer_path(
    username: str,
    password: str,
    remote_cd: str,
    mget_target: str,
    local_cwd: str,
    retries: int = MAX_RETRIES,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
) -> bool:
    """Transfer mget_target (file or directory) with reconnect-and-retry on disconnect.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_cd: path to cd into inside smbclient before mget.
        mget_target: name to pass to mget (supports wildcards).
        local_cwd: local directory where mget places downloaded files.
        retries: Maximal number of retries.
        log_file: File to log failed transfers.

    Returns:
        True on success, False after exhausting retries.
    """
    remote_cd = remote_cd.replace("\\", "/")
    os.makedirs(local_cwd, exist_ok=True)

    commands = [f'cd "{remote_cd}"', "recurse", "prompt", f"mget {mget_target}"]

    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"  [retry {attempt}/{retries}] {mget_target}")
            time.sleep(RETRY_DELAY)

        _, had_disconnect, rc = run_smbclient(username, password, commands, cwd=local_cwd, smb_server=smb_server)

        if not had_disconnect and rc == 0:
            return True

        if not had_disconnect:
            # Non-zero exit for a reason other than disconnect — still retry
            print(f"  [warn] smbclient exited with code {rc} for {mget_target}")

    print(f"  [error] failed to transfer {mget_target} after {retries} attempts — skipping")
    if log_file is not None:
        try:
            with open(log_file, 'a') as file:
                file.write(f"[error] failed to transfer {mget_target} after {retries} attempts — skipping")
        except Exception as e:
            print(f"Error: {e}")
    return False


def _sort_key(name):
    return int(name) if name.isdigit() else name


def iterative_n5_transfer(
    username: str,
    password: str,
    remote_dir: str,
    n5_name: str,
    output_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
):
    """Phase 2: transfer an N5 dataset setup-by-setup, scale-by-scale.
    For s0, s1, s2, and s3 (highest resolutions) each top-level chunk directory is transferred
    individually so a single disconnect only affects one small piece.
    All other scales are transferred as a single unit.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Path to cd into inside smbclient before mget.
        n5_name: Name to pass to mget (supports wildcards).
        output_dir: Output directory.
        log_file: Log file to store files which were not transferred.


    """
    # Normalise separators
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{n5_name}"
    local_n5 = os.path.join(output_dir, n5_name)

    print("\n=== Iterative N5 transfer mode ===")

    # Root attributes.json
    print(f"\n-- {n5_name}/attributes.json")
    transfer_path(username, password, remote_dir, f"{n5_name}/attributes.json", output_dir, smb_server=smb_server)

    # Discover setups
    setups = list_remote_dirs(username, password, full_remote, output_dir,
                              local_fallback=local_n5, smb_server=smb_server)
    setup_names = sorted(s for s in setups if re.match(r"^setup\d+$", s))
    if not setup_names:
        print("  [warn] no setup* directories found — nothing to transfer")
        return

    print(f"\n  Found setups: {setup_names}")

    for setup in setup_names:
        print(f"\n{'=' * 60}\nTransferring {setup}\n{'=' * 60}")

        setup_remote = f"{full_remote}/{setup}"
        tp_remote = f"{setup_remote}/timepoint0"
        tp_local = os.path.join(local_n5, setup, "timepoint0")

        # Discover scales
        scales = list_remote_dirs(
            username, password, tp_remote, output_dir,
            local_fallback=tp_local, smb_server=smb_server,
        )
        scale_names = sorted(s for s in scales if re.match(r"^s\d+$", s))
        if not scale_names:
            print(f"  [warn] no scale directories found in {setup}/timepoint0")
            continue

        print(f"  Scales: {scale_names}")

        for scale in scale_names:
            scale_remote = f"{tp_remote}/{scale}"
            scale_local = os.path.join(tp_local, scale)

            if scale in ["s0", "s1", "s2", "s3"]:
                # Enumerate top-level chunk directories and transfer individually
                print(f"\n  -- {setup}/timepoint0/{scale}  (per-subdirectory mode)")
                subdirs = list_remote_dirs(
                    username, password, scale_remote, output_dir,
                    local_fallback=scale_local, smb_server=smb_server,
                )
                chunk_dirs = sorted(subdirs, key=_sort_key)
                if not chunk_dirs:
                    print(f"  [warn] no chunk directories found in {scale}")
                    continue

                print(f"{scale} chunk directories: {chunk_dirs[0]} … {chunk_dirs[-1]} ({len(chunk_dirs)} total)")
                for chunk_dir in chunk_dirs:
                    print(f"    -> {scale}/{chunk_dir}")
                    transfer_path(
                        username, password,
                        remote_cd=scale_remote,
                        mget_target=chunk_dir,
                        local_cwd=scale_local,
                        log_file=log_file,
                        smb_server=smb_server,
                    )
            else:
                # Transfer the entire scale in one shot
                print(f"\n  -- {setup}/timepoint0/{scale}")
                transfer_path(
                    username, password,
                    remote_cd=tp_remote,
                    mget_target=scale,
                    local_cwd=tp_local,
                    log_file=log_file,
                    smb_server=smb_server,
                )

    print("\n=== Iterative transfer complete ===")


def main():
    parser = argparse.ArgumentParser(
        description="Resilient SMB transfer for N5 data with automatic disconnect recovery."
    )
    parser.add_argument("-u", "--username", help="GWDG username, e.g. schilling40")
    parser.add_argument("-p", "--remote_parent_dir", help="Remote parent directory on the SMB share")
    parser.add_argument("-d", "--remote_data", help="Remote file or folder on the SMB share (N5 root)")
    parser.add_argument("-o", "--output-dir", default=os.getcwd(), help="Local output directory (default: cwd)")
    parser.add_argument("-s", "--smb_server", type=str, default=SMB_SERVER,
                        help="SMB server to transfer files from. Default: //wfs-medizin.top.gwdg.de/ukon-all$/ukon100")
    parser.add_argument("-l", "--log_file", type=str, default=None,
                        help="Log transfer errors. Default: transfer_log.txt in output directory.")
    args = parser.parse_args()

    output_dir = os.path.realpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    password = getpass.getpass("Enter password: ")

    if "\\" not in args.remote_parent_dir:
        warnings.warn("Ensure that path to parent directory contains double \\ or is quoted.")

    p = pathlib.PureWindowsPath(args.remote_parent_dir)
    remote_dir = p.as_posix()
    n5_name = args.remote_data
    log_file = args.log_file if args.log_file is not None else os.path.join(output_dir, "transfer_log.txt")

    # Phase 1: attempt bulk transfer
    print("Connecting to SMB server and starting bulk transfer...")
    commands = [f'cd "{remote_dir}"', "recurse", "prompt", f"mget {n5_name}"]
    _, had_disconnect, rc = run_smbclient(args.username, password, commands,
                                          cwd=output_dir, smb_server=args.smb_server)

    if not had_disconnect and rc == 0:
        print("File transfer completed successfully.")
        sys.exit(0)

    if not had_disconnect:
        print(f"Transfer failed (exit code {rc}).")
        sys.exit(rc)

    # Phase 2: iterative mode
    print("\nDisconnect detected — switching to iterative transfer mode.")
    iterative_n5_transfer(args.username, password, remote_dir, n5_name, output_dir,
                          log_file=log_file, smb_server=args.smb_server)


if __name__ == "__main__":
    main()
