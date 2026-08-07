#!/usr/bin/env python3
"""
Resilient SMB transfer for N5 data with automatic disconnect recovery.

Transfers an N5 dataset in either direction, falling back from a single bulk
transfer to a per-subdirectory iterative mode when an SMB connection drop is
detected. Each piece is retried independently without re-prompting for the
password.

Download (default): copy an N5 dataset from the SMB share to a local directory.
Ingest (--ingest): upload a local N5 dataset to the SMB share.

Usage:
    python smb_transfer_resilient.py -u <username> -p <remote_parent_dir> -d <n5_name> [-o local_dir]
    python smb_transfer_resilient.py --ingest -u <username> -p <remote_parent_dir> -d <n5_name> -o <local_source_dir>

Pass --setup to restrict the transfer to specific setup(s) of the N5 dataset, e.g.
--setup 0 1 for setup0 and setup1, instead of the whole dataset.

Data directories that are not in N5 format are detected automatically (no top-level
setupN directories) and transferred file by file, which suits arbitrary trees with a
few large files on an unstable connection. Pass --generic to force this per-file mode.
"""

import argparse
import getpass
import os
import pathlib
import posixpath
import re
import sys
import time  # noqa: F401  (tests patch smb.time.sleep)
import warnings
from typing import Optional

# The generic SMB primitives live in the package so that flamingo_tools.convert_data can use
# the same implementation. Importing them by name keeps them patchable through this module.
from flamingo_tools.data_transfer_utils import (
    MAX_RETRIES,
    RETRY_DELAY,
    SMB_SERVER,
    UKON_OLD,
    UPLOAD_ERROR_TOKENS,
    append_log,
    build_remote_size_map,
    ensure_remote_path,
    list_remote_dirs,
    remote_dir_exists,
    remote_size_map_with_retry,
    run_smbclient,
    run_with_retry,
    transfer_path,
    upload_path,
)

# Kept as a module-level alias: the existing tests and call sites use the private name.
_run_with_retry = run_with_retry


def _sort_key(name):
    return int(name) if name.isdigit() else name


def _normalize_setup(value: str) -> str:
    """Normalize a --setup CLI value to 'setupN' form (accepts '0' or 'setup0').

    Args:
        value: Raw --setup value from the command line.

    Returns:
        Normalized setup directory name, e.g. "setup0".
    """
    value = value.strip()
    if re.match(r"^setup\d+$", value):
        return value
    if value.isdigit():
        return f"setup{value}"
    raise ValueError(f"invalid --setup value: {value!r} (expected a number or 'setupN')")


def find_truncated_chunks(
    local_n5: str, min_bytes: int = 4, subdirs: Optional[list[str]] = None,
) -> list[str]:
    """Find N5 chunk files too small to contain a valid header.

    Mirrors the check zarr's N5 decoder performs (struct.unpack(">H", chunk[2:4])), which
    requires every chunk file to be at least 4 bytes. A file below that is almost always
    left over from an interrupted mget mid-transfer (frequently 0 bytes).

    Args:
        local_n5: Local root directory of the transferred N5 dataset.
        min_bytes: Minimum valid chunk file size.
        subdirs: Restrict the scan to these subdirectories of local_n5 (e.g. requested
            setups). Scans the whole local_n5 tree when not given.

    Returns:
        Paths of undersized chunk files, relative to local_n5.
    """
    bad = []
    roots = [os.path.join(local_n5, s) for s in subdirs] if subdirs else [local_n5]
    for root_dir in roots:
        if not os.path.isdir(root_dir):
            continue
        for root, _dirs, files in os.walk(root_dir):
            for fname in files:
                if fname == "attributes.json":
                    continue
                path = os.path.join(root, fname)
                if os.path.getsize(path) < min_bytes:
                    bad.append(os.path.relpath(path, local_n5))
    return bad


def verify_and_repair_n5(
    username: str,
    password: str,
    remote_dir: str,
    n5_name: str,
    output_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    max_passes: int = 3,
    setup_filter: Optional[list[str]] = None,
) -> None:
    """Scan a transferred N5 dataset for truncated chunk files and re-fetch them individually.

    A chunk file corrupted by a mid-transfer disconnect (typically left at 0 bytes) can
    survive a later `mget` retry of its parent directory, since that retry may itself be
    interrupted, or the specific file skipped without affecting the overall exit code. This
    runs as an independent pass after the transfer finishes: every chunk file's size is
    checked against the minimum N5 header size, and any undersized file is re-downloaded on
    its own. Runs for up to `max_passes` rounds, since a freshly repaired file can be
    corrupted again by a new disconnect.

    Reuses the password already held in memory from the initial transfer — no additional
    password prompt is needed.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote parent directory (already forward-slash normalised).
        n5_name: Name of the N5 dataset (top-level directory name).
        output_dir: Local output directory containing the transferred N5 dataset.
        log_file: File to log chunks that remain corrupt after all repair passes.
        smb_server: SMB server to connect to.
        max_passes: Maximum number of verify-and-repair rounds.
        setup_filter: Restrict verification to these setup(s) (e.g. ["setup0"]).
            Verifies the whole dataset when not given.
    """
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{n5_name}"
    local_n5 = os.path.join(output_dir, n5_name)

    if not os.path.isdir(local_n5):
        print(f"  [warn] local N5 directory not found for verification: {local_n5}")
        return

    for attempt in range(1, max_passes + 1):
        bad_files = find_truncated_chunks(local_n5, subdirs=setup_filter)
        if not bad_files:
            label = "all chunk files look complete" if attempt == 1 else "all chunk files repaired"
            print(f"\n=== Verification pass {attempt}: {label} ===")
            return

        print(f"\n=== Verification pass {attempt}: found {len(bad_files)} truncated chunk file(s) ===")
        for rel_path in bad_files:
            rel_dir = os.path.dirname(rel_path)
            fname = os.path.basename(rel_path)
            remote_cd = f"{full_remote}/{rel_dir}" if rel_dir else full_remote
            local_cwd = os.path.join(local_n5, rel_dir) if rel_dir else local_n5
            print(f"  -> re-fetching {rel_path}")
            transfer_path(
                username, password,
                remote_cd=remote_cd,
                mget_target=fname,
                local_cwd=local_cwd,
                log_file=log_file,
                smb_server=smb_server,
            )

    remaining = find_truncated_chunks(local_n5, subdirs=setup_filter)
    if remaining:
        print(f"\n[error] {len(remaining)} chunk file(s) still truncated after {max_passes} repair passes:")
        for rel_path in remaining:
            print(f"  {rel_path}")
        for rel_path in remaining:
            append_log(log_file, f"[error] chunk file still truncated after repair: {rel_path}")


def iterative_n5_transfer(
    username: str,
    password: str,
    remote_dir: str,
    n5_name: str,
    output_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    setup_filter: Optional[list[str]] = None,
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
        setup_filter: Restrict transfer to these setup(s) (e.g. ["setup0"]). Transfers
            all discovered setups when not given.

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

    if setup_filter:
        missing = [s for s in setup_filter if s not in setup_names]
        for s in missing:
            print(f"  [warn] requested setup not found remotely: {s}")
        setup_names = [s for s in setup_names if s in setup_filter]
        if not setup_names:
            print("  [warn] none of the requested setups were found remotely — nothing to transfer")
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


def _local_dirs(path: str, pattern: Optional[str] = None) -> list[str]:
    """Return sorted subdirectory names of a local directory, optionally filtered by a regex."""
    if not os.path.isdir(path):
        return []
    names = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    if pattern is not None:
        names = [d for d in names if re.match(pattern, d)]
    return sorted(names, key=_sort_key)


def _local_files(path: str) -> list[str]:
    """Return sorted file names directly inside a local directory."""
    if not os.path.isdir(path):
        return []
    return sorted(f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f)))


def _smb_quote(name: str) -> str:
    """Wrap a single file name in double quotes so a name with spaces stays one mget/put mask."""
    return f'"{name}"'


def _put_top_level_files(
    username, password, remote_dir, local_dir, log_file=None, smb_server=SMB_SERVER, quote=False,
):
    """Upload every file directly inside local_dir into an already-existing remote_dir.

    Set quote=True for arbitrary trees whose file names may contain spaces.
    """
    for fname in _local_files(local_dir):
        print(f"    -> put {fname}")
        upload_path(
            username, password, remote_dir=remote_dir,
            local_target=_smb_quote(fname) if quote else fname,
            local_cwd=local_dir, is_dir=False, ensure=False,
            log_file=log_file, smb_server=smb_server,
        )


def iterative_n5_upload(
    username: str,
    password: str,
    remote_dir: str,
    n5_name: str,
    source_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    setup_filter: Optional[list[str]] = None,
    base: Optional[str] = None,
):
    """Phase 2 (ingest): upload an N5 dataset setup-by-setup, scale-by-scale.

    Mirror of iterative_n5_transfer for the upload direction. The dataset
    structure is discovered by walking the local source tree. For s0, s1, s2,
    and s3 each top-level chunk directory is uploaded individually so a single
    disconnect only affects one small piece. All other scales are uploaded as a
    single unit. Every attributes.json is uploaded explicitly, including the
    scale-level files that the per-chunk mput does not cover.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote parent directory (already forward-slash normalised).
        n5_name: Name of the N5 dataset (top-level directory name).
        source_dir: Local directory containing the N5 dataset.
        log_file: Log file to store files which were not transferred.
        smb_server: SMB server to connect to.
        setup_filter: Restrict upload to these setup(s) (e.g. ["setup0"]).
            Uploads all discovered setups when not given.
        base: Prefix assumed to already exist on the share. Defaults to remote_dir.
    """
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{n5_name}"
    local_n5 = os.path.join(source_dir, n5_name)
    base = base if base is not None else remote_dir

    print("\n=== Iterative N5 ingest mode ===")

    ensure_remote_path(username, password, base=base, target=full_remote,
                       local_cwd=local_n5, smb_server=smb_server)

    # Root-level files (attributes.json).
    print(f"\n-- {n5_name}/ (root files)")
    _put_top_level_files(username, password, full_remote, local_n5,
                         log_file=log_file, smb_server=smb_server)

    setup_names = _local_dirs(local_n5, r"^setup\d+$")
    if not setup_names:
        print("  [warn] no setup* directories found locally — nothing to transfer")
        return

    if setup_filter:
        missing = [s for s in setup_filter if s not in setup_names]
        for s in missing:
            print(f"  [warn] requested setup not found locally: {s}")
        setup_names = [s for s in setup_names if s in setup_filter]
        if not setup_names:
            print("  [warn] none of the requested setups were found locally — nothing to transfer")
            return

    print(f"\n  Found setups: {setup_names}")

    for setup in setup_names:
        print(f"\n{'=' * 60}\nUploading {setup}\n{'=' * 60}")

        setup_remote = f"{full_remote}/{setup}"
        setup_local = os.path.join(local_n5, setup)
        ensure_remote_path(username, password, base=full_remote, target=setup_remote,
                           local_cwd=setup_local, smb_server=smb_server)
        _put_top_level_files(username, password, setup_remote, setup_local,
                             log_file=log_file, smb_server=smb_server)

        for tp in _local_dirs(setup_local):
            tp_remote = f"{setup_remote}/{tp}"
            tp_local = os.path.join(setup_local, tp)
            ensure_remote_path(username, password, base=setup_remote, target=tp_remote,
                               local_cwd=tp_local, smb_server=smb_server)
            _put_top_level_files(username, password, tp_remote, tp_local,
                                 log_file=log_file, smb_server=smb_server)

            scale_names = _local_dirs(tp_local, r"^s\d+$")
            if not scale_names:
                print(f"  [warn] no scale directories found in {setup}/{tp}")
                continue

            print(f"  Scales: {scale_names}")

            for scale in scale_names:
                scale_remote = f"{tp_remote}/{scale}"
                scale_local = os.path.join(tp_local, scale)

                if scale in ["s0", "s1", "s2", "s3"]:
                    # Upload each top-level chunk directory individually.
                    print(f"\n  -- {setup}/{tp}/{scale}  (per-subdirectory mode)")
                    ensure_remote_path(username, password, base=tp_remote, target=scale_remote,
                                       local_cwd=scale_local, smb_server=smb_server)
                    _put_top_level_files(username, password, scale_remote, scale_local,
                                         log_file=log_file, smb_server=smb_server)
                    chunk_dirs = _local_dirs(scale_local)
                    if not chunk_dirs:
                        print(f"  [warn] no chunk directories found in {scale}")
                        continue
                    print(f"{scale} chunk directories: {chunk_dirs[0]} … {chunk_dirs[-1]} "
                          f"({len(chunk_dirs)} total)")
                    for chunk_dir in chunk_dirs:
                        print(f"    -> {scale}/{chunk_dir}")
                        upload_path(
                            username, password,
                            remote_dir=scale_remote,
                            local_target=chunk_dir,
                            local_cwd=scale_local,
                            is_dir=True, ensure=False,
                            log_file=log_file, smb_server=smb_server,
                        )
                else:
                    # Upload the entire scale in one shot.
                    print(f"\n  -- {setup}/{tp}/{scale}")
                    upload_path(
                        username, password,
                        remote_dir=tp_remote,
                        local_target=scale,
                        local_cwd=tp_local,
                        is_dir=True, ensure=False,
                        log_file=log_file, smb_server=smb_server,
                    )

    print("\n=== Iterative ingest complete ===")


def verify_and_repair_upload(
    username: str,
    password: str,
    remote_dir: str,
    n5_name: str,
    source_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    max_passes: int = 3,
    setup_filter: Optional[list[str]] = None,
    base: Optional[str] = None,
) -> None:
    """Verify an uploaded N5 dataset against the local source and re-upload mismatches.

    Inverse of verify_and_repair_n5. A file is re-uploaded when it is missing on
    the share or its remote size differs from the local size (a mid-transfer
    disconnect can leave a short remote file that a later mput retry does not
    fix). Runs for up to max_passes rounds, since a freshly repaired file can be
    corrupted again by a new disconnect. Reuses the password already in memory.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote parent directory (already forward-slash normalised).
        n5_name: Name of the N5 dataset (top-level directory name).
        source_dir: Local directory containing the N5 dataset.
        log_file: File to log files that remain mismatched after all passes.
        smb_server: SMB server to connect to.
        max_passes: Maximum number of verify-and-repair rounds.
        setup_filter: Restrict verification to these setup(s). Verifies the whole
            dataset when not given.
        base: Prefix assumed to already exist on the share. Defaults to remote_dir.
    """
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{n5_name}"
    local_n5 = os.path.join(source_dir, n5_name)
    base = base if base is not None else remote_dir

    if not os.path.isdir(local_n5):
        print(f"  [warn] local N5 directory not found for verification: {local_n5}")
        return

    def _local_rel_files() -> list[str]:
        if setup_filter:
            roots = [os.path.join(local_n5, s) for s in setup_filter]
            rel_files = _local_files(local_n5)  # root-level files (attributes.json)
        else:
            roots = [local_n5]
            rel_files = []
        for root_dir in roots:
            if not os.path.isdir(root_dir):
                continue
            for root, _dirs, files in os.walk(root_dir):
                for fname in files:
                    rel_files.append(os.path.relpath(os.path.join(root, fname), local_n5))
        # Normalise to posix and de-duplicate.
        return sorted({pathlib.PurePath(r).as_posix() for r in rel_files})

    for attempt in range(1, max_passes + 1):
        size_map = build_remote_size_map(username, password, full_remote, local_n5, smb_server=smb_server)
        if size_map is None:
            print("  [warn] could not list remote dataset for verification — "
                  "relying on per-piece retries instead")
            return

        bad_files = []
        for rel in _local_rel_files():
            local_size = os.path.getsize(os.path.join(local_n5, rel))
            remote_size = size_map.get(rel)
            if remote_size is None or remote_size != local_size:
                bad_files.append(rel)

        if not bad_files:
            label = "all files match" if attempt == 1 else "all files repaired"
            print(f"\n=== Verification pass {attempt}: {label} ===")
            return

        print(f"\n=== Verification pass {attempt}: found {len(bad_files)} mismatched file(s) ===")
        for rel in bad_files:
            rel_dir = os.path.dirname(rel)
            fname = os.path.basename(rel)
            remote_target = f"{full_remote}/{rel_dir}" if rel_dir else full_remote
            local_cwd = os.path.join(local_n5, rel_dir) if rel_dir else local_n5
            print(f"  -> re-uploading {rel}")
            upload_path(
                username, password,
                remote_dir=remote_target, local_target=fname, local_cwd=local_cwd,
                is_dir=False, base=base, ensure=True,
                log_file=log_file, smb_server=smb_server,
            )

    size_map = build_remote_size_map(username, password, full_remote, local_n5, smb_server=smb_server)
    remaining = []
    if size_map is not None:
        for rel in _local_rel_files():
            local_size = os.path.getsize(os.path.join(local_n5, rel))
            if size_map.get(rel) != local_size:
                remaining.append(rel)
    if remaining:
        print(f"\n[error] {len(remaining)} file(s) still mismatched after {max_passes} repair passes:")
        for rel in remaining:
            print(f"  {rel}")
        for rel in remaining:
            append_log(log_file, f"[error] file still mismatched after repair: {rel}")


def _looks_like_n5_local(path: str) -> bool:
    """Return True if a local directory has the N5 layout (top-level setupN directories)."""
    return bool(_local_dirs(path, r"^setup\d+$"))


def _looks_like_n5_remote(
    username: str,
    password: str,
    full_remote: str,
    cwd: str,
    smb_server: str = SMB_SERVER,
) -> bool:
    """Return True if a remote directory has the N5 layout (top-level setupN directories).

    On a disconnected listing this returns False, so the caller falls back to the generic
    per-file path, whose size-based verification is safe for any tree.
    """
    dirs = list_remote_dirs(username, password, full_remote, cwd, smb_server=smb_server)
    return any(re.match(r"^setup\d+$", d) for d in dirs)


def _remote_size_map_with_retry(
    username: str,
    password: str,
    full_remote: str,
    cwd: str,
    smb_server: str = SMB_SERVER,
    retries: int = MAX_RETRIES,
) -> Optional[dict]:
    """Call build_remote_size_map, retrying on a dropped listing (the listing is cheap metadata)."""
    return remote_size_map_with_retry(
        username, password, full_remote, cwd, retries=retries, smb_server=smb_server,
    )


def generic_iterative_download(
    username: str,
    password: str,
    remote_dir: str,
    data_name: str,
    output_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
) -> None:
    """Download an arbitrary (non-N5) directory tree from the share, one file at a time.

    Each file is an independent retry unit, which suits trees with a few very large files on
    an unstable connection. The remote tree is discovered with a single recursive listing.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote parent directory (already forward-slash normalised).
        data_name: Name of the directory to download (top-level directory name).
        output_dir: Local directory that receives the tree.
        log_file: File to log files which were not transferred.
        smb_server: SMB server to connect to.
    """
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{data_name}"
    local_root = os.path.join(output_dir, data_name)

    print("\n=== Generic per-file download mode ===")
    size_map = _remote_size_map_with_retry(username, password, full_remote, output_dir,
                                           smb_server=smb_server)
    if not size_map:
        print("  [warn] could not list the remote tree — nothing transferred")
        return

    print(f"  Found {len(size_map)} file(s) to transfer")
    for rel in sorted(size_map):
        reldir, fname = posixpath.split(rel)
        remote_cd = f"{full_remote}/{reldir}" if reldir else full_remote
        local_cwd = os.path.join(local_root, *reldir.split("/")) if reldir else local_root
        print(f"    -> {rel}")
        transfer_path(
            username, password,
            remote_cd=remote_cd, mget_target=_smb_quote(fname), local_cwd=local_cwd,
            log_file=log_file, smb_server=smb_server,
        )

    print("\n=== Generic download complete ===")


def verify_and_repair_download_generic(
    username: str,
    password: str,
    remote_dir: str,
    data_name: str,
    output_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    max_passes: int = 3,
) -> None:
    """Verify a downloaded tree against remote file sizes and re-fetch mismatches.

    Download twin of verify_and_repair_upload: the remote listing is authoritative, and every
    local file that is missing or whose size differs is re-fetched individually. This catches a
    truncated large file that the N5 <4-byte chunk check cannot see.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote parent directory (already forward-slash normalised).
        data_name: Name of the downloaded directory (top-level directory name).
        output_dir: Local directory containing the downloaded tree.
        log_file: File to log files that remain mismatched after all passes.
        smb_server: SMB server to connect to.
        max_passes: Maximum number of verify-and-repair rounds.
    """
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{data_name}"
    local_root = os.path.join(output_dir, data_name)

    def _refetch(rel):
        reldir, fname = posixpath.split(rel)
        transfer_path(
            username, password,
            remote_cd=f"{full_remote}/{reldir}" if reldir else full_remote,
            mget_target=_smb_quote(fname),
            local_cwd=os.path.join(local_root, *reldir.split("/")) if reldir else local_root,
            log_file=log_file, smb_server=smb_server,
        )

    def _bad_files(size_map):
        bad = []
        for rel, rsize in size_map.items():
            lp = os.path.join(local_root, *rel.split("/"))
            if not os.path.exists(lp) or os.path.getsize(lp) != rsize:
                bad.append(rel)
        return bad

    for attempt in range(1, max_passes + 1):
        size_map = build_remote_size_map(username, password, full_remote, output_dir,
                                         smb_server=smb_server)
        if size_map is None:
            print("  [warn] could not list the remote tree for verification — "
                  "relying on per-file retries instead")
            return
        bad = _bad_files(size_map)
        if not bad:
            label = "all files match" if attempt == 1 else "all files repaired"
            print(f"\n=== Verification pass {attempt}: {label} ===")
            return
        print(f"\n=== Verification pass {attempt}: found {len(bad)} mismatched/missing file(s) ===")
        for rel in bad:
            print(f"  -> re-fetching {rel}")
            _refetch(rel)

    size_map = build_remote_size_map(username, password, full_remote, output_dir, smb_server=smb_server)
    remaining = _bad_files(size_map) if size_map is not None else []
    if remaining:
        print(f"\n[error] {len(remaining)} file(s) still mismatched after {max_passes} repair passes:")
        for rel in remaining:
            print(f"  {rel}")
        for rel in remaining:
            append_log(log_file, f"[error] file still mismatched after repair: {rel}")


def generic_iterative_upload(
    username: str,
    password: str,
    remote_dir: str,
    data_name: str,
    source_dir: str,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    base: Optional[str] = None,
) -> None:
    """Upload an arbitrary (non-N5) directory tree to the share, one file at a time.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote parent directory (already forward-slash normalised).
        data_name: Name of the directory to upload (top-level directory name).
        source_dir: Local directory containing the tree.
        log_file: File to log files which were not transferred.
        smb_server: SMB server to connect to.
        base: Prefix assumed to already exist on the share. Defaults to remote_dir.
    """
    remote_dir = remote_dir.replace("\\", "/")
    full_remote = f"{remote_dir}/{data_name}"
    local_root = os.path.join(source_dir, data_name)
    base = base if base is not None else remote_dir

    print("\n=== Generic per-file ingest mode ===")
    ensure_remote_path(username, password, base=base, target=full_remote,
                       local_cwd=local_root, smb_server=smb_server)

    for root, _dirs, _files in os.walk(local_root):
        rel = os.path.relpath(root, local_root)
        if rel == ".":
            remote_subdir = full_remote
        else:
            rel_posix = pathlib.PurePath(rel).as_posix()
            remote_subdir = f"{full_remote}/{rel_posix}"
            parent_remote = posixpath.dirname(remote_subdir)
            print(f"\n  -- {data_name}/{rel_posix}")
            ensure_remote_path(username, password, base=parent_remote, target=remote_subdir,
                               local_cwd=root, smb_server=smb_server)
        _put_top_level_files(username, password, remote_subdir, root,
                             log_file=log_file, smb_server=smb_server, quote=True)

    print("\n=== Generic ingest complete ===")


def _run_download(args, password, remote_dir, n5_name, output_dir, log_file, setup_filter):
    """Download a dataset from the share (Phase 1 bulk → Phase 2 iterative → verify).

    Detects N5 vs a generic tree once, so the Phase-1-success verify and the Phase-2 fallback
    both use the matching path. A generic tree is transferred and verified per file.
    """
    full_remote = f"{remote_dir}/{n5_name}"
    is_n5 = not args.generic and (
        bool(setup_filter)
        or _looks_like_n5_remote(args.username, password, full_remote, output_dir,
                                 smb_server=args.smb_server)
    )

    def _verify():
        if is_n5:
            verify_and_repair_n5(args.username, password, remote_dir, n5_name, output_dir,
                                 log_file=log_file, smb_server=args.smb_server, setup_filter=setup_filter)
        else:
            verify_and_repair_download_generic(args.username, password, remote_dir, n5_name,
                                               output_dir, log_file=log_file, smb_server=args.smb_server)

    print("Connecting to SMB server and starting bulk transfer...")
    if setup_filter:
        local_n5_dir = os.path.join(output_dir, n5_name)
        os.makedirs(local_n5_dir, exist_ok=True)
        mget_targets = ["attributes.json"] + setup_filter
        commands = [f'cd "{remote_dir}/{n5_name}"', "recurse", "prompt"] + [f"mget {t}" for t in mget_targets]
        phase1_cwd = local_n5_dir
    else:
        commands = [f'cd "{remote_dir}"', "recurse", "prompt", f"mget {n5_name}"]
        phase1_cwd = output_dir

    _, had_disconnect, rc = run_smbclient(args.username, password, commands,
                                          cwd=phase1_cwd, smb_server=args.smb_server)

    if not had_disconnect and rc == 0:
        print("File transfer completed successfully.")
        if setup_filter:
            for s in setup_filter:
                if not os.path.isdir(os.path.join(output_dir, n5_name, s)):
                    print(f"  [warn] requested setup not found after transfer: {s}")
        _verify()
        sys.exit(0)

    if not had_disconnect:
        print(f"Transfer failed (exit code {rc}).")
        sys.exit(rc)

    print("\nDisconnect detected — switching to iterative transfer mode.")
    if is_n5:
        iterative_n5_transfer(args.username, password, remote_dir, n5_name, output_dir,
                              log_file=log_file, smb_server=args.smb_server, setup_filter=setup_filter)
    else:
        generic_iterative_download(args.username, password, remote_dir, n5_name, output_dir,
                                   log_file=log_file, smb_server=args.smb_server)
    _verify()


def _run_ingest(args, password, remote_dir, n5_name, source_dir, log_file, setup_filter):
    """Upload a local dataset to the share (Phase 1 bulk → Phase 2 iterative → verify).

    Detects N5 vs a generic tree from the local source. A generic tree is uploaded per file in
    the Phase-2 fallback; verification is size-based for both (verify_and_repair_upload).
    """
    local_n5 = os.path.join(source_dir, n5_name)
    if not os.path.isdir(local_n5):
        raise SystemExit(f"Local dataset not found: {local_n5}")

    is_n5 = not args.generic and (bool(setup_filter) or _looks_like_n5_local(local_n5))

    # The parent directory must exist so mput does not dump the dataset into the
    # share root. Create the full chain only on explicit opt-in.
    base = remote_dir
    if args.create_parents:
        ensure_remote_path(args.username, password, base="", target=remote_dir,
                           local_cwd=source_dir, smb_server=args.smb_server)
    else:
        exists = remote_dir_exists(args.username, password, remote_dir, source_dir,
                                   smb_server=args.smb_server)
        if exists is False:
            raise SystemExit(
                f"Remote parent directory does not exist: {remote_dir}\n"
                "Create it first or pass --create-parents to create it automatically."
            )

    print("Connecting to SMB server and starting bulk ingest...")
    full_remote = f"{remote_dir}/{n5_name}"
    if setup_filter:
        ensure_remote_path(args.username, password, base=base, target=full_remote,
                           local_cwd=local_n5, smb_server=args.smb_server)
        commands = [f'cd "{full_remote}"', "put attributes.json", "recurse", "prompt"] + \
                   [f"mput {s}" for s in setup_filter]
        phase1_cwd = local_n5
    else:
        commands = [f'cd "{remote_dir}"', "recurse", "prompt", f"mput {n5_name}"]
        phase1_cwd = source_dir

    lines, had_disconnect, rc = run_smbclient(args.username, password, commands,
                                              cwd=phase1_cwd, smb_server=args.smb_server)
    upload_error = any(tok in line for line in lines for tok in UPLOAD_ERROR_TOKENS)

    if not had_disconnect and rc == 0 and not upload_error:
        print("File ingest completed successfully.")
        verify_and_repair_upload(args.username, password, remote_dir, n5_name, source_dir,
                                 log_file=log_file, smb_server=args.smb_server,
                                 setup_filter=setup_filter, base=base)
        sys.exit(0)

    if not had_disconnect:
        print("Bulk ingest reported an error — switching to iterative ingest mode.")
    else:
        print("\nDisconnect detected — switching to iterative ingest mode.")
    if is_n5:
        iterative_n5_upload(args.username, password, remote_dir, n5_name, source_dir,
                            log_file=log_file, smb_server=args.smb_server,
                            setup_filter=setup_filter, base=base)
    else:
        generic_iterative_upload(args.username, password, remote_dir, n5_name, source_dir,
                                 log_file=log_file, smb_server=args.smb_server, base=base)
    verify_and_repair_upload(args.username, password, remote_dir, n5_name, source_dir,
                             log_file=log_file, smb_server=args.smb_server,
                             setup_filter=setup_filter, base=base)


def main():
    parser = argparse.ArgumentParser(
        description="Resilient SMB transfer for N5 data with automatic disconnect recovery. "
                    "Downloads from the share by default; pass --ingest to upload."
    )
    parser.add_argument("-u", "--username", help="GWDG username, e.g. schilling40")
    parser.add_argument("-p", "--remote_parent_dir", help="Remote parent directory on the SMB share")
    parser.add_argument("-d", "--remote_data", help="N5 root directory name (on the share and locally)")
    parser.add_argument("-o", "--output-dir", default=os.getcwd(),
                        help="Local directory. Download: destination for the dataset. "
                             "Ingest: parent directory that contains the dataset to upload. "
                             "Default: cwd.")
    parser.add_argument("-s", "--smb_server", type=str, default=SMB_SERVER,
                        help=f"SMB server to transfer to/from. Default: {SMB_SERVER}")
    parser.add_argument("-l", "--log_file", type=str, default=None,
                        help="Log transfer errors. Default: transfer_log.txt in output directory.")
    parser.add_argument("--setup", nargs="+", default=None,
                        help="Restrict transfer to specific setup(s) of the N5 dataset, "
                             "e.g. '--setup 0 1' for setup0 and setup1. Accepts bare numbers "
                             "or 'setupN'. Default: transfer the entire dataset.")
    parser.add_argument("--ingest", action="store_true",
                        help="Upload a local N5 dataset to the share instead of downloading.")
    parser.add_argument("--create-parents", action="store_true",
                        help="Ingest only: create the remote parent directory chain if it is "
                             "missing, instead of aborting. Use with care on a shared server.")
    parser.add_argument("--generic", action="store_true",
                        help="Force the generic per-file transfer, bypassing N5 detection. Suits "
                             "arbitrary directory trees. Non-N5 data is detected automatically too.")
    args = parser.parse_args()

    if args.generic and args.setup:
        parser.error("--generic and --setup are mutually exclusive (--setup applies to N5 data only)")

    output_dir = os.path.realpath(args.output_dir)
    if not args.ingest:
        os.makedirs(output_dir, exist_ok=True)
    elif not os.path.isdir(output_dir):
        parser.error(f"local source directory does not exist: {output_dir}")

    password = getpass.getpass("Enter password: ")

    if "\\" not in args.remote_parent_dir:
        warnings.warn("Ensure that path to parent directory contains double \\ or is quoted.")

    p = pathlib.PureWindowsPath(args.remote_parent_dir)
    remote_dir = p.as_posix()
    n5_name = args.remote_data
    log_file = args.log_file if args.log_file is not None else os.path.join(output_dir, "transfer_log.txt")

    setup_filter = None
    if args.setup:
        try:
            setup_filter = list(dict.fromkeys(_normalize_setup(s) for s in args.setup))
        except ValueError as e:
            parser.error(str(e))

    if args.ingest:
        _run_ingest(args, password, remote_dir, n5_name, output_dir, log_file, setup_filter)
    else:
        _run_download(args, password, remote_dir, n5_name, output_dir, log_file, setup_filter)


if __name__ == "__main__":
    main()
