#!/usr/bin/env python3
"""
Resilient SMB transfer for individual files and directories.

Generic counterpart to smb_transfer_resilient.py: transfers a single file, a single
directory (whole-tree passthrough), or an arbitrary list of files described by a
tab-separated manifest (columns "UKONspezial" for the remote path, "local" for the
local path), with the same disconnect-retry mechanics plus a post-transfer
empty-file check.

Download (default): share -> local.
Ingest (--ingest): local -> share.

Usage:
    python smb_transfer_files.py -u <user> -r <remote_path> -l <local_path>
    python smb_transfer_files.py -u <user> --manifest files.tsv [--ingest]
"""

import argparse
import getpass
import os
import pathlib
import posixpath
import sys
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from flamingo_tools.data_transfer_utils import (
    MAX_RETRIES,
    SMB_SERVER,
    UPLOAD_ERROR_TOKENS,
    append_log,
    ensure_remote_path,
    remote_dir_exists,
    run_with_retry,
    transfer_path,
    upload_path,
)

MANIFEST_REMOTE_COL = "UKONspezial"
MANIFEST_LOCAL_COL = "local"
EMPTY_FILE_RETRIES = 3


@dataclass
class FileJob:
    """One single-file transfer unit, direction-agnostic (--ingest picks source/dest)."""

    remote_dir: str
    remote_name: str
    local_dir: str
    local_name: str
    label: str


@dataclass
class DirJob:
    """One whole-directory transfer unit, passed through to transfer_path/upload_path.

    `transferred_name` is the basename mget/mput preserves (the remote basename on
    download, the local basename on ingest). `rename_to` is the desired final
    basename, set only when it differs from `transferred_name`.
    """

    remote_dir: str
    local_dir: str
    transferred_name: str
    rename_to: Optional[str]


def split_remote_path(path: str) -> tuple[str, str]:
    """Normalize a Windows-or-posix remote path and split into (parent_dir, basename)."""
    posix = pathlib.PureWindowsPath(path).as_posix().rstrip("/")
    return posixpath.split(posix)


def split_local_path(path: str) -> tuple[str, str]:
    """Split a local path into (parent_dir, basename)."""
    return os.path.split(path.rstrip(os.sep))


def download_file(
    username, password, remote_cd, remote_name, local_cwd, local_name,
    retries: int = MAX_RETRIES, log_file: Optional[str] = None, smb_server: str = SMB_SERVER,
) -> bool:
    """Download one remote file, renaming it locally if remote_name != local_name.

    Uses `reget`, not `get` — smbclient resumes from the local file's current size on
    `reget` instead of restarting, so a disconnect-triggered retry keeps whatever was
    already transferred. `reget` behaves like a plain `get` when no local file exists
    yet, so this is a strict improvement over `get` with no downside.
    """
    remote_cd = remote_cd.replace("\\", "/")
    os.makedirs(local_cwd, exist_ok=True)
    commands = [f'cd "{remote_cd}"', f'reget "{remote_name}" "{local_name}"']
    return run_with_retry(
        username, password, commands, local_cwd=local_cwd, label=f"{remote_cd}/{remote_name}",
        retries=retries, log_file=log_file, smb_server=smb_server, error_tokens=None,
    )


def upload_file(
    username, password, remote_cd, remote_name, local_cwd, local_name,
    ensure: bool = True, retries: int = MAX_RETRIES, log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
) -> bool:
    """Upload one local file, renaming it remotely if local_name != remote_name.

    No `reput` exists in smbclient, so a retry always re-uploads the whole file (the
    same limitation upload_path already has).
    """
    remote_cd = remote_cd.replace("\\", "/").rstrip("/")
    if ensure:
        ensure_remote_path(
            username, password, base=remote_cd.rsplit("/", 1)[0], target=remote_cd,
            local_cwd=local_cwd, retries=retries, smb_server=smb_server,
        )
    commands = [f'cd "{remote_cd}"', f'put "{local_name}" "{remote_name}"']
    return run_with_retry(
        username, password, commands, local_cwd=local_cwd, label=f"{remote_cd}/{remote_name}",
        retries=retries, log_file=log_file, smb_server=smb_server, error_tokens=UPLOAD_ERROR_TOKENS,
    )


def transfer_single_file(
    username, password, job: FileJob, ingest: bool, retries: int = MAX_RETRIES,
    log_file: Optional[str] = None, smb_server: str = SMB_SERVER,
    empty_retries: int = EMPTY_FILE_RETRIES, ensure_remote: bool = True,
) -> bool:
    """Transfer one FileJob, then verify the result is non-empty.

    Download: retries the whole transfer up to `empty_retries` times if the local
    file is empty despite an apparently successful attempt (that budget is separate
    from `retries`, which only covers disconnects inside a single attempt). Ingest:
    checks the local source before uploading — an empty source is skipped, not
    retried, since there is nothing to try again.
    """
    if ingest:
        src_path = os.path.join(job.local_dir, job.local_name)
        if os.path.getsize(src_path) == 0:
            message = f"[warn] {job.label}: local source file is empty — skipping upload"
            print(f"  {message}")
            append_log(log_file, message)
            return False
        return upload_file(
            username, password, job.remote_dir, job.remote_name, job.local_dir, job.local_name,
            ensure=ensure_remote, retries=retries, log_file=log_file, smb_server=smb_server,
        )

    dest_path = os.path.join(job.local_dir, job.local_name)
    for attempt in range(1, empty_retries + 1):
        ok = download_file(
            username, password, job.remote_dir, job.remote_name, job.local_dir, job.local_name,
            retries=retries, log_file=log_file, smb_server=smb_server,
        )
        if not ok:
            return False
        if os.path.getsize(dest_path) > 0:
            return True
        print(f"  [warn] {job.label}: downloaded file is empty (attempt {attempt}/{empty_retries})")
        if attempt < empty_retries:
            print(f"  [retry {attempt + 1}/{empty_retries}] re-downloading {job.label} (empty result)")

    message = f"[error] {job.label}: still empty after {empty_retries} attempts — skipping"
    print(f"  {message}")
    append_log(log_file, message)
    return False


def _warn_empty_files_in_tree(root: str, log_file: Optional[str] = None) -> None:
    """Log a [warn] for every zero-byte file under root.

    Reporting only, no automatic re-fetch of individual files — that is what the
    N5-specific verify-and-repair passes in smb_transfer_resilient.py already do for
    the case that needs it.
    """
    for dirpath, _dirs, files in os.walk(root):
        for fname in files:
            path = os.path.join(dirpath, fname)
            if os.path.getsize(path) == 0:
                message = f"[warn] {path} is empty after transfer"
                print(f"  {message}")
                append_log(log_file, message)


def transfer_directory(
    username, password, job: DirJob, ingest: bool, retries: int = MAX_RETRIES,
    log_file: Optional[str] = None, smb_server: str = SMB_SERVER,
) -> bool:
    """Transfer a whole directory tree, passed straight through to transfer_path/upload_path."""
    if not ingest:
        ok = transfer_path(
            username, password, job.remote_dir, job.transferred_name, job.local_dir,
            retries=retries, log_file=log_file, smb_server=smb_server,
        )
        if not ok:
            return False
        transferred_path = os.path.join(job.local_dir, job.transferred_name)
        _warn_empty_files_in_tree(transferred_path, log_file)
        if job.rename_to:
            dst = os.path.join(job.local_dir, job.rename_to)
            if os.path.exists(dst):
                print(f"  [warn] cannot rename {transferred_path} to {dst}: destination already exists")
            else:
                os.rename(transferred_path, dst)
                print(f"  [info] renamed {transferred_path} -> {dst}")
        return True

    if job.rename_to:
        print(
            f"  [warn] ingest of a whole directory cannot rename the remote target "
            f"('{job.transferred_name}' -> requested '{job.rename_to}'); uploading "
            f"under the original name '{job.transferred_name}' instead"
        )
    return upload_path(
        username, password, job.remote_dir, job.transferred_name, job.local_dir,
        is_dir=True, retries=retries, log_file=log_file, smb_server=smb_server,
    )


def _classify_local(path: str) -> str:
    """Return 'dir', 'file', or 'missing' for a local filesystem path."""
    if os.path.isdir(path):
        return "dir"
    if os.path.isfile(path):
        return "file"
    return "missing"


def _classify_remote(username, password, path: str, smb_server: str) -> str:
    """Return 'dir' or 'file' for a remote path.

    A non-directory is always treated as 'file': remote_dir_exists cannot
    distinguish a plain file from a missing path without an extra remote round
    trip, and a genuinely missing file simply fails (and logs) at transfer time.
    """
    is_dir = remote_dir_exists(username, password, path, os.getcwd(), smb_server=smb_server)
    return "dir" if is_dir else "file"


def _build_file_job(args) -> FileJob:
    """Both --remote and --local are exact paths; basenames may differ."""
    remote_dir, remote_name = split_remote_path(args.remote)
    local_dir, local_name = split_local_path(args.local)
    return FileJob(remote_dir, remote_name, local_dir, local_name, label=local_name)


def _build_file_job_into_dir(args, local_is_dir: bool) -> FileJob:
    """One side is a plain file, the other an existing directory to place it into.

    Borrows the file's basename for the directory side, mirroring the dir-vs-file
    idiom in flamingo_tools/extract_block_util.py.
    """
    if local_is_dir:
        remote_dir, remote_name = split_remote_path(args.remote)
        local_dir, local_name = args.local.rstrip(os.sep) or os.sep, remote_name
    else:
        local_dir, local_name = split_local_path(args.local)
        remote_dir, remote_name = args.remote.rstrip("/") or "/", local_name
    return FileJob(remote_dir, remote_name, local_dir, local_name, label=local_name)


def _build_dir_job(args, ingest: bool) -> DirJob:
    remote_parent, remote_base = split_remote_path(args.remote)
    local_parent, local_base = split_local_path(args.local)
    if ingest:
        return DirJob(
            remote_dir=remote_parent, local_dir=local_parent, transferred_name=local_base,
            rename_to=remote_base if remote_base != local_base else None,
        )
    return DirJob(
        remote_dir=remote_parent, local_dir=local_parent, transferred_name=remote_base,
        rename_to=local_base if local_base != remote_base else None,
    )


def resolve_endpoints(args, username, password, parser):
    """Classify --remote/--local and build the single job to run.

    Returns ("dir", DirJob) or ("file", FileJob).
    """
    local_kind = _classify_local(args.local)
    if args.ingest and local_kind == "missing":
        parser.error(f"local source does not exist: {args.local}")

    remote_kind = _classify_remote(username, password, args.remote, args.smb_server)

    if args.ingest:
        if local_kind == "dir":
            if remote_kind == "file":
                parser.error(f"--local is a directory but --remote is an existing file: {args.remote}")
            return "dir", _build_dir_job(args, ingest=True)
        if remote_kind == "dir":
            return "file", _build_file_job_into_dir(args, local_is_dir=False)
        return "file", _build_file_job(args)

    if remote_kind == "dir":
        if local_kind == "file":
            parser.error(f"--remote is a directory but --local is an existing file: {args.local}")
        return "dir", _build_dir_job(args, ingest=False)
    if local_kind == "dir":
        return "file", _build_file_job_into_dir(args, local_is_dir=True)
    return "file", _build_file_job(args)


def read_manifest(manifest_path: str, log_file: Optional[str] = None) -> list[FileJob]:
    """Read a tab-separated manifest into a list of FileJob.

    Header columns: 'UKONspezial' (remote path) and 'local' (local path). A row with
    either column blank is logged and skipped rather than aborting the whole run,
    matching the log-and-continue philosophy already used by run_with_retry.
    """
    table = pd.read_csv(manifest_path, sep="\t")
    missing_cols = {MANIFEST_REMOTE_COL, MANIFEST_LOCAL_COL} - set(table.columns)
    if missing_cols:
        raise ValueError(f"manifest {manifest_path} is missing column(s): {sorted(missing_cols)}")

    jobs = []
    for i, row in table.iterrows():
        remote_value = row[MANIFEST_REMOTE_COL]
        local_value = row[MANIFEST_LOCAL_COL]
        if pd.isna(remote_value) or pd.isna(local_value):
            missing_col = MANIFEST_REMOTE_COL if pd.isna(remote_value) else MANIFEST_LOCAL_COL
            message = f"[error] manifest row {i}: missing '{missing_col}' value — skipping"
            print(f"  {message}")
            append_log(log_file, message)
            continue
        remote_dir, remote_name = split_remote_path(str(remote_value))
        local_dir, local_name = split_local_path(str(local_value))
        jobs.append(FileJob(
            remote_dir, remote_name, local_dir, local_name, label=f"manifest row {i}: {local_name}",
        ))
    return jobs


def validate_args(args, parser) -> None:
    have_remote_local = args.remote is not None or args.local is not None
    if args.manifest is not None and have_remote_local:
        parser.error("--manifest and --remote/--local are mutually exclusive.")
    if args.manifest is None and not (args.remote is not None and args.local is not None):
        parser.error("either --manifest, or both --remote and --local, must be given.")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Resilient SMB transfer for individual files. Downloads from the share "
            "by default; pass --ingest to upload. Transfers a single file, a single "
            "directory (whole-tree passthrough), or a list of files described by a "
            "tab-separated manifest."
        )
    )
    parser.add_argument("-u", "--username", required=True, help="GWDG username, e.g. schilling40")
    parser.add_argument(
        "-r", "--remote", default=None,
        help="Path on the SMB share. File or directory. Source on download, "
             "destination on --ingest. Mutually exclusive with --manifest.",
    )
    parser.add_argument(
        "-l", "--local", default=None,
        help="Local filesystem path. File or directory. Destination on download, "
             "source on --ingest. If it names an existing directory (download) or "
             "--remote names an existing directory (ingest), the missing side's "
             "file name is taken from the other side's basename. Mutually exclusive "
             "with --manifest.",
    )
    parser.add_argument(
        "--manifest", default=None,
        help="Tab-separated file listing one transfer per row, with header columns "
             "'UKONspezial' (remote path) and 'local' (local path). Direction is "
             "controlled by --ingest, same as for --remote/--local. Mutually "
             "exclusive with --remote/--local.",
    )
    parser.add_argument("--ingest", action="store_true", help="Upload to the share instead of downloading.")
    parser.add_argument(
        "-s", "--smb_server", type=str, default=SMB_SERVER,
        help=f"SMB server to transfer to/from. Default: {SMB_SERVER}",
    )
    parser.add_argument(
        "--log_file", type=str, default=None,
        help="Log transfer errors. Default: transfer_log.txt next to --local "
             "(single-transfer mode) or in the current directory (manifest mode).",
    )
    parser.add_argument(
        "--retries", type=int, default=MAX_RETRIES,
        help=f"Maximum smbclient retry attempts per file on disconnect. Default: {MAX_RETRIES}.",
    )
    parser.add_argument(
        "--empty-retries", type=int, default=EMPTY_FILE_RETRIES,
        help="Maximum extra attempts to re-download a file that came back empty "
             f"despite an apparently successful transfer. Download only. Default: {EMPTY_FILE_RETRIES}.",
    )
    args = parser.parse_args()
    validate_args(args, parser)

    password = getpass.getpass("Enter password: ")

    if args.manifest is not None:
        log_file = args.log_file if args.log_file is not None else os.path.join(os.getcwd(), "transfer_log.txt")
        jobs = read_manifest(args.manifest, log_file=log_file)
        n_total = len(jobs)
        n_ok = 0
        ensured_dirs: set = set()
        for job in jobs:
            ensure_remote = job.remote_dir not in ensured_dirs
            ok = transfer_single_file(
                args.username, password, job, ingest=args.ingest, retries=args.retries,
                log_file=log_file, smb_server=args.smb_server, empty_retries=args.empty_retries,
                ensure_remote=ensure_remote,
            )
            ensured_dirs.add(job.remote_dir)
            n_ok += int(ok)
        print(f"[summary] {n_ok}/{n_total} files transferred successfully")
        sys.exit(0 if n_ok == n_total else 1)

    local_dest_dir = args.local if os.path.isdir(args.local) else os.path.dirname(os.path.abspath(args.local))
    log_file = args.log_file if args.log_file is not None else os.path.join(local_dest_dir, "transfer_log.txt")

    kind, job = resolve_endpoints(args, args.username, password, parser)
    if kind == "dir":
        ok = transfer_directory(
            args.username, password, job, ingest=args.ingest, retries=args.retries,
            log_file=log_file, smb_server=args.smb_server,
        )
    else:
        ok = transfer_single_file(
            args.username, password, job, ingest=args.ingest, retries=args.retries,
            log_file=log_file, smb_server=args.smb_server, empty_retries=args.empty_retries,
        )
    print(f"[summary] transfer {'succeeded' if ok else 'failed'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
