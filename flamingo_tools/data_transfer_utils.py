"""Resilient data transfer and file access over unstable connections.

Two independent layers live here:

- SMB helpers that drive the ``smbclient`` binary. They need ``smbclient`` at call time only,
  so this module still imports on a system without it (for example Windows).
- Local I/O helpers that retry plain file reads. Use them for data on a mounted network drive.
  Never memory-map such data: a page fault that fails on a memory-mapped network file raises
  ``EXCEPTION_IN_PAGE_ERROR`` on Windows or ``SIGBUS`` on Linux. Neither is a catchable Python
  exception, so the process dies without a traceback.
"""

import errno
import os
import re
import subprocess
import time

from dataclasses import dataclass
from typing import Any, Callable, Optional

UKON_OLD = "//wfs-medizin.top.gwdg.de/ukon-all$/ukon100"
SMB_SERVER = "//wfs-medizin-spezial.top.gwdg.de/ukon-all$"
MAX_RETRIES = 10
RETRY_DELAY = 5  # seconds

# The per-operation timeout passed to smbclient with -t. Its own default is 20 seconds, which a
# multi-GB sequential read exceeds; the man page recommends raising it when requests time out.
SMB_TIMEOUT = 60  # seconds

# smbclient often exits 0 even when a cd/mput/put failed, so upload success
# cannot rely on the return code alone. These tokens in the streamed output
# mark a failed upload unit.
UPLOAD_ERROR_TOKENS = (
    "NT_STATUS_OBJECT_NAME_NOT_FOUND",
    "NT_STATUS_OBJECT_PATH_NOT_FOUND",
    "NT_STATUS_ACCESS_DENIED",
    "NT_STATUS_NO_SUCH_FILE",
)

# A transfer can stop in the middle of a file while smbclient still exits 0, so the return code
# alone cannot be trusted. These tokens mean the session itself is broken and the only correct
# answer is to reconnect and try again. Keep per-file errors such as NT_STATUS_NO_SUCH_FILE out of
# this tuple: with recurse on, one missing file inside a directory transfer would otherwise mark
# the whole unit failed and retry forever.
TRANSPORT_ERROR_TOKENS = (
    "NT_STATUS_CONNECTION_DISCONNECTED",
    "NT_STATUS_IO_TIMEOUT",
    "NT_STATUS_CONNECTION_RESET",
    "NT_STATUS_CONNECTION_ABORTED",
    "NT_STATUS_NETWORK_NAME_DELETED",
    "NT_STATUS_USER_SESSION_DELETED",
    "NT_STATUS_UNEXPECTED_NETWORK_ERROR",
)


def append_log(log_file: Optional[str], message: str) -> None:
    """Append a message to a log file, ignoring any error.

    A failure to log must never abort a transfer or a conversion.

    Args:
        log_file: File to append to. Nothing happens if this is None.
        message: Message to append. A newline is added.
    """
    if log_file is None:
        return
    try:
        with open(log_file, "a") as file:
            file.write(f"{message}\n")
    except Exception as e:
        print(f"Error: {e}")


#
# Local file access with retry, for data on a mounted network drive.
#


@dataclass
class RetryConfig:
    """Retry behavior for local file access on an unstable mount.

    Attributes:
        max_retries: Maximal number of attempts per operation.
        retry_delay: Delay before the second attempt, in seconds.
        max_retry_delay: Upper bound for the delay, in seconds.
        backoff: Factor applied to the delay after every failed attempt.
            The SMB helpers use a fixed delay. A dropped mount can need tens of seconds to
            re-establish, so the delay grows here while the first retry stays fast.
        log_file: File that records every failed attempt.
    """

    max_retries: int = MAX_RETRIES
    retry_delay: float = RETRY_DELAY
    max_retry_delay: float = 60.0
    backoff: float = 2.0
    log_file: Optional[str] = None


def retry_io(func: Callable[[], Any], label: str, config: Optional[RetryConfig] = None) -> Any:
    """Call a function and retry it with exponential backoff on an I/O error.

    Args:
        func: Function to call. It must take no argument and must be safe to repeat.
        label: Short name of the operation, used in the messages.
        config: Retry behavior. Defaults are used if this is None.

    Returns:
        The return value of func.

    Raises:
        OSError: The last error, after all attempts failed.
    """
    config = RetryConfig() if config is None else config
    last_error = None

    for attempt in range(1, config.max_retries + 1):
        try:
            return func()
        # TimeoutError and the Windows network errors are all subclasses of OSError.
        except OSError as error:
            last_error = error
            message = f"  [warn] I/O error on {label} (attempt {attempt}/{config.max_retries}): {error}"
            print(message)
            append_log(config.log_file, message)
            if attempt == config.max_retries:
                break
            delay = min(config.retry_delay * config.backoff ** (attempt - 1), config.max_retry_delay)
            print(f"  [retry {attempt + 1}/{config.max_retries}] {label} in {delay:.0f}s")
            time.sleep(delay)

    message = f"  [error] {label} failed after {config.max_retries} attempts"
    print(message)
    append_log(config.log_file, message)
    raise last_error


def wait_for_path(path: str, config: Optional[RetryConfig] = None) -> os.stat_result:
    """Stat a path, retrying until the mount answers.

    On Windows this stat is what makes the operating system re-establish a dropped mapped drive.

    Args:
        path: The file path to stat.
        config: Retry behavior.

    Returns:
        The stat result.
    """
    return retry_io(lambda: os.stat(path), f"stat {path}", config)


def read_into(path: str, offset: int, buf: Any, config: Optional[RetryConfig] = None) -> int:
    """Read from a file into a buffer, retrying the whole buffer on an I/O error.

    The file handle is opened again on every attempt. A handle that was open when the connection
    dropped stays broken after the reconnect.

    Args:
        path: The file path to read from.
        offset: Byte offset in the file where the read starts.
        buf: Writable buffer to fill completely. A numpy array or a memoryview.
        config: Retry behavior.

    Returns:
        The number of bytes read, which always equals the size of buf.
    """
    view = buf if isinstance(buf, memoryview) else memoryview(buf)
    if view.format != "B":
        view = view.cast("B")
    n_bytes = view.nbytes

    def _read():
        n_read = 0
        with open(path, "rb") as f:
            f.seek(offset)
            while n_read < n_bytes:
                n = f.readinto(view[n_read:])
                # A short read that does not advance means the connection dropped or the file is
                # truncated. Raise so that it is retried instead of silently truncating the data.
                if not n:
                    raise OSError(
                        errno.EIO,
                        f"Short read from {path}: got {n_read} of {n_bytes} bytes at offset {offset}",
                    )
                n_read += n
        return n_read

    return retry_io(_read, f"read {os.path.basename(path)} at offset {offset}", config)


def copy_file_resilient(
    src: str,
    dst: str,
    config: Optional[RetryConfig] = None,
    block_size: int = 64 << 20,
) -> int:
    """Copy a file block by block, resuming from the bytes already present in the destination.

    Args:
        src: Source file path, typically on a mounted network drive.
        dst: Destination file path on local storage.
        config: Retry behavior.
        block_size: Size of one copied block in bytes.

    Returns:
        The number of bytes in the destination file.

    Raises:
        RuntimeError: The destination size does not match the source size after the copy.
    """
    config = RetryConfig() if config is None else config
    src_size = wait_for_path(src, config).st_size

    dst_dir = os.path.dirname(os.path.abspath(dst))
    os.makedirs(dst_dir, exist_ok=True)

    done = os.path.getsize(dst) if os.path.exists(dst) else 0
    if done > src_size:
        print(f"  [warn] {dst} is larger than the source; starting the copy again")
        os.remove(dst)
        done = 0
    if done == src_size:
        print(f"  [skip] {os.path.basename(src)} is already staged ({src_size} bytes)")
        return done
    if done:
        print(f"  [resume] {os.path.basename(src)} at {done}/{src_size} bytes")

    buf = bytearray(block_size)
    with open(dst, "r+b" if done else "wb") as out:
        out.seek(done)
        while done < src_size:
            n = min(block_size, src_size - done)
            view = memoryview(buf)[:n]
            read_into(src, done, view, config)
            out.write(view)
            done += n

    final_size = os.path.getsize(dst)
    if final_size != src_size:
        raise RuntimeError(f"Copy of {src} to {dst} is incomplete: {final_size} of {src_size} bytes.")
    return final_size


#
# SMB access through the smbclient binary.
#


def run_smbclient(
    username: str,
    password: str,
    commands: list[str],
    cwd: str,
    smb_server: str = SMB_SERVER,
    timeout: int = SMB_TIMEOUT,
) -> tuple[list[str], bool, int]:
    """Run smbclient with the given command list; stream output in real time.
    Terminates the process immediately on the first transport failure.

    Args:
        username: GWDG username.
        password: GWDG password.
        commands: smbclient command list to run.
        cwd: Current working directory.
        smb_server: SMB server to connect to.
        timeout: The per-operation timeout in seconds, passed to smbclient with -t.

    Returns:
      lines, had_disconnect, returncode. `had_disconnect` is True for any token in
      TRANSPORT_ERROR_TOKENS, not only for a dropped connection.

    """
    cmd = ["smbclient", smb_server, "-U", f"GWDG/{username}%{password}", "-t", str(timeout)]
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
        if any(token in line for token in TRANSPORT_ERROR_TOKENS) and not had_disconnect:
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
        cwd: Local working directory for the smbclient process.
        local_fallback: Path to local data to check directory structure.
        smb_server: SMB server to connect to.

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


def run_with_retry(
    username: str,
    password: str,
    commands: list[str],
    local_cwd: str,
    label: str,
    retries: int = MAX_RETRIES,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    error_tokens: Optional[tuple[str, ...]] = None,
) -> bool:
    """Run an smbclient command list with reconnect-and-retry on disconnect.

    Shared retry scaffold for both download (mget) and upload (mput/put) units.

    Args:
        username: GWDG username.
        password: GWDG password.
        commands: smbclient command list to run each attempt.
        local_cwd: local working directory passed to smbclient.
        label: short name of the transfer unit, used in log and retry messages.
        retries: Maximal number of retries.
        log_file: File to log failed transfers.
        smb_server: SMB server to connect to.
        error_tokens: Substrings whose presence in the output marks a failed
            attempt even when the return code is 0 (smbclient exits 0 on some
            failed uploads). None disables the output scan (download behaviour).

    Returns:
        True on success, False after exhausting retries.
    """
    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"  [retry {attempt}/{retries}] {label}")
            time.sleep(RETRY_DELAY)

        lines, had_disconnect, rc = run_smbclient(
            username, password, commands, cwd=local_cwd, smb_server=smb_server,
        )

        if not had_disconnect and rc == 0:
            if error_tokens and any(tok in line for line in lines for tok in error_tokens):
                print(f"  [warn] smbclient reported an error for {label}")
            else:
                return True
        elif not had_disconnect:
            # Non-zero exit for a reason other than disconnect — still retry
            print(f"  [warn] smbclient exited with code {rc} for {label}")

    print(f"  [error] failed to transfer {label} after {retries} attempts — skipping")
    append_log(log_file, f"[error] failed to transfer {label} after {retries} attempts — skipping")
    return False


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
    """Download mget_target (file or directory) with reconnect-and-retry on disconnect.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_cd: path to cd into inside smbclient before mget.
        mget_target: name to pass to mget (supports wildcards).
        local_cwd: local directory where mget places downloaded files.
        retries: Maximal number of retries.
        log_file: File to log failed transfers.
        smb_server: SMB server to connect to.

    Returns:
        True on success, False after exhausting retries.
    """
    remote_cd = remote_cd.replace("\\", "/")
    os.makedirs(local_cwd, exist_ok=True)

    commands = [f'cd "{remote_cd}"', "recurse", "prompt", f"mget {mget_target}"]
    return run_with_retry(
        username, password, commands, local_cwd=local_cwd, label=mget_target,
        retries=retries, log_file=log_file, smb_server=smb_server, error_tokens=None,
    )


def resumable_download(
    username: str,
    password: str,
    remote_cd: str,
    name: str,
    local_cwd: str,
    expected_size: Optional[int] = None,
    retries: int = MAX_RETRIES,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
    timeout: int = SMB_TIMEOUT,
) -> bool:
    """Download one file, continuing an interrupted transfer where it stopped.

    Uses smbclient's `reget`, which restarts at the end of the local file, so an attempt that
    stops part way through a multi-GB file is continued instead of started again. Unlike
    `transfer_path` this takes a single name and does not send `recurse`, so the recursive
    directory walker is not involved.

    Success is decided by the local file size rather than by the return code, because smbclient
    can abort a transfer mid-file and still exit 0.

    NOTE: `reget` continues at the end of the local file without comparing what is already there
    against the remote file. This suits raw acquisition data, which does not change on the share.
    Delete the local file first if the remote file may have been replaced.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_cd: Remote directory to cd into before the download.
        name: Name of the file inside remote_cd.
        local_cwd: Local directory that receives the file.
        expected_size: Size of the remote file in bytes, used to decide success.
            Falls back to the smbclient return code when it is not known.
        retries: Maximal number of attempts.
        log_file: File to log a failed download.
        smb_server: SMB server to connect to.
        timeout: The per-operation timeout in seconds.

    Returns:
        True on success, False after exhausting the attempts.
    """
    remote_cd = remote_cd.replace("\\", "/")
    os.makedirs(local_cwd, exist_ok=True)
    local_path = os.path.join(local_cwd, name)

    def _local_size():
        return os.path.getsize(local_path) if os.path.exists(local_path) else 0

    commands = [f'cd "{remote_cd}"', f'reget "{name}"']

    for attempt in range(1, retries + 1):
        have = _local_size()

        if expected_size is not None:
            if have == expected_size:
                if attempt > 1:
                    print(f"  [done] {name} is complete ({expected_size} bytes)")
                return True
            if have > expected_size:
                print(f"  [warn] {name} is larger than the remote file; downloading it again")
                os.remove(local_path)
                have = 0

        if attempt > 1:
            print(f"  [retry {attempt}/{retries}] {name} from {have} bytes")
            time.sleep(RETRY_DELAY)

        _, had_disconnect, rc = run_smbclient(
            username, password, commands, cwd=local_cwd, smb_server=smb_server, timeout=timeout,
        )
        got = _local_size()

        if expected_size is None:
            if not had_disconnect and rc == 0:
                return True
        else:
            if got == expected_size:
                return True
            gained = got - have
            print(f"  [warn] {name} stopped at {got}/{expected_size} bytes (+{gained} this attempt)")
            if gained <= 0 and attempt > 1:
                append_log(log_file, f"[warn] no progress on {name} at {got} bytes")

    message = f"[error] failed to download {name} after {retries} attempts, {_local_size()} bytes"
    print(f"  {message}")
    append_log(log_file, message)
    return False


def remote_dir_exists(
    username: str,
    password: str,
    remote_path: str,
    cwd: str,
    smb_server: str = SMB_SERVER,
) -> Optional[bool]:
    """Check whether a directory exists on the SMB share.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_path: Remote directory to check.
        cwd: Local working directory for the smbclient process.
        smb_server: SMB server to connect to.

    Returns:
        True if the directory exists, False if it is missing, None if the
        connection dropped before the answer could be determined.
    """
    remote_path = remote_path.replace("\\", "/")
    lines, had_disconnect, _ = run_smbclient(
        username, password, [f'cd "{remote_path}"', "ls"], cwd, smb_server=smb_server,
    )
    if had_disconnect:
        return None
    missing_tokens = ("NT_STATUS_OBJECT_NAME_NOT_FOUND", "NT_STATUS_OBJECT_PATH_NOT_FOUND")
    if any(tok in line for line in lines for tok in missing_tokens):
        return False
    return True


def ensure_remote_path(
    username: str,
    password: str,
    base: str,
    target: str,
    local_cwd: str,
    retries: int = MAX_RETRIES,
    smb_server: str = SMB_SERVER,
) -> bool:
    """Create a remote directory (and missing parents below `base`) via mkdir.

    `mput`/`put` never create the directory they upload into, so the target
    directory must exist first. smbclient `mkdir` creates a single level only,
    so one mkdir per cumulative path component is issued. mkdir on an existing
    directory returns NT_STATUS_OBJECT_NAME_COLLISION, which is harmless, so the
    operation is idempotent. Only a disconnect triggers a retry.

    Args:
        username: GWDG username.
        password: GWDG password.
        base: Prefix assumed to already exist; never created.
        target: Full remote directory that must exist afterwards.
        local_cwd: Local working directory for the smbclient process.
        retries: Maximal number of retries on disconnect.
        smb_server: SMB server to connect to.

    Returns:
        True on success, False after exhausting retries.
    """
    base_norm = base.replace("\\", "/").strip("/")
    target_clean = target.replace("\\", "/")
    lead_slash = "/" if target_clean.startswith("/") else ""
    target_parts = target_clean.strip("/").split("/")
    base_parts = base_norm.split("/") if base_norm else []

    # Number of leading components already covered by base (0 if base is not a prefix).
    start = len(base_parts) if target_parts[:len(base_parts)] == base_parts else 0
    commands = [
        f'mkdir "{lead_slash}{"/".join(target_parts[: i + 1])}"'
        for i in range(start, len(target_parts))
    ]
    if not commands:
        return True

    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"  [retry {attempt}/{retries}] mkdir {target}")
            time.sleep(RETRY_DELAY)
        _, had_disconnect, _ = run_smbclient(
            username, password, commands, cwd=local_cwd, smb_server=smb_server,
        )
        if not had_disconnect:
            return True

    print(f"  [error] failed to create remote directory {target} after {retries} attempts")
    return False


def upload_path(
    username: str,
    password: str,
    remote_dir: str,
    local_target: str,
    local_cwd: str,
    is_dir: Optional[bool] = None,
    base: Optional[str] = None,
    ensure: bool = True,
    retries: int = MAX_RETRIES,
    log_file: Optional[str] = None,
    smb_server: str = SMB_SERVER,
) -> bool:
    """Upload a single file or directory with reconnect-and-retry on disconnect.

    Mirror of `transfer_path` for the ingest direction. A directory is uploaded
    with `recurse; prompt; mput <dir>` (creates the remote subtree). A single
    file is uploaded with `put <file>`, because with recurse ON `mput` filters
    directory names and would skip a file such as attributes.json.

    Args:
        username: GWDG username.
        password: GWDG password.
        remote_dir: Remote directory to cd into before the upload; must exist
            (created here when `ensure` is True).
        local_target: File or directory name inside local_cwd to upload.
        local_cwd: Local source directory containing local_target.
        is_dir: Whether local_target is a directory. Inferred when None.
        base: Prefix assumed to exist for `ensure_remote_path`. Defaults to the
            parent of remote_dir.
        ensure: Create remote_dir first when True.
        retries: Maximal number of retries.
        log_file: File to log failed transfers.
        smb_server: SMB server to connect to.

    Returns:
        True on success, False after exhausting retries.
    """
    remote_dir = remote_dir.replace("\\", "/").rstrip("/")
    if is_dir is None:
        is_dir = os.path.isdir(os.path.join(local_cwd, local_target))

    if ensure:
        ensure_base = base if base is not None else remote_dir.rsplit("/", 1)[0]
        ensure_remote_path(
            username, password, base=ensure_base, target=remote_dir,
            local_cwd=local_cwd, retries=retries, smb_server=smb_server,
        )

    if is_dir:
        commands = [f'cd "{remote_dir}"', "recurse", "prompt", f"mput {local_target}"]
    else:
        commands = [f'cd "{remote_dir}"', f"put {local_target}"]

    return run_with_retry(
        username, password, commands, local_cwd=local_cwd, label=local_target,
        retries=retries, log_file=log_file, smb_server=smb_server,
        error_tokens=UPLOAD_ERROR_TOKENS,
    )


def build_remote_size_map(
    username: str,
    password: str,
    full_remote: str,
    cwd: str,
    smb_server: str = SMB_SERVER,
    timeout: int = SMB_TIMEOUT,
) -> Optional[dict]:
    """Return a map of remote file paths to sizes for a remote directory tree.

    Runs a single recursive `ls` and parses the per-directory blocks smbclient
    prints. Paths are relative to full_remote, forward-slash separated, matching
    the layout of the local dataset.

    Args:
        username: GWDG username.
        password: GWDG password.
        full_remote: Remote root directory.
        cwd: Local working directory for the smbclient process.
        smb_server: SMB server to connect to.
        timeout: The per-operation timeout in seconds.

    Returns:
        {relative_path: size_in_bytes} for every remote file, or None if the
        connection dropped or the output could not be parsed.
    """
    full_remote = full_remote.replace("\\", "/")
    fr = full_remote.strip("/")
    lines, had_disconnect, _ = run_smbclient(
        username, password, [f'cd "{full_remote}"', "recurse", "prompt", "ls"],
        cwd, smb_server=smb_server, timeout=timeout,
    )
    if had_disconnect:
        return None

    size_map: dict = {}
    cur_dir = ""
    # Attribute letters are optional (a plain file may list only "name size date").
    entry_re = re.compile(r"^\s+(.+?)\s+(?:([DARHSN]+)\s+)?(\d+)\s+\w")
    for line in lines:
        # smbclient closes each listing with a "N blocks of size M. K blocks available"
        # summary line that would otherwise match the entry pattern.
        if "blocks of size" in line:
            continue
        m = entry_re.match(line)
        if m:
            name, attrs, size = m.group(1).strip(), m.group(2) or "", m.group(3)
            if name in (".", ".."):
                continue
            if "D" in attrs:
                continue
            rel = f"{cur_dir}/{name}" if cur_dir else name
            size_map[rel] = int(size)
            continue
        stripped = line.strip()
        if not stripped:
            continue
        # Directory header line, e.g. "\path\to\n5\setup0".
        header = stripped.replace("\\", "/").strip("/")
        if header == fr:
            cur_dir = ""
        elif header.startswith(fr + "/"):
            cur_dir = header[len(fr) + 1:]
        elif stripped.startswith("\\") or stripped.startswith("/"):
            cur_dir = header

    return size_map if size_map else None


def remote_size_map_with_retry(
    username: str,
    password: str,
    full_remote: str,
    cwd: str,
    retries: int = MAX_RETRIES,
    smb_server: str = SMB_SERVER,
    timeout: int = SMB_TIMEOUT,
) -> Optional[dict]:
    """Build a remote size map, retrying on disconnect.

    A recursive listing is cheap metadata, so it is worth retrying on its own.

    Args:
        username: GWDG username.
        password: GWDG password.
        full_remote: Remote root directory.
        cwd: Local working directory for the smbclient process.
        retries: Maximal number of retries.
        smb_server: SMB server to connect to.
        timeout: The per-operation timeout in seconds.

    Returns:
        {relative_path: size_in_bytes}, or None after exhausting retries.
    """
    for attempt in range(1, retries + 1):
        if attempt > 1:
            print(f"  [retry {attempt}/{retries}] listing {full_remote}")
            time.sleep(RETRY_DELAY)
        size_map = build_remote_size_map(
            username, password, full_remote, cwd, smb_server=smb_server, timeout=timeout,
        )
        if size_map is not None:
            return size_map
    return None
