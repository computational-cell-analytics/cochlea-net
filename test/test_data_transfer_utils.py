import errno
import os
import tempfile
import unittest
from unittest import mock


class FlakyFile:
    """File wrapper whose readinto fails a fixed number of times."""

    def __init__(self, wrapped, budget):
        self._wrapped = wrapped
        self._budget = budget

    def __getattr__(self, name):
        return getattr(self._wrapped, name)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return self._wrapped.__exit__(*args)

    def readinto(self, buf):
        if self._budget["failures"] > 0:
            self._budget["failures"] -= 1
            raise OSError(errno.EIO, "simulated connection loss")
        return self._wrapped.readinto(buf)


def flaky_open(target, failures):
    """Return an open() replacement that fails the first `failures` reads of `target`."""
    real_open = open
    budget = {"failures": failures}

    def _open(path, *args, **kwargs):
        handle = real_open(path, *args, **kwargs)
        if os.fspath(path) != target:
            return handle
        return FlakyFile(handle, budget)

    return _open, budget


class TestRetryIo(unittest.TestCase):
    def test_succeeds_after_failures(self):
        from flamingo_tools.data_transfer_utils import RetryConfig, retry_io

        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise OSError(errno.EIO, "boom")
            return "value"

        config = RetryConfig(max_retries=5, retry_delay=0.0, max_retry_delay=0.0)
        with mock.patch("time.sleep"):
            self.assertEqual(retry_io(flaky, "flaky", config), "value")
        self.assertEqual(calls["n"], 3)

    def test_raises_after_exhaustion(self):
        from flamingo_tools.data_transfer_utils import RetryConfig, retry_io

        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise OSError(errno.EIO, "boom")

        config = RetryConfig(max_retries=3, retry_delay=0.0, max_retry_delay=0.0)
        with mock.patch("time.sleep"), self.assertRaises(OSError):
            retry_io(always_fails, "always", config)
        self.assertEqual(calls["n"], 3)

    def test_logs_every_attempt(self):
        from flamingo_tools.data_transfer_utils import RetryConfig, retry_io

        with tempfile.TemporaryDirectory() as tmp:
            log_file = os.path.join(tmp, "log.txt")
            config = RetryConfig(max_retries=2, retry_delay=0.0, log_file=log_file)
            with mock.patch("time.sleep"), self.assertRaises(OSError):
                retry_io(lambda: (_ for _ in ()).throw(OSError("boom")), "unit", config)
            with open(log_file) as f:
                lines = f.read().splitlines()
        # Two failed attempts plus the final give-up line.
        self.assertEqual(len(lines), 3)
        self.assertIn("unit", lines[-1])


class TestReadInto(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "data.bin")
        self.payload = bytes(range(256)) * 40
        with open(self.path, "wb") as f:
            f.write(self.payload)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_a_range(self):
        from flamingo_tools.data_transfer_utils import read_into

        buf = bytearray(100)
        n = read_into(self.path, 500, buf)
        self.assertEqual(n, 100)
        self.assertEqual(bytes(buf), self.payload[500:600])

    def test_recovers_from_a_dropped_read(self):
        from flamingo_tools.data_transfer_utils import RetryConfig, read_into
        import flamingo_tools.data_transfer_utils as dtu

        fake_open, budget = flaky_open(self.path, failures=2)
        buf = bytearray(100)
        config = RetryConfig(max_retries=5, retry_delay=0.0, max_retry_delay=0.0)
        with mock.patch.object(dtu, "open", fake_open, create=True), mock.patch("time.sleep"):
            read_into(self.path, 500, buf, config)
        self.assertEqual(budget["failures"], 0)
        self.assertEqual(bytes(buf), self.payload[500:600])

    def test_short_read_is_an_error(self):
        from flamingo_tools.data_transfer_utils import RetryConfig, read_into

        buf = bytearray(64)
        config = RetryConfig(max_retries=2, retry_delay=0.0, max_retry_delay=0.0)
        with mock.patch("time.sleep"), self.assertRaises(OSError):
            read_into(self.path, len(self.payload) - 8, buf, config)


class TestCopyFileResilient(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = os.path.join(self.tmp.name, "src.bin")
        self.dst = os.path.join(self.tmp.name, "out", "dst.bin")
        self.payload = os.urandom(5000)
        with open(self.src, "wb") as f:
            f.write(self.payload)

    def tearDown(self):
        self.tmp.cleanup()

    def _copy(self, block_size=1024):
        from flamingo_tools.data_transfer_utils import RetryConfig, copy_file_resilient

        config = RetryConfig(max_retries=3, retry_delay=0.0, max_retry_delay=0.0)
        return copy_file_resilient(self.src, self.dst, config, block_size=block_size)

    def test_full_copy(self):
        self.assertEqual(self._copy(), len(self.payload))
        with open(self.dst, "rb") as f:
            self.assertEqual(f.read(), self.payload)

    def test_resumes_from_a_partial_destination(self):
        os.makedirs(os.path.dirname(self.dst), exist_ok=True)
        with open(self.dst, "wb") as f:
            f.write(self.payload[:2048])
        self._copy()
        with open(self.dst, "rb") as f:
            self.assertEqual(f.read(), self.payload)

    def test_skips_a_complete_destination(self):
        self._copy()
        with mock.patch("flamingo_tools.data_transfer_utils.read_into") as read_mock:
            self._copy()
        read_mock.assert_not_called()

    def test_restarts_an_oversized_destination(self):
        os.makedirs(os.path.dirname(self.dst), exist_ok=True)
        with open(self.dst, "wb") as f:
            f.write(self.payload + b"extra")
        self._copy()
        with open(self.dst, "rb") as f:
            self.assertEqual(f.read(), self.payload)


class TestTransportFailureDetection(unittest.TestCase):
    """smbclient can abort mid-file and still exit 0, so the output has to be scanned."""

    def _run(self, output_lines, rc=0):
        import flamingo_tools.data_transfer_utils as dtu

        class FakeProc:
            def __init__(self):
                self.stdout = iter(output_lines)
                self.returncode = rc
                self.stdin = mock.MagicMock()
                self.terminated = False

            def terminate(self):
                self.terminated = True

            def wait(self):
                return self.returncode

        proc = FakeProc()
        with mock.patch.object(dtu.subprocess, "Popen", return_value=proc) as popen:
            lines, had_disconnect, code = dtu.run_smbclient("u", "p", ["ls"], cwd=".")
        return lines, had_disconnect, code, popen.call_args[0][0], proc

    def test_io_timeout_with_exit_zero_is_a_failure(self):
        # The exact regression: NT_STATUS_IO_TIMEOUT used to be reported as a successful transfer.
        _, had_disconnect, code, _, proc = self._run(
            ["NT_STATUS_IO_TIMEOUT listing \\path\\to\\file.raw\n"], rc=0
        )
        self.assertTrue(had_disconnect)
        self.assertEqual(code, 0)
        self.assertTrue(proc.terminated)

    def test_disconnect_is_still_a_failure(self):
        _, had_disconnect, _, _, _ = self._run(["NT_STATUS_CONNECTION_DISCONNECTED\n"])
        self.assertTrue(had_disconnect)

    def test_clean_output_is_a_success(self):
        _, had_disconnect, _, _, proc = self._run(["getting file x of 10 bytes\n"])
        self.assertFalse(had_disconnect)
        self.assertFalse(proc.terminated)

    def test_per_file_error_is_not_a_transport_failure(self):
        # A missing file inside a directory transfer must not mark the whole unit as broken.
        _, had_disconnect, _, _, _ = self._run(["NT_STATUS_NO_SUCH_FILE listing \\x\n"])
        self.assertFalse(had_disconnect)

    def test_passes_the_timeout_to_smbclient(self):
        from flamingo_tools.data_transfer_utils import SMB_TIMEOUT

        _, _, _, argv, _ = self._run(["ok\n"])
        self.assertIn("-t", argv)
        self.assertEqual(argv[argv.index("-t") + 1], str(SMB_TIMEOUT))
        self.assertEqual(SMB_TIMEOUT, 60)


class FakeSmbclient:
    """Stand-in for run_smbclient that grows a local file by a fixed amount per call."""

    def __init__(self, local_path, total, per_call, fail_lines=("NT_STATUS_IO_TIMEOUT listing x",)):
        self.local_path = local_path
        self.total = total
        self.per_call = per_call
        self.fail_lines = fail_lines
        self.calls = []

    def __call__(self, username, password, commands, cwd, smb_server=None, timeout=None):
        self.calls.append(list(commands))
        have = os.path.getsize(self.local_path) if os.path.exists(self.local_path) else 0
        wrote = min(self.per_call, self.total - have)
        with open(self.local_path, "ab") as f:
            f.write(b"\0" * wrote)
        if have + wrote < self.total:
            # Aborted part way, but smbclient still exits 0.
            return list(self.fail_lines), True, 0
        return [], False, 0


class TestResumableDownload(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.local_cwd = os.path.join(self.tmp.name, "stage")
        os.makedirs(self.local_cwd)
        self.name = "big.raw"
        self.local_path = os.path.join(self.local_cwd, self.name)
        self.total = 1000

    def tearDown(self):
        self.tmp.cleanup()

    def _download(self, fake, retries=10, expected_size=None):
        import flamingo_tools.data_transfer_utils as dtu

        with mock.patch.object(dtu, "run_smbclient", fake), mock.patch("time.sleep"):
            return dtu.resumable_download(
                "u", "p", remote_cd="P/dir", name=self.name, local_cwd=self.local_cwd,
                expected_size=self.total if expected_size is None else expected_size,
                retries=retries,
            )

    def test_uses_reget_without_recurse(self):
        fake = FakeSmbclient(self.local_path, self.total, self.total)
        self.assertTrue(self._download(fake))
        self.assertEqual(fake.calls[0], ['cd "P/dir"', 'reget "big.raw"'])
        flat = " ".join(c for call in fake.calls for c in call)
        self.assertNotIn("recurse", flat)
        self.assertNotIn("mget", flat)

    def test_resumes_until_the_size_matches(self):
        fake = FakeSmbclient(self.local_path, self.total, 300)
        self.assertTrue(self._download(fake))
        self.assertEqual(os.path.getsize(self.local_path), self.total)
        # 300 + 300 + 300 + 100 -> four attempts, each keeping what it gained.
        self.assertEqual(len(fake.calls), 4)

    def test_returns_false_when_attempts_run_out(self):
        fake = FakeSmbclient(self.local_path, self.total, 100)
        self.assertFalse(self._download(fake, retries=3))
        self.assertEqual(os.path.getsize(self.local_path), 300)

    def test_complete_file_needs_no_transfer(self):
        with open(self.local_path, "wb") as f:
            f.write(b"\0" * self.total)
        fake = FakeSmbclient(self.local_path, self.total, self.total)
        self.assertTrue(self._download(fake))
        self.assertEqual(fake.calls, [])

    def test_oversized_local_file_is_replaced(self):
        with open(self.local_path, "wb") as f:
            f.write(b"\0" * (self.total + 500))
        fake = FakeSmbclient(self.local_path, self.total, self.total)
        self.assertTrue(self._download(fake))
        self.assertEqual(os.path.getsize(self.local_path), self.total)
        self.assertEqual(len(fake.calls), 1)

    def test_falls_back_to_the_return_code_without_an_expected_size(self):
        import flamingo_tools.data_transfer_utils as dtu

        fake = FakeSmbclient(self.local_path, self.total, self.total)
        with mock.patch.object(dtu, "run_smbclient", fake), mock.patch("time.sleep"):
            ok = dtu.resumable_download(
                "u", "p", remote_cd="P/dir", name=self.name, local_cwd=self.local_cwd,
                expected_size=None,
            )
        self.assertTrue(ok)
        self.assertEqual(len(fake.calls), 1)


class TestWaitForPath(unittest.TestCase):
    def test_missing_path_raises(self):
        from flamingo_tools.data_transfer_utils import RetryConfig, wait_for_path

        config = RetryConfig(max_retries=2, retry_delay=0.0, max_retry_delay=0.0)
        with mock.patch("time.sleep"), self.assertRaises(OSError):
            wait_for_path("/does/not/exist", config)


if __name__ == "__main__":
    unittest.main()
