"""Unit tests for smb_transfer_resilient.py.

`run_smbclient` is replaced by a recorder so the tests need neither a real
smbclient binary nor a network connection. Run with:

    python -m pytest scripts/data_transfer/test_smb_transfer_resilient.py
    python -m unittest scripts.data_transfer.test_smb_transfer_resilient   # from repo root
"""

import contextlib
import os
import sys
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smb_transfer_resilient as smb  # noqa: E402
import flamingo_tools.data_transfer_utils as dtu  # noqa: E402


@contextmanager
def patch_both(name, new=None, **kwargs):
    """Install one replacement for `name` in both the script and the package module.

    The SMB primitives live in flamingo_tools.data_transfer_utils and are imported by name into
    smb_transfer_resilient. A moved function resolves its helpers in the package module, while a
    function that stayed in the script resolves them in the script module. A single shared
    replacement keeps a side_effect sequence consistent no matter which module resolves the call.
    """
    replacement = mock.MagicMock(**kwargs) if new is None else new
    targets = [module for module in (dtu, smb) if hasattr(module, name)]
    with contextlib.ExitStack() as stack:
        for module in targets:
            stack.enter_context(mock.patch.object(module, name, replacement))
        yield replacement


def patch_run_smbclient(recorder):
    """Replace run_smbclient with a recorder in both modules."""
    return patch_both("run_smbclient", new=recorder)


class Recorder:
    """Stand-in for run_smbclient. Records (commands, cwd) and returns scripted results."""

    def __init__(self, responses=None):
        # responses: list of (lines, had_disconnect, rc); the last is reused when exhausted.
        self.responses = responses or [([], False, 0)]
        self.calls = []

    def __call__(self, username, password, commands, cwd, smb_server=smb.SMB_SERVER, **kwargs):
        self.calls.append((list(commands), cwd))
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]

    @property
    def command_lists(self):
        return [c for c, _ in self.calls]


def _touch(path, size=1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"x" * size)


def _make_n5(root):
    """Create a minimal N5 fixture: root + setup0/timepoint0/{s0 (chunks 0,1), s4}."""
    n5 = os.path.join(root, "n5")
    _touch(os.path.join(n5, "attributes.json"), 50)
    tp = os.path.join(n5, "setup0", "timepoint0")
    _touch(os.path.join(tp, "s0", "attributes.json"), 80)
    _touch(os.path.join(tp, "s0", "0", "0"), 1024)
    _touch(os.path.join(tp, "s0", "1", "0"), 1024)
    _touch(os.path.join(tp, "s4", "attributes.json"), 60)
    _touch(os.path.join(tp, "s4", "0", "0"), 128)
    return n5


def _make_generic_tree(root):
    """Create a non-N5 tree: data/top.txt, data/sub/a.bin, data/sub/dir/big.bin."""
    data = os.path.join(root, "data")
    _touch(os.path.join(data, "top.txt"), 10)
    _touch(os.path.join(data, "sub", "a.bin"), 20)
    _touch(os.path.join(data, "sub", "dir", "big.bin"), 30)
    return data


class TestEnsureRemotePath(unittest.TestCase):
    def test_mkdir_chain_below_base(self):
        rec = Recorder()
        with patch_run_smbclient(rec):
            ok = smb.ensure_remote_path(
                "u", "p", base="P/parent",
                target="P/parent/n5/setup0/timepoint0/s0", local_cwd=".",
            )
        self.assertTrue(ok)
        self.assertEqual(len(rec.calls), 1)
        self.assertEqual(rec.command_lists[0], [
            'mkdir "P/parent/n5"',
            'mkdir "P/parent/n5/setup0"',
            'mkdir "P/parent/n5/setup0/timepoint0"',
            'mkdir "P/parent/n5/setup0/timepoint0/s0"',
        ])

    def test_create_all_when_base_empty(self):
        rec = Recorder()
        with patch_run_smbclient(rec):
            smb.ensure_remote_path("u", "p", base="", target="/UKON/x/n5", local_cwd=".")
        self.assertEqual(rec.command_lists[0], [
            'mkdir "/UKON"', 'mkdir "/UKON/x"', 'mkdir "/UKON/x/n5"',
        ])

    def test_nothing_to_create(self):
        rec = Recorder()
        with patch_run_smbclient(rec):
            ok = smb.ensure_remote_path("u", "p", base="P/n5", target="P/n5", local_cwd=".")
        self.assertTrue(ok)
        self.assertEqual(rec.calls, [])


class TestUploadPath(unittest.TestCase):
    def test_file_uses_put(self):
        rec = Recorder()
        with patch_run_smbclient(rec):
            smb.upload_path("u", "p", remote_dir="R", local_target="attributes.json",
                            local_cwd=".", is_dir=False, ensure=False)
        self.assertEqual(rec.command_lists[0], ['cd "R"', "put attributes.json"])

    def test_dir_uses_mput(self):
        rec = Recorder()
        with patch_run_smbclient(rec):
            smb.upload_path("u", "p", remote_dir="R", local_target="5",
                            local_cwd=".", is_dir=True, ensure=False)
        self.assertEqual(rec.command_lists[0], ['cd "R"', "recurse", "prompt", "mput 5"])

    def test_ensure_creates_dir_first(self):
        rec = Recorder()
        with patch_run_smbclient(rec):
            smb.upload_path("u", "p", remote_dir="R/sub", local_target="5",
                            local_cwd=".", is_dir=True, base="R", ensure=True)
        self.assertEqual(rec.command_lists[0], ['mkdir "R/sub"'])
        self.assertEqual(rec.command_lists[1], ['cd "R/sub"', "recurse", "prompt", "mput 5"])


class TestIterativeUpload(unittest.TestCase):
    def test_command_sequence(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec):
                smb.iterative_n5_upload("u", "p", remote_dir="P", n5_name="n5",
                                        source_dir=tmp, base="P")
        self.assertEqual(rec.command_lists, [
            ['mkdir "P/n5"'],
            ['cd "P/n5"', "put attributes.json"],
            ['mkdir "P/n5/setup0"'],
            ['mkdir "P/n5/setup0/timepoint0"'],
            ['mkdir "P/n5/setup0/timepoint0/s0"'],
            ['cd "P/n5/setup0/timepoint0/s0"', "put attributes.json"],
            ['cd "P/n5/setup0/timepoint0/s0"', "recurse", "prompt", "mput 0"],
            ['cd "P/n5/setup0/timepoint0/s0"', "recurse", "prompt", "mput 1"],
            ['cd "P/n5/setup0/timepoint0"', "recurse", "prompt", "mput s4"],
        ])

    def test_setup_filter(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            n5 = _make_n5(tmp)
            # A second setup that must be skipped by the filter.
            _touch(os.path.join(n5, "setup1", "timepoint0", "s4", "0", "0"), 1)
            rec = Recorder()
            with patch_run_smbclient(rec):
                smb.iterative_n5_upload("u", "p", remote_dir="P", n5_name="n5",
                                        source_dir=tmp, base="P", setup_filter=["setup0"])
        joined = " ".join(c for cmds in rec.command_lists for c in cmds)
        self.assertIn("setup0", joined)
        self.assertNotIn("setup1", joined)


class TestPhase1Ingest(unittest.TestCase):
    def _args(self, create_parents=False):
        return types.SimpleNamespace(
            username="u", smb_server=smb.SMB_SERVER, create_parents=create_parents, generic=False,
        )

    def test_bulk_non_filtered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()  # preflight ls + bulk mput both succeed
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb, "verify_and_repair_upload"):
                with self.assertRaises(SystemExit) as cm:
                    smb._run_ingest(self._args(), "p", "P", "n5", tmp, "log.txt", None)
        self.assertEqual(cm.exception.code, 0)
        self.assertIn(['cd "P"', "ls"], rec.command_lists)          # preflight
        self.assertIn(['cd "P"', "recurse", "prompt", "mput n5"], rec.command_lists)

    def test_bulk_filtered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb, "verify_and_repair_upload"):
                with self.assertRaises(SystemExit):
                    smb._run_ingest(self._args(), "p", "P", "n5", tmp, "log.txt", ["setup0"])
        self.assertIn(['mkdir "P/n5"'], rec.command_lists)
        self.assertIn(['cd "P/n5"', "put attributes.json", "recurse", "prompt", "mput setup0"],
                      rec.command_lists)

    def test_missing_parent_aborts(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder([(["NT_STATUS_OBJECT_NAME_NOT_FOUND listing \\P"], False, 1)])
            with patch_run_smbclient(rec):
                with self.assertRaises(SystemExit):
                    smb._run_ingest(self._args(), "p", "P", "n5", tmp, "log.txt", None)
        # Only the preflight ran; no mput was attempted.
        joined = " ".join(c for cmds in rec.command_lists for c in cmds)
        self.assertNotIn("mput", joined)


class TestRemoteSizeMap(unittest.TestCase):
    FIXTURE = [
        r"\P\n5",
        "  .                                   D        0  Mon Jul 21 10:00:00 2025",
        "  ..                                  D        0  Mon Jul 21 10:00:00 2025",
        "  attributes.json                     A       50  Mon Jul 21 10:00:00 2025",
        "  setup0                              D        0  Mon Jul 21 10:00:00 2025",
        "",
        r"\P\n5\setup0\timepoint0\s0",
        "  attributes.json                     A       80  Mon Jul 21 10:00:00 2025",
        "  0                                   D        0  Mon Jul 21 10:00:00 2025",
        "",
        r"\P\n5\setup0\timepoint0\s0\0",
        "  0                                          1024  Mon Jul 21 10:00:00 2025",
        "",
        "\t\t63305 blocks of size 524288. 12345 blocks available",
    ]

    def test_parse(self):
        rec = Recorder([(self.FIXTURE, False, 0)])
        with patch_run_smbclient(rec):
            size_map = smb.build_remote_size_map("u", "p", "P/n5", ".")
        self.assertEqual(size_map, {
            "attributes.json": 50,
            "setup0/timepoint0/s0/attributes.json": 80,
            "setup0/timepoint0/s0/0/0": 1024,
        })

    def test_disconnect_returns_none(self):
        rec = Recorder([([], True, 0)])
        with patch_run_smbclient(rec):
            self.assertIsNone(smb.build_remote_size_map("u", "p", "P/n5", "."))


class TestVerifyAndRepairUpload(unittest.TestCase):
    def test_reuploads_mismatched_file(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            # Pass 1: root attributes.json size wrong; pass 2: everything matches.
            good = {
                "attributes.json": 50,
                "setup0/timepoint0/s0/attributes.json": 80,
                "setup0/timepoint0/s0/0/0": 1024,
                "setup0/timepoint0/s0/1/0": 1024,
                "setup0/timepoint0/s4/attributes.json": 60,
                "setup0/timepoint0/s4/0/0": 128,
            }
            bad = dict(good, **{"attributes.json": 3})
            rec = Recorder()
            with patch_run_smbclient(rec), \
                 patch_both("build_remote_size_map", side_effect=[bad, good]):
                smb.verify_and_repair_upload("u", "p", "P", "n5", tmp)
        # The mismatched root attributes.json was re-uploaded via put.
        self.assertIn(['cd "P/n5"', "put attributes.json"], rec.command_lists)


class TestRetryScaffold(unittest.TestCase):
    def test_success_after_disconnects(self):
        rec = Recorder([([], True, 0), ([], True, 0), ([], False, 0)])
        with patch_run_smbclient(rec), \
             mock.patch.object(smb.time, "sleep"):
            ok = smb._run_with_retry("u", "p", ["ls"], local_cwd=".", label="x")
        self.assertTrue(ok)
        self.assertEqual(len(rec.calls), 3)

    def test_exhaustion_logs_single_newline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "log.txt")
            rec = Recorder([([], True, 0)])
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb.time, "sleep"):
                ok = smb._run_with_retry("u", "p", ["ls"], local_cwd=".", label="x",
                                         retries=2, log_file=log)
            self.assertFalse(ok)
            with open(log) as f:
                content = f.read()
        self.assertTrue(content.endswith("\n"))
        self.assertEqual(content.count("\n"), 1)

    def test_upload_error_token_fails_unit(self):
        rec = Recorder([(["NT_STATUS_ACCESS_DENIED opening remote file"], False, 0)])
        with patch_run_smbclient(rec), \
             mock.patch.object(smb.time, "sleep"):
            ok = smb._run_with_retry("u", "p", ["put x"], local_cwd=".", label="x",
                                     retries=2, error_tokens=smb.UPLOAD_ERROR_TOKENS)
        self.assertFalse(ok)


class TestDownloadUnchanged(unittest.TestCase):
    def test_transfer_path_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder()
            with patch_run_smbclient(rec):
                ok = smb.transfer_path("u", "p", remote_cd="R", mget_target="setup0",
                                       local_cwd=os.path.join(tmp, "dest"))
        self.assertTrue(ok)
        self.assertEqual(rec.command_lists[0], ['cd "R"', "recurse", "prompt", "mget setup0"])


class TestDetection(unittest.TestCase):
    def test_looks_like_n5_local(self):
        with tempfile.TemporaryDirectory() as tmp:
            n5 = _make_n5(tmp)
            data = _make_generic_tree(tmp)
            self.assertTrue(smb._looks_like_n5_local(n5))
            self.assertFalse(smb._looks_like_n5_local(data))

    def test_looks_like_n5_remote_true(self):
        ls = ["  setup0   D   0  Mon Jul 21 10:00:00 2025",
              "  data.bin   A   5  Mon Jul 21 10:00:00 2025"]
        rec = Recorder([(ls, False, 0)])
        with patch_run_smbclient(rec):
            self.assertTrue(smb._looks_like_n5_remote("u", "p", "P/n5", "."))

    def test_looks_like_n5_remote_false(self):
        ls = ["  top.txt   A   5  Mon Jul 21 10:00:00 2025",
              "  sub   D   0  Mon Jul 21 10:00:00 2025"]
        rec = Recorder([(ls, False, 0)])
        with patch_run_smbclient(rec):
            self.assertFalse(smb._looks_like_n5_remote("u", "p", "P/data", "."))

    def test_looks_like_n5_remote_disconnect_is_false(self):
        rec = Recorder([([], True, 0)])
        with patch_run_smbclient(rec):
            self.assertFalse(smb._looks_like_n5_remote("u", "p", "P/data", "."))


class TestGenericDownload(unittest.TestCase):
    def test_command_sequence_and_local_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            size_map = {"top.txt": 10, "sub/a.bin": 20, "sub/dir/big.bin": 30}
            rec = Recorder()
            with patch_run_smbclient(rec), \
                 patch_both("build_remote_size_map", return_value=size_map):
                smb.generic_iterative_download("u", "p", "P", "data", tmp)
            self.assertEqual(rec.command_lists, [
                ['cd "P/data/sub"', "recurse", "prompt", 'mget "a.bin"'],
                ['cd "P/data/sub/dir"', "recurse", "prompt", 'mget "big.bin"'],
                ['cd "P/data"', "recurse", "prompt", 'mget "top.txt"'],
            ])
            self.assertTrue(os.path.isdir(os.path.join(tmp, "data", "sub", "dir")))

    def test_quotes_filename_with_spaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = Recorder()
            with patch_run_smbclient(rec), \
                 patch_both("build_remote_size_map", return_value={"a b.txt": 5}):
                smb.generic_iterative_download("u", "p", "P", "data", tmp)
            self.assertEqual(rec.command_lists[0],
                             ['cd "P/data"', "recurse", "prompt", 'mget "a b.txt"'])


class TestGenericDownloadVerify(unittest.TestCase):
    def test_refetches_size_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "data", "sub", "dir", "big.bin"), 5)  # short
            rec = Recorder()
            maps = [{"sub/dir/big.bin": 30}, {"sub/dir/big.bin": 5}]  # mismatch, then match
            with patch_run_smbclient(rec), \
                 patch_both("build_remote_size_map", side_effect=maps):
                smb.verify_and_repair_download_generic("u", "p", "P", "data", tmp)
            self.assertIn(['cd "P/data/sub/dir"', "recurse", "prompt", 'mget "big.bin"'],
                          rec.command_lists)


class TestGenericUpload(unittest.TestCase):
    def test_command_sequence(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_generic_tree(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec):
                smb.generic_iterative_upload("u", "p", "P", "data", tmp, base="P")
            self.assertEqual(rec.command_lists, [
                ['mkdir "P/data"'],
                ['cd "P/data"', 'put "top.txt"'],
                ['mkdir "P/data/sub"'],
                ['cd "P/data/sub"', 'put "a.bin"'],
                ['mkdir "P/data/sub/dir"'],
                ['cd "P/data/sub/dir"', 'put "big.bin"'],
            ])


@contextmanager
def _patch_download(is_n5):
    with mock.patch.object(smb, "_looks_like_n5_remote", return_value=is_n5), \
         mock.patch.object(smb, "iterative_n5_transfer") as it_n5, \
         mock.patch.object(smb, "generic_iterative_download") as it_gen, \
         mock.patch.object(smb, "verify_and_repair_n5") as v_n5, \
         mock.patch.object(smb, "verify_and_repair_download_generic") as v_gen:
        yield it_n5, it_gen, v_n5, v_gen


class TestDownloadDispatch(unittest.TestCase):
    def _args(self, generic=False):
        return types.SimpleNamespace(username="u", smb_server=smb.SMB_SERVER, generic=generic)

    def test_n5_success_uses_n5_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=True) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], False, 0)])):
                with self.assertRaises(SystemExit):
                    smb._run_download(self._args(), "p", "P", "n5", tmp, "log", None)
        v_n5.assert_called_once()
        v_gen.assert_not_called()
        it_n5.assert_not_called()
        it_gen.assert_not_called()

    def test_generic_success_uses_generic_verify(self):
        # Regression guard: a truncated large generic file must be caught by the size verify
        # even when Phase 1 reports success.
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=False) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], False, 0)])):
                with self.assertRaises(SystemExit):
                    smb._run_download(self._args(), "p", "P", "data", tmp, "log", None)
        v_gen.assert_called_once()
        v_n5.assert_not_called()

    def test_n5_fallback_uses_n5_iterative(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=True) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], True, 0)])):
                smb._run_download(self._args(), "p", "P", "n5", tmp, "log", None)
        it_n5.assert_called_once()
        it_gen.assert_not_called()
        v_n5.assert_called_once()

    def test_generic_fallback_uses_generic_iterative(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=False) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], True, 0)])):
                smb._run_download(self._args(), "p", "P", "data", tmp, "log", None)
        it_gen.assert_called_once()
        it_n5.assert_not_called()
        v_gen.assert_called_once()

    def test_generic_flag_forces_generic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=True) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], True, 0)])):
                # is_n5 detection would say N5, but --generic overrides it.
                smb._run_download(self._args(generic=True), "p", "P", "n5", tmp, "log", None)
        it_gen.assert_called_once()
        it_n5.assert_not_called()


class TestIngestDispatch(unittest.TestCase):
    def _args(self, generic=False):
        return types.SimpleNamespace(username="u", smb_server=smb.SMB_SERVER,
                                     generic=generic, create_parents=False)

    def _run(self, make_tree, name, generic=False):
        with tempfile.TemporaryDirectory() as tmp:
            make_tree(tmp)
            with patch_run_smbclient(Recorder([([], True, 0)])), \
                 mock.patch.object(smb, "remote_dir_exists", return_value=True), \
                 mock.patch.object(smb, "iterative_n5_upload") as up_n5, \
                 mock.patch.object(smb, "generic_iterative_upload") as up_gen, \
                 mock.patch.object(smb, "verify_and_repair_upload") as v_up:
                smb._run_ingest(self._args(generic), "p", "P", name, tmp, "log", None)
            return up_n5, up_gen, v_up

    def test_n5_fallback_uses_n5_upload(self):
        up_n5, up_gen, v_up = self._run(_make_n5, "n5")
        up_n5.assert_called_once()
        up_gen.assert_not_called()
        v_up.assert_called_once()

    def test_generic_fallback_uses_generic_upload(self):
        up_n5, up_gen, v_up = self._run(_make_generic_tree, "data")
        up_gen.assert_called_once()
        up_n5.assert_not_called()
        v_up.assert_called_once()


class TestSmbQuote(unittest.TestCase):
    def test_quote(self):
        self.assertEqual(smb._smb_quote("a b.txt"), '"a b.txt"')


if __name__ == "__main__":
    unittest.main()
