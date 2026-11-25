"""Unit tests for flamingo_tools/data_transfer.py.

`run_smbclient` is replaced by a recorder so the tests need neither a real
smbclient binary nor a network connection.
"""

import contextlib
import io
import os
import tempfile
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import flamingo_tools.data_transfer as smb
import flamingo_tools.data_transfer_utils as dtu


@contextmanager
def patch_both(name, new=None, **kwargs):
    """Install one replacement for `name` in both data_transfer and data_transfer_utils.

    The SMB primitives live in flamingo_tools.data_transfer_utils and are imported by name into
    flamingo_tools.data_transfer. A primitive resolves its helpers in data_transfer_utils, while a
    driver resolves them in data_transfer. A single shared
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

    def test_size_map_skips_complete_units(self):
        # s0/0 and s4 are complete on the share, s0/1/0 is missing.
        size_map = {"attributes.json": 50, "setup0/timepoint0/s0/attributes.json": 80,
                    "setup0/timepoint0/s0/0/0": 1024, "setup0/timepoint0/s4/attributes.json": 60,
                    "setup0/timepoint0/s4/0/0": 128}
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec):
                smb.iterative_n5_upload("u", "p", remote_dir="P", n5_name="n5",
                                        source_dir=tmp, base="P", size_map=size_map)
        mputs = [cmds for cmds in rec.command_lists if any(c.startswith("mput") for c in cmds)]
        self.assertEqual(mputs, [['cd "P/n5/setup0/timepoint0/s0"', "recurse", "prompt", "mput 1"]])
        # The small attributes.json files are always put.
        self.assertIn(['cd "P/n5"', "put attributes.json"], rec.command_lists)

    def test_size_mismatch_uploads_the_unit(self):
        size_map = {"setup0/timepoint0/s4/attributes.json": 60, "setup0/timepoint0/s4/0/0": 64}
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec):
                smb.iterative_n5_upload("u", "p", remote_dir="P", n5_name="n5",
                                        source_dir=tmp, base="P", size_map=size_map)
        self.assertIn(['cd "P/n5/setup0/timepoint0"', "recurse", "prompt", "mput s4"], rec.command_lists)


class TestPhase1Ingest(unittest.TestCase):
    def _args(self, create_parents=False):
        return types.SimpleNamespace(
            username="u", smb_server=smb.SMB_SERVER, create_parents=create_parents, generic=False, resume=False,
        )

    def test_bulk_non_filtered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()  # preflight + bulk mput both succeed
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb, "verify_and_repair_upload"):
                smb._run_ingest(self._args(), "p", "P", "n5", tmp, "log.txt", None)
        self.assertIn(['cd "P"'], rec.command_lists)          # preflight
        self.assertIn(['cd "P"', "recurse", "prompt", "mput n5"], rec.command_lists)

    def test_bulk_filtered(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb, "verify_and_repair_upload"):
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
            size_map = dtu.build_remote_size_map("u", "p", "P/n5", ".")
        self.assertEqual(size_map, {
            "attributes.json": 50,
            "setup0/timepoint0/s0/attributes.json": 80,
            "setup0/timepoint0/s0/0/0": 1024,
        })

    def test_disconnect_returns_none(self):
        rec = Recorder([([], True, 0)])
        with patch_run_smbclient(rec):
            self.assertIsNone(dtu.build_remote_size_map("u", "p", "P/n5", "."))


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

    def test_logs_failed_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _make_n5(tmp)
            log_file = os.path.join(tmp, "log.txt")
            with patch_run_smbclient(Recorder()), mock.patch.object(dtu.time, "sleep"), \
                 patch_both("build_remote_size_map", return_value=None):
                smb.verify_and_repair_upload("u", "p", "P", "n5", tmp, log_file=log_file)
            with open(log_file) as f:
                self.assertIn("could not list the remote tree P/n5 for verification", f.read())


class TestRetryScaffold(unittest.TestCase):
    def test_success_after_disconnects(self):
        rec = Recorder([([], True, 0), ([], True, 0), ([], False, 0)])
        with patch_run_smbclient(rec), \
             mock.patch.object(dtu.time, "sleep"):
            ok = dtu.run_with_retry("u", "p", ["ls"], local_cwd=".", label="x")
        self.assertTrue(ok)
        self.assertEqual(len(rec.calls), 3)

    def test_exhaustion_logs_single_newline(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            log = os.path.join(tmp, "log.txt")
            rec = Recorder([([], True, 0)])
            with patch_run_smbclient(rec), \
                 mock.patch.object(dtu.time, "sleep"):
                ok = dtu.run_with_retry("u", "p", ["ls"], local_cwd=".", label="x",
                                        retries=2, log_file=log)
            self.assertFalse(ok)
            with open(log) as f:
                content = f.read()
        self.assertTrue(content.endswith("\n"))
        self.assertEqual(content.count("\n"), 1)

    def test_upload_error_token_fails_unit(self):
        rec = Recorder([(["NT_STATUS_ACCESS_DENIED opening remote file"], False, 0)])
        with patch_run_smbclient(rec), \
             mock.patch.object(dtu.time, "sleep"):
            ok = dtu.run_with_retry("u", "p", ["put x"], local_cwd=".", label="x",
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


class TestIterativeDownload(unittest.TestCase):
    def test_min_scale(self):
        listing = {
            "P/n5": ["setup0"],
            "P/n5/setup0/timepoint0": ["s0", "s1", "s2", "s4"],
            "P/n5/setup0/timepoint0/s2": ["0"],
        }
        with tempfile.TemporaryDirectory() as tmp, \
             patch_both("list_remote_dirs", side_effect=lambda u, p, path, *a, **k: listing[path]), \
             patch_both("transfer_path") as tp:
            smb.iterative_n5_transfer("u", "p", "P", "n5", tmp, min_scale=2)
        fetched = [(c.kwargs["remote_cd"], c.kwargs["mget_target"]) for c in tp.call_args_list]
        self.assertEqual(fetched, [
            ("P/n5", "attributes.json"),
            ("P/n5/setup0", "attributes.json"),
            ("P/n5/setup0/timepoint0", "attributes.json"),
            ("P/n5/setup0/timepoint0/s2", "attributes.json"),
            ("P/n5/setup0/timepoint0/s2", "0"),
            ("P/n5/setup0/timepoint0", "s4"),
        ])

    def test_empty_listing_is_logged(self):
        # A failed listing returns [] without a disconnect; the skipped scale must reach the log.
        listing = {"P/n5": ["setup0"], "P/n5/setup0/timepoint0": ["s2"], "P/n5/setup0/timepoint0/s2": []}
        with tempfile.TemporaryDirectory() as tmp, \
             patch_both("list_remote_dirs", side_effect=lambda u, p, path, *a, **k: listing[path]), \
             patch_both("transfer_path"):
            log_file = os.path.join(tmp, "log.txt")
            smb.iterative_n5_transfer("u", "p", "P", "n5", tmp, log_file=log_file)
            with open(log_file) as f:
                self.assertIn("no chunk directories found in P/n5/setup0/timepoint0/s2", f.read())


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
        with patch_run_smbclient(rec), mock.patch.object(dtu.time, "sleep"):
            self.assertFalse(smb._looks_like_n5_remote("u", "p", "P/data", "."))

    def test_list_remote_dirs_retries_disconnect(self):
        ls = ["  setup0   D   0  Mon Jul 21 10:00:00 2025"]
        rec = Recorder([([], True, 0), (ls, False, 0)])
        with patch_run_smbclient(rec), mock.patch.object(dtu.time, "sleep"):
            self.assertEqual(smb.list_remote_dirs("u", "p", "P/n5", "."), ["setup0"])
        self.assertEqual(len(rec.calls), 2)


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

    def _partial_tree(self, tmp):
        """top.txt is complete, sub/a.bin is short, sub/dir/big.bin is missing."""
        _touch(os.path.join(tmp, "data", "top.txt"), 10)
        _touch(os.path.join(tmp, "data", "sub", "a.bin"), 5)
        return {"top.txt": 10, "sub/a.bin": 20, "sub/dir/big.bin": 30}

    def test_resume_keeps_complete_files_and_continues_short_ones(self):
        with tempfile.TemporaryDirectory() as tmp:
            size_map = self._partial_tree(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec), patch_both("build_remote_size_map", return_value=size_map), \
                 mock.patch.object(smb, "resumable_download") as reget:
                smb.generic_iterative_download("u", "p", "P", "data", tmp, resume=True)
            self.assertEqual(rec.command_lists, [['cd "P/data/sub/dir"', "recurse", "prompt", 'mget "big.bin"']])
            reget.assert_called_once()
            self.assertEqual(reget.call_args.args[2:5], ("P/data/sub", "a.bin", os.path.join(tmp, "data", "sub")))
            self.assertEqual(reget.call_args.kwargs["expected_size"], 20)

    def test_without_resume_every_file_is_fetched_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            size_map = self._partial_tree(tmp)
            rec = Recorder()
            with patch_run_smbclient(rec), patch_both("build_remote_size_map", return_value=size_map), \
                 mock.patch.object(smb, "resumable_download") as reget:
                smb.generic_iterative_download("u", "p", "P", "data", tmp)
            self.assertEqual(len(rec.calls), 3)
            reget.assert_not_called()


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

    def test_resume_continues_short_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "data", "big.bin"), 5)
            rec = Recorder()
            maps = [{"big.bin": 30}, {"big.bin": 5}]
            with patch_run_smbclient(rec), patch_both("build_remote_size_map", side_effect=maps), \
                 mock.patch.object(smb, "resumable_download") as reget:
                smb.verify_and_repair_download_generic("u", "p", "P", "data", tmp, resume=True)
            self.assertEqual(rec.calls, [])
            self.assertEqual(reget.call_args.args[2:4], ("P/data", "big.bin"))
            self.assertEqual(reget.call_args.kwargs["expected_size"], 30)

    def test_retries_dropped_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "data", "a.bin"), 5)
            log_file = os.path.join(tmp, "log.txt")
            with patch_run_smbclient(Recorder()), mock.patch.object(dtu.time, "sleep"), \
                 patch_both("build_remote_size_map", side_effect=[None, {"a.bin": 5}]) as listing:
                smb.verify_and_repair_download_generic("u", "p", "P", "data", tmp, log_file=log_file)
            self.assertEqual(listing.call_count, 2)
            self.assertFalse(os.path.exists(log_file))

    def test_logs_failed_listing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "data", "a.bin"), 5)
            log_file = os.path.join(tmp, "log.txt")
            with patch_run_smbclient(Recorder()), mock.patch.object(dtu.time, "sleep"), \
                 patch_both("build_remote_size_map", return_value=None) as listing:
                smb.verify_and_repair_download_generic("u", "p", "P", "data", tmp, log_file=log_file)
            self.assertEqual(listing.call_count, dtu.MAX_RETRIES)
            with open(log_file) as f:
                self.assertIn("could not list the remote tree P/data for verification", f.read())


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
    with mock.patch.object(smb, "remote_entry", return_value=(True, 0)), \
         mock.patch.object(smb, "_looks_like_n5_remote", return_value=is_n5), \
         mock.patch.object(smb, "iterative_n5_transfer") as it_n5, \
         mock.patch.object(smb, "generic_iterative_download") as it_gen, \
         mock.patch.object(smb, "verify_and_repair_n5") as v_n5, \
         mock.patch.object(smb, "verify_and_repair_download_generic") as v_gen:
        yield it_n5, it_gen, v_n5, v_gen


class TestDownloadDispatch(unittest.TestCase):
    def _args(self, generic=False, resume=False):
        return types.SimpleNamespace(username="u", smb_server=smb.SMB_SERVER, generic=generic, resume=resume)

    def test_n5_success_uses_n5_verify(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=True) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], False, 0)])):
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

    def test_missing_remote_dir_aborts(self):
        rec = Recorder([(["cd \\P\\n5\\: NT_STATUS_OBJECT_NAME_NOT_FOUND"], False, 1)])
        with tempfile.TemporaryDirectory() as tmp, patch_run_smbclient(rec):
            with self.assertRaises(SystemExit):
                smb._run_download(self._args(), "p", "P", "n5", tmp, "log", None)
        # The file probe finds no file named n5, then the directory check fails.
        self.assertEqual(rec.command_lists, [['cd "P"', 'ls "n5"'], ['cd "P/n5"']])

    def test_directory_entry_skips_directory_check(self):
        probe = (["  data                                D        0  Mon Jul 21 10:00:00 2025"], False, 0)
        rec = Recorder([probe, ([], False, 0)])
        with tempfile.TemporaryDirectory() as tmp, patch_run_smbclient(rec), \
             mock.patch.object(smb, "verify_and_repair_download_generic"):
            smb._run_download(self._args(generic=True), "p", "P", "data", tmp, "log", None)
        self.assertEqual(rec.command_lists, [
            ['cd "P"', 'ls "data"'],
            ['cd "P"', "recurse", "prompt", "mget data"],
        ])

    def test_file_uses_resumable_download(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(smb, "remote_entry", return_value=(False, 123)), \
             mock.patch.object(smb, "resumable_download") as download, \
             patch_run_smbclient(Recorder()) as rec:
            smb._run_download(self._args(), "p", "P", "big.tif", tmp, "log", None)
        download.assert_called_once()
        self.assertEqual(download.call_args.args[2:5], ("P", "big.tif", tmp))
        self.assertEqual(download.call_args.kwargs["expected_size"], 123)
        self.assertEqual(rec.calls, [])

    def test_file_rejects_setup(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(smb, "remote_entry", return_value=(False, 123)), \
             mock.patch.object(smb, "resumable_download") as download:
            with self.assertRaises(SystemExit):
                smb._run_download(self._args(), "p", "P", "big.tif", tmp, "log", ["setup0"])
        download.assert_not_called()

    def test_generic_flag_forces_generic(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=True) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], True, 0)])):
                # is_n5 detection would say N5, but --generic overrides it.
                smb._run_download(self._args(generic=True), "p", "P", "n5", tmp, "log", None)
        it_gen.assert_called_once()
        it_n5.assert_not_called()

    def test_resume_with_local_directory_skips_bulk_transfer(self):
        rec = Recorder()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "1_raw"))
            with _patch_download(is_n5=True) as (it_n5, it_gen, v_n5, v_gen), patch_run_smbclient(rec):
                smb._run_download(self._args(resume=True), "p", "P", "1_raw", tmp, "log", None)
        # No bulk mget: the generic verify fetches only the missing files, also for an N5.
        self.assertEqual(rec.calls, [])
        v_gen.assert_called_once()
        self.assertTrue(v_gen.call_args.kwargs["resume"])
        v_n5.assert_not_called()
        it_n5.assert_not_called()
        it_gen.assert_not_called()

    def test_resume_without_local_directory_runs_bulk_transfer(self):
        probe = (["  1_raw                               D        0  Mon Jul 21 10:00:00 2025"], False, 0)
        rec = Recorder([probe, ([], False, 0)])
        with tempfile.TemporaryDirectory() as tmp, patch_run_smbclient(rec), \
             mock.patch.object(smb, "verify_and_repair_download_generic") as v_gen:
            smb._run_download(self._args(generic=True, resume=True), "p", "P", "1_raw", tmp, "log", None)
        self.assertEqual(rec.command_lists[-1], ['cd "P"', "recurse", "prompt", "mget 1_raw"])
        self.assertTrue(v_gen.call_args.kwargs["resume"])

    def test_resume_reaches_the_generic_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            with _patch_download(is_n5=False) as (it_n5, it_gen, v_n5, v_gen), \
                 patch_run_smbclient(Recorder([([], True, 1)])):
                smb._run_download(self._args(resume=True), "p", "P", "1_raw", tmp, "log", None)
        self.assertTrue(it_gen.call_args.kwargs["resume"])
        self.assertTrue(v_gen.call_args.kwargs["resume"])


class TestTransferDataCli(unittest.TestCase):
    def _run_cli(self, argv, stdin=""):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch("sys.argv", ["transfer_data", "-u", "u", "-p", "P", "-d", "data", "-o", tmp] + argv), \
             mock.patch("sys.stdin", io.StringIO(stdin)), \
             mock.patch.object(smb.getpass, "getpass", return_value="prompted") as prompt, \
             mock.patch.object(smb, "_run_download") as download:
            smb.transfer_data_cli()
        return prompt, download

    def test_password_from_stdin(self):
        prompt, download = self._run_cli(["--password-stdin"], stdin="pass word\nrest\n")
        prompt.assert_not_called()
        self.assertEqual(download.call_args.args[1], "pass word")

    def test_password_from_prompt(self):
        prompt, download = self._run_cli([])
        prompt.assert_called_once()
        self.assertEqual(download.call_args.args[1], "prompted")

    def test_password_from_stdin_drops_crlf(self):
        _, download = self._run_cli(["--password-stdin"], stdin="pass\r\n")
        self.assertEqual(download.call_args.args[1], "pass")

    def test_empty_stdin_stops(self):
        with self.assertRaises(SystemExit):
            self._run_cli(["--password-stdin"], stdin="\n")

    def test_resume_rejects_setup(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self._run_cli(["--resume", "--setup", "0"])

    def test_resume_applies_to_ingest(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            argv = ["transfer_data", "-u", "u", "-p", "P", "-d", "data", "-o", tmp, "--ingest", "--resume"]
            with mock.patch("sys.argv", argv), \
                 mock.patch.object(smb.getpass, "getpass", return_value="p"), \
                 mock.patch.object(smb, "_run_ingest") as ingest:
                smb.transfer_data_cli()
        self.assertTrue(ingest.call_args.args[0].resume)

    def test_password_stdin_on_terminal_stops(self):
        terminal = io.StringIO("visible\n")
        terminal.isatty = lambda: True
        with mock.patch("sys.stdin", terminal), self.assertRaises(SystemExit) as stop:
            smb._read_password(True)
        self.assertIn("pipe", str(stop.exception))


class TestIngestDispatch(unittest.TestCase):
    def _args(self, generic=False, resume=False):
        return types.SimpleNamespace(username="u", smb_server=smb.SMB_SERVER,
                                     generic=generic, create_parents=False, resume=resume)

    def _run(self, make_tree, name, generic=False, resume=False, remote=None, disconnect=True):
        """Run the ingest. `remote` is the remote_entry of the dataset; returns the mocks and the recorder."""
        rec = Recorder([([], disconnect, 0)])
        with tempfile.TemporaryDirectory() as tmp:
            make_tree(tmp)
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb, "require_remote_dir"), \
                 mock.patch.object(smb, "remote_entry", return_value=remote), \
                 mock.patch.object(smb, "remote_size_map_with_retry", return_value={"x": 1}) as listing, \
                 mock.patch.object(smb, "iterative_n5_upload") as up_n5, \
                 mock.patch.object(smb, "generic_iterative_upload") as up_gen, \
                 mock.patch.object(smb, "verify_and_repair_upload") as v_up:
                smb._run_ingest(self._args(generic, resume), "p", "P", name, tmp, "log", None)
        self.listing = listing
        self.mputs = [cmds for cmds in rec.command_lists if any(c.startswith("mput") for c in cmds)]
        return up_n5, up_gen, v_up

    def test_resume_with_remote_dataset_skips_bulk_ingest(self):
        up_n5, up_gen, v_up = self._run(_make_n5, "n5", resume=True, remote=(True, 0))
        self.assertEqual(self.mputs, [])
        self.assertEqual(up_n5.call_args.kwargs["size_map"], {"x": 1})
        up_gen.assert_not_called()
        v_up.assert_called_once()

    def test_resume_generic_goes_to_the_repair(self):
        up_n5, up_gen, v_up = self._run(_make_generic_tree, "data", resume=True, remote=(True, 0))
        self.assertEqual(self.mputs, [])
        up_n5.assert_not_called()
        up_gen.assert_not_called()
        v_up.assert_called_once()

    def test_resume_without_remote_dataset_runs_bulk_ingest(self):
        up_n5, _, v_up = self._run(_make_n5, "n5", resume=True, remote=None, disconnect=False)
        self.assertEqual(self.mputs, [['cd "P"', "recurse", "prompt", "mput n5"]])
        self.listing.assert_not_called()
        up_n5.assert_not_called()
        v_up.assert_called_once()

    def test_resume_fallback_skips_complete_units(self):
        up_n5, up_gen, _ = self._run(_make_n5, "n5", resume=True, remote=None)
        self.assertEqual(up_n5.call_args.kwargs["size_map"], {"x": 1})
        up_gen.assert_not_called()

    def test_fallback_without_resume_uploads_everything(self):
        up_n5, _, _ = self._run(_make_n5, "n5")
        self.assertIsNone(up_n5.call_args.kwargs["size_map"])
        self.listing.assert_not_called()

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

    def _ingest_file(self, remote_entry):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "big file.tif"), 10)
            log = os.path.join(tmp, "log.txt")
            rec = Recorder()
            with patch_run_smbclient(rec), \
                 mock.patch.object(smb, "remote_entry", return_value=remote_entry):
                smb._run_ingest(self._args(), "p", "P", "big file.tif", tmp, log, None)
            logged = os.path.exists(log) and os.path.getsize(log) > 0
        return rec, logged

    def test_file_uses_put(self):
        rec, logged = self._ingest_file(remote_entry=(False, 10))
        self.assertEqual(rec.command_lists, [['cd "P"'], ['cd "P"', 'put "big file.tif"']])
        self.assertFalse(logged)

    def test_file_size_mismatch_is_logged(self):
        _, logged = self._ingest_file(remote_entry=(False, 4))
        self.assertTrue(logged)

    def test_file_missing_on_share_is_logged(self):
        _, logged = self._ingest_file(remote_entry=None)
        self.assertTrue(logged)


class TestSmbQuote(unittest.TestCase):
    def test_quote(self):
        self.assertEqual(smb._smb_quote("a b.txt"), '"a b.txt"')


if __name__ == "__main__":
    unittest.main()
