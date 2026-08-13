"""Unit tests for smb_transfer_files.py.

`run_smbclient` (in flamingo_tools.data_transfer_utils) is replaced by a recorder so
the tests need neither a real smbclient binary nor a network connection. Run with:

    python -m unittest test_smb_transfer_files   # from scripts/data_transfer/
"""

import argparse
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smb_transfer_files as sf  # noqa: E402
import flamingo_tools.data_transfer_utils as dtu  # noqa: E402


class Recorder:
    """Stand-in for run_smbclient. Records (commands, cwd) and returns scripted results."""

    def __init__(self, responses=None):
        # responses: list of (lines, had_disconnect, rc); the last is reused when exhausted.
        self.responses = responses or [([], False, 0)]
        self.calls = []

    def __call__(self, username, password, commands, cwd, smb_server=dtu.SMB_SERVER, **kwargs):
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


class TestSplitHelpers(unittest.TestCase):
    def test_split_remote_path_backslash(self):
        self.assertEqual(sf.split_remote_path("\\P\\n5\\file.txt"), ("/P/n5", "file.txt"))

    def test_split_remote_path_forward_slash(self):
        self.assertEqual(sf.split_remote_path("/P/n5/file.txt"), ("/P/n5", "file.txt"))

    def test_split_remote_path_trailing_slash_dir(self):
        self.assertEqual(sf.split_remote_path("\\P\\n5\\sub\\"), ("/P/n5", "sub"))

    def test_split_local_path(self):
        self.assertEqual(sf.split_local_path("/tmp/x/y.bin"), ("/tmp/x", "y.bin"))

    def test_split_local_path_trailing_sep(self):
        self.assertEqual(sf.split_local_path("/tmp/x/y/"), ("/tmp/x", "y"))


class TestDownloadFile(unittest.TestCase):
    def test_uses_reget_with_rename(self):
        rec = Recorder()
        with mock.patch.object(dtu, "run_smbclient", rec):
            with tempfile.TemporaryDirectory() as tmp:
                ok = sf.download_file("u", "p", "R", "foo.raw", tmp, "bar.raw")
        self.assertTrue(ok)
        self.assertEqual(rec.command_lists[0], ['cd "R"', 'reget "foo.raw" "bar.raw"'])

    def test_same_name(self):
        rec = Recorder()
        with mock.patch.object(dtu, "run_smbclient", rec):
            with tempfile.TemporaryDirectory() as tmp:
                sf.download_file("u", "p", "R", "same.raw", tmp, "same.raw")
        self.assertEqual(rec.command_lists[0], ['cd "R"', 'reget "same.raw" "same.raw"'])


class TestUploadFile(unittest.TestCase):
    def test_uses_put_with_rename(self):
        rec = Recorder()
        with mock.patch.object(dtu, "run_smbclient", rec):
            with tempfile.TemporaryDirectory() as tmp:
                ok = sf.upload_file("u", "p", "R/sub", "remote.raw", tmp, "local.raw", ensure=False)
        self.assertTrue(ok)
        self.assertEqual(rec.command_lists[0], ['cd "R/sub"', 'put "local.raw" "remote.raw"'])

    def test_ensure_creates_dir_first(self):
        rec = Recorder()
        with mock.patch.object(dtu, "run_smbclient", rec):
            with tempfile.TemporaryDirectory() as tmp:
                sf.upload_file("u", "p", "R/sub", "x.raw", tmp, "x.raw", ensure=True)
        self.assertEqual(len(rec.calls), 2)
        self.assertEqual(rec.command_lists[0], ['mkdir "R/sub"'])
        self.assertEqual(rec.command_lists[1], ['cd "R/sub"', 'put "x.raw" "x.raw"'])

    def test_ensure_false_skips_mkdir(self):
        rec = Recorder()
        with mock.patch.object(dtu, "run_smbclient", rec):
            with tempfile.TemporaryDirectory() as tmp:
                sf.upload_file("u", "p", "R/sub", "x.raw", tmp, "x.raw", ensure=False)
        self.assertEqual(len(rec.calls), 1)

    def test_upload_error_token_triggers_retry(self):
        rec = Recorder(responses=[
            (["some", "NT_STATUS_ACCESS_DENIED"], False, 0),
            ([], False, 0),
        ])
        with mock.patch.object(dtu, "run_smbclient", rec), mock.patch.object(dtu.time, "sleep"):
            with tempfile.TemporaryDirectory() as tmp:
                ok = sf.upload_file("u", "p", "R", "x.raw", tmp, "x.raw", ensure=False)
        self.assertTrue(ok)
        self.assertEqual(len(rec.calls), 2)


class TestTransferSingleFileEmptyRetry(unittest.TestCase):
    def test_download_success_nonempty_first_try(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = sf.FileJob("R", "remote.raw", tmp, "local.raw", label="local.raw")

            def fake_download(*args, **kwargs):
                _touch(os.path.join(tmp, "local.raw"), size=10)
                return True

            with mock.patch.object(sf, "download_file", side_effect=fake_download) as m:
                ok = sf.transfer_single_file("u", "p", job, ingest=False)
            self.assertTrue(ok)
            self.assertEqual(m.call_count, 1)

    def test_download_empty_then_succeeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = sf.FileJob("R", "remote.raw", tmp, "local.raw", label="local.raw")
            calls = []

            def fake_download(*args, **kwargs):
                calls.append(1)
                size = 0 if len(calls) == 1 else 10
                _touch(os.path.join(tmp, "local.raw"), size=size)
                return True

            with mock.patch.object(sf, "download_file", side_effect=fake_download):
                ok = sf.transfer_single_file("u", "p", job, ingest=False, empty_retries=3)
            self.assertTrue(ok)
            self.assertEqual(len(calls), 2)

    def test_download_empty_all_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = sf.FileJob("R", "remote.raw", tmp, "local.raw", label="local.raw")

            def fake_download(*args, **kwargs):
                _touch(os.path.join(tmp, "local.raw"), size=0)
                return True

            log_file = os.path.join(tmp, "log.txt")
            with mock.patch.object(sf, "download_file", side_effect=fake_download) as m:
                ok = sf.transfer_single_file(
                    "u", "p", job, ingest=False, empty_retries=3, log_file=log_file,
                )
            self.assertFalse(ok)
            self.assertEqual(m.call_count, 3)
            with open(log_file) as f:
                content = f.read()
            self.assertIn("still empty after 3 attempts", content)

    def test_download_transfer_itself_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            job = sf.FileJob("R", "remote.raw", tmp, "local.raw", label="local.raw")
            with mock.patch.object(sf, "download_file", return_value=False) as m:
                ok = sf.transfer_single_file("u", "p", job, ingest=False, empty_retries=3)
            self.assertFalse(ok)
            self.assertEqual(m.call_count, 1)

    def test_upload_empty_source_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "local.raw"), size=0)
            job = sf.FileJob("R", "remote.raw", tmp, "local.raw", label="local.raw")
            log_file = os.path.join(tmp, "log.txt")
            with mock.patch.object(sf, "upload_file") as m:
                ok = sf.transfer_single_file("u", "p", job, ingest=True, log_file=log_file)
            self.assertFalse(ok)
            m.assert_not_called()
            with open(log_file) as f:
                content = f.read()
            self.assertIn("local source file is empty", content)

    def test_upload_nonempty_source_calls_upload_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "local.raw"), size=10)
            job = sf.FileJob("R", "remote.raw", tmp, "local.raw", label="local.raw")
            with mock.patch.object(sf, "upload_file", return_value=True) as m:
                ok = sf.transfer_single_file("u", "p", job, ingest=True)
            self.assertTrue(ok)
            self.assertEqual(m.call_count, 1)


class TestTransferDirectory(unittest.TestCase):
    def test_download_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            job = sf.DirJob(remote_dir="R", local_dir=tmp, transferred_name="data", rename_to=None)
            with mock.patch.object(sf, "transfer_path", return_value=True) as m:
                ok = sf.transfer_directory("u", "p", job, ingest=False)
            self.assertTrue(ok)
            m.assert_called_once_with(
                "u", "p", "R", "data", tmp, retries=sf.MAX_RETRIES,
                log_file=None, smb_server=sf.SMB_SERVER,
            )

    def test_download_rename_after_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            _touch(os.path.join(tmp, "data", "f.bin"), 5)
            job = sf.DirJob(
                remote_dir="R", local_dir=tmp, transferred_name="data", rename_to="renamed",
            )
            with mock.patch.object(sf, "transfer_path", return_value=True):
                ok = sf.transfer_directory("u", "p", job, ingest=False)
            self.assertTrue(ok)
            self.assertFalse(os.path.exists(os.path.join(tmp, "data")))
            self.assertTrue(os.path.isdir(os.path.join(tmp, "renamed")))

    def test_download_rename_skipped_if_destination_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            os.makedirs(os.path.join(tmp, "renamed"))
            job = sf.DirJob(
                remote_dir="R", local_dir=tmp, transferred_name="data", rename_to="renamed",
            )
            with mock.patch.object(sf, "transfer_path", return_value=True):
                ok = sf.transfer_directory("u", "p", job, ingest=False)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(os.path.join(tmp, "data")))

    def test_download_flags_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _touch(os.path.join(tmp, "data", "empty.bin"), 0)
            job = sf.DirJob(remote_dir="R", local_dir=tmp, transferred_name="data", rename_to=None)
            log_file = os.path.join(tmp, "log.txt")
            with mock.patch.object(sf, "transfer_path", return_value=True):
                sf.transfer_directory("u", "p", job, ingest=False, log_file=log_file)
            with open(log_file) as f:
                content = f.read()
            self.assertIn("empty.bin", content)
            self.assertIn("is empty after transfer", content)

    def test_ingest_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            job = sf.DirJob(remote_dir="R", local_dir=tmp, transferred_name="data", rename_to=None)
            with mock.patch.object(sf, "upload_path", return_value=True) as m:
                ok = sf.transfer_directory("u", "p", job, ingest=True)
            self.assertTrue(ok)
            m.assert_called_once_with(
                "u", "p", "R", "data", tmp, is_dir=True,
                retries=sf.MAX_RETRIES, log_file=None, smb_server=sf.SMB_SERVER,
            )

    def test_ingest_rename_mismatch_warns_and_uploads_original_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "data"))
            job = sf.DirJob(
                remote_dir="R", local_dir=tmp, transferred_name="data", rename_to="other",
            )
            with mock.patch.object(sf, "upload_path", return_value=True) as m:
                ok = sf.transfer_directory("u", "p", job, ingest=True)
            self.assertTrue(ok)
            m.assert_called_once_with(
                "u", "p", "R", "data", tmp, is_dir=True,
                retries=sf.MAX_RETRIES, log_file=None, smb_server=sf.SMB_SERVER,
            )


class TestReadManifest(unittest.TestCase):
    def test_basic_with_differing_basenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write("UKONspezial\tlocal\n")
                f.write("\\archiv\\data\\raw_0001.tif\t/local/data/renamed_0001.tif\n")
                f.write("\\archiv\\data\\raw_0002.tif\t/local/data/raw_0002.tif\n")
            jobs = sf.read_manifest(manifest)
        self.assertEqual(len(jobs), 2)
        self.assertEqual(jobs[0].remote_dir, "/archiv/data")
        self.assertEqual(jobs[0].remote_name, "raw_0001.tif")
        self.assertEqual(jobs[0].local_dir, "/local/data")
        self.assertEqual(jobs[0].local_name, "renamed_0001.tif")
        self.assertNotEqual(jobs[0].remote_name, jobs[0].local_name)
        self.assertEqual(jobs[1].remote_name, jobs[1].local_name)

    def test_missing_column_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write("UKONspezial\twrong_col\n")
                f.write("\\archiv\\a.tif\tsomething\n")
            with self.assertRaises(ValueError):
                sf.read_manifest(manifest)

    def test_blank_cell_logged_and_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = os.path.join(tmp, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write("UKONspezial\tlocal\n")
                f.write("\\archiv\\a.tif\t\n")
                f.write("\\archiv\\b.tif\t/local/b.tif\n")
            log_file = os.path.join(tmp, "log.txt")
            jobs = sf.read_manifest(manifest, log_file=log_file)
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0].local_name, "b.tif")
            with open(log_file) as f:
                content = f.read()
            self.assertIn("missing 'local' value", content)


class TestManifestEndToEnd(unittest.TestCase):
    def test_mixed_basenames_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_root = os.path.join(tmp, "local")
            os.makedirs(local_root)
            manifest = os.path.join(tmp, "manifest.tsv")
            with open(manifest, "w") as f:
                f.write("UKONspezial\tlocal\n")
                f.write(f"\\archiv\\a.tif\t{os.path.join(local_root, 'renamed_a.tif')}\n")
                f.write(f"\\archiv\\b.tif\t{os.path.join(local_root, 'b.tif')}\n")
            jobs = sf.read_manifest(manifest)

            def fake_download(
                username, password, remote_cd, remote_name, local_cwd, local_name, **kwargs,
            ):
                _touch(os.path.join(local_cwd, local_name), size=10)
                return True

            results = []
            with mock.patch.object(sf, "download_file", side_effect=fake_download):
                for job in jobs:
                    results.append(sf.transfer_single_file("u", "p", job, ingest=False))
            self.assertEqual(results, [True, True])
            self.assertTrue(os.path.isfile(os.path.join(local_root, "renamed_a.tif")))
            self.assertTrue(os.path.isfile(os.path.join(local_root, "b.tif")))


class TestValidateArgs(unittest.TestCase):
    def _parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-r", "--remote", default=None)
        parser.add_argument("-l", "--local", default=None)
        parser.add_argument("--manifest", default=None)
        return parser

    def test_manifest_and_remote_local_mutually_exclusive(self):
        args = argparse.Namespace(remote="R", local="L", manifest="m.tsv")
        with self.assertRaises(SystemExit):
            sf.validate_args(args, self._parser())

    def test_neither_given(self):
        args = argparse.Namespace(remote=None, local=None, manifest=None)
        with self.assertRaises(SystemExit):
            sf.validate_args(args, self._parser())

    def test_only_remote_given(self):
        args = argparse.Namespace(remote="R", local=None, manifest=None)
        with self.assertRaises(SystemExit):
            sf.validate_args(args, self._parser())

    def test_manifest_alone_is_valid(self):
        args = argparse.Namespace(remote=None, local=None, manifest="m.tsv")
        sf.validate_args(args, self._parser())

    def test_remote_and_local_is_valid(self):
        args = argparse.Namespace(remote="R", local="L", manifest=None)
        sf.validate_args(args, self._parser())


class TestResolveEndpoints(unittest.TestCase):
    def _parser(self):
        return argparse.ArgumentParser()

    def test_download_both_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = os.path.join(tmp, "existing")
            os.makedirs(local_dir)
            args = argparse.Namespace(
                remote="\\archiv\\data", local=local_dir, ingest=False, smb_server="s",
            )
            with mock.patch.object(sf, "remote_dir_exists", return_value=True):
                kind, job = sf.resolve_endpoints(args, "u", "p", self._parser())
        self.assertEqual(kind, "dir")
        self.assertIsInstance(job, sf.DirJob)
        self.assertEqual(job.transferred_name, "data")
        self.assertEqual(job.rename_to, "existing")

    def test_download_remote_dir_local_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_path = os.path.join(tmp, "not_yet_there")
            args = argparse.Namespace(
                remote="\\archiv\\data", local=local_path, ingest=False, smb_server="s",
            )
            with mock.patch.object(sf, "remote_dir_exists", return_value=True):
                kind, job = sf.resolve_endpoints(args, "u", "p", self._parser())
        self.assertEqual(kind, "dir")
        self.assertEqual(job.rename_to, "not_yet_there")

    def test_download_mixed_dir_file_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_file = os.path.join(tmp, "existing.txt")
            _touch(local_file, 3)
            args = argparse.Namespace(
                remote="\\archiv\\data", local=local_file, ingest=False, smb_server="s",
            )
            with mock.patch.object(sf, "remote_dir_exists", return_value=True):
                with self.assertRaises(SystemExit):
                    sf.resolve_endpoints(args, "u", "p", self._parser())

    def test_download_file_into_existing_dir_borrows_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_dir = os.path.join(tmp, "dest")
            os.makedirs(local_dir)
            args = argparse.Namespace(
                remote="\\archiv\\file.tif", local=local_dir, ingest=False, smb_server="s",
            )
            with mock.patch.object(sf, "remote_dir_exists", return_value=False):
                kind, job = sf.resolve_endpoints(args, "u", "p", self._parser())
        self.assertEqual(kind, "file")
        self.assertEqual(job.local_name, "file.tif")
        self.assertEqual(job.local_dir, local_dir)

    def test_ingest_local_missing_errors(self):
        args = argparse.Namespace(remote="R", local="/no/such/path", ingest=True, smb_server="s")
        with self.assertRaises(SystemExit):
            sf.resolve_endpoints(args, "u", "p", self._parser())

    def test_ingest_file_into_existing_remote_dir_borrows_basename(self):
        with tempfile.TemporaryDirectory() as tmp:
            local_file = os.path.join(tmp, "x.tif")
            _touch(local_file, 3)
            args = argparse.Namespace(
                remote="/archiv/dest", local=local_file, ingest=True, smb_server="s",
            )
            with mock.patch.object(sf, "remote_dir_exists", return_value=True):
                kind, job = sf.resolve_endpoints(args, "u", "p", self._parser())
        self.assertEqual(kind, "file")
        self.assertEqual(job.remote_name, "x.tif")
        self.assertEqual(job.remote_dir, "/archiv/dest")


if __name__ == "__main__":
    unittest.main()
