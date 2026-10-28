"""Unit tests for smb_transfer_converted_stitching.py.

Run with `python -m unittest test_smb_transfer_converted_stitching` from this directory.
"""

import os
import sys
import tempfile
import unittest
from contextlib import ExitStack
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import smb_transfer_converted_stitching as cs  # noqa: E402


def _run_main(output_dir, xml_on_share=True, remote_dir_ok=True):
    """Run main with every SMB helper replaced except iterative_n5_transfer, which the test patches."""

    def fake_transfer(*args, mget_target, local_cwd, **kwargs):
        if mget_target == "*xml*" and xml_on_share:
            open(os.path.join(local_cwd, "dataset.xml"), "w").close()
        return True

    def fake_require(*args, **kwargs):
        if not remote_dir_ok:
            raise SystemExit("Cannot open the remote directory")

    argv = ["prog", "-u", "u", "-p", "P/sample", "-d", "2_converted_stitching/", "-o", output_dir]
    with ExitStack() as stack:
        stack.enter_context(mock.patch.object(sys, "argv", argv))
        stack.enter_context(mock.patch.object(cs.getpass, "getpass", return_value="pw"))
        stack.enter_context(mock.patch.object(cs, "require_remote_dir", side_effect=fake_require))
        stack.enter_context(mock.patch.object(
            cs, "list_remote_dirs", return_value=["dataset.n5", "interestpoints.n5", "other"]))
        stack.enter_context(mock.patch.object(cs, "transfer_path", side_effect=fake_transfer))
        stack.enter_context(mock.patch.object(cs, "verify_and_repair_download_generic"))
        stack.enter_context(mock.patch.object(cs, "verify_and_repair_n5"))
        cs.main()


class TestMain(unittest.TestCase):
    def test_transfers_image_n5_from_min_scale(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cs, "iterative_n5_transfer") as it_n5:
            _run_main(tmp)
        it_n5.assert_called_once()
        args, kwargs = it_n5.call_args
        self.assertEqual(args[2:4], ("P/sample/2_converted_stitching", "dataset.n5"))
        self.assertEqual(kwargs["min_scale"], 2)

    def test_no_xml_stops_before_image_transfer(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(cs, "iterative_n5_transfer") as it_n5:
                with self.assertRaises(SystemExit):
                    _run_main(tmp, xml_on_share=False)
            it_n5.assert_not_called()

    def test_wrong_remote_dir_creates_no_local_folder(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(cs, "iterative_n5_transfer"):
            with self.assertRaises(SystemExit):
                _run_main(tmp, remote_dir_ok=False)
            self.assertFalse(os.path.exists(os.path.join(tmp, "2_converted_stitching")))


if __name__ == "__main__":
    unittest.main()
