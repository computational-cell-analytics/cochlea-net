import errno
import os
import unittest
import xml.etree.ElementTree as ET
from shutil import rmtree
from unittest import mock

import numpy as np

SETTINGS_TEMPLATE = """<Camera Settings>
AOI width {width}
AOI height {height}
Number of planes saved {depth}
Plane spacing {spacing}
<Start Position>
X {x}
Y {y}
Z {z}
"""

RAW_SHAPE = (300, 32, 48)


def make_raw_input(root, n_tiles=2, shape=RAW_SHAPE):
    """Write synthetic flamingo raw tiles with their settings files.

    Args:
        root: Directory that receives one subfolder per tile.
        n_tiles: Number of tiles to write.
        shape: Shape of one tile in (z, y, x).

    Returns:
        {tile_index: volume} for the written tiles.
    """
    volumes = {}
    for tile in range(n_tiles):
        folder = os.path.join(root, f"tile{tile}")
        os.makedirs(folder, exist_ok=True)
        name = f"S000_t000000_V000_R000{tile}_X000_C00_I0_D0_P00000"
        data = np.random.default_rng(tile).integers(0, 4000, size=shape, dtype="uint16")
        data.tofile(os.path.join(folder, f"{name}.raw"))
        with open(os.path.join(folder, f"{name}_Settings.txt"), "w") as f:
            f.write(SETTINGS_TEMPLATE.format(
                width=shape[2], height=shape[1], depth=shape[0], spacing=0.38,
                x=1.0 + tile, y=2.0, z=3.0,
            ))
        volumes[tile] = data
    return volumes


def read_setup(out_path, setup_id, scale=0):
    from elf.io import open_file

    with open_file(out_path, "r") as f:
        return f[f"setup{setup_id}/timepoint0/s{scale}"][:]


class TestDataConversion(unittest.TestCase):
    folder = "./tmp"

    def setUp(self):
        from flamingo_tools import create_test_data

        # TODO Create flamingo metadata.
        create_test_data(self.folder)

    def tearDown(self):
        rmtree(self.folder)

    def test_convert_lightsheet_to_bdv(self):
        from flamingo_tools import convert_lightsheet_to_bdv

        out_path = os.path.join(self.folder, "converted_data.n5")
        convert_lightsheet_to_bdv(self.folder, out_path=out_path, metadata_file_name_pattern=None)

        self.assertTrue(os.path.exists(out_path))
        xml_path = out_path.replace(".n5", ".xml")
        self.assertTrue(os.path.exists(xml_path))


class TestRawConversion(unittest.TestCase):
    """Streaming conversion of raw tiles, including resume and the guard rails."""

    folder = "./tmp_raw"

    def setUp(self):
        self.root = os.path.join(self.folder, "input")
        os.makedirs(self.root, exist_ok=True)
        self.volumes = make_raw_input(self.root)
        self.out_path = os.path.join(self.folder, "out.n5")

    def tearDown(self):
        rmtree(self.folder)

    def _convert(self, out_path=None, **kwargs):
        from flamingo_tools import convert_lightsheet_to_bdv

        kwargs.setdefault("slab_memory", 0.001)  # forces more than one slab per tile
        convert_lightsheet_to_bdv(
            self.root, out_path=self.out_path if out_path is None else out_path,
            file_ext=".raw", **kwargs
        )

    def test_streams_the_source_volumes(self):
        self._convert()
        for tile, expected in self.volumes.items():
            self.assertTrue(np.array_equal(read_setup(self.out_path, tile), expected))

    def test_writes_the_full_pyramid(self):
        from elf.io import open_file

        self._convert()
        with open_file(self.out_path, "r") as f:
            for level in range(6):
                self.assertIn(f"setup0/timepoint0/s{level}", f)

    def test_rerun_is_a_no_op(self):
        self._convert()
        self._convert()
        for tile, expected in self.volumes.items():
            self.assertTrue(np.array_equal(read_setup(self.out_path, tile), expected))

    def test_missing_views_stays_unique(self):
        self._convert()
        self._convert()
        seqdesc = ET.parse(self.out_path.replace(".n5", ".xml")).getroot().find("SequenceDescription")
        self.assertEqual(len(seqdesc.findall("MissingViews")), 1)

    def test_resume_after_an_interruption(self):
        import flamingo_tools.data_conversion as dc
        from flamingo_tools.data_conversion import ConversionState

        reference = os.path.join(self.folder, "reference.n5")
        self._convert(out_path=reference)

        real_iter = dc._iter_slabs
        budget = {"slabs": 1}

        def failing_iter(*args, **kwargs):
            for z0, z1, slab in real_iter(*args, **kwargs):
                if budget["slabs"] <= 0:
                    raise OSError(errno.EIO, "simulated connection loss")
                budget["slabs"] -= 1
                yield z0, z1, slab

        with mock.patch.object(dc, "_iter_slabs", failing_iter):
            with self.assertRaises(OSError):
                self._convert()

        entry = ConversionState.load_or_create(self.out_path).setup("c0-t0-i0-d0")
        self.assertGreater(entry["z_done"], 0)
        self.assertLess(entry["z_done"], RAW_SHAPE[0])
        self.assertFalse(entry["done"])

        self._convert()
        for tile, expected in self.volumes.items():
            self.assertTrue(np.array_equal(read_setup(self.out_path, tile), expected))
        for level in range(1, 6):
            self.assertTrue(np.array_equal(
                read_setup(self.out_path, 0, level), read_setup(reference, 0, level)
            ))

    def test_recovers_from_a_dropped_read(self):
        import flamingo_tools.data_transfer_utils as dtu
        from test_data_transfer_utils import flaky_open

        raw_file = os.path.join(self.root, "tile0", "S000_t000000_V000_R0000_X000_C00_I0_D0_P00000.raw")
        fake_open, budget = flaky_open(raw_file, failures=3)
        with mock.patch.object(dtu, "open", fake_open, create=True), mock.patch("time.sleep"):
            self._convert(retry_config=dtu.RetryConfig(max_retries=5, retry_delay=0.0))
        self.assertEqual(budget["failures"], 0)
        self.assertTrue(np.array_equal(read_setup(self.out_path, 0), self.volumes[0]))

    def test_output_without_progress_file_is_refused(self):
        from flamingo_tools.data_conversion import state_path_for

        self._convert()
        os.remove(state_path_for(self.out_path))
        with self.assertRaises(RuntimeError):
            self._convert()

    def test_restart_converts_again(self):
        from flamingo_tools.data_conversion import state_path_for

        self._convert()
        os.remove(state_path_for(self.out_path))
        self._convert(restart=True)
        self.assertTrue(np.array_equal(read_setup(self.out_path, 0), self.volumes[0]))

    def test_truncated_raw_file_is_reported(self):
        raw_file = os.path.join(self.root, "tile0", "S000_t000000_V000_R0000_X000_C00_I0_D0_P00000.raw")
        with open(raw_file, "r+b") as f:
            f.truncate(os.path.getsize(raw_file) - 1024)
        with self.assertRaises(RuntimeError):
            self._convert()

    def test_metadata_count_mismatch_is_reported(self):
        os.remove(os.path.join(
            self.root, "tile1", "S000_t000000_V000_R0001_X000_C00_I0_D0_P00000_Settings.txt"
        ))
        with self.assertRaises(RuntimeError):
            self._convert()

    def test_stage_dir_converts_and_cleans_up(self):
        stage_dir = os.path.join(self.folder, "stage")
        self._convert(stage_dir=stage_dir)
        self.assertTrue(np.array_equal(read_setup(self.out_path, 1), self.volumes[1]))
        self.assertEqual(os.listdir(stage_dir), [])

    def test_smb_mode_downloads_every_file(self):
        """Drive the SMB code path with smbclient replaced by a local copy."""
        import flamingo_tools.data_conversion as dc

        size_map = {}
        for dirpath, _, names in os.walk(self.root):
            for name in names:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, self.root).replace(os.sep, "/")
                size_map[rel] = os.path.getsize(path)

        def fake_transfer_path(username, password, remote_cd, mget_target, local_cwd, **kwargs):
            rel_dir = remote_cd[len(self.root):].strip("/")
            source = os.path.join(self.root, *rel_dir.split("/"), mget_target) if rel_dir \
                else os.path.join(self.root, mget_target)
            os.makedirs(local_cwd, exist_ok=True)
            with open(source, "rb") as src, open(os.path.join(local_cwd, mget_target), "wb") as dst:
                dst.write(src.read())
            return True

        with mock.patch.object(dc, "remote_size_map_with_retry", return_value=size_map), \
             mock.patch.object(dc, "transfer_path", side_effect=fake_transfer_path):
            self._convert(username="tester", password="secret",
                          stage_dir=os.path.join(self.folder, "smb_stage"))

        for tile, expected in self.volumes.items():
            self.assertTrue(np.array_equal(read_setup(self.out_path, tile), expected))

    def test_dry_run_writes_nothing(self):
        self._convert(dry_run=True)
        self.assertFalse(os.path.exists(self.out_path))


if __name__ == "__main__":
    unittest.main()
