import errno
import os
import unittest
import xml.etree.ElementTree as ET
from tempfile import TemporaryDirectory
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


class FakeShare:
    """Serve smbclient `reget` requests from a local directory.

    With max_bytes_per_call set, an attempt stops early and reports NT_STATUS_IO_TIMEOUT while
    still exiting 0, which is exactly how the real failure presented.
    """

    def __init__(self, root, max_bytes_per_call=None):
        self.root = root
        self.max_bytes_per_call = max_bytes_per_call
        self.calls = []

    def __call__(self, username, password, commands, cwd, smb_server=None, timeout=None):
        self.calls.append(list(commands))
        remote_cd = commands[0].split('"')[1]
        name = commands[1].split('"')[1]

        rel_dir = remote_cd[len(self.root):].strip("/")
        source = os.path.join(self.root, *rel_dir.split("/"), name) if rel_dir \
            else os.path.join(self.root, name)
        total = os.path.getsize(source)

        os.makedirs(cwd, exist_ok=True)
        local_path = os.path.join(cwd, name)
        have = os.path.getsize(local_path) if os.path.exists(local_path) else 0

        want = total - have
        if self.max_bytes_per_call is not None:
            want = min(want, self.max_bytes_per_call)
        with open(source, "rb") as src:
            src.seek(have)
            chunk = src.read(want)
        with open(local_path, "ab") as dst:
            dst.write(chunk)

        if have + len(chunk) < total:
            # Report the abort the way smbclient did: a status line, no disconnect flag and
            # exit code 0. Success must therefore be decided by the local size alone.
            return [f"NT_STATUS_IO_TIMEOUT listing \\{name}"], False, 0
        return [], False, 0


def read_setup(out_path, setup_id, scale=0):
    from elf.io import open_file

    with open_file(out_path, "r") as f:
        return f[f"setup{setup_id}/timepoint0/s{scale}"][:]


class TestDataConversion(unittest.TestCase):
    def setUp(self):
        from flamingo_tools import create_test_data

        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.folder = tmp_dir.name
        create_test_data(self.folder)

    def test_convert_lightsheet_to_bdv(self):
        from flamingo_tools import convert_lightsheet_to_bdv

        out_path = os.path.join(self.folder, "converted_data.n5")
        convert_lightsheet_to_bdv(self.folder, out_path=out_path, metadata_file_name_pattern=None)

        self.assertTrue(os.path.exists(out_path))
        xml_path = out_path.replace(".n5", ".xml")
        self.assertTrue(os.path.exists(xml_path))


class TestRawConversion(unittest.TestCase):
    """Streaming conversion of raw tiles, including resume and the guard rails."""

    def setUp(self):
        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.folder = tmp_dir.name
        self.root = os.path.join(self.folder, "input")
        os.makedirs(self.root, exist_ok=True)
        self.volumes = make_raw_input(self.root)
        self.out_path = os.path.join(self.folder, "out.n5")

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

    def test_every_level_has_the_gzip_level_and_its_factors(self):
        import json
        from flamingo_tools.data_conversion import GZIP_LEVEL

        self._convert()
        for level in range(6):
            path = os.path.join(self.out_path, "setup0", "timepoint0", f"s{level}", "attributes.json")
            with open(path) as f:
                attributes = json.load(f)
            self.assertEqual(attributes["compression"], {"type": "gzip", "level": GZIP_LEVEL})
            self.assertEqual(attributes["downsamplingFactors"], [2 ** level] * 3)

    def test_pyramid_matches_pybdv_downsample(self):
        from elf.io import open_file
        from pybdv.downsample import downsample

        self._convert()
        in_key = "setup0/timepoint0/s0"
        for level in range(1, 6):
            downsample(self.out_path, in_key, f"reference/s{level}", [2, 2, 2], "mean")
            in_key = f"reference/s{level}"
        with open_file(self.out_path, "r") as f:
            for level in range(1, 6):
                self.assertTrue(np.array_equal(read_setup(self.out_path, 0, level), f[f"reference/s{level}"][:]))

    def test_n_threads_defaults_to_the_slurm_allocation(self):
        import flamingo_tools.data_conversion as dc

        with mock.patch.dict(os.environ, {"SLURM_CPUS_PER_TASK": "3"}), \
             mock.patch.object(dc, "_convert_tile", wraps=dc._convert_tile) as convert_tile:
            self._convert()
        self.assertEqual({call.args[4] for call in convert_tile.call_args_list}, {3})

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

    def _size_map(self):
        size_map = {}
        for dirpath, _, names in os.walk(self.root):
            for name in names:
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, self.root).replace(os.sep, "/")
                size_map[rel] = os.path.getsize(path)
        return size_map

    @staticmethod
    def _retry_config(**kwargs):
        from flamingo_tools.data_transfer_utils import RetryConfig

        kwargs.setdefault("retry_delay", 0.0)
        return RetryConfig(**kwargs)

    def _convert_over_smb(self, share, **kwargs):
        import flamingo_tools.data_conversion as dc
        import flamingo_tools.data_transfer_utils as dtu

        with mock.patch.object(dc, "remote_size_map_with_retry", return_value=self._size_map()), \
             mock.patch.object(dtu, "run_smbclient", share), mock.patch("time.sleep"):
            self._convert(username="tester", password="secret",
                          stage_dir=os.path.join(self.folder, "smb_stage"), **kwargs)

    def test_smb_mode_downloads_every_file(self):
        share = FakeShare(self.root)
        self._convert_over_smb(share)
        for tile, expected in self.volumes.items():
            self.assertTrue(np.array_equal(read_setup(self.out_path, tile), expected))

    def test_smb_mode_resumes_a_truncated_download(self):
        """Regression: a mid-file NT_STATUS_IO_TIMEOUT used to abort the whole conversion.

        smbclient exits 0 after such an abort, so the transfer looked successful and the size
        check then raised instead of continuing the download.
        """
        # A raw tile is 921600 bytes, so this truncates each one into several attempts.
        share = FakeShare(self.root, max_bytes_per_call=300000)
        self._convert_over_smb(share)

        for tile, expected in self.volumes.items():
            self.assertTrue(np.array_equal(read_setup(self.out_path, tile), expected))

        regets = [c for c in share.calls if any("reget" in part for part in c)]
        self.assertGreater(len(regets), len(self.volumes), "the download was not resumed")
        flat = " ".join(part for call in share.calls for part in call)
        self.assertNotIn("recurse", flat)
        self.assertNotIn("mget", flat)

    def test_single_attempt_still_reports_a_truncated_download(self):
        """With no retries left the run must fail loudly rather than convert a partial file.

        This is the behavior before the fix, and it confirms the test above passes because the
        download is resumed and not because the truncation goes unnoticed.
        """
        share = FakeShare(self.root, max_bytes_per_call=300000)
        with self.assertRaises(RuntimeError) as raised:
            self._convert_over_smb(share, retry_config=self._retry_config(max_retries=1))
        self.assertIn("the share reports", str(raised.exception))

    def test_dry_run_writes_nothing(self):
        self._convert(dry_run=True)
        self.assertFalse(os.path.exists(self.out_path))


class TestPyramidCascade(unittest.TestCase):
    """The pyramid is computed from the s0 slabs in memory, with the values of pybdv."""

    # Slab depth 128: the last slab is partial, two slabs make one s1 chunk, and the z size of
    # s3 is odd (57).
    SHAPE = (456, 200, 136)

    def setUp(self):
        tmp_dir = TemporaryDirectory()
        self.addCleanup(tmp_dir.cleanup)
        self.folder = tmp_dir.name
        self.root = os.path.join(self.folder, "input")
        self.out_path = os.path.join(self.folder, "out.n5")

    def _make_input(self, shape=SHAPE):
        make_raw_input(self.root, n_tiles=1, shape=shape)

    def _convert(self, **kwargs):
        from flamingo_tools import convert_lightsheet_to_bdv

        convert_lightsheet_to_bdv(self.root, out_path=self.out_path, file_ext=".raw", slab_memory=0.001, **kwargs)

    def _assert_matches_pybdv(self, scale_factors):
        from elf.io import open_file
        from pybdv.downsample import downsample

        in_key = "setup0/timepoint0/s0"
        for level, factor in enumerate(scale_factors, 1):
            downsample(self.out_path, in_key, f"reference/s{level}", factor, "mean")
            in_key = f"reference/s{level}"
        with open_file(self.out_path, "r") as f:
            for level in range(1, len(scale_factors) + 1):
                self.assertTrue(np.array_equal(read_setup(self.out_path, 0, level), f[f"reference/s{level}"][:]))

    def test_matches_pybdv_without_reading_s0_back(self):
        import flamingo_tools.data_conversion as dc

        self._make_input()
        with mock.patch.object(dc, "_write_pyramid_level", wraps=dc._write_pyramid_level) as read_back:
            self._convert()
        read_back.assert_not_called()
        self._assert_matches_pybdv([[2, 2, 2]] * 5)

    def test_resumed_tile_reads_s0_back(self):
        import flamingo_tools.data_conversion as dc

        self._make_input()
        real_iter = dc._iter_slabs
        budget = {"slabs": 2}

        def failing_iter(*args, **kwargs):
            for z0, z1, slab in real_iter(*args, **kwargs):
                if budget["slabs"] <= 0:
                    raise OSError(errno.EIO, "simulated connection loss")
                budget["slabs"] -= 1
                yield z0, z1, slab

        with mock.patch.object(dc, "_iter_slabs", failing_iter):
            with self.assertRaises(OSError):
                self._convert()
        with mock.patch.object(dc, "_write_pyramid_level", wraps=dc._write_pyramid_level) as read_back:
            self._convert()
        self.assertEqual(read_back.call_count, 5)
        self._assert_matches_pybdv([[2, 2, 2]] * 5)

    def test_factor_that_does_not_divide_the_chunk_depth_reads_s0_back(self):
        import flamingo_tools.data_conversion as dc

        self._make_input()
        with mock.patch.object(dc, "_write_pyramid_level", wraps=dc._write_pyramid_level) as read_back:
            self._convert(scale_factors=[[3, 3, 3]])
        self.assertEqual(read_back.call_count, 1)
        self._assert_matches_pybdv([[3, 3, 3]])

    def test_axis_shorter_than_its_factor_reads_s0_back(self):
        import flamingo_tools.data_conversion as dc

        # s4 has 1 plane, so the z window of s5 is incomplete.
        self._make_input((20, 64, 64))
        with mock.patch.object(dc, "_write_pyramid_level", wraps=dc._write_pyramid_level) as read_back:
            self._convert()
        self.assertEqual(read_back.call_count, 5)
        self._assert_matches_pybdv([[2, 2, 2]] * 5)


class TestIntegerBlockMean(unittest.TestCase):
    def test_matches_the_truncated_float_mean(self):
        from pybdv.downsample import get_downsampler
        from flamingo_tools.data_conversion import _integer_block_mean

        float_mean = get_downsampler("mean")
        rng = np.random.default_rng(0)
        for dtype, high in (("uint8", 255), ("uint16", 65535), ("uint32", 2 ** 32 - 1)):
            data = rng.integers(0, high, size=(8, 12, 16), dtype=dtype, endpoint=True)
            for factor in ([2, 2, 2], [1, 2, 2], [2, 3, 4]):
                out_shape = tuple(sh // sf for sh, sf in zip(data.shape, factor))
                expected = float_mean(data, factor, out_shape).astype(dtype)
                result = _integer_block_mean(data, factor, out_shape)
                self.assertEqual(result.dtype, np.dtype(dtype))
                self.assertTrue(np.array_equal(result, expected), (dtype, factor))

    def test_matches_pybdv_for_incomplete_windows(self):
        from pybdv.downsample import get_downsampler, sample_shape
        from flamingo_tools.data_conversion import _integer_block_mean

        float_mean = get_downsampler("mean")
        rng = np.random.default_rng(0)
        factor = (2, 2, 2)
        # pybdv gives an axis that is shorter than its factor 1 voxel, and drops the rest of an
        # axis that the factor does not divide.
        for shape in ((1, 6, 5), (3, 4, 5), (2, 1, 3)):
            data = rng.integers(0, 65535, size=shape, dtype="uint16", endpoint=True)
            out_shape = sample_shape(shape, factor)
            expected = float_mean(data, factor, out_shape).astype("uint16")
            result = _integer_block_mean(data, factor, out_shape)
            self.assertTrue(np.array_equal(result, expected), shape)


class TestPrefetch(unittest.TestCase):
    def test_yields_every_item_in_order(self):
        from flamingo_tools.data_conversion import _prefetch

        self.assertEqual(list(_prefetch(iter(range(10)))), list(range(10)))

    def test_reader_error_reaches_the_caller(self):
        from flamingo_tools.data_conversion import _prefetch

        def failing_reader():
            yield 0
            yield 1
            raise OSError(errno.EIO, "simulated read error")

        received = []
        with self.assertRaises(OSError):
            for item in _prefetch(failing_reader()):
                received.append(item)
        self.assertEqual(received, [0, 1])

    def test_reader_stops_when_the_caller_stops(self):
        import itertools
        import threading
        from flamingo_tools.data_conversion import _prefetch

        produced = []

        def endless_reader():
            for i in itertools.count():
                produced.append(i)
                yield i

        n_threads = threading.active_count()
        items = _prefetch(endless_reader())
        self.assertEqual(next(items), 0)
        items.close()
        self.assertEqual(threading.active_count(), n_threads)
        # One item was taken, one waits in the queue, one waits to be put.
        self.assertLessEqual(len(produced), 3)


if __name__ == "__main__":
    unittest.main()
