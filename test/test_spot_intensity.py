import unittest

import numpy as np
import pandas as pd
import zarr


# Anisotropic voxels, so that a swapped axis order changes the result.
VOXEL_SIZE = (0.5, 0.5, 1.0)  # x, y, z
SHAPE = (40, 60, 60)  # Z, Y, X
BACKGROUND = 100.0
AMPLITUDE = 1000.0
SIGMA_UM = 0.6


def _volume(spots_um, shape=SHAPE, amplitude=AMPLITUDE, background=BACKGROUND, noise=0.0, seed=0):
    grid = np.meshgrid(*(np.arange(s) for s in shape), indexing="ij")
    voxel_zyx = np.asarray(VOXEL_SIZE)[::-1]
    volume = np.full(shape, background, dtype="float64")
    for spot in spots_um:
        center = np.asarray(spot)[::-1] / voxel_zyx
        dist2 = sum(((g - c) * v) ** 2 for g, c, v in zip(grid, center, voxel_zyx))
        volume += amplitude * np.exp(-dist2 / (2 * SIGMA_UM ** 2))
    if noise > 0:
        volume += np.random.default_rng(seed).normal(0, noise, size=shape)
    return volume


def _cube_mean(volume, position_um, halo):
    center = np.round(np.asarray(position_um)[::-1] / np.asarray(VOXEL_SIZE)[::-1]).astype(int)
    return volume[tuple(slice(max(c - halo, 0), c + halo + 1) for c in center)].mean()


def _spots(positions_um):
    positions_um = np.asarray(positions_um, dtype="float64")
    return pd.DataFrame({
        "spot_id": np.arange(1, len(positions_um) + 1),
        "x": positions_um[:, 0], "y": positions_um[:, 1], "z": positions_um[:, 2],
    })


class TestMeasureSpotIntensities(unittest.TestCase):

    def setUp(self):
        from flamingo_tools.synapse_detection.spot_intensity import measure_spot_intensities
        self.measure = measure_spot_intensities
        # Positions on the voxel grid, in micrometer (x, y, z). The volume spans 30 x 30 x 40 um.
        self.positions = [(10.0, 10.0, 12.0), (20.0, 15.0, 25.0)]

    def test_isolated_spots(self):
        volume = _volume(self.positions)
        result = self.measure({"stain": volume}, _spots(self.positions), voxel_size=VOXEL_SIZE, n_threads=1)
        # The default signal is the mean of the 3 x 3 x 3 voxels around the spot.
        expected = [_cube_mean(volume, position, 1) for position in self.positions]
        np.testing.assert_allclose(result.stain_signal, expected, rtol=1e-6)
        self.assertTrue(np.all(result.stain_signal < AMPLITUDE + BACKGROUND))
        np.testing.assert_allclose(result.stain_background, BACKGROUND, atol=1.0)
        np.testing.assert_allclose(result.stain_ratio, result.stain_signal / result.stain_background)
        np.testing.assert_allclose(result.stain_net_signal, result.stain_signal - result.stain_background)

        # The integrated intensity is the sum of the Gaussian over the core voxels.
        core = (0.8, 0.8, 1.0)
        center = np.asarray(self.positions[0])[::-1] / np.asarray(VOXEL_SIZE)[::-1]
        grid = np.meshgrid(*(np.arange(s) for s in SHAPE), indexing="ij")
        in_core = sum(((g - c) * v / r) ** 2 for g, c, v, r in zip(
            grid, center, np.asarray(VOXEL_SIZE)[::-1], core[::-1])) <= 1
        expected = np.sum(volume[in_core] - BACKGROUND)
        self.assertAlmostEqual(result.stain_integrated[0], expected, delta=0.01 * expected)
        self.assertEqual(result.n_core_voxels[0], in_core.sum())
        self.assertAlmostEqual(result.nn_distance[0], np.linalg.norm(np.subtract(*self.positions)))

    def test_axis_order(self):
        """A spot shows up at its own position, not at the position with x and z swapped."""
        volume = _volume([(5.0, 20.0, 25.0)])
        spots = _spots([(5.0, 20.0, 25.0), (25.0, 20.0, 5.0), (30.0, 20.0, 5.0)])
        result = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, signal_halo=0, n_threads=1)
        self.assertAlmostEqual(result.stain_signal[0], AMPLITUDE + BACKGROUND, places=3)
        self.assertAlmostEqual(result.stain_signal[1], BACKGROUND, places=3)
        # x = 30 um is voxel 60, outside of the volume.
        self.assertTrue(np.isnan(result.stain_signal[2]))

    def test_snr_with_noise(self):
        volume = _volume(self.positions, noise=10.0)
        result = self.measure({"stain": volume}, _spots(self.positions), voxel_size=VOXEL_SIZE, signal_halo=0,
                              n_threads=1)
        np.testing.assert_allclose(result.stain_noise, 10.0, rtol=0.2)
        np.testing.assert_allclose(result.stain_snr, AMPLITUDE / 10.0, rtol=0.25)

    def test_blocks_equal_single_block(self):
        positions = [(x, y, z) for x in (6.0, 15.0, 24.0) for y in (6.0, 24.0) for z in (8.0, 30.0)]
        volume = _volume(positions, noise=5.0)
        spots = _spots(positions)
        chunked = zarr.array(volume, chunks=(8, 8, 8))
        blocked = self.measure({"stain": chunked}, spots, voxel_size=VOXEL_SIZE, block_shape=(8, 8, 8), n_threads=4)
        single = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, block_shape=SHAPE, n_threads=1)
        pd.testing.assert_frame_equal(blocked, single)

    def test_border_spot(self):
        positions = [(0.0, 0.0, 0.0), (29.5, 29.5, 39.0)]
        volume = _volume(positions)
        result = self.measure({"stain": volume}, _spots(positions), voxel_size=VOXEL_SIZE, n_threads=1)
        # The 3 x 3 x 3 cube is clipped to the 2 x 2 x 2 voxels inside the volume.
        np.testing.assert_allclose(result.stain_signal, [_cube_mean(volume, p, 1) for p in positions], rtol=1e-6)
        self.assertEqual(volume[:2, :2, :2].mean(), _cube_mean(volume, positions[0], 1))
        self.assertTrue(np.all(result.n_background_voxels > 0))
        single = self.measure({"stain": volume}, _spots(positions), voxel_size=VOXEL_SIZE, signal_halo=0, n_threads=1)
        np.testing.assert_allclose(single.stain_signal, AMPLITUDE + BACKGROUND, rtol=1e-6)

    def test_signal_halo(self):
        volume = _volume(self.positions, noise=5.0)
        spots = _spots(self.positions)
        for halo in (0, 1, 2):
            result = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, signal_halo=halo, n_threads=1)
            expected = [_cube_mean(volume, position, halo) for position in self.positions]
            np.testing.assert_allclose(result.stain_signal, expected, rtol=1e-6)
            np.testing.assert_allclose(result.stain_ratio, result.stain_signal / result.stain_background)
            np.testing.assert_allclose(
                result.stain_snr, (result.stain_signal - result.stain_background) / result.stain_noise
            )
        # The halo reaches beyond the outer shell radius of 1 voxel in x and y.
        result = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, signal_halo=3, n_threads=1,
                              core_radius=0.4, background_radii=(0.45, 0.5))
        expected = [_cube_mean(volume, position, 3) for position in self.positions]
        np.testing.assert_allclose(result.stain_signal, expected, rtol=1e-6)

    def test_neighbor_exclusion(self):
        """A neighbor in the shell raises the background without the exclusion only."""
        positions = [(15.0, 15.0, 20.0), (17.5, 15.0, 20.0)]
        volume = _volume(positions)
        spots = _spots(positions)
        excluded = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, n_threads=1,
                                background_radii=(1.5, 3.0), exclusion_radius=1.5)
        included = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, n_threads=1,
                                background_radii=(1.5, 3.0), exclusion_radius=1e-6)
        self.assertLess(excluded.n_background_voxels[0], included.n_background_voxels[0])
        self.assertLess(excluded.stain_background[0], included.stain_background[0])
        self.assertAlmostEqual(excluded.stain_background[0], BACKGROUND, delta=1.0)

    def test_saturation_and_channels(self):
        volume = np.clip(_volume(self.positions, amplitude=70000.0), 0, 65535).astype("uint16")
        dim = _volume(self.positions, amplitude=500.0).astype("float32")
        result = self.measure({"bright": volume, "dim": dim}, _spots(self.positions), voxel_size=VOXEL_SIZE,
                              signal_halo=0, n_threads=1)
        self.assertTrue(np.all(result.bright_n_saturated >= 1))
        self.assertTrue(np.all(result.bright_signal == 65535))
        self.assertTrue(result.dim_n_saturated.isna().all())
        np.testing.assert_allclose(result.dim_signal, 600.0, rtol=1e-5)

    def test_core_larger_than_shell(self):
        """A core beyond the outer shell radius is not cut off by the shell window."""
        volume = _volume(self.positions)
        core = (3.0, 3.0, 3.0)
        result = self.measure({"stain": volume}, _spots(self.positions[:1]), voxel_size=VOXEL_SIZE, n_threads=1,
                              core_radius=core, background_radii=(1.0, 2.0))
        center = np.asarray(self.positions[0])[::-1] / np.asarray(VOXEL_SIZE)[::-1]
        grid = np.meshgrid(*(np.arange(s) for s in SHAPE), indexing="ij")
        in_core = sum(((g - c) * v / r) ** 2 for g, c, v, r in zip(
            grid, center, np.asarray(VOXEL_SIZE)[::-1], core[::-1])) <= 1
        self.assertEqual(result.n_core_voxels[0], in_core.sum())

    def test_saturation_in_signal_cube(self):
        """A saturated voxel in a corner of the signal cube counts, also outside of the core."""
        volume = np.full(SHAPE, 100, dtype="uint16")
        center = np.round(np.asarray(self.positions[0])[::-1] / np.asarray(VOXEL_SIZE)[::-1]).astype(int)
        volume[tuple(center + 2)] = 65535
        spots = _spots(self.positions[:1])
        for halo, expected in ((1, 0), (2, 1)):
            result = self.measure({"stain": volume}, spots, voxel_size=VOXEL_SIZE, signal_halo=halo, n_threads=1)
            self.assertEqual(result.stain_n_saturated[0], expected)

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            self.measure({"a": np.zeros((4, 4, 4)), "b": np.zeros((4, 4, 5))}, _spots([(1.0, 1.0, 1.0)]))
        with self.assertRaises(ValueError):
            self.measure({"a": np.zeros(SHAPE)}, _spots([(1.0, 1.0, 1.0)]), background_radii=(3.0, 2.0))
        with self.assertRaises(ValueError):
            self.measure({"a": np.zeros(SHAPE)}, _spots([(1.0, 1.0, 1.0)]), signal_halo=-1)


class TestSpotIntensityTable(unittest.TestCase):

    def test_tif_input(self):
        import os
        import tempfile

        import imageio.v3 as imageio
        from flamingo_tools.synapse_detection.spot_intensity import spot_intensity_table

        positions = [(10.0, 10.0, 12.0)]
        with tempfile.TemporaryDirectory() as tmp:
            image_path = os.path.join(tmp, "CTBP2.tif")
            imageio.imwrite(image_path, _volume(positions).astype("float32"))
            table_path = os.path.join(tmp, "spots.tsv")
            _spots(positions).to_csv(table_path, sep="\t", index=False)
            output_path = os.path.join(tmp, "out.tsv")
            spot_intensity_table(table_path, [image_path], output_path, input_key=None, voxel_size=VOXEL_SIZE,
                                 signal_halo=0)
            result = pd.read_csv(output_path, sep="\t")
            measures_path = os.path.join(tmp, "measures.tsv")
            spot_intensity_table(table_path, [image_path], measures_path, id_column="spot_id", input_key=None,
                                 voxel_size=VOXEL_SIZE, signal_halo=0)
            measures = pd.read_csv(measures_path, sep="\t")
            with self.assertRaises(ValueError):
                spot_intensity_table(table_path, [image_path], id_column="label_id", input_key=None)
        self.assertIn("CTBP2_snr", result.columns)
        self.assertAlmostEqual(result.CTBP2_signal[0], AMPLITUDE + BACKGROUND, places=2)
        # The id-only output has the layout of an object-measures table: no positions.
        self.assertEqual(measures.columns[0], "spot_id")
        self.assertFalse({"x", "y", "z"} & set(measures.columns))
        self.assertEqual(list(measures.columns[1:]), [c for c in result.columns if c not in ("spot_id", "x", "y", "z")])


if __name__ == "__main__":
    unittest.main()
