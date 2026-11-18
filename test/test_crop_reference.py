import unittest

import numpy as np


class TestCropReference(unittest.TestCase):

    def test_reference_order(self):
        from flamingo_tools.annotation.crop_reference import reference_candidates, select_reference

        self.assertEqual(reference_candidates("ihc"), ("IHC_v11b", "IHC_v11", "IHC_v10", "IHC_v9"))
        self.assertEqual(reference_candidates("sgn"), ("SGN_v3", "SGN_v2"))
        self.assertEqual(reference_candidates("sgn", resized=True), ("SGN_resized_v2",))
        with self.assertRaises(ValueError):
            reference_candidates("ihc", resized=True)
        sources = {"IHC_v4b": "segmentation", "IHC_v10": "segmentation", "IHC_v11": "segmentation",
                   "IHC_v11b": "image", "SGN_v2": "segmentation", "Vglut3": "image"}
        # IHC_v11b is not a segmentation source here, so the next candidate wins.
        self.assertEqual(select_reference(sources, reference_candidates("ihc")), "IHC_v11")
        self.assertEqual(select_reference(sources, reference_candidates("sgn")), "SGN_v2")
        self.assertEqual(select_reference(sources, ["IHC_v9", "IHC_v4b"]), "IHC_v4b")
        self.assertIsNone(select_reference(sources, ["IHC_v9"]))

    def test_names(self):
        from flamingo_tools.annotation.crop_reference import cell_type_of, crop_candidates

        cochlea, triplet, stain, resized = crop_candidates("G_EK_000233_L_crop_0593-1018-0276_Vglut3")
        self.assertEqual((cochlea, stain, resized), ("G_EK_000233_L", "Vglut3", False))
        np.testing.assert_array_equal(triplet, [593, 1018, 276])
        # A legacy name from the training data tables.
        cochlea, triplet, stain, _ = crop_candidates("M226L_Vglut3_2216-0850-1675")
        self.assertEqual((cochlea, stain), ("M_LR_000226_L", "Vglut3"))
        np.testing.assert_array_equal(triplet, [2216, 850, 1675])
        self.assertIsNone(crop_candidates("171R_Vglut3_apexIHC_HCAT_C1")[1])
        self.assertEqual(cell_type_of("PV"), "sgn")
        self.assertEqual(cell_type_of("Vglut3"), "ihc")

    def test_positions(self):
        from flamingo_tools.annotation.crop_reference import position_hypotheses, read_padded, verify_position

        rng = np.random.default_rng(0)
        volume = rng.integers(0, 1000, size=(200, 120, 120)).astype("uint16")
        shape = (40, 64, 64)
        hypotheses = position_hypotheses(np.array([19.0, 22.8, 38.0]), shape, voxel_size=0.38)
        self.assertEqual(hypotheses[0], (80, 28, 18))
        self.assertEqual(hypotheses[1], (-1, -10, 6))
        self.assertEqual(len(hypotheses), 4)

        def open_array(cochlea, source):
            return volume

        # The crop is the end of a 512 box in z: only the third hypothesis matches.
        start = hypotheses[2]
        image = read_padded(volume, start, shape)
        self.assertEqual(verify_position(image, "c", "Vglut3", hypotheses, open_array), start)
        self.assertIsNone(verify_position(image + 1000, "c", "Vglut3", hypotheses[:1], open_array))

        padded = read_padded(volume, (-5, 100, 0), (10, 30, 4))
        self.assertTrue((padded[:5] == 0).all())
        self.assertTrue((padded[:, 20:] == 0).all())
        np.testing.assert_array_equal(padded[5:, :20], volume[0:5, 100:120, 0:4])


if __name__ == "__main__":
    unittest.main()
