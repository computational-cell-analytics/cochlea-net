import importlib.util
import os
import re
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(REPO_ROOT, "scripts", "figures")


@unittest.skipIf(
    any(importlib.util.find_spec(name) is None for name in ("matplotlib", "sklearn")),
    "the figure scripts need matplotlib and scikit-learn",
)
class TestPaperFigures(unittest.TestCase):
    """Check the panel registry and the panel selection of paper_cochleanet_figures.py."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, FIGURE_DIR)
        try:
            import paper_cochleanet_figures
        finally:
            sys.path.remove(FIGURE_DIR)
        cls.paper = paper_cochleanet_figures

    def test_panel_ids(self):
        for panel_id in list(self.paper.PANELS) + list(self.paper.VARIANTS):
            self.assertIsNotNone(re.fullmatch(r"S?\d+[a-z]", panel_id), panel_id)
        self.assertLessEqual(set(self.paper.VARIANTS), set(self.paper.PANELS))

    def test_paper_cohorts_are_registered(self):
        from flamingo_tools.postprocessing.sgn_subtype_utils import COCHLEAE
        for name, cochleae in self.paper.PAPER_COHORTS.items():
            for cochlea in cochleae:
                self.assertIn(cochlea, self.paper.COCHLEA_DICT, f"{cochlea} of cohort {name}")
        for name, cochleae in self.paper.SUBTYPE_COHORTS.items():
            for cochlea in cochleae:
                self.assertIn(cochlea, COCHLEAE, f"{cochlea} of subtype cohort {name}")

    def test_no_cohort_lookup(self):
        # The published panels must not follow later changes of util.COHORT_DICT.
        with open(os.path.join(FIGURE_DIR, "paper_cochleanet_figures.py")) as f:
            source = f.read()
        self.assertNotIn("cohort_cochleae", source)
        self.assertNotIn("cohort_postnatal", source)

    def test_select_panels(self):
        select = self.paper.select_panels
        self.assertEqual(select(["2"]), ["2c", "2d"])
        self.assertEqual(select(["s4B"]), ["S4b"])
        self.assertEqual(select(["S3", "2c"]), ["2c", "S3a", "S3b", "S3c", "S3d"])
        self.assertEqual(select(None), list(self.paper.PANELS))
        for token in ("7", "S", "2z", "S10"):
            with self.assertRaises(ValueError):
                select([token])


if __name__ == "__main__":
    unittest.main()
