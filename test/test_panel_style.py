import importlib.util
import os
import re
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURE_DIR = os.path.join(REPO_ROOT, "scripts", "figures")


@unittest.skipIf(importlib.util.find_spec("matplotlib") is None, "the panel style needs matplotlib")
class TestPanelStyle(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import matplotlib
        matplotlib.use("Agg")
        sys.path.insert(0, FIGURE_DIR)
        try:
            import panel_style
        finally:
            sys.path.remove(FIGURE_DIR)
        cls.panel_style = panel_style

    def test_save_panel_size_and_fonts(self):
        width_mm, height_mm = 40, 30
        with tempfile.TemporaryDirectory() as tmp_dir, self.panel_style.style():
            fig, ax = self.panel_style.new_panel(width_mm, height_mm)
            ax.set_xlabel("Length fraction")
            stem = os.path.join(tmp_dir, "panel")
            self.panel_style.save_panel(fig, stem + ".pdf")
            self.assertTrue(os.path.isfile(stem + ".png"))
            with open(stem + ".pdf", "rb") as f:
                pdf = f.read()

        media_box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", pdf)
        self.assertIsNotNone(media_box)
        expected = [size / 25.4 * 72 for size in (width_mm, height_mm)]
        np.testing.assert_allclose([float(value) for value in media_box.groups()], expected, atol=0.5)
        # pdf.fonttype 42 embeds the font program as TrueType, which keeps the text editable.
        self.assertIn(b"/FontFile2", pdf)
        self.assertNotIn(b"/CreationDate", pdf)

    def test_style_does_not_leak(self):
        import matplotlib.pyplot as plt

        before = plt.rcParams["font.size"], plt.rcParams["axes.spines.top"]
        with self.panel_style.style():
            self.assertEqual(plt.rcParams["font.size"], self.panel_style.FONT_SIZE)
            self.assertFalse(plt.rcParams["axes.spines.top"])
        self.assertEqual((plt.rcParams["font.size"], plt.rcParams["axes.spines.top"]), before)

    def test_new_panel_rejects_oversize(self):
        with self.assertRaises(ValueError):
            self.panel_style.new_panel(181, 50)
        with self.assertRaises(ValueError):
            self.panel_style.new_panel(50, 216)

    def test_swarm_offsets(self):
        # 5.0 and 5.1 overlap, 9.0 is alone, the three values near 1 form one group.
        values = [5.0, 1.0, 9.0, 5.1, 1.05, 1.1]
        offsets = self.panel_style.swarm_offsets(values, y_tolerance=0.2, step=0.1)
        self.assertEqual(offsets[2], 0)
        np.testing.assert_allclose(sorted([offsets[0], offsets[3]]), [-0.05, 0.05])
        np.testing.assert_allclose(sorted([offsets[1], offsets[4], offsets[5]]), [-0.1, 0.0, 0.1])

    def test_label_points_do_not_overlap(self):
        import matplotlib.pyplot as plt

        # Four points in a tight cluster and one near the right edge of the axis.
        x, y = [5.0, 5.2, 5.1, 4.9, 9.8], [5.0, 5.1, 4.8, 4.9, 5.0]
        with self.panel_style.style():
            fig, ax = self.panel_style.new_panel(60, 40)
            ax.scatter(x, y)
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            annotations = self.panel_style.label_points(ax, x, y, ["M_01L", "M_01R", "M_05L", "M_05R", "M_02R"])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            boxes = [annotation.get_window_extent(renderer) for annotation in annotations]
            axis_box = ax.get_window_extent(renderer)
        plt.close(fig)
        for i in range(len(boxes)):
            self.assertLessEqual(boxes[i].x1, axis_box.x1 + 1)
            for j in range(i + 1, len(boxes)):
                self.assertFalse(boxes[i].overlaps(boxes[j]), f"labels {i} and {j} overlap")

    def test_text_colors(self):
        ps = self.panel_style
        self.assertEqual(ps.text_color_on(ps.COLORS["dark_gray"]), "#FFFFFF")
        self.assertEqual(ps.text_color_on(ps.COLORS["light_orange"]), ps.COLORS["black"])
        for color in ps.COLORS.values():
            self.assertGreaterEqual(ps.contrast_ratio(ps.readable(color), "#FFFFFF"), ps.MIN_TEXT_CONTRAST)


if __name__ == "__main__":
    unittest.main()
