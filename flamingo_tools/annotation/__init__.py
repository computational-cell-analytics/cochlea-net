"""Correct the instance annotation of IHC and SGN crops in napari, cell by cell.

`flamingo_tools.correct_annotation` finds the cells to correct (`issues`), shows each cell with its neighbours,
and completes the cell from a few edited slices (`fill`). `controller` holds the correction loop and `session`
stores its stages. doc/annotation.md describes the use.

The tool is work in progress: it is ready for tests on IHC crops, and its interface can change.
"""
