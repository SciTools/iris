# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.quickplot.scatter` function."""

import pytest

from iris.tests import _shared_utils
from iris.tests.unit.plot import TestGraphicStringCoord

if _shared_utils.MPL_AVAILABLE:
    import iris.quickplot as qplt


@_shared_utils.skip_plot
class TestStringCoordPlot(TestGraphicStringCoord):
    parent_setup = TestGraphicStringCoord._setup

    @pytest.fixture(autouse=True)
    def _setup(self, parent_setup):
        self.cube = self.cube[0, :]

    def test_xaxis_labels(self):
        qplt.scatter(self.cube.coord("str_coord"), self.cube)
        self.assert_bounds_tick_labels("xaxis")

    def test_yaxis_labels(self):
        qplt.scatter(self.cube, self.cube.coord("str_coord"))
        self.assert_bounds_tick_labels("yaxis")
