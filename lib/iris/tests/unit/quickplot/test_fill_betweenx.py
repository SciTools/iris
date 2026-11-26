# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.quickplot.fill_betweenx` function."""

from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests import _shared_utils

if _shared_utils.MPL_AVAILABLE:
    import matplotlib.pyplot as plt

    import iris.quickplot as qplt


@_shared_utils.skip_plot
class TestAxisLabels(_shared_utils.GraphicsTest):
    def test_labels(self):
        depth = DimCoord([0.0, 10.0, 20.0], long_name="depth", units="m")
        x1 = Cube([1.0, 2.0, 3.0], long_name="temperature", units="K")
        x1.add_dim_coord(depth, 0)
        x2 = x1 + 3
        qplt.fill_betweenx(depth, x1, x2)
        ax = plt.gca()
        assert ax.get_xlabel() == "Temperature / K"
        assert ax.get_ylabel() == "Depth / m"
