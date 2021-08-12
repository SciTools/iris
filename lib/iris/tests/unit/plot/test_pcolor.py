# Copyright Iris contributors
#
# This file is part of Iris and is released under the LGPL license.
# See COPYING and COPYING.LESSER in the root of the repository for full
# licensing details.
"""Unit tests for the `iris.plot.pcolor` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests  # isort:skip

from iris.tests.unit.plot import TestGraphicStringCoord
from iris.tests.unit.plot._blockplot_common import (
    Mixin2dCoordsContigTol,
    Mixin2dCoordsPlot,
    MixinStringCoordPlot,
)

if tests.MPL_AVAILABLE:
    import iris.plot as iplt

    PLOT_FUNCTION_TO_TEST = iplt.pcolor


@tests.skip_plot
class TestStringCoordPlot(MixinStringCoordPlot, TestGraphicStringCoord):
    def blockplot_func(self):
        return PLOT_FUNCTION_TO_TEST


@tests.skip_plot
class Test2dCoords(tests.IrisTest, Mixin2dCoordsPlot):
    def setUp(self):
        self.blockplot_setup()

    def blockplot_func(self):
        return PLOT_FUNCTION_TO_TEST


@tests.skip_plot
class Test2dContigTol(tests.IrisTest, Mixin2dCoordsContigTol):
    # Extra call kwargs expected.
    additional_kwargs = dict(antialiased=True, snap=False)

    def blockplot_func(self):
        return PLOT_FUNCTION_TO_TEST


if __name__ == "__main__":
    tests.main()
