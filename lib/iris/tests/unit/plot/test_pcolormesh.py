# Copyright Iris contributors
#
# This file is part of Iris and is released under the BSD license.
# See LICENSE in the root of the repository for full licensing details.
"""Unit tests for the `iris.plot.pcolormesh` function."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
from typing import Any

from iris.tests import _shared_utils
from iris.tests.unit.plot import TestGraphicStringCoord
from iris.tests.unit.plot._blockplot_common import (
    Mixin2dCoordsContigTol,
    Mixin2dCoordsPlot,
    MixinStringCoordPlot,
)

if _shared_utils.MPL_AVAILABLE:
    import iris.plot as iplt

    PLOT_FUNCTION_TO_TEST = iplt.pcolormesh


@_shared_utils.skip_plot
class TestStringCoordPlot(MixinStringCoordPlot, TestGraphicStringCoord):
    def blockplot_func(self):
        return PLOT_FUNCTION_TO_TEST


@_shared_utils.skip_plot
class Test2dCoords(Mixin2dCoordsPlot):
    def blockplot_func(self):
        return PLOT_FUNCTION_TO_TEST


@_shared_utils.skip_plot
class Test2dContigTol(Mixin2dCoordsContigTol):
    # Extra call kwargs expected -- unlike 'pcolor', there are none.
    additional_kwargs: dict[str, Any] = {}

    def blockplot_func(self):
        return PLOT_FUNCTION_TO_TEST
