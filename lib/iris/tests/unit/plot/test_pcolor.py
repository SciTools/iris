# (C) British Crown Copyright 2014 - 2018, Met Office
#
# This file is part of Iris.
#
# Iris is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the
# Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Iris is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with Iris.  If not, see <http://www.gnu.org/licenses/>.
"""Unit tests for the `iris.plot.pcolor` function."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.tests.stock import simple_2d
from iris.tests.unit.plot import TestGraphicStringCoord
from iris.tests.unit.plot._blockplot_common import \
    MixinStringCoordPlot, Mixin2dCoordsPlot, Mixin2dCoordsContigTol


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
