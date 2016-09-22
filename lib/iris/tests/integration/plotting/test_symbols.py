# (C) British Crown Copyright 2016, Met Office
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
"""Integration tests for the package :mod:`iris.symbols`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa
import six

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    from iris.plot import symbols as iplt_symbols
    import iris.symbols as symbols


@tests.skip_plot
class TestSymbols(tests.GraphicsTest):
    def test_cloud_cover(self):
        iplt_symbols(list(range(10)),
                     [0] * 10,
                     [symbols.CLOUD_COVER[i] for i in range(10)],
                     0.375)
        plt.axis('off')
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
