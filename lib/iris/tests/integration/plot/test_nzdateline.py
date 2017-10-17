# (C) British Crown Copyright 2017, Met Office
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
"""
Test set up of limited area map extents which bridge the date line.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import iris
# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import matplotlib.pyplot as plt
    from iris.plot import contour, contourf, pcolormesh, pcolor,\
        points, scatter


@tests.skip_plot
@tests.skip_data
class TestExtent(tests.IrisTest):
    def test_dateline(self):
        dpath = tests.get_data_path(['PP', 'nzgust.pp'])
        cube = iris.load_cube(dpath)
        pcolormesh(cube)
        # Ensure that the limited area expected for NZ is set.
        # This is set in longitudes with the datum set to the
        # International Date Line.
        self.assertTrue(-10 < plt.gca().get_xlim()[0] < -5 and
                        5 < plt.gca().get_xlim()[1] < 10)


if __name__ == "__main__":
    tests.main()
