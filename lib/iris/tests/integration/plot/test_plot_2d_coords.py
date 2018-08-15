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
Test plots with two dimensional coordinates.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before
# importing anything else
import iris.tests as tests

import iris

# Run tests in no graphics mode if matplotlib is not available.
if tests.MPL_AVAILABLE:
    import iris.quickplot as qplt


@tests.skip_data
def simple_cube():
    path = tests.get_data_path(('NetCDF', 'ORCA2', 'votemper.nc'))
    cube = iris.load_cube(path)
    return cube[0, 0]


@tests.skip_plot
@tests.skip_data
class Test(tests.GraphicsTest):
    def test_2d_coord_bounds(self):
        cube = simple_cube()
        qplt.pcolormesh(cube)
        self.check_graphic()


if __name__ == "__main__":
    tests.main()
