# (C) British Crown Copyright 2018, Met Office
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
"""Integration tests for :class:`iris.time.PartialDateTime`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.time import PartialDateTime


class Test(tests.IrisTest):

    @tests.skip_data
    def test_cftime_interface(self):
        # The `netcdf4` Python module introduced new calendar classes by v1.2.7
        # This test is primarily of this interface, so the
        # final test assertion is simple.
        filename = tests.get_data_path(('PP', 'structured', 'small.pp'))
        cube = iris.load_cube(filename)
        pdt = PartialDateTime(year=1992, month=10, day=1, hour=2)
        time_constraint = iris.Constraint(time=lambda cell: cell < pdt)
        sub_cube = cube.extract(time_constraint)
        self.assertEqual(sub_cube.coord('time').points.shape, (1,))


if __name__ == "__main__":
    tests.main()
