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
"""Integration tests for :func:`iris.util.new_axis`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import iris
from iris.util import new_axis


class Test(tests.IrisTest):

    @tests.skip_data
    def test_lazy_data(self):
        filename = tests.get_data_path(('PP', 'globClim1', 'theta.pp'))
        cube = iris.load_cube(filename)
        new_cube = new_axis(cube, 'time')
        self.assertTrue(cube.has_lazy_data())
        self.assertTrue(new_cube.has_lazy_data())
        self.assertEqual(new_cube.shape, (1,) + cube.shape)


if __name__ == "__main__":
    tests.main()
