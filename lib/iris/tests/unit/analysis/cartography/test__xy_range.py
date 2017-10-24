# (C) British Crown Copyright 2015, Met Office
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

"""Unit tests for :func:`iris.analysis.cartography._xy_range`"""

# Import iris.tests first so that some things can be initialised before
# importing anything else.

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

import iris.tests as tests
import iris.tests.stock as stock
import numpy as np
from iris.analysis.cartography import _xy_range


class Test(tests.IrisTest):
    def test_bounds_mismatch(self):
        cube = stock.realistic_3d()
        cube.coord('grid_longitude').guess_bounds()

        with self.assertRaisesRegexp(ValueError, 'bounds'):
            result_non_circ = _xy_range(cube)

    def test_non_circular(self):
        cube = stock.realistic_3d()
        assert not cube.coord('grid_longitude').circular

        result_non_circ = _xy_range(cube)
        self.assertEqual(result_non_circ, ((-5.0, 5.0), (-4.0, 4.0)))

    @tests.skip_data
    def test_geog_cs_circular(self):
        cube = stock.global_pp()
        assert cube.coord('longitude').circular

        result = _xy_range(cube)
        np.testing.assert_array_almost_equal(
            result, ((0, 360), (-90, 90)), decimal=0)

    @tests.skip_data
    def test_geog_cs_regional(self):
        cube = stock.global_pp()
        cube = cube[10:20, 20:30]
        assert not cube.coord('longitude').circular

        result = _xy_range(cube)
        np.testing.assert_array_almost_equal(
            result, ((75, 108.75), (42.5, 65)), decimal=0)


if __name__ == '__main__':
    tests.main()
