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
"""Integration tests for regridding."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests
import biggus
import iris


@tests.skip_data
class TestLazyIntersection(tests.IrisTest):
    def setUp(self):
        path = tests.get_data_path(
            ('NetCDF', 'global', 'xyt',
             'SMALL_hires_wind_u_for_ipcc4.nc'))
        self.cube = iris.load_cube(path)

    def test_intersection(self):
        self.icube = self.cube.intersection(latitude=(49.4, 50.4),
                                            longitude=(199.5, 200.3),
                                            ignore_bounds=True)
        self.assertTrue(self.icube.has_lazy_data())
        lazy_shape = self.icube.shape
        tmp = self.icube.data
        self.assertFalse(self.icube.has_lazy_data())
        shape = self.icube.shape
        self.assertEqual(lazy_shape, shape)


if __name__ == "__main__":
    tests.main()
