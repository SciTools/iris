# (C) British Crown Copyright 2013 - 2016, Met Office
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
Test the iris.analysis.interpolate module.

"""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# Import iris tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

import iris.analysis._interpolate_private as interpolate
from iris.coords import DimCoord
from iris.cube import Cube
from iris.tests.test_interpolation import normalise_order


class Test_linear__circular_wrapping(tests.IrisTest):
    def _create_cube(self, longitudes):
        # Return a Cube with circular longitude with the given values.
        data = np.arange(12).reshape((3, 4)) * 0.1
        cube = Cube(data)
        lon = DimCoord(longitudes, standard_name='longitude',
                       units='degrees', circular=True)
        cube.add_dim_coord(lon, 1)
        return cube

    def test_symmetric(self):
        # Check we can interpolate from a Cube defined over [-180, 180).
        cube = self._create_cube([-180, -90, 0, 90])
        samples = [('longitude', np.arange(-360, 720, 45))]
        result = interpolate.linear(cube, samples, extrapolation_mode='nan')
        normalise_order(result)
        self.assertCMLApproxData(result, ('analysis', 'interpolation',
                                          'linear', 'circular_wrapping',
                                          'symmetric'))

    def test_positive(self):
        # Check we can interpolate from a Cube defined over [0, 360).
        cube = self._create_cube([0, 90, 180, 270])
        samples = [('longitude', np.arange(-360, 720, 45))]
        result = interpolate.linear(cube, samples, extrapolation_mode='nan')
        normalise_order(result)
        self.assertCMLApproxData(result, ('analysis', 'interpolation',
                                          'linear', 'circular_wrapping',
                                          'positive'))


if __name__ == "__main__":
    tests.main()
