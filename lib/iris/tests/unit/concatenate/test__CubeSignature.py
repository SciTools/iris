# (C) British Crown Copyright 2014 - 2015, Met Office
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
"""Test class :class:`iris._concatenate._CubeSignature`."""

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

from cf_units import Unit
import numpy as np

import iris
from iris._concatenate import _CubeSignature as CubeSignature
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.util import new_axis


class Test__coordinate_dim_metadata_equality(tests.IrisTest):
    def setUp(self):
        nt = 10
        data = np.arange(nt, dtype=np.float32)
        cube = Cube(data, standard_name='air_temperature', units='K')
        # Temporal coordinate.
        t_units = Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        t_coord = DimCoord(points=np.arange(nt),
                           standard_name='time',
                           units=t_units)
        cube.add_dim_coord(t_coord, 0)

        # Increasing 1D time-series cube.
        self.series_inc_cube = cube
        self.series_inc = CubeSignature(self.series_inc_cube)

        # Decreasing 1D time-series cube.
        self.series_dec_cube = self.series_inc_cube.copy()
        self.series_dec_cube.remove_coord('time')
        t_tmp = DimCoord(points=t_coord.points[::-1],
                         standard_name='time',
                         units=t_units)
        self.series_dec_cube.add_dim_coord(t_tmp, 0)
        self.series_dec = CubeSignature(self.series_dec_cube)

        # Scalar 0D time-series cube with scalar time coordinate.
        cube = Cube(0, standard_name='air_temperature', units='K')
        cube.add_aux_coord(DimCoord(points=nt,
                                    standard_name='time',
                                    units=t_units))
        self.scalar_cube = cube

    def test_scalar_non_common_axis(self):
        scalar = CubeSignature(self.scalar_cube)
        self.assertNotEqual(self.series_inc.dim_metadata, scalar.dim_metadata)
        self.assertNotEqual(self.series_dec.dim_metadata, scalar.dim_metadata)

    def test_1d_single_value_common_axis(self):
        # Manually promote scalar time cube to be a 1d cube.
        single = CubeSignature(new_axis(self.scalar_cube, 'time'))
        self.assertEqual(self.series_inc.dim_metadata, single.dim_metadata)
        self.assertEqual(self.series_dec.dim_metadata, single.dim_metadata)

    def test_increasing_common_axis(self):
        series_inc = self.series_inc
        series_dec = self.series_dec
        self.assertEqual(series_inc.dim_metadata, series_inc.dim_metadata)
        self.assertNotEqual(series_inc.dim_metadata, series_dec.dim_metadata)

    def test_decreasing_common_axis(self):
        series_inc = self.series_inc
        series_dec = self.series_dec
        self.assertNotEqual(series_dec.dim_metadata, series_inc.dim_metadata)
        self.assertEqual(series_dec.dim_metadata, series_dec.dim_metadata)

    def test_circular(self):
        series_inc = self.series_inc
        circular_cube = self.series_inc_cube.copy()
        circular_cube.coord('time').circular = True
        circular = CubeSignature(circular_cube)
        self.assertNotEqual(circular.dim_metadata, series_inc.dim_metadata)
        self.assertEqual(circular.dim_metadata, circular.dim_metadata)


if __name__ == '__main__':
    tests.main()
