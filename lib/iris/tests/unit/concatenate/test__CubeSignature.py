# (C) British Crown Copyright 2014, Met Office
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

# import iris tests first so that some things can be initialised
# before importing anything else.
import iris.tests as tests

import numpy as np

import iris
from iris._concatenate import _CubeSignature as CubeSignature
from iris.coords import AuxCoord, DimCoord
from iris.cube import Cube
from iris.unit import Unit
from iris.util import new_axis


class Test__coordinate_metadata_equality(tests.IrisTest):
    def setUp(self):
        data = np.arange(60, dtype=np.float32).reshape(5, 3, 4)
        cube = Cube(data, standard_name='air_temperature', units='K')
        # Temporal coordinate.
        t_units = Unit('hours since 1970-01-01 00:00:00', calendar='gregorian')
        t_coord = DimCoord(points=np.arange(5),
                           standard_name='time',
                           units=t_units)
        cube.add_dim_coord(t_coord, 0)
        # Spatial coordinates.
        y_coord = DimCoord(points=np.arange(3),
                           standard_name='latitude',
                           units='degrees')
        cube.add_dim_coord(y_coord, 1)
        x_coord = DimCoord(points=np.arange(4),
                           standard_name='longitude',
                           units='degrees')
        cube.add_dim_coord(x_coord, 2)
        # Scalar coordinates.
        z_coord = AuxCoord([0], "height", units="m")
        cube.add_aux_coord(z_coord)
        # Auxiliary coordinates.
        aux1_coord = AuxCoord([0, 1, 2], long_name='aux1', units='1')
        cube.add_aux_coord(aux1_coord, (1,))
        aux2_coord = AuxCoord(data[0], long_name='aux2', units='1')
        cube.add_aux_coord(aux2_coord, (1, 2))

        # Baseline vector time-series cubes.
        self.series_inc_cube = cube
        self.series_inc = CubeSignature(self.series_inc_cube)

        self.series_dec_cube = self.series_inc_cube.copy()
        self.series_dec_cube.remove_coord('time')
        t_tmp = DimCoord(points=t_coord.points[::-1],
                         standard_name='time',
                         units=t_units)
        self.series_dec_cube.add_dim_coord(t_tmp, 0)
        self.series_dec = CubeSignature(self.series_dec_cube)

        # Scalar time-series cube.
        cube = Cube(data[0], standard_name='air_temperature', units='K')
        cube.add_dim_coord(y_coord, 0)
        cube.add_dim_coord(x_coord, 1)
        cube.add_aux_coord(z_coord)
        cube.add_aux_coord(aux1_coord, (0,))
        cube.add_aux_coord(aux2_coord, (0, 1))
        cube.add_aux_coord(DimCoord(points=t_coord.points[-1] + 1,
                                    standard_name='time',
                                    units=t_units))
        self.scalar_cube = cube

        # Increasing time-series cube.
        cube = Cube(data, standard_name='air_temperature', units='K')
        t_tmp = DimCoord(t_coord.points + t_coord.points.size,
                         standard_name='time',
                         units=t_units)
        cube.add_dim_coord(t_tmp, 0)
        cube.add_dim_coord(y_coord, 1)
        cube.add_dim_coord(x_coord, 2)
        cube.add_aux_coord(z_coord)
        cube.add_aux_coord(aux1_coord, (1,))
        cube.add_aux_coord(aux2_coord, (1, 2))
        self.inc_cube = cube
        self.inc = CubeSignature(self.inc_cube)

        # Decreasing time-series cube.
        t_tmp = DimCoord(t_coord.points[::-1] + t_coord.points.size,
                         standard_name='time',
                         units=t_units)
        self.dec_cube = cube.copy()
        self.dec_cube.remove_coord('time')
        self.dec_cube.add_dim_coord(t_tmp, (0,))
        self.dec = CubeSignature(self.dec_cube)

    def test_scalar_non_common_axis(self):
        scalar = CubeSignature(self.scalar_cube)
        self.assertFalse(self.series_inc.dim_metadata == scalar.dim_metadata)
        self.assertFalse(self.series_inc.aux_metadata == scalar.aux_metadata)
        self.assertFalse(self.series_dec.dim_metadata == scalar.dim_metadata)
        self.assertFalse(self.series_dec.aux_metadata == scalar.aux_metadata)

    def test_scalar_common_axis(self):
        scalar = CubeSignature(new_axis(self.scalar_cube, 'time'))
        self.assertTrue(self.series_inc.dim_metadata == scalar.dim_metadata)
        self.assertTrue(self.series_inc.aux_metadata == scalar.aux_metadata)
        self.assertTrue(self.series_dec.dim_metadata == scalar.dim_metadata)
        self.assertTrue(self.series_dec.aux_metadata == scalar.aux_metadata)

    def test_increasing_common_axis(self):
        self.assertTrue(self.series_inc.dim_metadata == self.inc.dim_metadata)
        self.assertTrue(self.series_inc.aux_metadata == self.inc.aux_metadata)
        self.assertFalse(self.series_dec.dim_metadata == self.inc.dim_metadata)
        self.assertTrue(self.series_dec.aux_metadata == self.inc.aux_metadata)

    def test_decreasing_common_axis(self):
        self.assertFalse(self.series_inc.dim_metadata == self.dec.dim_metadata)
        self.assertTrue(self.series_inc.aux_metadata == self.dec.aux_metadata)
        self.assertTrue(self.series_dec.dim_metadata == self.dec.dim_metadata)
        self.assertTrue(self.series_dec.aux_metadata == self.dec.aux_metadata)


if __name__ == '__main__':
    tests.main()
