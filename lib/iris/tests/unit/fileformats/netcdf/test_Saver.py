# (C) British Crown Copyright 2013, Met Office
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
"""Unit tests for the `iris.fileformats.netcdf.Saver` class."""

# Import iris.tests first so that some things can be initialised before
# importing anything else.
import iris.tests as tests

import numpy as np

from iris.coord_systems import GeogCS, TransverseMercator
from iris.coords import DimCoord
from iris.cube import Cube
from iris.fileformats.netcdf import Saver


class Test_write(tests.IrisTest):
    def _transverse_mercator_cube(self, ellipsoid=None):
        data = np.arange(12).reshape(3, 4)
        cube = Cube(data, 'air_pressure_anomaly')
        trans_merc = TransverseMercator(49.0, -2.0, -400000.0, 100000.0,
                                        0.9996012717, ellipsoid)
        coord = DimCoord(range(3), 'projection_y_coordinate', units='m',
                         coord_system=trans_merc)
        cube.add_dim_coord(coord, 0)
        coord = DimCoord(range(4), 'projection_x_coordinate', units='m',
                         coord_system=trans_merc)
        cube.add_dim_coord(coord, 1)
        return cube

    def test_transverse_mercator(self):
        # Create a Cube with a transverse Mercator coordinate system.
        ellipsoid = GeogCS(6377563.396, 6356256.909)
        cube = self._transverse_mercator_cube(ellipsoid)
        with self.temp_filename('nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path, ('unit', 'fileformats', 'netcdf', 'Saver',
                                     'write', 'transverse_mercator.cdl'))

    def test_transverse_mercator_no_ellipsoid(self):
        # Create a Cube with a transverse Mercator coordinate system.
        cube = self._transverse_mercator_cube()
        with self.temp_filename('nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path, ('unit', 'fileformats', 'netcdf', 'Saver',
                                     'write',
                                     'transverse_mercator_no_ellipsoid.cdl'))

    def _simple_cube(self, dtype):
        data = np.arange(12, dtype=dtype).reshape(3, 4)
        points = np.arange(3, dtype=dtype)
        bounds = np.arange(6, dtype=dtype).reshape(3, 2)
        cube = Cube(data, 'air_pressure_anomaly')
        coord = DimCoord(points, bounds=bounds)
        cube.add_dim_coord(coord, 0)
        return cube

    def test_little_endian(self):
        # Create a Cube with little-endian data.
        cube = self._simple_cube('<f4')
        with self.temp_filename('nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path, ('unit', 'fileformats', 'netcdf', 'Saver',
                                     'write', 'endian.cdl'), flags='')

    def test_big_endian(self):
        # Create a Cube with big-endian data.
        cube = self._simple_cube('>f4')
        with self.temp_filename('nc') as nc_path:
            with Saver(nc_path, 'NETCDF4') as saver:
                saver.write(cube)
            self.assertCDL(nc_path, ('unit', 'fileformats', 'netcdf', 'Saver',
                                     'write', 'endian.cdl'), flags='')


if __name__ == "__main__":
    tests.main()
