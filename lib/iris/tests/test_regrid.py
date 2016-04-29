# (C) British Crown Copyright 2010 - 2016, Met Office
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

from __future__ import (absolute_import, division, print_function)
from six.moves import (filter, input, map, range, zip)  # noqa

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy as np

import iris
from iris import load_cube
from iris.analysis._interpolate_private import regrid_to_max_resolution
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coord_systems import GeogCS


@tests.skip_data
class TestRegrid(tests.IrisTest):
    @staticmethod
    def patch_data(cube):
        # Workaround until regrid can handle factories
        for factory in cube.aux_factories:
            cube.remove_aux_factory(factory)

        # Remove coords that share lat/lon dimensions
        dim = cube.coord_dims(cube.coord('grid_longitude'))[0]
        for coord in cube.coords(contains_dimension=dim, dim_coords=False):
            cube.remove_coord(coord)
        dim = cube.coord_dims(cube.coord('grid_latitude'))[0]
        for coord in cube.coords(contains_dimension=dim, dim_coords=False):
            cube.remove_coord(coord)

    def setUp(self):
        self.theta_p_alt_path = tests.get_data_path(
            ('PP', 'COLPEX', 'small_colpex_theta_p_alt.pp'))
        self.theta_constraint = iris.Constraint('air_potential_temperature')
        self.airpress_constraint = iris.Constraint('air_pressure')
        self.level_constraint = iris.Constraint(model_level_number=1)
        self.multi_level_constraint = iris.Constraint(
            model_level_number=lambda c: 1 <= c < 6)
        self.forecast_constraint = iris.Constraint(
            forecast_period=lambda dt: 0.49 < dt < 0.51)

    def test_regrid_low_dimensional(self):
        theta = load_cube(
            self.theta_p_alt_path,
            (self.theta_constraint
             & self.level_constraint
             & self.forecast_constraint)
        )
        airpress = load_cube(
            self.theta_p_alt_path,
            (self.airpress_constraint
             & self.level_constraint
             & self.forecast_constraint)
        )
        TestRegrid.patch_data(theta)
        TestRegrid.patch_data(airpress)

        # 0-dimensional
        theta_0 = theta[0, 0]
        airpress_0 = airpress[0, 0]
        theta0_regridded = theta_0.regridded(airpress_0, mode='nearest')
        airpress0_regridded = airpress_0.regridded(theta_0, mode='nearest')
        self.assertEqual(theta_0, theta0_regridded)
        self.assertEqual(airpress_0, airpress0_regridded)
        self.assertCMLApproxData(
            theta0_regridded,
            ('regrid', 'theta_on_airpress_0d.cml'))
        self.assertCMLApproxData(
            airpress0_regridded,
            ('regrid', 'airpress_on_theta_0d.cml'))

        # 1-dimensional
        theta_1 = theta[0, 1:4]
        airpress_1 = airpress[0, 0:4]
        self.assertCMLApproxData(
            theta_1.regridded(airpress_1, mode='nearest'),
            ('regrid', 'theta_on_airpress_1d.cml'))
        self.assertCMLApproxData(
            airpress_1.regridded(theta_1, mode='nearest'),
            ('regrid', 'airpress_on_theta_1d.cml'))

        # 2-dimensional
        theta_2 = theta[1:3, 1:4]
        airpress_2 = airpress[0:4, 0:4]
        self.assertCMLApproxData(
            theta_2.regridded(airpress_2, mode='nearest'),
            ('regrid', 'theta_on_airpress_2d.cml'))
        self.assertCMLApproxData(
            airpress_2.regridded(theta_2, mode='nearest'),
            ('regrid', 'airpress_on_theta_2d.cml'))

    def test_regrid_3d(self):
        theta = load_cube(
            self.theta_p_alt_path,
            (self.theta_constraint
             & self.multi_level_constraint
             & self.forecast_constraint)
        )
        airpress = load_cube(
            self.theta_p_alt_path,
            (self.airpress_constraint
             & self.multi_level_constraint
             & self.forecast_constraint)
        )
        TestRegrid.patch_data(theta)
        TestRegrid.patch_data(airpress)

        theta_part = theta[:, 1:3, 1:4]
        airpress_part = airpress[:, 0:4, 0:4]
        self.assertCMLApproxData(
            theta_part.regridded(airpress_part, mode='nearest'),
            ('regrid', 'theta_on_airpress_3d.cml'))
        self.assertCMLApproxData(
            airpress_part.regridded(theta_part, mode='nearest'),
            ('regrid', 'airpress_on_theta_3d.cml'))

    def test_regrid_max_resolution(self):
        low = Cube(np.arange(12).reshape((3, 4)))
        cs = GeogCS(6371229)
        low.add_dim_coord(DimCoord(np.array([-1, 0, 1], dtype=np.int32), 'latitude', units='degrees', coord_system=cs), 0)
        low.add_dim_coord(DimCoord(np.array([-1, 0, 1, 2], dtype=np.int32), 'longitude', units='degrees', coord_system=cs), 1)

        med = Cube(np.arange(20).reshape((4, 5)))
        cs = GeogCS(6371229)
        med.add_dim_coord(DimCoord(np.array([-1, 0, 1, 2], dtype=np.int32), 'latitude', units='degrees', coord_system=cs), 0)
        med.add_dim_coord(DimCoord(np.array([-2, -1, 0, 1, 2], dtype=np.int32), 'longitude', units='degrees', coord_system=cs), 1)

        high = Cube(np.arange(30).reshape((5, 6)))
        cs = GeogCS(6371229)
        high.add_dim_coord(DimCoord(np.array([-2, -1, 0, 1, 2], dtype=np.int32), 'latitude', units='degrees', coord_system=cs), 0)
        high.add_dim_coord(DimCoord(np.array([-2, -1, 0, 1, 2, 3], dtype=np.int32), 'longitude', units='degrees', coord_system=cs), 1)

        cubes = regrid_to_max_resolution([low, med, high], mode='nearest')
        self.assertCMLApproxData(cubes, ('regrid', 'low_med_high.cml'))


class TestRegridBilinear(tests.IrisTest):
    def setUp(self):
        self.cs = GeogCS(6371229)

        # Source cube candidate for regridding.
        cube = Cube(np.arange(12, dtype=np.float32).reshape(3, 4), long_name='unknown')
        cube.units = '1'
        cube.add_dim_coord(DimCoord(np.array([1, 2, 3]), 'latitude', units='degrees', coord_system=self.cs), 0)
        cube.add_dim_coord(DimCoord(np.array([1, 2, 3, 4]), 'longitude', units='degrees', coord_system=self.cs), 1)
        self.source = cube

        # Cube with a smaller grid in latitude and longitude than the source grid by taking the coordinate mid-points.
        cube = Cube(np.arange(6, dtype=np.float).reshape(2, 3))
        cube.units = '1'
        cube.add_dim_coord(DimCoord(np.array([1.5, 2.5]), 'latitude', units='degrees', coord_system=self.cs), 0)
        cube.add_dim_coord(DimCoord(np.array([1.5, 2.5, 3.5]), 'longitude', units='degrees', coord_system=self.cs), 1)
        self.smaller = cube

        # Cube with a larger grid in latitude and longitude than the source grid by taking the coordinate mid-points and extrapolating at extremes.
        cube = Cube(np.arange(20, dtype=np.float).reshape(4, 5))
        cube.units = '1'
        cube.add_dim_coord(DimCoord(np.array([0.5, 1.5, 2.5, 3.5]), 'latitude', units='degrees', coord_system=self.cs), 0)
        cube.add_dim_coord(DimCoord(np.array([0.5, 1.5, 2.5, 3.5, 4.5]), 'longitude', units='degrees', coord_system=self.cs), 1)
        self.larger = cube

    def test_bilinear_smaller_lon_left(self):
        # Anchor smaller grid from the first point in longitude and perform mid-point linear interpolation in latitude.
        self.smaller.coord('longitude').points = self.smaller.coord('longitude').points - 0.5
        self.assertCMLApproxData(self.source.regridded(self.smaller), ('regrid', 'bilinear_smaller_lon_align_left.cml'))

    def test_bilinear_smaller(self):
        # Perform mid-point bilinear interpolation over both latitude and longitude.
        self.assertCMLApproxData(self.source.regridded(self.smaller), ('regrid', 'bilinear_smaller.cml'))

    def test_bilinear_smaller_lon_right(self):
        # Anchor smaller grid from the last point in longitude and perform mid-point linear interpolation in latitude.
        self.smaller.coord('longitude').points = self.smaller.coord('longitude').points + 0.5
        self.assertCMLApproxData(self.source.regridded(self.smaller), ('regrid', 'bilinear_smaller_lon_align_right.cml'))

    def test_bilinear_larger_lon_left(self):
        # Extrapolate first point of longitude with others aligned to source grid, and perform linear interpolation with extrapolation over latitude.
        coord = iris.coords.DimCoord(np.array([0.5, 1, 2, 3, 4]), 'longitude', units='degrees', coord_system=self.cs)
        self.larger.remove_coord('longitude')
        self.larger.add_dim_coord(coord, 1)
        self.assertCMLApproxData(self.source.regridded(self.larger), ('regrid', 'bilinear_larger_lon_extrapolate_left.cml'))

    def test_bilinear_larger(self):
        # Perform mid-point bi-linear interpolation with extrapolation over latitude and longitude.
        self.assertCMLApproxData(self.source.regridded(self.larger), ('regrid', 'bilinear_larger.cml'))

    def test_bilinear_larger_lon_right(self):
        # Extrapolate last point of longitude with others aligned to source grid, and perform linear interpolation with extrapolation over latitude.
        coord = iris.coords.DimCoord(np.array([1, 2, 3, 4, 4.5]), 'longitude', units='degrees', coord_system=self.cs)
        self.larger.remove_coord('longitude')
        self.larger.add_dim_coord(coord, 1)
        self.assertCMLApproxData(self.source.regridded(self.larger), ('regrid', 'bilinear_larger_lon_extrapolate_right.cml'))


if __name__ == "__main__":
    tests.main()
