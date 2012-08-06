# (C) British Crown Copyright 2010 - 2012, Met Office
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


# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import numpy

import iris
from iris import load_strict
from iris.analysis.interpolate import regrid_to_max_resolution
from iris.cube import Cube
from iris.coords import DimCoord
from iris.coord_systems import LatLonCS, GeoPosition


def old_style(cube):
    cube.coord("model_level_number")._TEST_COMPAT_override_axis = "z"
    cube.coord("model_level_number")._TEST_COMPAT_definitive = True
    cube.coord("model_level_number")._TEST_COMPAT_force_explicit = True
    cube.coord("sigma")._TEST_COMPAT_override_axis = "z"
    cube.coord("sigma")._TEST_COMPAT_definitive = False
    cube.coord("grid_latitude")._TEST_COMPAT_definitive = True
    cube.coord("grid_latitude")._TEST_COMPAT_force_explicit = True
    cube.coord("grid_longitude")._TEST_COMPAT_definitive = True
    cube.coord("grid_longitude")._TEST_COMPAT_force_explicit = True
    # TODO: Workaround until regrid can handle factories
    cube.remove_aux_factory(cube.aux_factories[0])
    cube.remove_coord("surface_altitude")
    
    return cube


@iris.tests.skip_data
class TestRegrid(tests.IrisTest):
    def setUp(self):
        self.theta_path = tests.get_data_path(('PP', 'COLPEX', 'theta_and_orog.pp'))
        self.uwind_path = tests.get_data_path(('PP', 'COLPEX', 'uwind_and_orog.pp'))
        self.theta_constraint = iris.Constraint('air_potential_temperature')
        self.uwind_constraint = iris.Constraint('eastward_wind')
        self.level_constraint = iris.Constraint(model_level_number=1)
        self.multi_level_constraint = iris.Constraint(model_level_number=lambda c: 1 <= c < 6)
        self.forecast_constraint = iris.Constraint(forecast_period=1.5)
        
    def test_regrid_low_dimensional(self):
        theta = load_strict(self.theta_path, self.theta_constraint & self.level_constraint & self.forecast_constraint)
        uwind = load_strict(self.uwind_path, self.uwind_constraint & self.level_constraint & self.forecast_constraint)
        theta = old_style(theta)
        uwind = old_style(uwind)
        
        # 0-dimensional
        theta0 = theta[0, 0]
        uwind0 = uwind[0, 0]
        self.assertCMLApproxData(theta0.regridded(uwind0, mode='nearest'), ('regrid', 'theta_on_uwind_0d.cml'))
        self.assertCMLApproxData(uwind0.regridded(theta0, mode='neatest'), ('regrid', 'uwind_on_theta_0d.cml'))

        # 1-dimensional
        theta1 = theta[0, 1:4]
        uwind1 = uwind[0, 0:4]
        self.assertCMLApproxData(theta1.regridded(uwind1, mode='nearest'), ('regrid', 'theta_on_uwind_1d.cml'))
        self.assertCMLApproxData(uwind1.regridded(theta1, mode='nearest'), ('regrid', 'uwind_on_theta_1d.cml'))

        # 2-dimensional
        theta2 = theta[1:3, 1:4]
        uwind2 = uwind[0:4, 0:4]
        self.assertCMLApproxData(theta2.regridded(uwind2, mode='nearest'), ('regrid', 'theta_on_uwind_2d.cml'))
        self.assertCMLApproxData(uwind2.regridded(theta2, mode='nearest'), ('regrid', 'uwind_on_theta_2d.cml'))

    def test_regrid_3d(self):
        theta = load_strict(self.theta_path, self.theta_constraint & self.multi_level_constraint & self.forecast_constraint)
        uwind = load_strict(self.uwind_path, self.uwind_constraint & self.multi_level_constraint & self.forecast_constraint)
        theta = old_style(theta)
        uwind = old_style(uwind)

        theta = theta[:, 1:3, 1:4]
        uwind = uwind[:, 0:4, 0:4]
        self.assertCMLApproxData(theta.regridded(uwind, mode='nearest'), ('regrid', 'theta_on_uwind_3d.cml'))
        self.assertCMLApproxData(uwind.regridded(theta, mode='nearest'), ('regrid', 'uwind_on_theta_3d.cml'))

    def test_regrid_max_resolution(self):
        low = Cube(numpy.arange(12).reshape((3, 4)))
        cs = LatLonCS('datum', 'pm', GeoPosition(90, 0), 0)
        low.add_dim_coord(DimCoord(numpy.array([-1, 0, 1], dtype=numpy.int32), 'latitude', units='degrees', coord_system=cs), 0)
        low.add_dim_coord(DimCoord(numpy.array([-1, 0, 1, 2], dtype=numpy.int32), 'longitude', units='degrees', coord_system=cs), 1)
        low.coord("latitude")._TEST_COMPAT_force_explicit = True
        low.coord("latitude")._TEST_COMPAT_definitive = True
        low.coord("longitude")._TEST_COMPAT_force_explicit = True
        low.coord("longitude")._TEST_COMPAT_definitive = True

        med = Cube(numpy.arange(20).reshape((4, 5)))
        cs = LatLonCS('datum', 'pm', GeoPosition(90, 0), 0)
        med.add_dim_coord(DimCoord(numpy.array([-1, 0, 1, 2], dtype=numpy.int32), 'latitude', units='degrees', coord_system=cs), 0)
        med.add_dim_coord(DimCoord(numpy.array([-2, -1, 0, 1, 2], dtype=numpy.int32), 'longitude', units='degrees', coord_system=cs), 1)
        med.coord("latitude")._TEST_COMPAT_force_explicit = True
        med.coord("latitude")._TEST_COMPAT_definitive = True
        med.coord("longitude")._TEST_COMPAT_force_explicit = True
        med.coord("longitude")._TEST_COMPAT_definitive = True

        high = Cube(numpy.arange(30).reshape((5, 6)))
        cs = LatLonCS('datum', 'pm', GeoPosition(90, 0), 0)
        high.add_dim_coord(DimCoord(numpy.array([-2, -1, 0, 1, 2], dtype=numpy.int32), 'latitude', units='degrees', coord_system=cs), 0)
        high.add_dim_coord(DimCoord(numpy.array([-2, -1, 0, 1, 2, 3], dtype=numpy.int32), 'longitude', units='degrees', coord_system=cs), 1)
        high.coord("latitude")._TEST_COMPAT_force_explicit = True
        high.coord("latitude")._TEST_COMPAT_definitive = True
        high.coord("longitude")._TEST_COMPAT_force_explicit = True
        high.coord("longitude")._TEST_COMPAT_definitive = True

        cubes = regrid_to_max_resolution([low, med, high], mode='nearest')
        self.assertCMLApproxData(cubes, ('regrid', 'low_med_high.cml'))


class TestRegridBilinear(tests.IrisTest):
    def setUp(self):
        self.cs = LatLonCS(None, None, GeoPosition(90, 0), 0)
        
        # Source cube candidate for regridding.
        cube = Cube(numpy.arange(12, dtype=numpy.float32).reshape(3, 4), long_name='unknown')
        cube.units = '1'
        cube.add_dim_coord(DimCoord(numpy.array([1, 2, 3]), 'latitude', units='degrees', coord_system=self.cs), 0)
        cube.add_dim_coord(DimCoord(numpy.array([1, 2, 3, 4]), 'longitude', units='degrees', coord_system=self.cs), 1)
        cube.coord("latitude")._TEST_COMPAT_force_explicit = True
        cube.coord("latitude")._TEST_COMPAT_definitive = True
        cube.coord("longitude")._TEST_COMPAT_force_explicit = True
        cube.coord("longitude")._TEST_COMPAT_definitive = True
        self.source = cube
        
        # Cube with a smaller grid in latitude and longitude than the source grid by taking the coordinate mid-points.
        cube = Cube(numpy.arange(6, dtype=numpy.float).reshape(2, 3))
        cube.units = '1'
        cube.add_dim_coord(DimCoord(numpy.array([1.5, 2.5]), 'latitude', units='degrees', coord_system=self.cs), 0)
        cube.add_dim_coord(DimCoord(numpy.array([1.5, 2.5, 3.5]), 'longitude', units='degrees', coord_system=self.cs), 1)
        cube.coord("latitude")._TEST_COMPAT_force_explicit = True
        cube.coord("latitude")._TEST_COMPAT_definitive = True
        cube.coord("longitude")._TEST_COMPAT_force_explicit = True
        cube.coord("longitude")._TEST_COMPAT_definitive = True
        self.smaller = cube
        
        # Cube with a larger grid in latitude and longitude than the source grid by taking the coordinate mid-points and extrapolating at extremes.
        cube = Cube(numpy.arange(20, dtype=numpy.float).reshape(4, 5))
        cube.units = '1'
        cube.add_dim_coord(DimCoord(numpy.array([0.5, 1.5, 2.5, 3.5]), 'latitude', units='degrees', coord_system=self.cs), 0)
        cube.add_dim_coord(DimCoord(numpy.array([0.5, 1.5, 2.5, 3.5, 4.5]), 'longitude', units='degrees', coord_system=self.cs), 1)
        cube.coord("latitude")._TEST_COMPAT_force_explicit = True
        cube.coord("latitude")._TEST_COMPAT_definitive = True
        cube.coord("longitude")._TEST_COMPAT_force_explicit = True
        cube.coord("longitude")._TEST_COMPAT_definitive = True
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
        coord = iris.coords.DimCoord(numpy.array([0.5, 1, 2, 3, 4]), 'longitude', units='degrees', coord_system=self.cs)
        self.larger.remove_coord('longitude')
        self.larger.add_dim_coord(coord, 1)
        self.assertCMLApproxData(self.source.regridded(self.larger), ('regrid', 'bilinear_larger_lon_extrapolate_left.cml'))
        
    def test_bilinear_larger(self):
        # Perform mid-point bi-linear interpolation with extrapolation over latitude and longitude.
        self.assertCMLApproxData(self.source.regridded(self.larger), ('regrid', 'bilinear_larger.cml'))
        
    def test_bilinear_larger_lon_right(self):
        # Extrapolate last point of longitude with others aligned to source grid, and perform linear interpolation with extrapolation over latitude.
        coord = iris.coords.DimCoord(numpy.array([1, 2, 3, 4, 4.5]), 'longitude', units='degrees', coord_system=self.cs)
        self.larger.remove_coord('longitude')
        self.larger.add_dim_coord(coord, 1)
        self.assertCMLApproxData(self.source.regridded(self.larger), ('regrid', 'bilinear_larger_lon_extrapolate_right.cml'))


if __name__ == "__main__":
    tests.main()
