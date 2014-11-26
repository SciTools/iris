# (C) British Crown Copyright 2010 - 2014, Met Office
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
"""Unit tests for iris.fileformats.grib_save_rules"""

from __future__ import (absolute_import, division, print_function)

# import iris tests first so that some things can be initialised before importing anything else
import iris.tests as tests

import gribapi
import numpy as np
import mock
import warnings

import iris.cube
import iris.coords
import iris.fileformats.grib._save_rules as grib_save_rules


class Test_set_fixed_surfaces(tests.IrisTest):
    @mock.patch.object(gribapi, "grib_set_long")
    def test_altitude_point(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1, 2, 3, 4, 5])
        cube.add_aux_coord(iris.coords.AuxCoord([12345], "altitude", units="m"))

        grib_save_rules.set_fixed_surfaces(cube, grib)

        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 102)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 12345)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_height_point(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1, 2, 3, 4, 5])
        cube.add_aux_coord(iris.coords.AuxCoord([12345], "height", units="m"))

        grib_save_rules.set_fixed_surfaces(cube, grib)

        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 103)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 12345)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_no_vertical(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube([1, 2, 3, 4, 5])
        grib_save_rules.set_fixed_surfaces(cube, grib)
        mock_set_long.assert_any_call(grib, "typeOfFirstFixedSurface", 1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "scaledValueOfFirstFixedSurface", 0)
        mock_set_long.assert_any_call(grib, "typeOfSecondFixedSurface", -1)
        mock_set_long.assert_any_call(grib, "scaleFactorOfSecondFixedSurface", 255)
        mock_set_long.assert_any_call(grib, "scaledValueOfSecondFixedSurface", -1)


class Test_phenomenon(tests.IrisTest):
    @mock.patch.object(gribapi, "grib_set_long")
    def test_phenom_unknown(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        # Force reset of warnings registry to avoid suppression of
        # repeated warnings. warnings.resetwarnings() does not do this.
        if hasattr(grib_save_rules, '__warningregistry__'):
            grib_save_rules.__warningregistry__.clear()
        with warnings.catch_warnings():
            # This should issue a warning about unrecognised data
            warnings.simplefilter("error")
            with self.assertRaises(UserWarning):
                grib_save_rules.set_discipline_and_parameter(cube, grib)
        # do it all again, and this time check the results
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        grib_save_rules.set_discipline_and_parameter(cube, grib)
        mock_set_long.assert_any_call(grib, "discipline", 255)
        mock_set_long.assert_any_call(grib, "parameterCategory", 255)
        mock_set_long.assert_any_call(grib, "parameterNumber", 255)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_phenom_known_standard_name(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]),
                              standard_name='sea_surface_temperature')
        grib_save_rules.set_discipline_and_parameter(cube, grib)
        mock_set_long.assert_any_call(grib, "discipline", 10)
        mock_set_long.assert_any_call(grib, "parameterCategory", 3)
        mock_set_long.assert_any_call(grib, "parameterNumber", 0)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_phenom_known_long_name(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]),
                              long_name='cloud_mixing_ratio')
        grib_save_rules.set_discipline_and_parameter(cube, grib)
        mock_set_long.assert_any_call(grib, "discipline", 0)
        mock_set_long.assert_any_call(grib, "parameterCategory", 1)
        mock_set_long.assert_any_call(grib, "parameterNumber", 22)


class Test_type_of_statistical_processing(tests.IrisTest):
    @mock.patch.object(gribapi, "grib_set_long")
    def test_stats_type_min(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        time_unit = iris.unit.Unit('hours since 1970-01-01 00:00:00')
        time_coord = iris.coords.DimCoord([0.0],
                                          bounds=[0.0, 1],
                                          standard_name='time',
                                          units=time_unit)
        cube.add_aux_coord(time_coord, ())
        cube.add_cell_method(iris.coords.CellMethod('maximum', time_coord))
        grib_save_rules.type_of_statistical_processing(cube, grib, time_coord)
        mock_set_long.assert_any_call(grib, "typeOfStatisticalProcessing", 2)

    @mock.patch.object(gribapi, "grib_set_long")
    def test_stats_type_max(self, mock_set_long):
        grib = None
        cube = iris.cube.Cube(np.array([1.0]))
        time_unit = iris.unit.Unit('hours since 1970-01-01 00:00:00')
        time_coord = iris.coords.DimCoord([0.0],
                                          bounds=[0.0, 1],
                                          standard_name='time',
                                          units=time_unit)
        cube.add_aux_coord(time_coord, ())
        cube.add_cell_method(iris.coords.CellMethod('minimum', time_coord))
        grib_save_rules.type_of_statistical_processing(cube, grib, time_coord)
        mock_set_long.assert_any_call(grib, "typeOfStatisticalProcessing", 3)


if __name__ == "__main__":
    tests.main()
