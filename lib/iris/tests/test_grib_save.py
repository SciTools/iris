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

import os
import warnings
import datetime
from distutils.version import StrictVersion

import cf_units
import numpy as np

import iris
import iris.cube
import iris.coord_systems
import iris.coords

if tests.GRIB_AVAILABLE:
    import gribapi


@tests.skip_data
@tests.skip_grib
class TestLoadSave(tests.TestGribMessage):

    def setUp(self):
        self.skip_keys = []
        if gribapi.__version__ < '1.13':
            self.skip_keys = ['g2grid', 'gridDescriptionSectionPresent',
                              'latitudeOfLastGridPointInDegrees',
                              'iDirectionIncrementInDegrees',
                              'longitudeOfLastGridPointInDegrees',
                              'latLonValues', 'distinctLatitudes',
                              'distinctLongitudes', 'lengthOfHeaders',
                              'values', 'x']

    def test_latlon_forecast_plev(self):
        source_grib = tests.get_data_path(("GRIB", "uk_t", "uk_t.grib2"))
        cubes = iris.load(source_grib)
        with self.temp_filename(suffix='.grib2') as temp_file_path:
            iris.save(cubes, temp_file_path)
            expect_diffs = {'totalLength': (4837, 4832),
                            'productionStatusOfProcessedData': (0, 255),
                            'scaleFactorOfRadiusOfSphericalEarth': (4294967295,
                                                                    0),
                            'shapeOfTheEarth': (0, 1),
                            'scaledValueOfRadiusOfSphericalEarth': (4294967295,
                                                                    6367470),
                            'typeOfGeneratingProcess': (0, 255),
                            'generatingProcessIdentifier': (128, 255),
                            }
            self.assertGribMessageDifference(source_grib, temp_file_path,
                                             expect_diffs, self.skip_keys,
                                             skip_sections=[2])

    def test_rotated_latlon(self):
        source_grib = tests.get_data_path(("GRIB", "rotated_nae_t",
                                           "sensible_pole.grib2"))
        cubes = iris.load(source_grib)
        with self.temp_filename(suffix='.grib2') as temp_file_path:
            iris.save(cubes, temp_file_path)
            expect_diffs = {'totalLength': (648196, 648191),
                            'productionStatusOfProcessedData': (0, 255),
                            'scaleFactorOfRadiusOfSphericalEarth': (4294967295,
                                                                    0),
                            'shapeOfTheEarth': (0, 1),
                            'scaledValueOfRadiusOfSphericalEarth': (4294967295,
                                                                    6367470),
                            'iDirectionIncrement': (109994, 109993),
                            'longitudeOfLastGridPoint': (392109982, 32106370),
                            'latitudeOfLastGridPoint': (19419996, 19419285),
                            'typeOfGeneratingProcess': (0, 255),
                            'generatingProcessIdentifier': (128, 255),
                            }
            self.assertGribMessageDifference(source_grib, temp_file_path,
                                             expect_diffs, self.skip_keys,
                                             skip_sections=[2])

    def test_time_mean(self):
        # This test for time-mean fields also tests negative forecast time.
        try:
            iris.fileformats.grib.hindcast_workaround = True
            source_grib = tests.get_data_path(("GRIB", "time_processed",
                                               "time_bound.grib2"))
            cubes = iris.load(source_grib)
            expect_diffs = {'totalLength': (21232, 21227),
                            'productionStatusOfProcessedData': (0, 255),
                            'scaleFactorOfRadiusOfSphericalEarth': (4294967295,
                                                                    0),
                            'shapeOfTheEarth': (0, 1),
                            'scaledValueOfRadiusOfSphericalEarth': (4294967295,
                                                                    6367470),
                            'longitudeOfLastGridPoint': (356249908, 356249810),
                            'latitudeOfLastGridPoint': (-89999938, -89999944),
                            'typeOfGeneratingProcess': (0, 255),
                            'generatingProcessIdentifier': (128, 255),
                            'typeOfTimeIncrement': (2, 255)
                            }
            self.skip_keys.append('stepType')
            self.skip_keys.append('stepTypeInternal')
            with self.temp_filename(suffix='.grib2') as temp_file_path:
                iris.save(cubes, temp_file_path)
                self.assertGribMessageDifference(source_grib, temp_file_path,
                                                 expect_diffs, self.skip_keys,
                                             skip_sections=[2])
        finally:
            iris.fileformats.grib.hindcast_workaround = False


@tests.skip_data
@tests.skip_grib
class TestCubeSave(tests.IrisTest):
    # save fabricated cubes

    def _load_basic(self):
        path = tests.get_data_path(("GRIB", "uk_t", "uk_t.grib2"))
        return iris.load(path)[0]

    def test_params(self):
        # TODO
        pass

    def test_originating_centre(self):
        # TODO
        pass

    def test_irregular(self):
        cube = self._load_basic()
        lat_coord = cube.coord("latitude")
        cube.remove_coord("latitude")

        new_lats = np.append(lat_coord.points[:-1], lat_coord.points[0])  # Irregular
        cube.add_aux_coord(iris.coords.AuxCoord(new_lats, "latitude", units="degrees", coord_system=lat_coord.coord_system), 0)

        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)

    def test_non_latlon(self):
        cube = self._load_basic()
        cube.coord(dimensions=[0]).coord_system = None
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)

    def test_forecast_period(self):
        # unhandled unit
        cube = self._load_basic()
        cube.coord("forecast_period").units = cf_units.Unit("years")
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)

    def test_unhandled_vertical(self):
        # unhandled level type
        cube = self._load_basic()
        # Adjust the 'pressure' coord to make it into an "unrecognised Z coord"
        p_coord = cube.coord("pressure")
        p_coord.rename("not the messiah")
        p_coord.units = 'K'
        p_coord.attributes['positive'] = 'up'
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        with self.assertRaises(iris.exceptions.TranslationError):
            iris.save(cube, saved_grib)
        os.remove(saved_grib)

    def test_scalar_int32_pressure(self):
        # Make sure we can save a scalar int32 coordinate with unit conversion.
        cube = self._load_basic()
        cube.coord("pressure").points = np.array([200], dtype=np.int32)
        cube.coord("pressure").units = "hPa"
        with self.temp_filename(".grib2") as testfile:
            iris.save(cube, testfile)

    def test_bounded_level(self):
        cube = iris.load_cube(tests.get_data_path(("GRIB", "uk_t",
                                                   "uk_t.grib2")))
        # Changing pressure to altitude due to grib api bug:
        # https://github.com/SciTools/iris/pull/715#discussion_r5901538
        cube.remove_coord("pressure")
        cube.add_aux_coord(iris.coords.AuxCoord(
            1030.0, long_name='altitude', units='m',
            bounds=np.array([111.0, 1949.0])))
        with self.temp_filename(".grib2") as testfile:
            iris.save(cube, testfile)
            with open(testfile, "rb") as saved_file:
                g = gribapi.grib_new_from_file(saved_file)
                self.assertEqual(
                    gribapi.grib_get_double(g,
                                            "scaledValueOfFirstFixedSurface"),
                    111.0)
                self.assertEqual(
                    gribapi.grib_get_double(g,
                                            "scaledValueOfSecondFixedSurface"),
                    1949.0)


@tests.skip_grib
class TestHandmade(tests.IrisTest):

    def _lat_lon_cube_no_time(self):
        """Returns a cube with a latitude and longitude suitable for testing saving to PP/NetCDF etc."""
        cube = iris.cube.Cube(np.arange(12, dtype=np.int32).reshape((3, 4)))
        cs = iris.coord_systems.GeogCS(6371229)
        cube.add_dim_coord(iris.coords.DimCoord(np.arange(4) * 90 + -180, 'longitude', units='degrees', coord_system=cs), 1)
        cube.add_dim_coord(iris.coords.DimCoord(np.arange(3) * 45 + -90, 'latitude', units='degrees', coord_system=cs), 0)

        return cube

    def _cube_time_no_forecast(self):
        cube = self._lat_lon_cube_no_time()
        unit = cf_units.Unit('hours since epoch', calendar=cf_units.CALENDAR_GREGORIAN)
        dt = datetime.datetime(2010, 12, 31, 12, 0)
        cube.add_aux_coord(iris.coords.AuxCoord(np.array([unit.date2num(dt)], dtype=np.float64), 'time', units=unit))
        return cube

    def _cube_with_forecast(self):
        cube = self._cube_time_no_forecast()
        cube.add_aux_coord(iris.coords.AuxCoord(np.array([6], dtype=np.int32), 'forecast_period', units='hours'))
        return cube

    def _cube_with_pressure(self):
        cube = self._cube_with_forecast()
        cube.add_aux_coord(iris.coords.DimCoord(np.int32(10), 'air_pressure', units='Pa'))
        return cube

    def _cube_with_time_bounds(self):
        cube = self._cube_with_pressure()
        cube.coord("time").bounds = np.array([[0, 100]])
        return cube

    def test_no_time_cube(self):
        cube = self._lat_lon_cube_no_time()
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)

    def test_cube_with_time_bounds(self):
        cube = self._cube_with_time_bounds()
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)


if __name__ == "__main__":
    tests.main()
