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

import os
import warnings
import datetime

import gribapi
import numpy

import iris
import iris.cube
import iris.coord_systems
import iris.coords


@iris.tests.skip_data
class TestLoadSave(tests.IrisTest):
    # load and save grib
    
    def setUp(self):
        iris.fileformats.grib.hindcast_workaround = True

    def tearDown(self):
        iris.fileformats.grib.hindcast_workaround = False
  
    def save_and_compare(self, source_grib, reference_text):
        """Load and save grib data, generate diffs, compare with expected diffs."""

        # load and save from Iris
        cubes = iris.load(source_grib)

        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        iris.save(cubes, saved_grib)

        # missing reference? (the expected diffs between source_grib and saved_grib)
        if not os.path.exists(reference_text):
            warnings.warn("Creating grib compare reference %s" % reference_text)
            os.system("grib_compare %s %s > %s" % (source_grib, saved_grib, reference_text))
    
        # generate and compare diffs
        compare_text = iris.util.create_temp_filename(suffix='.grib_compare.txt')
        os.system("grib_compare %s %s > %s" % (source_grib, saved_grib, compare_text))
        self.assertTextFile(compare_text, reference_text, "grib_compare output")
        
        os.remove(saved_grib)
        os.remove(compare_text)

    def test_latlon_forecast_plev(self):
        source_grib = tests.get_data_path(("GRIB", "uk_t", "uk_t.grib2"))
        reference_text = tests.get_result_path(("grib_save", "latlon_forecast_plev.grib_compare.txt"))
        self.save_and_compare(source_grib, reference_text)

    def test_rotated_latlon(self):
        source_grib = tests.get_data_path(("GRIB", "rotated_nae_t", "sensible_pole.grib2"))        
        reference_text = tests.get_result_path(("grib_save", "rotated_latlon.grib_compare.txt"))
        # TODO: Investigate small change in test result:
        #       long [iDirectionIncrement]: [109994] != [109993]
        #       Consider the change in dx_dy() to "InDegrees" too.
        self.save_and_compare(source_grib, reference_text)
    
# XXX Addressed in #1118 pending #1039 for hybrid levels
#    def test_hybrid_pressure_levels(self):
#        source_grib = tests.get_data_path(("GRIB", "ecmwf_standard", "t0.grib2"))
#        reference_text = tests.get_result_path(("grib_save", "hybrid_pressure.grib_compare.txt"))
#        self.save_and_compare(source_grib, reference_text)

    def test_time_mean(self):
        # This test for time-mean fields also tests negative forecast time.
        # Because the results depend on the presence of our api patch,
        # we currently have results for both a patched and unpatched api.
        # If the api ever allows -ve ft, we should revert to a single result.
        source_grib = tests.get_data_path(("GRIB", "time_processed",
                                           "time_bound.grib2"))
        reference_text = tests.get_result_path(("grib_save",
                                                "time_mean.grib_compare.txt"))
        # TODO: It's not ideal to have grib patch awareness here...
        try:
            self.save_and_compare(source_grib, reference_text)
        except:
            reference_text = tests.get_result_path((
                                        "grib_save",
                                        "time_mean.grib_compare.FT_PATCH.txt"))
            self.save_and_compare(source_grib, reference_text)


@iris.tests.skip_data
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
       
        new_lats = numpy.append(lat_coord.points[:-1], lat_coord.points[0])  # Irregular
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
        cube.coord("forecast_period").units = iris.unit.Unit("years")
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)
    
    def test_unhandled_vertical(self):
        # unhandled level type
        cube = self._load_basic()
        cube.coord("pressure").rename("not the messiah")
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)


class TestHandmade(tests.IrisTest):

    def _lat_lon_cube_no_time(self):
        """Returns a cube with a latitude and longitude suitable for testing saving to PP/NetCDF etc."""
        cube = iris.cube.Cube(numpy.arange(12, dtype=numpy.int32).reshape((3, 4))) 
        cs = iris.coord_systems.GeogCS(6371229)
        cube.add_dim_coord(iris.coords.DimCoord(numpy.arange(4) * 90 + -180, 'longitude', units='degrees', coord_system=cs), 1) 
        cube.add_dim_coord(iris.coords.DimCoord(numpy.arange(3) * 45 + -90, 'latitude', units='degrees', coord_system=cs), 0) 
        
        return cube
        
    def _cube_time_no_forecast(self):
        cube = self._lat_lon_cube_no_time()
        unit = iris.unit.Unit('hours since epoch', calendar=iris.unit.CALENDAR_GREGORIAN)
        dt = datetime.datetime(2010, 12, 31, 12, 0)
        cube.add_aux_coord(iris.coords.AuxCoord(numpy.array([unit.date2num(dt)], dtype=numpy.float64), 'time', units=unit)) 
        return cube
    
    def _cube_with_forecast(self):
        cube = self._cube_time_no_forecast()
        cube.add_aux_coord(iris.coords.AuxCoord(numpy.array([6], dtype=numpy.int32), 'forecast_period', units='hours')) 
        return cube
    
    def _cube_with_pressure(self):
        cube = self._cube_with_forecast()
        cube.add_aux_coord(iris.coords.DimCoord(numpy.int32(10), 'air_pressure', units='Pa')) 
        return cube

    def _cube_with_time_bounds(self):
        cube = self._cube_with_pressure()
        cube.coord("time").bounds = numpy.array([[0,100]]) 
        return cube
    
    def test_no_time_cube(self):
        cube = self._lat_lon_cube_no_time()
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)
        
    def test_cube_time_no_forecast(self):
        cube = self._cube_time_no_forecast()
        saved_grib = iris.util.create_temp_filename(suffix='.grib2')
        self.assertRaises(iris.exceptions.TranslationError, iris.save, cube, saved_grib)
        os.remove(saved_grib)

    def test_cube_with_forecast(self):
        cube = self._cube_with_forecast()
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


