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

import iris
import iris.coords
import iris.io
import iris.exceptions


def truipp_filename_callback(cube, field, filename):    
    #add some metadata from the filename
    basename = filename.split(".")[0]
    experiment_id = basename.split("__")[1] 
    cube.add_aux_coord(iris.coords.AuxCoord(experiment_id, long_name='experiment_id', units='no_unit'))

    #make sure we have the time coords we're expecting
    time_coord = cube.coord("time")
    delta_coord = cube.coord("forecast_period")

    if delta_coord.units != "hours":
        raise iris.exceptions.NotYetImplementedError("forecast delta not in hours")
   
    if len(time_coord.points) != 1:
        raise Exception("time not scalar")

    if len(delta_coord.points) != 1:
        raise Exception("delta time not scalar")

    #ensure the dates in the filename match the cube
    forecast_delta = delta_coord.points[0]
    dt = time_coord.units.num2date(time_coord.points[0] - forecast_delta)
    init_time_string = dt.strftime('%Y%m%d%H00')
    
    if not os.path.basename(filename).startswith(init_time_string):
        raise Exception("dates do not match: %s , %s, %s" % (init_time_string, filename[0:12], filename))


@iris.tests.skip_data
class TestCallbacks(tests.IrisTest):
    def test_invalid_signature_callback(self):
        def invalid_callback(cube, ):
            # should never get here
            pass
        
        fname = tests.get_data_path(["PP", "trui", "air_temp_init", "200812011200__qwqu12ff.initanl.pp"])
        self.assertRaises(TypeError, iris.load_strict, fname, callback=invalid_callback)        

    def test_invalid_return_type_callback(self):
        def invalid_callback(cube, field, filename):
            return "Not valid to return a string"
        fname = tests.get_data_path(["PP", "trui", "air_temp_init", "200812011200__qwqu12ff.initanl.pp"])
        
        with self.assertRaises(TypeError):
            iris.load_strict(fname, callback=invalid_callback) 

    def test_non_returning_callback(self):
        def drop_all_callback(c, f, fn):
            # Callback that filters out every cube
            raise iris.exceptions.IgnoreCubeException()
        
        fname = tests.get_data_path(["PP", "trui", "air_temp_init", "200812011200__qwqu12ff.initanl.pp"])
        
        # If the callback filters out all cubes, then the cube list should be empty (ie len 0).
        r = iris.load(fname, callback=drop_all_callback)
        self.assertEqual(len(r), 0)
    
    def test_deprecated_callbacks(self):
        # Tests that callback functions that return values are still supported but that warnings are generated
        
        def returns_cube(cube, field, filename):
            return cube
            
        def returns_no_cube(cube, field, filename):
            return iris.io.NO_CUBE
            
        fname = tests.get_data_path(["PP", "trui", "air_temp_init", "200812011200__qwqu12ff.initanl.pp"])
        
        # Catch all warnings for returns_cube
        with warnings.catch_warnings(record=True) as generated_warnings_cube:
            warnings.simplefilter("always")
            r = iris.load(fname, callback=returns_cube)
            
            # Test that our warnings are present in the generated warnings:
            gen_warnings_cube = [str(x.message) for x in generated_warnings_cube]
            self.assertIn(iris.io.CALLBACK_DEPRECATION_MSG, gen_warnings_cube, "Callback deprecation warning message not issued.")
        
        # Catch all warnings for returns_no_cube
        with warnings.catch_warnings(record=True) as generated_warnings_no_cube:
            warnings.simplefilter("always")  
            r = iris.load(fname, callback=returns_no_cube)
            
            # Test that our warnings are present in the generated warnings:
            gen_warnings_no_cube = [str(x.message) for x in generated_warnings_no_cube]
            self.assertIn(iris.io.CALLBACK_DEPRECATION_MSG, gen_warnings_no_cube, "Callback deprecation warning message not issued.")

    
    def test_grib_callback(self):
        def grib_thing_getter(cube, field, filename):
            cube.add_aux_coord(iris.coords.AuxCoord(field.extra_keys['_periodStartDateTime'], long_name='random element', units='no_unit'))
            
        fname = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
        cube = iris.load_strict(fname, callback=grib_thing_getter)
        self.assertCML(cube, ['uri_callback', 'grib_global.cml'])
    
    def test_pp_callback(self):
        fname = tests.get_data_path(["PP", "trui", "air_temp_T24", "200812011200__qwqg12ff.T24.pp"])
        cube = iris.load_strict(fname, callback=truipp_filename_callback)
        self.assertCML(cube, ['uri_callback', 'trui_t24.cml'])

        fname = tests.get_data_path(["PP", "trui", "air_temp_init", "200812011200__qwqu12ff.initanl.pp"])
        cube = iris.load_strict(fname, callback=truipp_filename_callback)
        self.assertCML(cube, ['uri_callback', 'trui_init.cml'])
   
        
if __name__ == "__main__":
    tests.main()
