# (C) British Crown Copyright 2010 - 2013, Met Office
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


@iris.tests.skip_data
class TestCallbacks(tests.IrisTest):
    def test_invalid_signature_callback(self):
        def invalid_callback(cube, ):
            # should never get here
            pass
        fname = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        with self.assertRaises(TypeError):
            iris.load_cube(fname, callback=invalid_callback)

    def test_invalid_return_type_callback(self):
        def invalid_callback(cube, field, filename):
            return 'Not valid to return a string'
        fname = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        with self.assertRaises(TypeError):
            iris.load_cube(fname, callback=invalid_callback)

    def test_non_returning_callback(self):
        def drop_all_callback(cube, field, filename):
            # Callback that filters out every cube
            raise iris.exceptions.IgnoreCubeException()
        fname = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        cubes = iris.load(fname, callback=drop_all_callback)
        # If the callback filters out all cubes, then the cube list
        # should be empty (ie len 0).
        self.assertEqual(len(cubes), 0)

    def test_deprecated_callbacks(self):
        # Tests that callback functions that return values are still supported but that warnings are generated
        
        def returns_cube(cube, field, filename):
            return cube
            
        def returns_no_cube(cube, field, filename):
            return iris.io.NO_CUBE
            
        fname = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        
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
            
        iris.fileformats.grib.hindcast_workaround = True
        fname = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
        cube = iris.load_cube(fname, callback=grib_thing_getter)
        try:
            self.assertCML(cube, ['uri_callback', 'grib_global.cml'])
        finally:
            iris.fileformats.grib.hindcast_workaround = False
    
    def test_pp_callback(self):
        def pp_callback(cube, field, filename):
            cube.local_attributes['filename'] = os.path.basename(filename)
            cube.local_attributes['lbyr'] = field.lbyr
        fname = tests.get_data_path(('PP', 'aPPglob1', 'global.pp'))
        cube = iris.load_cube(fname, callback=pp_callback)
        self.assertCML(cube, ['uri_callback', 'pp_global.cml'])


if __name__ == "__main__":
    tests.main()
