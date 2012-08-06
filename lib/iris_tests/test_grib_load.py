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

import matplotlib.pyplot as plt

import iris
import iris.fileformats.grib
import iris.plot as iplt
import iris.util
import iris.tests.stock


@iris.tests.skip_data
class TestGribLoad(tests.GraphicsTest):
    
    def test_load(self):
                
        cubes = iris.load(tests.get_data_path(('GRIB', 'rotated_uk', "uk_wrongparam.grib1")))
        cubes[0].coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
        cubes[0].coord("forecast_period")._TEST_COMPAT_definitive = True 
        cubes[0].coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
        cubes[0].coord("originating_centre")._TEST_COMPAT_definitive = False 
        cubes[0].coord("pressure")._TEST_COMPAT_definitive = False 
        self.assertCML(cubes, ("grib_load", "rotated.cml"))
        
        cubes = iris.load(tests.get_data_path(('GRIB', "time_processed", "time_bound.grib2")))
        cubes[0].coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
        cubes[0].coord("forecast_period")._TEST_COMPAT_definitive = False 
        cubes[0].coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
        cubes[0].coord("originating_centre")._TEST_COMPAT_definitive = False 
        cubes[0].coord("pressure")._TEST_COMPAT_definitive = False 
        cubes[0].coord("time")._TEST_COMPAT_definitive = False 
        self.assertCML(cubes, ("grib_load", "time_bound.cml"))
        
        cubes = iris.load(tests.get_data_path(('GRIB', "3_layer_viz", "3_layer.grib2")))
        cubes = iris.cube.CubeList([cubes[1], cubes[0], cubes[2]])
        cubes[0].coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
        cubes[0].coord("forecast_period")._TEST_COMPAT_definitive = False 
        cubes[0].coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
        cubes[0].coord("originating_centre")._TEST_COMPAT_definitive = False 
        cubes[0].coord("pressure")._TEST_COMPAT_definitive = False 
        cubes[0].coord("time")._TEST_COMPAT_definitive = False 
        cubes[1].coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
        cubes[1].coord("forecast_period")._TEST_COMPAT_definitive = False 
        cubes[1].coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
        cubes[1].coord("originating_centre")._TEST_COMPAT_definitive = False 
        cubes[1].coord("pressure")._TEST_COMPAT_definitive = False 
        cubes[1].coord("time")._TEST_COMPAT_definitive = False 
        cubes[2].coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
        cubes[2].coord("forecast_period")._TEST_COMPAT_definitive = False 
        cubes[2].coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
        cubes[2].coord("originating_centre")._TEST_COMPAT_definitive = False 
        cubes[2].coord("pressure")._TEST_COMPAT_definitive = False 
        cubes[2].coord("time")._TEST_COMPAT_definitive = False 
        self.assertCML(cubes, ("grib_load", "3_layer.cml"))
        
    def test_y_fastest(self):
        cubes = iris.load(tests.get_data_path(("GRIB", "y_fastest", "y_fast.grib2")))
        cubes[0].coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
        cubes[0].coord("forecast_period")._TEST_COMPAT_definitive = False 
        cubes[0].coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
        cubes[0].coord("originating_centre")._TEST_COMPAT_definitive = False 
        cubes[0].coord("pressure")._TEST_COMPAT_definitive = False 
        cubes[0].coord("time")._TEST_COMPAT_definitive = False 
        self.assertCML(cubes, ("grib_load", "y_fastest.cml"))
        iplt.contourf(cubes[0])
        iplt.gcm(cubes[0]).drawcoastlines()
        plt.title("y changes fastest")
        self.check_graphic()

    def test_ij_directions(self):
        
        def old_compat_load(name):
            cube = iris.load(tests.get_data_path(('GRIB', 'ij_directions', name)))[0]
            cube.coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
            cube.coord("forecast_period")._TEST_COMPAT_definitive = False 
            cube.coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
            cube.coord("originating_centre")._TEST_COMPAT_definitive = False 
            cube.coord("pressure")._TEST_COMPAT_definitive = False 
            cube.coord("time")._TEST_COMPAT_definitive = False 
            return [cube]
        
        cubes = old_compat_load("ipos_jpos.grib2")
        self.assertCML(cubes, ("grib_load", "ipos_jpos.cml"))
        iplt.contourf(cubes[0])
        iplt.gcm(cubes[0]).drawcoastlines()
        plt.title("ipos_jpos cube")
        self.check_graphic()

        cubes = old_compat_load("ipos_jneg.grib2")
        self.assertCML(cubes, ("grib_load", "ipos_jneg.cml"))
        iplt.contourf(cubes[0])
        iplt.gcm(cubes[0]).drawcoastlines()
        plt.title("ipos_jneg cube")
        self.check_graphic()

        cubes = old_compat_load("ineg_jneg.grib2")
        self.assertCML(cubes, ("grib_load", "ineg_jneg.cml"))
        iplt.contourf(cubes[0])
        iplt.gcm(cubes[0]).drawcoastlines()
        plt.title("ineg_jneg cube")
        self.check_graphic()

        cubes = old_compat_load("ineg_jpos.grib2")
        self.assertCML(cubes, ("grib_load", "ineg_jpos.cml"))
        iplt.contourf(cubes[0])
        iplt.gcm(cubes[0]).drawcoastlines()
        plt.title("ineg_jpos cube")
        self.check_graphic()
        
    def test_shape_of_earth(self):
        
        def old_compat_load(name):
            cube = iris.load(tests.get_data_path(('GRIB', 'shape_of_earth', name)))[0]
            cube.coord("forecast_period")._TEST_COMPAT_override_axis = "forecast_period"
            cube.coord("forecast_period")._TEST_COMPAT_definitive = False 
            cube.coord("originating_centre")._TEST_COMPAT_override_axis = "originating_centre"
            cube.coord("originating_centre")._TEST_COMPAT_definitive = False 
            cube.coord("pressure")._TEST_COMPAT_definitive = False 
            cube.coord("time")._TEST_COMPAT_definitive = False 
            return cube
        
        #pre-defined sphere
        cube = old_compat_load("0.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_0.cml"))

        #custom sphere
        cube = old_compat_load("1.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_1.cml"))

        #IAU65 oblate sphere 
        cube = old_compat_load("2.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_2.cml"))

        #custom oblate spheroid (km) 
        cube = old_compat_load("3.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_3.cml"))

        #IAG-GRS80 oblate spheroid 
        cube = old_compat_load("4.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_4.cml"))

        #WGS84
        cube = old_compat_load("5.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_5.cml"))

        #pre-defined sphere
        cube = old_compat_load("6.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_6.cml"))

        #custom oblate spheroid (m)
        cube = old_compat_load("7.grib2")
        self.assertCML(cube, ("grib_load", "earth_shape_7.cml"))

        #grib1 - same as grib2 shape 6, above
        cube = old_compat_load("global.grib1")
        cube.coord("forecast_period")._TEST_COMPAT_definitive = True 
        cube.coord("time")._TEST_COMPAT_definitive = True 
        self.assertCML(cube, ("grib_load", "earth_shape_grib1.cml"))

    def test_custom_rules(self):
        """Test custom rule evaluation."""
        # Default behaviour
#        data_path = tests.get_data_path(('GRIB', 'global_t', 'global.grib2'))
#        cube = iris.load_strict(data_path)
        cube = tests.stock.global_grib2()
        self.assertEqual(cube.name(), 'air_temperature')

        # Custom behaviour
        temp_path = iris.util.create_temp_filename()
        f = open(temp_path, 'w')
        f.write('\n'.join((
            'IF',
            'grib.edition == 2',
            'grib.discipline == 0',
            'grib.parameterCategory == 0',
            'grib.parameterNumber == 0',
            'THEN',
            'CMAttribute("long_name", "customised")',
            'CMAttribute("standard_name", None)')))
        f.close()
        iris.fileformats.grib.add_load_rules(temp_path)
        cube = tests.stock.global_grib2()
        self.assertEqual(cube.name(), 'customised')
        os.remove(temp_path)
        
        # Back to default
        iris.fileformats.grib.reset_load_rules()
        cube = tests.stock.global_grib2()
        self.assertEqual(cube.name(), 'air_temperature')
        
    def test_fp_units(self):

        """Test different units for forecast period (just the ones we care about)."""
        cube = iris.load_strict(tests.get_data_path(('GRIB', 'fp_units', 'minutes.grib2')))
        self.assertEqual(cube.coord("forecast_period").units, "hours")
        self.assertEqual(cube.coord("forecast_period").points[0], 24)

        cube = iris.load_strict(tests.get_data_path(('GRIB', 'fp_units', 'hours.grib2')))
        self.assertEqual(cube.coord("forecast_period").units, "hours")
        self.assertEqual(cube.coord("forecast_period").points[0], 24)

        cube = iris.load_strict(tests.get_data_path(('GRIB', 'fp_units', 'days.grib2')))
        self.assertEqual(cube.coord("forecast_period").units, "hours")
        self.assertEqual(cube.coord("forecast_period").points[0], 24)
        
        cube = iris.load_strict(tests.get_data_path(('GRIB', 'fp_units', 'seconds.grib2')))
        self.assertEqual(cube.coord("forecast_period").units, "hours")
        self.assertEqual(cube.coord("forecast_period").points[0], 24)
 

if __name__ == "__main__":
    tests.main()
    print "finished"
